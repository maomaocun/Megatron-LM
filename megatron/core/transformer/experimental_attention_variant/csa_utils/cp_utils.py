# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MCore-facing utilities for the DSv4 THD context-parallel path.

This module owns CP row mapping, boundary exchange, compressor-input layout,
and indexer top-k metadata. It reuses MCore's fused MLA RoPE and calls the
retained compaction kernel; ``csa.py`` calls final-index lowering directly.
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

from . import cp_layout_kernels
from .fused_sparse_attention import batch_of_row, indexer_forward_scores, indexer_topk

# =============================================================================
# RoPE Wrappers
# =============================================================================


def _thd_cp_position_ids(
    cu_seqlens_padded: torch.Tensor, global_start: int, local_rows: int
) -> torch.Tensor:
    """Map a consecutive CP row interval to positions within packed sequences."""
    global_rows = torch.arange(
        int(global_start),
        int(global_start) + int(local_rows),
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    sequence_ids = torch.bucketize(
        global_rows, cu_seqlens_padded[1:], out_int32=True, right=True
    ).clamp_max(cu_seqlens_padded.shape[0] - 2)
    sequence_starts = cu_seqlens_padded[sequence_ids]
    sequence_ends = cu_seqlens_padded[sequence_ids + 1]
    valid_rows = (global_rows >= sequence_starts) & (global_rows < sequence_ends)
    return torch.where(valid_rows, global_rows - sequence_starts, 0)


def apply_thd_cp_local_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows."""
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    recompute_backend = os.environ.get(
        'DSV4_FUSED_ROPE_RECOMPUTE_BACKEND', 'fused'
    ).strip().lower()
    if recompute_backend in {'unfused', 'reference', 'torch'}:
        # The Triton MLA RoPE kernel is physically in-place and has shown
        # rank-selective corruption in the CP + full-recompute path.  Use the
        # same cos/sin values through an out-of-place PyTorch expression for
        # this explicit safe backend. The default still keeps the fused
        # kernel. The formula mirrors
        # ``_mla_rope_fwd_inplace_kernel`` with REMOVE_INTERLEAVING=True.
        cos_pos = cos.index_select(0, position_ids.long()).view(x.shape[0], 1, pos_dim)
        sin_pos = sin.index_select(0, position_ids.long()).view(x.shape[0], 1, pos_dim)
        if inverse:
            sin_pos = -sin_pos
        content, rotary = torch.split(rope_input, [nope_dim, pos_dim], dim=-1)
        half = pos_dim // 2
        x_left = rotary[..., 0::2]
        x_right = rotary[..., 1::2]
        rotated = torch.stack(
            (
                x_left * cos_pos[..., :half] - x_right * sin_pos[..., :half],
                x_right * cos_pos[..., half:] + x_left * sin_pos[..., half:],
            ),
            dim=-1,
        ).flatten(-2)
        output = torch.cat((content, rotated), dim=-1)
    else:
        if inverse or rope_input.requires_grad:
            # The fused kernel is in-place. During checkpoint recompute the
            # q/KV projection output is a live autograd value; mutating that
            # value (or its squeeze view) makes the second forward depend on
            # an already retained graph and can produce rank-selective NaNs in
            # CP. Keep the fast no-grad forward in-place, but give
            # autograd/recompute private storage. The inverse path has always
            # needed the same protection.
            rope_input = rope_input.clone()
        output = fused_mla_rope_inplace(
            rope_input,
            cos,
            sin,
            nope_dim,
            pos_dim,
            cu_seqlens_q=cu_seqlens_padded,
            inverse=inverse,
            remove_interleaving=True,
            position_ids=position_ids,
        )
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


def apply_thd_cp_local_rope_unfused(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    config,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply unfused RoPE to a consecutive interval of packed CP rows."""
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])
    freqs = torch.index_select(rotary_pos_emb, 0, position_ids.long())

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    content, rotary = torch.split(rope_input, [nope_dim, pos_dim], dim=-1)
    rotary = _apply_rotary_pos_emb_bshd(
        rotary,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        mscale=1.0,
        mla_rotary_interleaved=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    output = torch.cat((content, rotary), dim=-1)
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


# =============================================================================
# Boundary Hidden Exchange
# =============================================================================


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange fixed left-boundary windows and scatter gradients back to senders."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup):
        """Receive fixed left-boundary hidden rows needed by this CP rank."""
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if tensor.shape[0] < d_window:
            raise RuntimeError(
                "DSv4 CP boundary exchange requires local rows >= D_window: "
                f"local_rows={tensor.shape[0]}, D_window={d_window}."
            )
        boundary = tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            send_tail = tensor[-d_window:].contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Send boundary gradients back to ranks that own those hidden rows."""
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        d_window = ctx.d_window
        grad_input = grad_boundary.new_zeros(ctx.input_shape)

        ops = []
        if cp_rank > 0:
            send_grad = grad_boundary.contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_grad, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            recv_grad = grad_boundary.new_empty(grad_boundary.shape)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_grad, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-d_window:] = recv_grad
        return grad_input, None, None


def exchange_cp_boundary_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
    csa_window_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Exchange hidden-state rows immediately left of this rank's token block."""
    d_comp = 8 if compress_ratio == 4 else compress_ratio if compress_ratio > 1 else 0
    d_window = max(int(csa_window_size), d_comp)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary_hidden = _LeftBoundaryExchange.apply(hidden_flat, d_window, cp_group)
    return boundary_hidden.reshape((d_window,) + tuple(hidden_states.shape[1:]))


# =============================================================================
# Compressed Metadata And Compressor Inputs
# =============================================================================


def prepare_cp_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity compressor input for this rank's token block.

    Returns:
        ``hidden_compact``: rank-local compressor input, shape
            ``(compact_group_capacity * ratio, ...)``.
        ``compressed_group_ids``: original per-sequence compressed group id for each
            compact group, shape ``(compact_group_capacity,)``. For example,
            with ``ratio=4``, ``comp_id=3`` maps to RoPE position ``12``.
        ``seq_to_rank_row``: map from global sequence-major compressed rows
            to their canonical rank-major all-gather rows.
            If rank 0 owns ``A0, A1`` and rank 1 owns ``B0, B1``, with four
            slots per rank, logical rows ``[A0, A1, B0, B1]`` are stored as
            ``[A0, A1, pad, pad | B0, B1, pad, pad]`` and map to ``[0, 1, 4, 5]``.
    """
    cp_size = int(cp_size)
    ratio = int(ratio)
    d_comp = 8 if ratio == 4 else ratio
    global_start = int(global_start)
    l_local = hidden_local.shape[0]
    group_alignment = 32 // math.gcd(32, ratio)
    c_cap = max(1, (l_local + d_comp) // ratio)
    c_cap = ((c_cap + group_alignment - 1) // group_alignment) * group_alignment
    hidden_compact, compressed_group_ids = cp_layout_kernels.CompressorInputCompact.apply(
        hidden_local, boundary_hidden, cu_seqlens, global_start, ratio, d_comp, c_cap
    )

    # A compressed group belongs to the rank containing its last token. From
    # that rank's first visible compressed row, its fixed-capacity
    # rank-major slot follows directly; no (seq, comp, valid) tensors or repack
    # kernel are needed.
    seq_major_rows = (l_local * cp_size) // ratio
    logical_rows = torch.arange(seq_major_rows, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    n_seq = cu_seqlens.shape[0] - 1
    seq_ids = torch.bucketize(
        logical_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    comp_ids = logical_rows - cu_seqlens_compressed[seq_ids]
    group_last_rows = cu_seqlens[seq_ids] + (comp_ids + 1) * ratio - 1
    owner_ranks = torch.div(group_last_rows, l_local, rounding_mode="floor").clamp_(0, cp_size - 1)

    rank_starts = torch.arange(cp_size, dtype=cu_seqlens.dtype, device=cu_seqlens.device) * l_local
    first_seq_ids = torch.bucketize(
        rank_starts, cu_seqlens[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    first_comp_ids = torch.div(
        (rank_starts - d_comp - cu_seqlens[first_seq_ids]).clamp_min_(0) + ratio - 1,
        ratio,
        rounding_mode="floor",
    )
    first_logical_rows = cu_seqlens_compressed[first_seq_ids] + first_comp_ids
    rank_slots = logical_rows - first_logical_rows[owner_ranks]
    rank_rows = owner_ranks * compressed_group_ids.shape[0] + rank_slots
    seq_to_rank_row = torch.where(logical_rows < cu_seqlens_compressed[-1], rank_rows, -1).to(
        torch.int32
    )
    return hidden_compact, compressed_group_ids, seq_to_rank_row


def _build_cp_indexer_layout_eager(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the indexer's packed local-Q/full-K metadata."""
    # Each real Q segment intersects its sequence with this rank's row interval,
    # while K keeps the sequence's full compressed segment. The final synthetic
    # segment holds CP capacity padding and has zero K rows. Causal offsets
    # restore each non-empty local Q segment's position in the original sequence.
    global_end = global_start + local_rows
    zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    local_starts = cu_seqlens_q[:-1].clamp_min(global_start)
    local_ends = cu_seqlens_q[1:].clamp_max(global_end)
    q_lens = (local_ends - local_starts).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    padding_q = (global_end - cu_seqlens_q[-1].clamp_min(global_start)).clamp_min(0)
    cu_q_topk = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
    cu_k_topk = torch.cat((cu_seqlens_compressed, cu_seqlens_compressed[-1:]))
    q_causal_offsets = torch.cat(
        (torch.where(q_lens > 0, local_starts - cu_seqlens_q[:-1], 0), zero)
    )
    return cu_q_topk, cu_k_topk, q_causal_offsets


def _build_cp_indexer_score_tile_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
    tile_start: int,
    tile_end: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build THD metadata for a compressed-K tile crossing packed documents."""
    global_end = int(global_start) + int(local_rows)
    q_zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    q_start = torch.tensor(global_start, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    q_end = torch.tensor(global_end, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    q_piece_start = torch.maximum(cu_seqlens_q[:-1], q_start)
    q_piece_end = torch.minimum(cu_seqlens_q[1:], q_end)
    q_lens = (q_piece_end - q_piece_start).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    q_padding = (q_end - torch.maximum(cu_seqlens_q[-1], q_start)).clamp_min(0)
    cu_q_tile = torch.cat((q_zero, q_prefix, (q_prefix[-1] + q_padding).view(1)))

    k_zero = torch.zeros(
        (1,), dtype=cu_seqlens_compressed.dtype, device=cu_seqlens_compressed.device
    )
    k_start = torch.tensor(
        int(tile_start), dtype=cu_seqlens_compressed.dtype, device=cu_seqlens_compressed.device
    )
    k_end = torch.tensor(
        int(tile_end), dtype=cu_seqlens_compressed.dtype, device=cu_seqlens_compressed.device
    )
    k_piece_start = torch.maximum(cu_seqlens_compressed[:-1], k_start)
    k_piece_end = torch.minimum(cu_seqlens_compressed[1:], k_end)
    k_lens = (k_piece_end - k_piece_start).clamp_min(0)
    k_prefix = torch.cumsum(k_lens, dim=0, dtype=torch.int32)
    cu_k_tile = torch.cat((k_zero, k_prefix, k_prefix[-1].view(1)))

    q_rel_start = q_piece_start - cu_seqlens_q[:-1]
    k_rel_start = k_piece_start - cu_seqlens_compressed[:-1]
    has_rows = (q_lens > 0) & (k_lens > 0)
    q_causal_offsets = torch.where(
        has_rows,
        q_rel_start - k_rel_start * int(ratio),
        torch.zeros_like(q_rel_start),
    )
    q_causal_offsets = torch.cat((q_causal_offsets, q_zero))
    # Match the synthetic tail segment present in both cumulative layouts so
    # row-to-segment mapping remains valid for capacity padding rows.
    k_lens_with_tail = torch.cat((k_lens, k_lens.new_zeros((1,))))
    k_rel_start_with_tail = torch.cat(
        (k_rel_start, k_rel_start.new_zeros((1,)))
    )
    return (
        cu_q_tile,
        cu_k_tile,
        q_causal_offsets,
        k_lens_with_tail,
        k_rel_start_with_tail,
    )


def _compile_cp_helper(fn):
    """Keep CP metadata compilation opt-in for long-context diagnostics."""
    if os.environ.get('DSV4_DISABLE_V4_HELPER_TORCH_COMPILE', '0').strip().lower() in {
        '1', 'true', 'yes', 'on'
    }:
        return fn
    return torch.compile(fn)


@_compile_cp_helper
def _build_cp_indexer_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed metadata on the compiled main attention path."""
    return _build_cp_indexer_layout_eager(
        cu_seqlens_q, cu_seqlens_compressed, global_start, local_rows
    )


def compute_cp_indexer_topk(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
    use_fused: bool,
    max_seqlen_kv: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Return local top-k and its local-Q/full-K packed layout."""
    topk_width = int(topk_width)
    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return None, None
    max_seqlen_kv = (
        int(max_seqlen_kv) if max_seqlen_kv is not None else int(max_seqlen_q) // int(ratio)
    )
    if max_seqlen_kv == 0:
        return None, None

    global_start = int(global_start)
    l_local = q_indexer_local.shape[0]
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    # CuDNN's radix top-k wrapper has a large fixed workspace on the H200
    # image.  For very long K, keep the same fused score kernel and perform
    # the final top-k over its bounded query chunk in PyTorch. This path is
    # explicit and opt-in; the default remains the production radix kernel.
    topk_backend = os.environ.get('DSV4_INDEXER_TOPK_BACKEND', '').strip().lower()
    if topk_backend in {'score_topk', 'score_tile_topk'}:
        if indexer_softmax_scale != 1.0:
            weights_for_scores = (
                weights_indexer_local.float() * float(indexer_softmax_scale)
            ).to(weights_indexer_local.dtype)
        else:
            weights_for_scores = weights_indexer_local

    if topk_backend == 'score_tile_topk':
        try:
            score_tile_size = int(
                os.environ.get('DSV4_INDEXER_SCORE_TILE_SIZE', '8192') or 8192
            )
        except ValueError:
            score_tile_size = 8192
        score_tile_size = max(score_tile_size, 1)
        best_values = None
        best_indices = None
        key_count = k_indexer_seq_major.shape[0]
        for tile_start in range(0, key_count, score_tile_size):
            tile_end = min(tile_start + score_tile_size, key_count)
            tile_len = tile_end - tile_start
            tile_k = k_indexer_seq_major.narrow(0, tile_start, tile_len)
            (
                tile_q_cu,
                tile_k_cu,
                tile_q_offsets,
                tile_k_lens,
                tile_k_rel_start,
            ) = _build_cp_indexer_score_tile_layout(
                cu_seqlens_q,
                cu_seqlens_compressed,
                global_start,
                l_local,
                tile_start,
                tile_end,
                ratio,
            )
            scores = indexer_forward_scores(
                q_indexer_local,
                tile_k,
                weights_for_scores,
                ratio,
                cu_seqlens_q=tile_q_cu,
                cu_seqlens_kv=tile_k_cu,
                max_seqlen_q=l_local,
                max_seqlen_kv=tile_len,
                q_causal_offsets=tile_q_offsets,
            )
            row_sequence_ids = batch_of_row(tile_q_cu, total_q=l_local)
            row_k_lens = tile_k_lens[row_sequence_ids]
            row_valid = torch.arange(
                l_local, device=q_indexer_local.device, dtype=torch.int64
            ) < tile_q_cu[-2].to(torch.int64)
            key_positions = torch.arange(
                scores.shape[-1], device=scores.device, dtype=torch.int64
            )
            scores = scores.masked_fill(
                key_positions.unsqueeze(0) >= row_k_lens.to(torch.int64).unsqueeze(1),
                float('-inf'),
            )
            selected_width = min(topk_width, tile_len)
            tile_values, tile_indices = torch.topk(
                scores, selected_width, dim=-1
            )
            tile_valid = torch.isfinite(tile_values)
            tile_indices = (
                tile_indices.to(torch.int32)
                + tile_k_rel_start[row_sequence_ids].to(torch.int32).unsqueeze(1)
            )
            tile_valid = tile_valid & row_valid.unsqueeze(1)
            tile_indices = tile_indices.masked_fill(~tile_valid, -1)
            if selected_width < topk_width:
                padding_shape = (l_local, topk_width - selected_width)
                tile_values = torch.cat(
                    (
                        tile_values,
                        torch.full(
                            padding_shape,
                            float('-inf'),
                            dtype=tile_values.dtype,
                            device=tile_values.device,
                        ),
                    ),
                    dim=-1,
                )
                tile_indices = torch.cat(
                    (
                        tile_indices,
                        torch.full(
                            padding_shape,
                            -1,
                            dtype=torch.int32,
                            device=tile_indices.device,
                        ),
                    ),
                    dim=-1,
                )
            if best_values is None:
                best_values = tile_values
                best_indices = tile_indices
            else:
                candidate_values = torch.cat((best_values, tile_values), dim=-1)
                candidate_indices = torch.cat((best_indices, tile_indices), dim=-1)
                best_values, selected = torch.topk(
                    candidate_values, topk_width, dim=-1
                )
                best_indices = torch.gather(candidate_indices, -1, selected)
                best_indices = best_indices.masked_fill(
                    ~torch.isfinite(best_values), -1
                )
                del candidate_values, candidate_indices, selected
            del scores, tile_values, tile_indices, tile_valid
        if best_indices is None:
            return None, None
        return best_indices, None

    if topk_backend == 'score_topk':
        if k_indexer_seq_major.shape[0] == 0:
            return None, None
        try:
            score_chunk_size = int(
                os.environ.get('DSV4_INDEXER_SCORE_CHUNK_SIZE', '128') or 128
            )
        except ValueError:
            score_chunk_size = 128
        score_chunk_size = max(score_chunk_size, 1)

        # The score wrapper returns [query_rows, compressed_K] fp32.  Process
        # bounded subchunks and immediately reduce each one to top-k; keeping
        # the outer query chunk intact would still allocate hundreds of MiB at
        # 256K even though the final indices are only a few MiB.
        index_chunks = []
        rows = q_indexer_local.shape[0]
        for local_start in range(0, rows, score_chunk_size):
            local_end = min(local_start + score_chunk_size, rows)
            # This inner loop deliberately stays eager.  The outer helper is
            # compiled for the normal path, but each score subchunk has a new
            # Python offset/shape; compiling it here creates one Inductor
            # graph per 128-row position at long context.
            sub_q_cu, sub_k_cu, sub_offsets = _build_cp_indexer_layout_eager(
                cu_seqlens_q,
                cu_seqlens_compressed,
                global_start + local_start,
                local_end - local_start,
            )
            scores = indexer_forward_scores(
                q_indexer_local[local_start:local_end],
                k_indexer_seq_major,
                weights_for_scores[local_start:local_end],
                ratio,
                cu_seqlens_q=sub_q_cu,
                cu_seqlens_kv=sub_k_cu,
                # The sub-layout contains only this bounded query interval.
                # Passing the original 256K stride here needlessly makes the
                # CuDNN score kernel size its workspace for the whole sample.
                max_seqlen_q=min(int(max_seqlen_q), local_end - local_start),
                max_seqlen_kv=int(max_seqlen_kv),
                q_causal_offsets=sub_offsets,
            )
            max_score_k = scores.shape[-1]
            selected_width = min(topk_width, max_score_k)
            if selected_width == 0:
                continue
            values, indices = torch.topk(scores, selected_width, dim=-1)
            selected_valid = torch.isfinite(values)
            indices = indices.to(torch.int32).masked_fill(~selected_valid, -1)
            if selected_width < topk_width:
                padding = torch.full(
                    (local_end - local_start, topk_width - selected_width),
                    -1,
                    dtype=torch.int32,
                    device=q_indexer_local.device,
                )
                indices = torch.cat((indices, padding), dim=-1)
            index_chunks.append(indices)
            del scores, values, selected_valid, indices
        if not index_chunks:
            return None, None
        return torch.cat(index_chunks, dim=0), None

    # The experimental score branches build their own eager metadata.  Keep
    # the compiled layout exclusively on the default radix path; otherwise a
    # long-context query chunk would pay for an unused graph specialization.
    cu_q_topk, cu_k_topk, q_causal_offsets = _build_cp_indexer_layout(
        cu_seqlens_q, cu_seqlens_compressed, global_start, l_local
    )

    if not use_fused:
        global_rows = torch.arange(
            global_start,
            global_start + l_local,
            dtype=cu_seqlens_q.dtype,
            device=cu_seqlens_q.device,
        )
        sequence_ids = torch.bucketize(
            global_rows, cu_seqlens_q[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_q.shape[0] - 2)
        positions = global_rows - cu_seqlens_q[sequence_ids]
        visible_k = torch.minimum(
            torch.div(positions + 1, int(ratio), rounding_mode="floor"),
            cu_seqlens_compressed[sequence_ids + 1] - cu_seqlens_compressed[sequence_ids],
        ).clamp_min(0)
        valid_q = (global_rows >= cu_seqlens_q[sequence_ids]) & (
            global_rows < cu_seqlens_q[sequence_ids + 1]
        )

        k_rows = torch.arange(
            k_indexer_seq_major.shape[0],
            dtype=cu_seqlens_compressed.dtype,
            device=cu_seqlens_compressed.device,
        )
        k_sequence_ids = torch.bucketize(
            k_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_compressed.shape[0] - 2)
        k_positions = k_rows - cu_seqlens_compressed[k_sequence_ids]
        output = torch.full(
            (l_local, topk_width), -1, dtype=torch.int32, device=q_indexer_local.device
        )
        selected_width = min(topk_width, k_indexer_seq_major.shape[0])
        for start in range(0, l_local, 128):
            end = min(start + 128, l_local)
            scores = torch.einsum(
                "rhd,kd->rhk", q_indexer_local[start:end].float(), k_indexer_seq_major.float()
            )
            scores = torch.relu(scores) * weights_indexer_local[start:end].float().unsqueeze(-1)
            scores = scores.sum(dim=1) * float(indexer_softmax_scale)
            valid_k = (
                (k_sequence_ids.unsqueeze(0) == sequence_ids[start:end].unsqueeze(1))
                & (k_positions.unsqueeze(0) < visible_k[start:end].unsqueeze(1))
                & valid_q[start:end].unsqueeze(1)
            )
            scores = scores.masked_fill(~valid_k, float("-inf"))
            values, rows = torch.topk(scores, selected_width, dim=-1)
            local_rows = k_positions[rows].to(torch.int32)
            output[start:end, :selected_width] = torch.where(torch.isfinite(values), local_rows, -1)
        return output, (cu_q_topk, cu_k_topk, q_causal_offsets)

    topk, _ = indexer_topk(
        q_indexer_local,
        k_indexer_seq_major,
        weights_indexer_local,
        topk=topk_width,
        ratio=ratio,
        indexer_softmax_scale=indexer_softmax_scale,
        cu_seqlens_q=cu_q_topk,
        cu_seqlens_kv=cu_k_topk,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_kv=int(max_seqlen_kv),
        q_causal_offsets=q_causal_offsets,
    )
    return topk, (cu_q_topk, cu_k_topk, q_causal_offsets)
