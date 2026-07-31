"""直接读取预分配 KV Buffer 的单 Token Triton Decode Attention。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_attention_kernel(
    query,
    key_buffer,
    value_buffer,
    lengths,
    attention_mask,
    output,
    scale,
    query_stride_batch,
    query_stride_head,
    query_stride_dim,
    mask_stride_batch,
    mask_stride_sequence,
    output_stride_batch,
    output_stride_head,
    output_stride_dim,
    mask_length,
    capacity: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_tokens: tl.constexpr,
    block_head_dim: tl.constexpr,
):
    """每个 Program 处理一个 Batch、一个 Query Head。"""

    program = tl.program_id(0)
    batch = program // num_query_heads
    query_head = program % num_query_heads
    groups = num_query_heads // num_kv_heads
    kv_head = query_head // groups

    dimensions = tl.arange(0, block_head_dim)
    query_offsets = (
        batch * query_stride_batch
        + query_head * query_stride_head
        + dimensions * query_stride_dim
    )
    query_vector = tl.load(
        query + query_offsets,
        mask=dimensions < head_dim,
        other=0.0,
    ).to(tl.float32)
    logical_length = tl.load(lengths + batch)

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((block_head_dim,), dtype=tl.float32)

    # Online Softmax 让显存和寄存器开销只与块大小、Head Dim 有关，
    # 不随 KV Capacity 线性增长。
    for start in range(0, capacity, block_tokens):
        positions = start + tl.arange(0, block_tokens)
        valid_tokens = (
            (positions < logical_length)
            & (positions < mask_length)
            & (positions < capacity)
        )
        kv_offsets = (
            (
                (
                    batch * capacity
                    + positions[:, None]
                )
                * num_kv_heads
                + kv_head
            )
            * head_dim
            + dimensions[None, :]
        )
        matrix_mask = valid_tokens[:, None] & (
            dimensions[None, :] < head_dim
        )
        key_block = tl.load(
            key_buffer + kv_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(
            key_block * query_vector[None, :],
            axis=1,
        ) * scale
        mask_values = tl.load(
            attention_mask
            + batch * mask_stride_batch
            + positions * mask_stride_sequence,
            mask=valid_tokens,
            other=-float("inf"),
        ).to(tl.float32)
        scores += mask_values
        scores = tl.where(valid_tokens, scores, -float("inf"))

        block_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max)
        block_sum = tl.sum(probabilities, axis=0)

        value_block = tl.load(
            value_buffer + kv_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * previous_scale
            + tl.sum(probabilities[:, None] * value_block, axis=0)
        )
        running_sum = running_sum * previous_scale + block_sum
        running_max = next_max

    context = accumulator / running_sum
    output_offsets = (
        batch * output_stride_batch
        + query_head * output_stride_head
        + dimensions * output_stride_dim
    )
    tl.store(
        output + output_offsets,
        context,
        mask=dimensions < head_dim,
    )


def triton_decode_attention(
    query: torch.Tensor,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    lengths: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """执行 B×H×1×D 的 GQA Decode Attention。"""

    tensors = (
        query,
        key_buffer,
        value_buffer,
        lengths,
        attention_mask,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton Decode Attention 的所有 Tensor 必须位于 CUDA")
    if (
        query.ndim != 4
        or query.shape[2] != 1
        or key_buffer.ndim != 4
        or key_buffer.shape != value_buffer.shape
        or attention_mask.ndim != 4
        or attention_mask.shape[2] != 1
    ):
        raise ValueError("Triton Decode Attention 收到非法 Shape")
    batch, query_heads, _, head_dim = query.shape
    capacity = key_buffer.shape[1]
    kv_heads = key_buffer.shape[2]
    if (
        batch != key_buffer.shape[0]
        or batch != attention_mask.shape[0]
        or lengths.shape != (batch,)
        or query_heads % kv_heads
        or head_dim != key_buffer.shape[3]
    ):
        raise ValueError("Triton Decode Attention 的 GQA 或 Batch Shape 不匹配")
    if lengths.dtype != torch.int64:
        raise ValueError("Triton Decode Attention 的 Length 必须是 i64")
    if not key_buffer.is_contiguous() or not value_buffer.is_contiguous():
        raise ValueError("Triton Decode Attention 要求连续 KV Buffer")
    if query.stride(-1) != 1:
        raise ValueError("Triton Decode Attention 要求 Query Head Dim 连续")

    output = torch.empty_like(query)
    block_head_dim = triton.next_power_of_2(head_dim)
    block_tokens = 32
    _decode_attention_kernel[(batch * query_heads,)](
        query,
        key_buffer,
        value_buffer,
        lengths,
        attention_mask,
        output,
        scale,
        query.stride(0),
        query.stride(1),
        query.stride(3),
        attention_mask.stride(0),
        attention_mask.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        attention_mask.shape[-1],
        capacity=capacity,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        block_tokens=block_tokens,
        block_head_dim=block_head_dim,
        num_warps=4,
    )
    return output
