"""多Token Causal GQA Prefill Attention的Triton Online Softmax实现。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_attention_kernel(
    query,
    key,
    value,
    attention_mask,
    output,
    scale,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    mask_stride_batch,
    mask_stride_query,
    mask_stride_key,
    output_stride_batch,
    output_stride_head,
    output_stride_token,
    sequence_length: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """每个Program计算一个Batch、Query Head和Query Token Block。"""

    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_query_heads
    query_head = batch_head % num_query_heads
    groups = num_query_heads // num_kv_heads
    kv_head = query_head // groups

    query_positions = query_block * block_m + tl.arange(0, block_m)
    dimensions = tl.arange(0, block_d)
    query_offsets = (
        batch * query_stride_batch
        + query_head * query_stride_head
        + query_positions[:, None] * query_stride_token
        + dimensions[None, :]
    )
    query_tile = tl.load(
        query + query_offsets,
        mask=(query_positions[:, None] < sequence_length)
        & (dimensions[None, :] < head_dim),
        other=0.0,
    )

    running_max = tl.full((block_m,), -float("inf"), tl.float32)
    running_sum = tl.zeros((block_m,), tl.float32)
    accumulator = tl.zeros((block_m, block_d), tl.float32)

    for start_n in range(0, sequence_length, block_n):
        key_positions = start_n + tl.arange(0, block_n)
        key_offsets = (
            batch * key_stride_batch
            + kv_head * key_stride_head
            + key_positions[:, None] * key_stride_token
            + dimensions[None, :]
        )
        matrix_mask = (
            (key_positions[:, None] < sequence_length)
            & (dimensions[None, :] < head_dim)
        )
        key_tile = tl.load(
            key + key_offsets,
            mask=matrix_mask,
            other=0.0,
        )
        scores = tl.dot(
            query_tile,
            tl.trans(key_tile),
            input_precision="ieee",
        ).to(tl.float32) * scale

        valid = (
            (query_positions[:, None] < sequence_length)
            & (key_positions[None, :] < sequence_length)
        )
        mask_offsets = (
            batch * mask_stride_batch
            + query_positions[:, None] * mask_stride_query
            + key_positions[None, :] * mask_stride_key
        )
        mask_values = tl.load(
            attention_mask + mask_offsets,
            mask=(query_positions[:, None] < sequence_length)
            & (key_positions[None, :] < sequence_length),
            other=-float("inf"),
        ).to(tl.float32)
        scores = tl.where(valid, scores + mask_values, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        block_sum = tl.sum(probabilities, axis=1)

        value_offsets = (
            batch * value_stride_batch
            + kv_head * value_stride_head
            + key_positions[:, None] * value_stride_token
            + dimensions[None, :]
        )
        value_tile = tl.load(
            value + value_offsets,
            mask=matrix_mask,
            other=0.0,
        )
        accumulator = (
            accumulator * previous_scale[:, None]
            + tl.dot(
                probabilities.to(value_tile.dtype),
                value_tile,
                input_precision="ieee",
            )
        )
        running_sum = running_sum * previous_scale + block_sum
        running_max = next_max

    context = accumulator / running_sum[:, None]
    output_offsets = (
        batch * output_stride_batch
        + query_head * output_stride_head
        + query_positions[:, None] * output_stride_token
        + dimensions[None, :]
    )
    tl.store(
        output + output_offsets,
        context,
        mask=(query_positions[:, None] < sequence_length)
        & (dimensions[None, :] < head_dim),
    )


def triton_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """执行B×H×T×D的动态Prompt Causal GQA Attention。"""

    tensors = (query, key, value, attention_mask)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton Prefill Attention的所有Tensor必须位于CUDA")
    if (
        query.ndim != 4
        or key.ndim != 4
        or key.shape != value.shape
        or attention_mask.ndim != 4
        or attention_mask.shape[1] != 1
    ):
        raise ValueError("Triton Prefill Attention收到非法Shape")
    batch, query_heads, tokens, head_dim = query.shape
    kv_batch, kv_heads, kv_tokens, kv_head_dim = key.shape
    if (
        batch != kv_batch
        or tokens != kv_tokens
        or head_dim != kv_head_dim
        or query_heads % kv_heads
        or attention_mask.shape != (batch, 1, tokens, tokens)
    ):
        raise ValueError("Triton Prefill Attention的GQA或Token Shape不匹配")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise ValueError("Triton Prefill Attention的Q/K/V DType必须一致")
    if query.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise ValueError(f"Triton Prefill Attention不支持DType {query.dtype}")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("Triton Prefill Attention要求Head Dim连续")
    if scale <= 0:
        raise ValueError("Triton Prefill Attention的scale必须为正数")

    output = torch.empty_like(query)
    block_m = 16 if tokens <= 16 else 32
    block_n = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    grid = (triton.cdiv(tokens, block_m), batch * query_heads)
    _prefill_attention_kernel[grid](
        query,
        key,
        value,
        attention_mask,
        output,
        scale,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        attention_mask.stride(0),
        attention_mask.stride(2),
        attention_mask.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        sequence_length=tokens,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=4,
    )
    return output
