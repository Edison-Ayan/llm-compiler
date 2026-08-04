"""Qwen2半维旋转RoPE的Triton实现。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    query,
    key,
    cosine,
    sine,
    query_output,
    key_output,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    cosine_stride_batch,
    cosine_stride_token,
    sine_stride_batch,
    sine_stride_token,
    query_output_stride_batch,
    query_output_stride_head,
    query_output_stride_token,
    key_output_stride_batch,
    key_output_stride_head,
    key_output_stride_token,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    """单次Launch覆盖Query和Key的所有Batch、Head与Token行。"""

    row = tl.program_id(0)
    total_heads = num_query_heads + num_kv_heads
    batch = row // (total_heads * num_tokens)
    head_token = row % (total_heads * num_tokens)
    combined_head = head_token // num_tokens
    token = head_token % num_tokens
    is_query = combined_head < num_query_heads
    query_head = combined_head
    key_head = combined_head - num_query_heads
    dimensions = tl.arange(0, block_dim)
    valid = dimensions < head_dim
    half = head_dim // 2

    query_base = (
        batch * query_stride_batch
        + query_head * query_stride_head
        + token * query_stride_token
    )
    key_base = (
        batch * key_stride_batch
        + key_head * key_stride_head
        + token * key_stride_token
    )
    query_values = tl.load(
        query + query_base + dimensions,
        mask=valid & is_query,
        other=0.0,
    )
    key_values = tl.load(
        key + key_base + dimensions,
        mask=valid & ~is_query,
        other=0.0,
    )
    values = tl.where(is_query, query_values, key_values)
    rotated_dimensions = tl.where(
        dimensions < half,
        dimensions + half,
        dimensions - half,
    )
    query_rotated = tl.load(
        query + query_base + rotated_dimensions,
        mask=valid & is_query,
        other=0.0,
    )
    key_rotated = tl.load(
        key + key_base + rotated_dimensions,
        mask=valid & ~is_query,
        other=0.0,
    )
    rotated = tl.where(is_query, query_rotated, key_rotated)
    rotated = tl.where(dimensions < half, -rotated, rotated)

    cosine_offsets = (
        batch * cosine_stride_batch
        + token * cosine_stride_token
        + dimensions
    )
    sine_offsets = (
        batch * sine_stride_batch
        + token * sine_stride_token
        + dimensions
    )
    cosine_values = tl.load(cosine + cosine_offsets, mask=valid)
    sine_values = tl.load(sine + sine_offsets, mask=valid)
    result = values * cosine_values + rotated * sine_values

    query_output_offsets = (
        batch * query_output_stride_batch
        + query_head * query_output_stride_head
        + token * query_output_stride_token
        + dimensions
    )
    key_output_offsets = (
        batch * key_output_stride_batch
        + key_head * key_output_stride_head
        + token * key_output_stride_token
        + dimensions
    )
    tl.store(
        query_output + query_output_offsets,
        result,
        mask=valid & is_query,
    )
    tl.store(
        key_output + key_output_offsets,
        result,
        mask=valid & ~is_query,
    )


@triton.jit
def _rope_flat_kernel(
    query,
    key,
    cosine,
    sine,
    query_output,
    key_output,
    query_stride_batch,
    query_stride_head,
    query_stride_token,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    cosine_stride_batch,
    cosine_stride_token,
    sine_stride_batch,
    sine_stride_token,
    query_output_stride_batch,
    query_output_stride_head,
    query_output_stride_token,
    key_output_stride_batch,
    key_output_stride_head,
    key_output_stride_token,
    query_elements: tl.constexpr,
    total_elements: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    block_elements: tl.constexpr,
):
    """把长Prefill的Q/K逻辑元素合并为大Block，减少Program调度开销。"""

    logical_offsets = (
        tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    )
    valid = logical_offsets < total_elements
    is_query = logical_offsets < query_elements
    query_logical = logical_offsets
    key_logical = logical_offsets - query_elements

    query_dimension = query_logical % head_dim
    query_rest = query_logical // head_dim
    query_token = query_rest % num_tokens
    query_rest = query_rest // num_tokens
    query_head = query_rest % num_query_heads
    query_batch = query_rest // num_query_heads

    key_dimension = key_logical % head_dim
    key_rest = key_logical // head_dim
    key_token = key_rest % num_tokens
    key_rest = key_rest // num_tokens
    key_head = key_rest % num_kv_heads
    key_batch = key_rest // num_kv_heads

    query_offsets = (
        query_batch * query_stride_batch
        + query_head * query_stride_head
        + query_token * query_stride_token
        + query_dimension
    )
    key_offsets = (
        key_batch * key_stride_batch
        + key_head * key_stride_head
        + key_token * key_stride_token
        + key_dimension
    )
    query_mask = valid & is_query
    key_mask = valid & ~is_query
    query_values = tl.load(query + query_offsets, mask=query_mask, other=0.0)
    key_values = tl.load(key + key_offsets, mask=key_mask, other=0.0)
    values = tl.where(is_query, query_values, key_values)

    half = head_dim // 2
    query_rotated_dimension = tl.where(
        query_dimension < half,
        query_dimension + half,
        query_dimension - half,
    )
    key_rotated_dimension = tl.where(
        key_dimension < half,
        key_dimension + half,
        key_dimension - half,
    )
    query_rotated = tl.load(
        query + query_offsets - query_dimension + query_rotated_dimension,
        mask=query_mask,
        other=0.0,
    )
    key_rotated = tl.load(
        key + key_offsets - key_dimension + key_rotated_dimension,
        mask=key_mask,
        other=0.0,
    )
    rotated = tl.where(is_query, query_rotated, key_rotated)
    selected_dimension = tl.where(
        is_query,
        query_dimension,
        key_dimension,
    )
    rotated = tl.where(selected_dimension < half, -rotated, rotated)

    cosine_query_offsets = (
        query_batch * cosine_stride_batch
        + query_token * cosine_stride_token
        + query_dimension
    )
    cosine_key_offsets = (
        key_batch * cosine_stride_batch
        + key_token * cosine_stride_token
        + key_dimension
    )
    sine_query_offsets = (
        query_batch * sine_stride_batch
        + query_token * sine_stride_token
        + query_dimension
    )
    sine_key_offsets = (
        key_batch * sine_stride_batch
        + key_token * sine_stride_token
        + key_dimension
    )
    cosine_offsets = tl.where(
        is_query,
        cosine_query_offsets,
        cosine_key_offsets,
    )
    sine_offsets = tl.where(
        is_query,
        sine_query_offsets,
        sine_key_offsets,
    )
    cosine_values = tl.load(cosine + cosine_offsets, mask=valid)
    sine_values = tl.load(sine + sine_offsets, mask=valid)
    result = values * cosine_values + rotated * sine_values

    query_output_offsets = (
        query_batch * query_output_stride_batch
        + query_head * query_output_stride_head
        + query_token * query_output_stride_token
        + query_dimension
    )
    key_output_offsets = (
        key_batch * key_output_stride_batch
        + key_head * key_output_stride_head
        + key_token * key_output_stride_token
        + key_dimension
    )
    tl.store(
        query_output + query_output_offsets,
        result,
        mask=query_mask,
    )
    tl.store(
        key_output + key_output_offsets,
        result,
        mask=key_mask,
    )


def triton_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对B×H×T×D的Query和Key执行共享位置系数的Qwen2 RoPE。"""

    tensors = (query, key, cosine, sine)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton RoPE的所有Tensor必须位于CUDA")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("Triton RoPE的Query和Key必须是四维Tensor")
    if cosine.ndim != 3 or sine.shape != cosine.shape:
        raise ValueError("Triton RoPE的Cosine和Sine必须是同Shape三维Tensor")
    batch, _, tokens, head_dim = query.shape
    if (
        key.shape[0] != batch
        or key.shape[2:] != (tokens, head_dim)
        or cosine.shape != (batch, tokens, head_dim)
    ):
        raise ValueError("Triton RoPE的Batch、Token或Head Dim不匹配")
    if head_dim <= 0 or head_dim % 2 or head_dim > 65536:
        raise ValueError("Triton RoPE要求不超过65536的正偶数Head Dim")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("Triton RoPE的所有Tensor DType必须一致")
    if query.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise ValueError(f"Triton RoPE不支持DType {query.dtype}")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("Triton RoPE要求所有Tensor的Head Dim连续")

    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    block_dim = max(16, triton.next_power_of_2(head_dim))
    query_heads = query.shape[1]
    kv_heads = key.shape[1]
    common_arguments = (
        query,
        key,
        cosine,
        sine,
        query_output,
        key_output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        cosine.stride(0),
        cosine.stride(1),
        sine.stride(0),
        sine.stride(1),
        query_output.stride(0),
        query_output.stride(1),
        query_output.stride(2),
        key_output.stride(0),
        key_output.stride(1),
        key_output.stride(2),
    )
    if tokens == 1:
        grid = (batch * (query_heads + kv_heads),)
        _rope_kernel[grid](
            *common_arguments,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
            num_tokens=tokens,
            head_dim=head_dim,
            block_dim=block_dim,
            num_warps=1 if head_dim <= 128 else 4,
        )
    else:
        query_elements = query.numel()
        total_elements = query_elements + key.numel()
        block_elements = 256
        grid = (triton.cdiv(total_elements, block_elements),)
        _rope_flat_kernel[grid](
            *common_arguments,
            query_elements=query_elements,
            total_elements=total_elements,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
            num_tokens=tokens,
            head_dim=head_dim,
            block_elements=block_elements,
            num_warps=4,
        )
    return query_output, key_output
