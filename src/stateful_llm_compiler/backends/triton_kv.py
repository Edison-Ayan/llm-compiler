"""`serve.kv.store` 的 Triton 位置写入 Kernel。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_store_kernel(
    key_buffer,
    value_buffer,
    key,
    value,
    positions,
    capacity,
    key_stride_batch,
    key_stride_head,
    key_stride_token,
    value_stride_batch,
    value_stride_head,
    value_stride_token,
    num_heads: tl.constexpr,
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    program = tl.program_id(0)
    programs_per_batch = num_heads * num_tokens
    batch = program // programs_per_batch
    local = program % programs_per_batch
    head = local // num_tokens
    token = local % num_tokens

    columns = tl.arange(0, block_size)
    position = tl.load(positions + batch) + token
    mask = (columns < head_dim) & (position < capacity)

    key_input_offset = (
        batch * key_stride_batch
        + head * key_stride_head
        + token * key_stride_token
        + columns
    )
    value_input_offset = (
        batch * value_stride_batch
        + head * value_stride_head
        + token * value_stride_token
        + columns
    )
    output_offset = (
        ((batch * capacity + position) * num_heads + head)
        * head_dim
        + columns
    )
    current_key = tl.load(
        key + key_input_offset,
        mask=mask,
        other=0.0,
    )
    current_value = tl.load(
        value + value_input_offset,
        mask=mask,
        other=0.0,
    )
    tl.store(key_buffer + output_offset, current_key, mask=mask)
    tl.store(value_buffer + output_offset, current_value, mask=mask)


def triton_kv_store(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """把 B×H×T×D 的当前 K/V 写入 B×Capacity×H×D Buffer。"""

    if not all(
        tensor.is_cuda
        for tensor in (
            key_buffer,
            value_buffer,
            key,
            value,
            positions,
        )
    ):
        raise ValueError("Triton KV Store 的所有 Tensor 必须位于 CUDA")
    if (
        key_buffer.shape != value_buffer.shape
        or key.shape != value.shape
        or key.ndim != 4
        or key_buffer.ndim != 4
    ):
        raise ValueError("Triton KV Store 收到非法 Shape")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("Triton KV Store 要求 K/V 的 Head Dim 连续")
    if not key_buffer.is_contiguous() or not value_buffer.is_contiguous():
        raise ValueError("Triton KV Store 的 Buffer 必须连续")
    batch, heads, tokens, head_dim = key.shape
    if (
        batch != key_buffer.shape[0]
        or heads != key_buffer.shape[2]
        or head_dim != key_buffer.shape[3]
    ):
        raise ValueError("Triton KV Store 的 Buffer 和当前 K/V 不匹配")

    block_size = triton.next_power_of_2(head_dim)
    _kv_store_kernel[(batch * heads * tokens,)](
        key_buffer,
        value_buffer,
        key,
        value,
        positions,
        key_buffer.shape[1],
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        num_heads=heads,
        num_tokens=tokens,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )
