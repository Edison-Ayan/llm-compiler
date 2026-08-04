"""`kernel.triton.linear` 的二维分块 GEMM 实现。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    input_pointer,
    weight_pointer,
    bias_pointer,
    output_pointer,
    rows,
    output_features: tl.constexpr,
    input_features: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """每个 Triton Program 计算输出矩阵中的一个 M×N Tile。"""

    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for start_k in range(0, input_features, block_k):
        current_k = start_k + offsets_k
        input_addresses = (
            input_pointer
            + offsets_m[:, None] * input_features
            + current_k[None, :]
        )
        # PyTorch Linear 的 Weight 物理布局是 N×K；这里按 K×N Tile 加载，
        # 从而直接计算 input @ weight.T，无需单独生成转置 Tensor。
        weight_addresses = (
            weight_pointer
            + offsets_n[None, :] * input_features
            + current_k[:, None]
        )
        input_tile = tl.load(
            input_addresses,
            mask=(offsets_m[:, None] < rows)
            & (current_k[None, :] < input_features),
            other=0.0,
        )
        weight_tile = tl.load(
            weight_addresses,
            mask=(offsets_n[None, :] < output_features)
            & (current_k[:, None] < input_features),
            other=0.0,
        )
        # IEEE 输入精度避免 FP32 Qwen2 差分被默认 TF32 舍入放大。
        accumulator += tl.dot(
            input_tile,
            weight_tile,
            input_precision="ieee",
        )

    if has_bias:
        bias = tl.load(
            bias_pointer + offsets_n,
            mask=offsets_n < output_features,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias[None, :]

    output_addresses = (
        output_pointer
        + offsets_m[:, None] * output_features
        + offsets_n[None, :]
    )
    tl.store(
        output_addresses,
        accumulator,
        mask=(offsets_m[:, None] < rows)
        & (offsets_n[None, :] < output_features),
    )


def triton_linear(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """执行二维或三维动态 Shape Linear，不经过 PyTorch 算子回退。"""

    if not tensor.is_cuda or not weight.is_cuda:
        raise ValueError("Triton Linear 只支持 CUDA Tensor")
    if tensor.ndim not in {2, 3} or weight.ndim != 2:
        raise ValueError("Triton Linear 只支持二维/三维输入和二维 Weight")
    if not tensor.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Triton Linear 当前要求 input 和 weight 连续")
    if tensor.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise ValueError(f"Triton Linear 不支持 DType {tensor.dtype}")
    if tensor.dtype != weight.dtype or tensor.device != weight.device:
        raise ValueError("Triton Linear 的 input 和 weight 必须同 DType、同设备")

    output_features, input_features = weight.shape
    if tensor.shape[-1] != input_features:
        raise ValueError("Triton Linear 的 input K 与 weight K 不匹配")
    if bias is not None:
        if (
            bias.shape != (output_features,)
            or bias.dtype != tensor.dtype
            or bias.device != tensor.device
            or not bias.is_contiguous()
        ):
            raise ValueError("Triton Linear 的 bias Shape、DType 或 Device 不匹配")

    rows = tensor.numel() // input_features
    output = torch.empty(
        (*tensor.shape[:-1], output_features),
        device=tensor.device,
        dtype=tensor.dtype,
    )
    block_m = _tile_size(rows)
    block_n = _tile_size(output_features)
    block_k = min(64, max(16, triton.next_power_of_2(input_features)))
    grid = (
        triton.cdiv(rows, block_m),
        triton.cdiv(output_features, block_n),
    )
    _linear_kernel[grid](
        tensor,
        weight,
        bias if bias is not None else weight,
        output,
        rows,
        output_features=output_features,
        input_features=input_features,
        has_bias=bias is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
    )
    return output


def _tile_size(dimension: int) -> int:
    """为小 Decode 和较大 Prefill Shape 选择简单、稳定的 Tile。"""

    if dimension <= 16:
        return 16
    if dimension <= 32:
        return 32
    return 64
