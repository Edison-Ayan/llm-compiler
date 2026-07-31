"""把高层 ServeIR 操作 Lower 到 Triton 的 GPU 执行器。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..cost_model import resolve_lowering_plan
from ..execution import (
    ExecutionError,
    PreallocatedKVCacheState,
    ReferenceExecutor,
)
from .triton_attention import triton_decode_attention
from .triton_kv import triton_kv_store
from .triton_rmsnorm import triton_rms_norm


def _inductor_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """与 Benchmark 保持一致的 Inductor RMSNorm 计算图。"""

    variance = tensor.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = tensor.float() * torch.rsqrt(variance + epsilon)
    return (normalized * weight.float()).to(tensor.dtype)


class TritonExecutor(ReferenceExecutor):
    """未注册 Triton Lowering 的 ATen 操作继续使用 PyTorch 参考语义。"""

    def __init__(self) -> None:
        super().__init__()
        self.lowering_trace: list[dict[str, Any]] = []
        self._inductor_kernels: dict[
            tuple[str, torch.dtype, int, float], Any
        ] = {}

    def _serve_rms_norm(
        self,
        operands: list[Any], attributes: dict[str, Any]
    ) -> torch.Tensor:
        if len(operands) != 2:
            raise ExecutionError("serve.rms_norm 需要 input 和 weight")
        tensor, weight = operands
        hidden_size = tensor.shape[-1]
        rows = tensor.numel() // hidden_size
        decision = resolve_lowering_plan(
            attributes.get("lowering_plan"),
            rows,
        )
        self.lowering_trace.append(
            {
                "backend": decision.backend,
                "source": decision.source,
                "rows": rows,
                "profile_rows": decision.profile_rows,
                "estimated_us": decision.estimated_us,
                "num_warps": decision.num_warps,
            }
        )

        epsilon = float(attributes["epsilon"])
        output_dtype = attributes.get("output_dtype")
        if decision.backend == "triton":
            return triton_rms_norm(
                tensor,
                weight,
                epsilon=epsilon,
                output_dtype=output_dtype,
                num_warps=decision.num_warps,
            )
        if decision.backend == "native":
            output = F.rms_norm(
                tensor,
                (hidden_size,),
                weight,
                epsilon,
            )
            return _cast_output(output, output_dtype)
        if decision.backend == "inductor":
            key = (
                str(tensor.device),
                tensor.dtype,
                hidden_size,
                epsilon,
            )
            compiled = self._inductor_kernels.get(key)
            if compiled is None:
                # dynamic=True 允许同一个已编译 Kernel 覆盖同 Hidden Size
                # 下的多个 Rows，避免每个 Bucket 重复触发编译。
                compiled = torch.compile(
                    _inductor_rms_norm,
                    fullgraph=True,
                    dynamic=True,
                    mode="default",
                )
                self._inductor_kernels[key] = compiled
            return _cast_output(
                compiled(tensor, weight, epsilon),
                output_dtype,
            )
        raise ExecutionError(
            f"不支持的 RMSNorm Lowering Backend：{decision.backend}"
        )

    @staticmethod
    def _serve_kv_store(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> PreallocatedKVCacheState:
        if len(operands) != 4 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.kv.store 需要 state、key、value 和 positions"
            )
        state, key, value, positions = operands
        return state.store(
            int(attributes.get("slot", 0)),
            key,
            value,
            positions,
            writer=triton_kv_store,
        )

    @staticmethod
    def _serve_decode_attention(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) != 3 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.decode_attention 需要 state、query 和 mask"
            )
        state, query, attention_mask = operands
        return state.decode_attention(
            int(attributes.get("slot", 0)),
            query,
            attention_mask,
            groups=int(attributes["groups"]),
            scale=float(attributes["scale"]),
            runner=triton_decode_attention,
        )


def _cast_output(
    tensor: torch.Tensor,
    output_dtype: str | torch.dtype | None,
) -> torch.Tensor:
    if output_dtype is None or isinstance(output_dtype, torch.dtype):
        return tensor.to(output_dtype) if output_dtype is not None else tensor
    dtypes = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
    }
    dtype = dtypes.get(output_dtype)
    if dtype is None:
        raise ExecutionError(f"不支持的输出 DType：{output_dtype}")
    return tensor.to(dtype)
