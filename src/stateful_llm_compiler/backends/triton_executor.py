"""把高层 ServeIR 操作 Lower 到 Triton 的 GPU 执行器。"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ..cost_model import resolve_lowering_plan
from ..execution import (
    ExecutionError,
    ExecutionResult,
    PreallocatedKVCacheState,
    ReferenceExecutor,
)
from ..ir import Module
from ..lowering import require_full_lowering
from .triton_attention import triton_decode_attention
from .triton_kv import triton_kv_store
from .triton_linear import triton_linear
from .triton_prefill_attention import triton_prefill_attention
from .triton_rmsnorm import triton_rms_norm
from .triton_rope import triton_rope


def _inductor_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    round_before_weight: bool,
) -> torch.Tensor:
    """与 Benchmark 保持一致的 Inductor RMSNorm 计算图。"""

    variance = tensor.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = tensor.float() * torch.rsqrt(variance + epsilon)
    if round_before_weight:
        return weight * normalized.to(tensor.dtype)
    return (normalized * weight.float()).to(tensor.dtype)


class TritonExecutor(ReferenceExecutor):
    """执行 Triton KernelIR；兼容模式仍允许参考执行器承接旧图。"""

    def __init__(self, *, strict: bool = False) -> None:
        super().__init__()
        self.strict = strict
        self.lowering_trace: list[dict[str, Any]] = []
        self._inductor_kernels: dict[
            tuple[str, torch.dtype, int, float, bool], Any
        ] = {}

    def run(
        self,
        module: Module,
        arguments: Sequence[Any],
        *,
        function_name: str | None = None,
    ) -> ExecutionResult:
        if self.strict:
            # 在执行任何节点之前检查整图，禁止运行到中途才发现 eager 回退。
            require_full_lowering(module)
        return super().run(
            module,
            arguments,
            function_name=function_name,
        )

    def _execute_operation(
        self,
        name: str,
        attributes: dict[str, Any],
        operands,
        environment: dict[str, Any],
    ) -> Any:
        runtime_operands = [environment[value.name] for value in operands]
        if name == "kernel.triton.rms_norm":
            return self._kernel_triton_rms_norm(
                runtime_operands,
                attributes,
            )
        if name == "kernel.cuda.rms_norm":
            return self._kernel_cuda_rms_norm(
                runtime_operands,
                attributes,
            )
        if name == "kernel.triton.linear":
            return self._kernel_triton_linear(runtime_operands, attributes)
        if name == "kernel.cublas.linear":
            return self._kernel_cublas_linear(runtime_operands, attributes)
        if name == "kernel.triton.prefill_attention":
            return self._kernel_triton_prefill_attention(
                runtime_operands,
                attributes,
            )
        if name == "kernel.cuda.prefill_attention":
            return self._kernel_cuda_prefill_attention(
                runtime_operands,
                attributes,
            )
        if name == "kernel.triton.rope":
            return self._kernel_triton_rope(runtime_operands, attributes)
        if name == "kernel.cuda.rope":
            return self._kernel_cuda_rope(runtime_operands, attributes)
        if name == "kernel.triton.kv_store":
            return self._serve_kv_store(runtime_operands, attributes)
        if name == "kernel.triton.kv_prefill_store":
            return self._kernel_triton_kv_prefill_store(
                runtime_operands,
                attributes,
            )
        if name == "kernel.triton.decode_attention":
            return self._serve_decode_attention(runtime_operands, attributes)
        if name == "kernel.cuda.decode_attention":
            return self._kernel_cuda_decode_attention(
                runtime_operands,
                attributes,
            )
        if name == "runtime.kv.length":
            return super()._serve_kv_length(runtime_operands, attributes)
        if name == "runtime.kv.advance":
            return super()._serve_kv_advance(runtime_operands, attributes)
        if name == "runtime.kv.init":
            return super()._serve_kv_init(runtime_operands, attributes)
        return super()._execute_operation(
            name,
            attributes,
            operands,
            environment,
        )

    @staticmethod
    def _kernel_triton_linear(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) not in {2, 3}:
            raise ExecutionError(
                "kernel.triton.linear 需要 input、weight 和可选 bias"
            )
        tensor, weight = operands[:2]
        bias = operands[2] if len(operands) == 3 else None
        if bool(attributes.get("has_bias")) != (bias is not None):
            raise ExecutionError(
                "kernel.triton.linear 的 has_bias 与运行时参数不一致"
            )
        return triton_linear(tensor, weight, bias)

    @staticmethod
    def _kernel_cuda_rms_norm(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        """保留官方归约、Rsqrt和BF16舍入顺序的CUDA RMSNorm路径。"""

        _require_cuda_tensors(operands, "kernel.cuda.rms_norm")
        return ReferenceExecutor._serve_rms_norm(operands, attributes)

    @staticmethod
    def _kernel_cublas_linear(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        """执行编译器显式选择的cuBLAS GEMM库调用。"""

        if len(operands) not in {2, 3}:
            raise ExecutionError(
                "kernel.cublas.linear需要input、weight和可选bias"
            )
        tensor, weight = operands[:2]
        bias = operands[2] if len(operands) == 3 else None
        if bool(attributes.get("has_bias")) != (bias is not None):
            raise ExecutionError(
                "kernel.cublas.linear的has_bias与运行时参数不一致"
            )
        return F.linear(tensor, weight, bias)

    @staticmethod
    def _kernel_triton_prefill_attention(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) != 4:
            raise ExecutionError(
                "kernel.triton.prefill_attention需要Q、K、V和mask"
            )
        return triton_prefill_attention(
            *operands,
            scale=float(attributes["scale"]),
        )

    @staticmethod
    def _kernel_cuda_prefill_attention(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        """用显式CUDA复合路径保留官方BF16 Attention舍入边界。"""

        _require_cuda_tensors(operands, "kernel.cuda.prefill_attention")
        return ReferenceExecutor._serve_prefill_attention(
            operands,
            attributes,
        )

    @staticmethod
    def _kernel_triton_rope(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(operands) != 4:
            raise ExecutionError(
                "kernel.triton.rope需要query、key、cosine和sine"
            )
        query, key, cosine, sine = operands
        if query.shape[-1] != int(attributes["head_dim"]):
            raise ExecutionError("kernel.triton.rope的Head Dim不匹配")
        return triton_rope(query, key, cosine, sine)

    @staticmethod
    def _kernel_cuda_rope(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """保留Mul和Add之间BF16物化边界的CUDA RoPE复合路径。"""

        _require_cuda_tensors(operands, "kernel.cuda.rope")
        return ReferenceExecutor._serve_rope(operands, attributes)

    @staticmethod
    def _kernel_triton_kv_prefill_store(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> PreallocatedKVCacheState:
        if len(operands) != 3 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "kernel.triton.kv_prefill_store需要state、key和value"
            )
        state, key, value = operands
        return state.prefill_store(
            int(attributes.get("slot", 0)),
            key,
            value,
            writer=triton_kv_store,
        )

    def _kernel_triton_rms_norm(
        self,
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        """执行已经明确 Lower 到 Triton 的 RMSNorm。"""

        if len(operands) != 2:
            raise ExecutionError("kernel.triton.rms_norm 需要 input 和 weight")
        tensor, weight = operands
        hidden_size = tensor.shape[-1]
        rows = tensor.numel() // hidden_size
        decision = resolve_lowering_plan(
            attributes.get("lowering_plan"),
            rows,
        )
        self.lowering_trace.append(
            {
                "backend": "triton",
                "source": "kernel_ir",
                "rows": rows,
                "profile_rows": decision.profile_rows,
                "estimated_us": decision.estimated_us,
                "num_warps": decision.num_warps,
            }
        )
        return triton_rms_norm(
            tensor,
            weight,
            epsilon=float(attributes["epsilon"]),
            output_dtype=attributes.get("output_dtype"),
            num_warps=decision.num_warps,
            round_before_weight=bool(
                attributes.get("round_before_weight", False)
            ),
        )

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
        round_before_weight = bool(
            attributes.get("round_before_weight", False)
        )
        if decision.backend == "triton":
            return triton_rms_norm(
                tensor,
                weight,
                epsilon=epsilon,
                output_dtype=output_dtype,
                num_warps=decision.num_warps,
                round_before_weight=round_before_weight,
            )
        if decision.backend == "native":
            if round_before_weight:
                variance = tensor.float().pow(2).mean(
                    dim=-1,
                    keepdim=True,
                )
                normalized = tensor.float() * torch.rsqrt(
                    variance + epsilon
                )
                return _cast_output(
                    weight * normalized.to(tensor.dtype),
                    output_dtype,
                )
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
                round_before_weight,
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
                compiled(
                    tensor,
                    weight,
                    epsilon,
                    round_before_weight,
                ),
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

    @staticmethod
    def _kernel_cuda_decode_attention(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        """显式执行与官方BF16 Decode Attention一致的CUDA复合路径。"""

        if len(operands) != 3 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "kernel.cuda.decode_attention需要state、query和mask"
            )
        state, query, attention_mask = operands
        _require_cuda_tensors(
            [query, attention_mask, *state.keys, *state.values, *state.lengths],
            "kernel.cuda.decode_attention",
        )
        slot = int(attributes.get("slot", 0))
        groups = int(attributes["groups"])
        scale = float(attributes["scale"])
        key_buffer, value_buffer, lengths = state._attention_inputs(
            slot,
            query,
            attention_mask,
            groups,
        )
        sequence = attention_mask.shape[-1]
        # 兼容路径有意重建原图的GQA展开和中间Tensor边界。快速路径则让
        # Triton核直接读取B×Capacity×H×D物理Buffer，避免这次物化。
        key = key_buffer[:, :sequence].permute(0, 2, 1, 3)
        value = value_buffer[:, :sequence].permute(0, 2, 1, 3)
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores * scale
        scores = scores.float() + attention_mask.float()
        valid = torch.arange(
            sequence,
            device=query.device,
        ).view(1, 1, 1, sequence) < lengths.view(-1, 1, 1, 1)
        scores = scores.masked_fill(~valid, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(query.dtype)
        return torch.matmul(probabilities, value)


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


def _require_cuda_tensors(operands: Sequence[Any], operation: str) -> None:
    """防止CUDA KernelIR在CPU上被误当成普通参考操作执行。"""

    tensors = [
        operand
        for operand in operands
        if isinstance(operand, torch.Tensor)
    ]
    if not tensors or not all(tensor.is_cuda for tensor in tensors):
        raise ExecutionError(f"{operation}的所有Tensor必须位于CUDA")
