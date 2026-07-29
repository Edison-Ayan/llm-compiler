"""把高层 ServeIR 操作 Lower 到 Triton 的 GPU 执行器。"""

from __future__ import annotations

from typing import Any

import torch

from ..execution import ExecutionError, ReferenceExecutor
from .triton_rmsnorm import triton_rms_norm


class TritonExecutor(ReferenceExecutor):
    """未注册 Triton Lowering 的 ATen 操作继续使用 PyTorch 参考语义。"""

    @staticmethod
    def _serve_rms_norm(
        operands: list[Any], attributes: dict[str, Any]
    ) -> torch.Tensor:
        if len(operands) != 2:
            raise ExecutionError("serve.rms_norm 需要 input 和 weight")
        tensor, weight = operands
        return triton_rms_norm(
            tensor,
            weight,
            epsilon=float(attributes["epsilon"]),
            output_dtype=attributes.get("output_dtype"),
        )

