"""把 Functional ATen Linear 规范化为 ServeIR Linear。"""

from __future__ import annotations

from ..ir import Module, StaticDim, TensorType
from ..pass_manager import CompilerPass, PassResult


class NormalizeLinearPass(CompilerPass):
    """识别当前后端支持的二维/三维静态权重 Linear。"""

    name = "normalize-linear"

    def run(self, module: Module) -> PassResult:
        normalized = 0
        skipped = 0
        for function in module.functions:
            for operation in function.block.operations:
                if operation.name != "aten.linear.default":
                    continue
                dimensions = _linear_dimensions(operation)
                if dimensions is None:
                    skipped += 1
                    continue
                input_features, output_features = dimensions
                operation.name = "serve.linear"
                operation.attributes.update(
                    {
                        "input_features": input_features,
                        "output_features": output_features,
                        "has_bias": len(operation.operands) == 3,
                        "source_op": "aten.linear.default",
                    }
                )
                normalized += 1
        return PassResult(
            self.name,
            changed=normalized > 0,
            statistics={
                "normalized": normalized,
                "skipped": skipped,
            },
        )


def _linear_dimensions(operation) -> tuple[int, int] | None:
    if len(operation.operands) not in {2, 3} or len(operation.results) != 1:
        return None
    input_type = operation.operands[0].type
    weight_type = operation.operands[1].type
    result_type = operation.results[0].type
    if not all(
        isinstance(type_, TensorType)
        for type_ in (input_type, weight_type, result_type)
    ):
        return None
    if len(input_type.shape) not in {2, 3} or len(weight_type.shape) != 2:
        return None
    input_features = weight_type.shape[1]
    output_features = weight_type.shape[0]
    if not isinstance(input_features, StaticDim) or not isinstance(
        output_features,
        StaticDim,
    ):
        return None
    return input_features.value, output_features.value
