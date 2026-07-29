"""RMSNorm 子图识别与融合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis import UseDefAnalysis
from ..ir import Function, Module, Operation, TensorType, Value
from ..pass_manager import CompilerPass, PassResult
from ..rewriter import IRRewriter, RewriteError


@dataclass
class RMSNormMatch:
    operations: list[Operation]
    output: Value
    input: Value
    weight: Value
    epsilon: float


class FuseRMSNormPass(CompilerPass):
    """把展开的 ATen RMSNorm 计算折叠为 `serve.rms_norm`。"""

    name = "fuse-rmsnorm"

    def run(self, module: Module) -> PassResult:
        fused = 0
        rejected_escape = 0
        for function in module.functions:
            while True:
                match = self._find_match(function)
                if match is None:
                    break
                rewriter = IRRewriter(function)
                try:
                    rewriter.replace_subgraph(
                        match.operations,
                        match.output,
                        name="serve.rms_norm",
                        operands=[match.input, match.weight],
                        result_type=match.output.type,
                        attributes={
                            "epsilon": match.epsilon,
                            "axis": -1,
                            "compute_dtype": "f32",
                            "output_dtype": _result_dtype(match.output),
                        },
                        result_hint="rms_norm",
                    )
                except RewriteError:
                    # 匹配结构存在但中间值逃逸时不能融合，避免改变程序语义。
                    rejected_escape += 1
                    break
                fused += 1
        return PassResult(
            self.name,
            changed=fused > 0,
            statistics={
                "fused": fused,
                "rejected_escape": rejected_escape,
            },
        )

    def _find_match(self, function: Function) -> RMSNormMatch | None:
        analysis = UseDefAnalysis(function)
        for candidate in function.block.operations:
            match = _match_from_output_cast(candidate, analysis)
            if match is not None:
                return match
        return None


def _match_from_output_cast(
    output_cast: Operation,
    analysis: UseDefAnalysis,
) -> RMSNormMatch | None:
    if output_cast.name != "aten.to.dtype" or len(output_cast.operands) != 1:
        return None

    weighted = analysis.producer(output_cast.operands[0])
    if not _is(weighted, "aten.mul.Tensor", operand_count=2):
        return None

    normal_mul = _producer_named(
        analysis, weighted.operands[0], "aten.mul.Tensor"
    )
    weight_cast = _producer_named(
        analysis, weighted.operands[1], "aten.to.dtype"
    )
    if normal_mul is None or weight_cast is None:
        # 乘法满足交换律，兼容权重和归一化结果顺序互换。
        normal_mul = _producer_named(
            analysis, weighted.operands[1], "aten.mul.Tensor"
        )
        weight_cast = _producer_named(
            analysis, weighted.operands[0], "aten.to.dtype"
        )
    if normal_mul is None or weight_cast is None:
        return None
    if len(weight_cast.operands) != 1:
        return None

    rsqrt, input_for_mul = _split_producer(
        normal_mul, analysis, "aten.rsqrt.default"
    )
    if rsqrt is None or input_for_mul is None:
        return None
    input_cast_2 = analysis.producer(input_for_mul)
    if not _is(input_cast_2, "aten.to.dtype", operand_count=1):
        return None

    add = _single_producer(analysis, rsqrt, "aten.add.Tensor")
    mean = _single_producer(analysis, add, "aten.mean.dim")
    power = _single_producer(analysis, mean, "aten.pow.Tensor_Scalar")
    if add is None or mean is None or power is None:
        return None
    input_cast_1 = _single_producer(
        analysis, power, "aten.to.dtype"
    )
    if input_cast_1 is None:
        return None
    # FP32 导出可能保留连续两次 cast，FP16 导出则让两个 cast 直接共享原输入。
    # 两种形式都表示同一个 RMSNorm 输入。
    same_cast_chain = input_cast_2.operands[0] is input_cast_1.results[0]
    same_original_input = (
        input_cast_2.operands[0] is input_cast_1.operands[0]
    )
    if not same_cast_chain and not same_original_input:
        return None

    power_args = _positional_args(power)
    mean_args = _positional_args(mean)
    add_args = _positional_args(add)
    if len(power_args) < 2 or power_args[1] != 2:
        return None
    if len(mean_args) < 3 or mean_args[1] != [-1] or mean_args[2] is not True:
        return None
    if len(add_args) < 2 or not isinstance(add_args[1], (int, float)):
        return None

    # 输入和权重 cast 可能同时服务残差分支，不能作为融合子图的一部分删除。
    # `serve.rms_norm` 接收 cast 后的值，后续的 Cast/DCE Pass 再决定是否消除它们。
    operations = [
        power,
        mean,
        add,
        rsqrt,
        normal_mul,
        weighted,
        output_cast,
    ]
    return RMSNormMatch(
        operations=operations,
        output=output_cast.results[0],
        input=input_for_mul,
        weight=weight_cast.results[0],
        epsilon=float(add_args[1]),
    )


def _single_producer(
    analysis: UseDefAnalysis,
    operation: Operation | None,
    name: str,
) -> Operation | None:
    if operation is None or len(operation.operands) != 1:
        return None
    return _producer_named(analysis, operation.operands[0], name)


def _producer_named(
    analysis: UseDefAnalysis,
    value: Value,
    name: str,
) -> Operation | None:
    operation = analysis.producer(value)
    if operation is None or operation.name != name:
        return None
    return operation


def _split_producer(
    operation: Operation,
    analysis: UseDefAnalysis,
    producer_name: str,
) -> tuple[Operation | None, Value | None]:
    if len(operation.operands) != 2:
        return None, None
    first = analysis.producer(operation.operands[0])
    second = analysis.producer(operation.operands[1])
    if first is not None and first.name == producer_name:
        return first, operation.operands[1]
    if second is not None and second.name == producer_name:
        return second, operation.operands[0]
    return None, None


def _is(
    operation: Operation | None,
    name: str,
    *,
    operand_count: int | None = None,
) -> bool:
    if operation is None or operation.name != name:
        return False
    return operand_count is None or len(operation.operands) == operand_count


def _positional_args(operation: Operation) -> list[Any]:
    arguments = operation.attributes.get("args", {})
    if not isinstance(arguments, dict):
        return []
    values = arguments.get("tuple", [])
    return values if isinstance(values, list) else []


def _result_dtype(value: Value) -> str:
    if isinstance(value.type, TensorType):
        return value.type.dtype
    return "f32"
