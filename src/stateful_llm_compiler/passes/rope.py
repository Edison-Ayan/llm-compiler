"""把Qwen2展开的旋转位置编码融合为双结果领域操作。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis import UseDefAnalysis
from ..ir import Module, Operation, StaticDim, TensorType, Value
from ..pass_manager import CompilerPass, PassResult


@dataclass(frozen=True)
class _RotationMatch:
    """描述一个Query或Key上的半维旋转子图。"""

    operations: tuple[Operation, ...]
    final_add: Operation
    tensor: Value
    cosine: Value
    sine: Value
    cosine_broadcast: Operation
    sine_broadcast: Operation


@dataclass(frozen=True)
class _RoPEMatch:
    """描述共享Cosine/Sine的一对Query和Key旋转。"""

    operations: tuple[Operation, ...]
    query_match: _RotationMatch
    key_match: _RotationMatch
    head_dim: int


class FuseRoPEPass(CompilerPass):
    """把两组展开的Qwen2 RoPE替换为一个双结果操作。"""

    name = "fuse-rope"

    def run(self, module: Module) -> PassResult:
        fused = 0
        rejected: set[int] = set()
        for function in module.functions:
            while True:
                analysis = UseDefAnalysis(function)
                match = _find_rope_match(function, analysis, rejected)
                if match is None:
                    break
                _replace_rope(function, match)
                fused += 1
        return PassResult(
            self.name,
            changed=fused > 0,
            statistics={
                "fused": fused,
                "rejected": len(rejected),
                "variant": "qwen2_half_rotation",
            },
        )


def _find_rope_match(function, analysis, rejected) -> _RoPEMatch | None:
    """寻找共享同一对Broadcast节点的Query/Key旋转。"""

    groups: dict[tuple[int, int], list[_RotationMatch]] = {}
    for operation in function.block.operations:
        if operation.name != "aten.add.Tensor":
            continue
        rotation = _match_rotation(operation, analysis)
        if rotation is None:
            rejected.add(id(operation))
            continue
        key = (
            id(rotation.cosine_broadcast),
            id(rotation.sine_broadcast),
        )
        groups.setdefault(key, []).append(rotation)

    operation_order = {
        id(operation): index
        for index, operation in enumerate(function.block.operations)
    }
    for rotations in groups.values():
        if len(rotations) != 2:
            rejected.update(id(item.final_add) for item in rotations)
            continue
        rotations.sort(key=lambda item: operation_order[id(item.final_add)])
        match = _combine_rotations(rotations[0], rotations[1], analysis)
        if match is not None:
            return match
        rejected.update(id(item.final_add) for item in rotations)
    return None


def _match_rotation(
    final_add: Operation,
    analysis: UseDefAnalysis,
) -> _RotationMatch | None:
    """从最终Add反向匹配x*cos+rotate_half(x)*sin。"""

    if len(final_add.operands) != 2 or len(final_add.results) != 1:
        return None
    if not _ssa_arguments_match(final_add, tuple(final_add.operands)):
        return None

    for direct_index in (0, 1):
        direct_mul = analysis.producer(final_add.operands[direct_index])
        rotated_mul = analysis.producer(final_add.operands[1 - direct_index])
        direct = _match_direct_product(direct_mul, analysis)
        rotated = _match_rotated_product(rotated_mul, analysis)
        if direct is None or rotated is None:
            continue
        tensor, cosine, cosine_broadcast = direct
        (
            rotated_tensor,
            sine,
            sine_broadcast,
            rotated_operations,
        ) = rotated
        if tensor is not rotated_tensor:
            continue
        if not _valid_single_rotation_contract(
            tensor,
            cosine,
            sine,
            final_add.results[0],
        ):
            continue
        operations = (
            cosine_broadcast,
            sine_broadcast,
            direct_mul,
            *rotated_operations,
            final_add,
        )
        return _RotationMatch(
            operations,
            final_add,
            tensor,
            cosine,
            sine,
            cosine_broadcast,
            sine_broadcast,
        )
    return None


def _match_direct_product(
    operation: Operation | None,
    analysis: UseDefAnalysis,
) -> tuple[Value, Value, Operation] | None:
    """匹配x与三维Cosine扩维结果的逐元素乘法。"""

    if not _is(operation, "aten.mul.Tensor", 2):
        return None
    if not _ssa_arguments_match(operation, tuple(operation.operands)):
        return None
    for broadcast_index in (0, 1):
        broadcast = analysis.producer(operation.operands[broadcast_index])
        base = _match_head_broadcast(broadcast)
        if base is not None:
            return operation.operands[1 - broadcast_index], base, broadcast
    return None


def _match_rotated_product(
    operation: Operation | None,
    analysis: UseDefAnalysis,
) -> tuple[Value, Value, Operation, tuple[Operation, ...]] | None:
    """匹配rotate_half(x)与三维Sine扩维结果的逐元素乘法。"""

    if not _is(operation, "aten.mul.Tensor", 2):
        return None
    if not _ssa_arguments_match(operation, tuple(operation.operands)):
        return None
    for broadcast_index in (0, 1):
        broadcast = analysis.producer(operation.operands[broadcast_index])
        base = _match_head_broadcast(broadcast)
        if base is None:
            continue
        rotated_value = operation.operands[1 - broadcast_index]
        rotated = _match_rotate_half(rotated_value, analysis)
        if rotated is not None:
            tensor, rotate_operations = rotated
            return (
                tensor,
                base,
                broadcast,
                (*rotate_operations, operation),
            )
    return None


def _match_head_broadcast(operation: Operation | None) -> Value | None:
    """匹配[B,T,D]在Head维插入一维得到[B,1,T,D]。"""

    if not _is(operation, "aten.unsqueeze.default", 1):
        return None
    if _literal_argument(operation, 1) not in {1, -3}:
        return None
    if not _ssa_arguments_match(operation, (operation.operands[0],)):
        return None
    source_type = operation.operands[0].type
    result_type = operation.results[0].type
    if not isinstance(source_type, TensorType) or not isinstance(
        result_type,
        TensorType,
    ):
        return None
    if len(source_type.shape) != 3 or len(result_type.shape) != 4:
        return None
    if result_type.shape != (
        source_type.shape[0],
        StaticDim(1),
        source_type.shape[1],
        source_type.shape[2],
    ):
        return None
    return operation.operands[0]


def _match_rotate_half(
    value: Value,
    analysis: UseDefAnalysis,
) -> tuple[Value, tuple[Operation, ...]] | None:
    """匹配cat(-x[..., D/2:], x[..., :D/2])。"""

    concatenate = analysis.producer(value)
    if not _is(concatenate, "aten.cat.default", 2):
        return None
    if _literal_argument(concatenate, 1) not in {-1, 3}:
        return None
    negative = analysis.producer(concatenate.operands[0])
    low_slice = analysis.producer(concatenate.operands[1])
    if not _is(negative, "aten.neg.default", 1):
        return None
    high_slice = analysis.producer(negative.operands[0])
    if not _is(low_slice, "aten.slice.Tensor", 1) or not _is(
        high_slice,
        "aten.slice.Tensor",
        1,
    ):
        return None
    tensor = low_slice.operands[0]
    if high_slice.operands[0] is not tensor:
        return None
    tensor_type = tensor.type
    if not isinstance(tensor_type, TensorType) or len(tensor_type.shape) != 4:
        return None
    head_dim = tensor_type.shape[3]
    if not isinstance(head_dim, StaticDim) or head_dim.value % 2:
        return None
    half = head_dim.value // 2
    if _slice_spec(low_slice) != (3, 0, half):
        return None
    high_spec = _slice_spec(high_slice)
    if high_spec is None or high_spec[:2] != (3, half):
        return None
    if high_spec[2] < head_dim.value:
        return None
    if not _ssa_arguments_match(negative, (high_slice.results[0],)):
        return None
    positional = _positional_arguments(concatenate)
    expected_values = [
        {"ssa": negative.results[0].name},
        {"ssa": low_slice.results[0].name},
    ]
    if positional != [expected_values, -1] and positional != [
        expected_values,
        3,
    ]:
        return None
    return tensor, (low_slice, high_slice, negative, concatenate)


def _combine_rotations(
    first: _RotationMatch,
    second: _RotationMatch,
    analysis: UseDefAnalysis,
) -> _RoPEMatch | None:
    if (
        first.cosine is not second.cosine
        or first.sine is not second.sine
        or first.cosine_broadcast is not second.cosine_broadcast
        or first.sine_broadcast is not second.sine_broadcast
    ):
        return None
    first_type = first.tensor.type
    second_type = second.tensor.type
    if not isinstance(first_type, TensorType) or not isinstance(
        second_type,
        TensorType,
    ):
        return None
    if (
        first_type.shape[0] != second_type.shape[0]
        or first_type.shape[2:] != second_type.shape[2:]
        or first_type.dtype != second_type.dtype
        or first_type.device != second_type.device
    ):
        return None
    head_dim = first_type.shape[3]
    if not isinstance(head_dim, StaticDim):
        return None

    unique_operations = []
    seen_operations: set[int] = set()
    for operation in (*first.operations, *second.operations):
        if id(operation) in seen_operations:
            continue
        seen_operations.add(id(operation))
        unique_operations.append(operation)
    operations = tuple(unique_operations)
    final_ids = {id(first.final_add), id(second.final_add)}
    operation_ids = {id(operation) for operation in operations}
    for operation in operations:
        if id(operation) in final_ids:
            continue
        for result in operation.results:
            for use in analysis.uses(result):
                if use.operation is None or id(use.operation) not in operation_ids:
                    return None
    return _RoPEMatch(operations, first, second, head_dim.value)


def _replace_rope(function, match: _RoPEMatch) -> None:
    removed = {id(operation) for operation in match.operations}
    insertion = min(
        index
        for index, operation in enumerate(function.block.operations)
        if id(operation) in removed
    )
    fused = Operation(
        "serve.rope",
        [
            match.query_match.tensor,
            match.key_match.tensor,
            match.query_match.cosine,
            match.query_match.sine,
        ],
        [
            match.query_match.final_add.results[0],
            match.key_match.final_add.results[0],
        ],
        attributes={
            "head_dim": match.head_dim,
            "variant": "qwen2_half_rotation",
        },
    )
    rebuilt = []
    for index, operation in enumerate(function.block.operations):
        if index == insertion:
            rebuilt.append(fused)
        if id(operation) not in removed:
            rebuilt.append(operation)
    function.block.operations = rebuilt


def _valid_single_rotation_contract(
    tensor: Value,
    cosine: Value,
    sine: Value,
    result: Value,
) -> bool:
    types = (tensor.type, cosine.type, sine.type, result.type)
    if not all(isinstance(type_, TensorType) for type_ in types):
        return False
    tensor_type, cosine_type, sine_type, result_type = types
    if len(tensor_type.shape) != 4 or len(cosine_type.shape) != 3:
        return False
    head_dim = tensor_type.shape[3]
    return (
        result_type == tensor_type
        and cosine_type == sine_type
        and cosine_type.shape
        == (tensor_type.shape[0], tensor_type.shape[2], head_dim)
        and tensor_type.dtype == cosine_type.dtype
        and tensor_type.device == cosine_type.device
        and isinstance(head_dim, StaticDim)
        and head_dim.value % 2 == 0
    )


def _is(operation, name: str, operand_count: int) -> bool:
    return (
        operation is not None
        and operation.name == name
        and len(operation.operands) == operand_count
        and len(operation.results) == 1
    )


def _slice_spec(operation: Operation) -> tuple[int, int, int] | None:
    dimension = _literal_argument(operation, 1)
    start = _literal_argument(operation, 2)
    end = _literal_argument(operation, 3)
    if not all(isinstance(value, int) for value in (dimension, start, end)):
        return None
    normalized_dimension = dimension if dimension >= 0 else dimension + 4
    return normalized_dimension, start, end


def _literal_argument(operation: Operation, index: int) -> Any:
    positional = _positional_arguments(operation)
    return positional[index] if len(positional) > index else None


def _positional_arguments(operation: Operation) -> list[Any]:
    arguments = operation.attributes.get("args")
    if not isinstance(arguments, dict) or set(arguments) != {"tuple"}:
        return []
    positional = arguments["tuple"]
    return positional if isinstance(positional, list) else []


def _ssa_arguments_match(
    operation: Operation,
    values: tuple[Value, ...],
) -> bool:
    positional = _positional_arguments(operation)
    expected = [{"ssa": value.name} for value in values]
    return positional[: len(values)] == expected
