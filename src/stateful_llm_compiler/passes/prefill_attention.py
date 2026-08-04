"""把展开的多Token GQA Prefill Attention融合为领域操作。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis import UseDefAnalysis
from ..ir import Module, Operation, StaticDim, TensorType, Value
from ..pass_manager import CompilerPass, PassResult


@dataclass(frozen=True)
class _PrefillAttentionMatch:
    operations: tuple[Operation, ...]
    final_matmul: Operation
    query: Value
    key: Value
    value: Value
    mask: Value
    groups: int
    scale: float


class FusePrefillAttentionPass(CompilerPass):
    """把GQA物化、两次Matmul和Softmax替换为Prefill Attention。"""

    name = "fuse-prefill-attention"

    def run(self, module: Module) -> PassResult:
        fused = 0
        rejected: set[int] = set()
        for function in module.functions:
            while True:
                analysis = UseDefAnalysis(function)
                match = None
                for operation in function.block.operations:
                    if operation.name != "aten.matmul.default":
                        continue
                    match = _match_from_context_matmul(operation, analysis)
                    if match is not None:
                        break
                    rejected.add(id(operation))
                if match is None:
                    break
                _replace_prefill_attention(function, match)
                fused += 1
        return PassResult(
            self.name,
            changed=fused > 0,
            statistics={
                "fused": fused,
                "rejected": len(rejected),
                "algorithm": "causal_online_softmax",
            },
        )


def _match_from_context_matmul(
    context_matmul: Operation,
    analysis: UseDefAnalysis,
) -> _PrefillAttentionMatch | None:
    if len(context_matmul.operands) != 2:
        return None
    probability_value = context_matmul.operands[0]
    value_repeat = analysis.producer(context_matmul.operands[1])
    if not _is(value_repeat, "aten.repeat_interleave.self_int", 1):
        return None
    if not _ssa_arguments_match(
        context_matmul,
        (probability_value, value_repeat.results[0]),
    ):
        return None
    probability_cast = analysis.producer(probability_value)
    if _is(probability_cast, "aten.to.dtype", 1):
        softmax_value = probability_cast.operands[0]
    else:
        probability_cast = None
        softmax_value = probability_value
    softmax = analysis.producer(softmax_value)
    if not _is(softmax, "aten.softmax.int", 1):
        return None
    if _literal_argument(softmax, 1) not in {-1, 3}:
        return None
    add = analysis.producer(softmax.operands[0])
    if not _is(add, "aten.add.Tensor", 2):
        return None

    score_value, mask = _split_score_and_mask(add, analysis)
    if score_value is None or mask is None:
        return None
    score_cast = analysis.producer(score_value)
    if _is(score_cast, "aten.to.dtype", 1):
        scaled_value = score_cast.operands[0]
    else:
        score_cast = None
        scaled_value = score_value
    scaling = analysis.producer(scaled_value)
    scale = _scaling_value(scaling)
    if scale is None:
        return None
    score_matmul = analysis.producer(scaling.operands[0])
    if not _is(score_matmul, "aten.matmul.default", 2):
        return None

    query = score_matmul.operands[0]
    key_transpose = analysis.producer(score_matmul.operands[1])
    if not _is(key_transpose, "aten.transpose.int", 1):
        return None
    if _transpose_dims(key_transpose) not in {(-2, -1), (2, 3)}:
        return None
    key_repeat = analysis.producer(key_transpose.operands[0])
    if not _is(key_repeat, "aten.repeat_interleave.self_int", 1):
        return None
    if not _ssa_arguments_match(
        score_matmul,
        (query, key_transpose.results[0]),
    ):
        return None

    key_spec = _repeat_spec(key_repeat)
    value_spec = _repeat_spec(value_repeat)
    if key_spec is None or key_spec != value_spec:
        return None
    groups, dimension = key_spec
    if groups <= 0 or dimension not in {1, -3}:
        return None
    key = key_repeat.operands[0]
    value = value_repeat.operands[0]
    if not _valid_prefill_contract(
        query,
        key,
        value,
        mask,
        context_matmul.results[0],
        groups,
    ):
        return None

    operations = [
        key_repeat,
        value_repeat,
        key_transpose,
        score_matmul,
        scaling,
    ]
    if score_cast is not None:
        operations.append(score_cast)
    operations.extend([add, softmax])
    if probability_cast is not None:
        operations.append(probability_cast)
    operations.append(context_matmul)
    if not _is_closed_subgraph(operations, context_matmul, analysis):
        return None
    return _PrefillAttentionMatch(
        tuple(operations),
        context_matmul,
        query,
        key,
        value,
        mask,
        groups,
        scale,
    )


def _replace_prefill_attention(function, match) -> None:
    removed = {id(operation) for operation in match.operations}
    fused = Operation(
        "serve.prefill_attention",
        [match.query, match.key, match.value, match.mask],
        match.final_matmul.results,
        attributes={
            "groups": match.groups,
            "scale": match.scale,
            # 当前前端把Causal约束编码在四维加性Mask中，融合不能假设任意
            # 方阵Mask都天然Causal，因此Kernel必须完整消费Mask语义。
            "causal": "mask",
            "algorithm": "online_softmax",
        },
    )
    rebuilt = []
    for operation in function.block.operations:
        if operation is match.final_matmul:
            rebuilt.append(fused)
        elif id(operation) not in removed:
            rebuilt.append(operation)
    function.block.operations = rebuilt


def _split_score_and_mask(
    add: Operation,
    analysis: UseDefAnalysis,
) -> tuple[Value | None, Value | None]:
    for index in (0, 1):
        candidate = add.operands[index]
        producer = analysis.producer(candidate)
        if producer is not None and producer.name in {
            "aten.to.dtype",
            "aten.mul.Tensor",
            "aten.div.Tensor",
        }:
            return candidate, add.operands[1 - index]
    return None, None


def _scaling_value(operation: Operation | None) -> float | None:
    if operation is None or len(operation.operands) != 1:
        return None
    positional = _positional_arguments(operation)
    score_argument = {"ssa": operation.operands[0].name}
    if (
        operation.name == "aten.mul.Tensor"
        and len(positional) == 2
    ):
        literal = (
            positional[1]
            if positional[0] == score_argument
            else positional[0]
            if positional[1] == score_argument
            else None
        )
        if isinstance(literal, (int, float)) and float(literal) > 0:
            return float(literal)
    if (
        operation.name == "aten.div.Tensor"
        and len(positional) == 2
        and positional[0] == score_argument
        and isinstance(positional[1], (int, float))
        and float(positional[1]) > 0
    ):
        return 1.0 / float(positional[1])
    return None


def _valid_prefill_contract(
    query: Value,
    key: Value,
    value: Value,
    mask: Value,
    context: Value,
    groups: int,
) -> bool:
    types = (query.type, key.type, value.type, mask.type, context.type)
    if not all(isinstance(type_, TensorType) for type_ in types):
        return False
    if any(len(type_.shape) != 4 for type_ in types):
        return False
    query_heads = query.type.shape[1]
    kv_heads = key.type.shape[1]
    tokens = query.type.shape[2]
    return (
        isinstance(query_heads, StaticDim)
        and isinstance(kv_heads, StaticDim)
        and query_heads.value == kv_heads.value * groups
        and key.type == value.type
        and query.type.shape[0] == key.type.shape[0] == mask.type.shape[0]
        and tokens == key.type.shape[2]
        and tokens == mask.type.shape[2] == mask.type.shape[3]
        and query.type.shape[3] == key.type.shape[3]
        and isinstance(mask.type.shape[1], StaticDim)
        and mask.type.shape[1].value == 1
        and context.type == query.type
        and query.type.device == key.type.device == mask.type.device
    )


def _is_closed_subgraph(operations, final, analysis) -> bool:
    operation_ids = {id(operation) for operation in operations}
    for operation in operations:
        if operation is final:
            continue
        for result in operation.results:
            for use in analysis.uses(result):
                if use.operation is None or id(use.operation) not in operation_ids:
                    return False
    return True


def _is(operation, name: str, operand_count: int) -> bool:
    return (
        operation is not None
        and operation.name == name
        and len(operation.operands) == operand_count
    )


def _literal_argument(operation: Operation, index: int) -> Any:
    positional = _positional_arguments(operation)
    return positional[index] if len(positional) > index else None


def _positional_arguments(operation: Operation) -> list[Any]:
    arguments = operation.attributes.get("args")
    if not isinstance(arguments, dict) or set(arguments) != {"tuple"}:
        return []
    positional = arguments["tuple"]
    return positional if isinstance(positional, list) else []


def _repeat_spec(operation: Operation) -> tuple[int, int] | None:
    repeats = _literal_argument(operation, 1)
    dimension = _literal_argument(operation, 2)
    if not isinstance(repeats, int) or not isinstance(dimension, int):
        return None
    return repeats, dimension


def _transpose_dims(operation: Operation) -> tuple[int, int] | None:
    first = _literal_argument(operation, 1)
    second = _literal_argument(operation, 2)
    if not isinstance(first, int) or not isinstance(second, int):
        return None
    return first, second


def _ssa_arguments_match(
    operation: Operation,
    values: tuple[Value, ...],
) -> bool:
    return _positional_arguments(operation) == [
        {"ssa": value.name} for value in values
    ]
