"""把物化 KV Tensor 的 Decode Attention 融合为直接读取物理 Buffer 的操作。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis import UseDefAnalysis
from ..ir import Effect, EffectKind, KVStateType, Module, Operation, Value
from ..pass_manager import CompilerPass, PassResult


@dataclass(frozen=True)
class _AttentionMatch:
    """一次完整的 GQA Decode Attention 匹配结果。"""

    operations: tuple[Operation, ...]
    final_matmul: Operation
    state: Value
    query: Value
    mask: Value
    slot: int
    groups: int
    scale: float


class FuseDecodeAttentionPass(CompilerPass):
    """把 kv.read + GQA + Softmax + Matmul 改写为一个领域操作。"""

    name = "fuse-decode-attention"

    def run(self, module: Module) -> PassResult:
        fused = 0
        rejected = 0
        for function in module.functions:
            while True:
                analysis = UseDefAnalysis(function)
                match = None
                for operation in function.block.operations:
                    if operation.name != "serve.kv.read":
                        continue
                    match = _match_attention(operation, analysis)
                    if match is not None:
                        break
                    rejected += 1
                if match is None:
                    break
                _replace_attention(function, match)
                fused += 1

        return PassResult(
            self.name,
            changed=fused > 0,
            statistics={
                "fused": fused,
                "rejected": rejected,
                "algorithm": "online_softmax",
            },
        )


def _match_attention(
    read: Operation,
    analysis: UseDefAnalysis,
) -> _AttentionMatch | None:
    if (
        len(read.operands) != 1
        or len(read.results) != 2
        or not isinstance(read.operands[0].type, KVStateType)
    ):
        return None
    state_type = read.operands[0].type
    if state_type.layout != "contiguous_bshd":
        return None

    key_repeat = _single_user(
        read.results[0],
        analysis,
        "aten.repeat_interleave.self_int",
    )
    value_repeat = _single_user(
        read.results[1],
        analysis,
        "aten.repeat_interleave.self_int",
    )
    if key_repeat is None or value_repeat is None:
        return None
    key_spec = _repeat_spec(key_repeat)
    value_spec = _repeat_spec(value_repeat)
    if key_spec is None or key_spec != value_spec:
        return None
    groups, dimension = key_spec
    if groups <= 0 or dimension not in {1, -3}:
        return None

    key_transpose = _single_user(
        key_repeat.results[0],
        analysis,
        "aten.transpose.int",
    )
    if key_transpose is None or _transpose_dims(key_transpose) not in {
        (-2, -1),
        (2, 3),
    }:
        return None
    score_matmul = _single_user(
        key_transpose.results[0],
        analysis,
        "aten.matmul.default",
    )
    if score_matmul is None or len(score_matmul.operands) != 2:
        return None
    query = _other_operand(score_matmul, key_transpose.results[0])
    if query is None:
        return None

    division = _single_user(
        score_matmul.results[0],
        analysis,
        "aten.div.Tensor",
    )
    divisor = _literal_argument(division, 1) if division is not None else None
    if not isinstance(divisor, (int, float)) or float(divisor) <= 0:
        return None

    score_value = division.results[0]
    score_cast = _optional_single_user(
        score_value,
        analysis,
        "aten.to.dtype",
    )
    if score_cast is not None:
        score_value = score_cast.results[0]

    add = _single_user(score_value, analysis, "aten.add.Tensor")
    if add is None or len(add.operands) != 2:
        return None
    mask = _other_operand(add, score_value)
    if mask is None:
        return None

    softmax = _single_user(
        add.results[0],
        analysis,
        "aten.softmax.int",
    )
    if softmax is None or _literal_argument(softmax, 1) not in {-1, 3}:
        return None
    probability_value = softmax.results[0]
    probability_cast = _optional_single_user(
        probability_value,
        analysis,
        "aten.to.dtype",
    )
    if probability_cast is not None:
        probability_value = probability_cast.results[0]

    context_matmul = _single_user(
        probability_value,
        analysis,
        "aten.matmul.default",
    )
    if (
        context_matmul is None
        or len(context_matmul.operands) != 2
        or value_repeat.results[0] not in context_matmul.operands
    ):
        return None

    operations = [
        read,
        key_repeat,
        value_repeat,
        key_transpose,
        score_matmul,
        division,
    ]
    if score_cast is not None:
        operations.append(score_cast)
    operations.extend([add, softmax])
    if probability_cast is not None:
        operations.append(probability_cast)
    operations.append(context_matmul)
    if not _is_closed_subgraph(
        operations,
        context_matmul,
        analysis,
    ):
        return None

    return _AttentionMatch(
        tuple(operations),
        context_matmul,
        read.operands[0],
        query,
        mask,
        int(read.attributes.get("slot", 0)),
        groups,
        1.0 / float(divisor),
    )


def _replace_attention(function, match: _AttentionMatch) -> None:
    removed = {id(operation) for operation in match.operations}
    resource = match.state.type.resource
    fused = Operation(
        "serve.decode_attention",
        [match.state, match.query, match.mask],
        # 复用原始 Context 结果，后续 Transpose/Reshape 不需要改写。
        match.final_matmul.results,
        attributes={
            "slot": match.slot,
            "groups": match.groups,
            "scale": match.scale,
            "layout": match.state.type.layout,
            "algorithm": "online_softmax",
        },
        effects=(Effect(EffectKind.READ, resource),),
    )
    rebuilt = []
    for operation in function.block.operations:
        if operation is match.final_matmul:
            rebuilt.append(fused)
        elif id(operation) not in removed:
            rebuilt.append(operation)
    function.block.operations = rebuilt


def _single_user(
    value: Value,
    analysis: UseDefAnalysis,
    expected_name: str,
) -> Operation | None:
    uses = analysis.uses(value)
    if len(uses) != 1 or uses[0].operation is None:
        return None
    operation = uses[0].operation
    return operation if operation.name == expected_name else None


def _optional_single_user(
    value: Value,
    analysis: UseDefAnalysis,
    expected_name: str,
) -> Operation | None:
    uses = analysis.uses(value)
    if len(uses) != 1 or uses[0].operation is None:
        return None
    operation = uses[0].operation
    return operation if operation.name == expected_name else None


def _other_operand(operation: Operation, value: Value) -> Value | None:
    others = [operand for operand in operation.operands if operand is not value]
    return others[0] if len(others) == 1 else None


def _literal_argument(
    operation: Operation | None,
    index: int,
) -> Any:
    if operation is None:
        return None
    arguments = operation.attributes.get("args")
    if not (
        isinstance(arguments, dict)
        and set(arguments) == {"tuple"}
        and isinstance(arguments["tuple"], list)
        and len(arguments["tuple"]) > index
    ):
        return None
    return arguments["tuple"][index]


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


def _is_closed_subgraph(
    operations: list[Operation],
    final: Operation,
    analysis: UseDefAnalysis,
) -> bool:
    operation_ids = {id(operation) for operation in operations}
    for operation in operations:
        if operation is final:
            continue
        for result in operation.results:
            for use in analysis.uses(result):
                if use.operation is None or id(use.operation) not in operation_ids:
                    return False
    return True
