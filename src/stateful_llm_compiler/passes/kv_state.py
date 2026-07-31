"""把 Tensor 形式的 KV Cache 改写为显式 ServeIR 状态操作。"""

from __future__ import annotations

import re

from ..analysis import UseDefAnalysis
from ..ir import (
    Effect,
    EffectKind,
    KVStateType,
    Module,
    Operation,
    StaticDim,
    TensorType,
    Value,
)
from ..pass_manager import CompilerPass, PassResult
from ..rewriter import IRRewriter


class MaterializeKVStatePass(CompilerPass):
    """识别两个 KV `cat`，并改写成一次状态追加和一次状态读取。"""

    name = "materialize-kv-state"

    def run(self, module: Module) -> PassResult:
        converted = 0
        rejected = 0
        for function in module.functions:
            result = _rewrite_function(function)
            converted += int(result is True)
            rejected += int(result is None)
        return PassResult(
            self.name,
            changed=converted > 0,
            statistics={
                "converted": converted,
                "rejected": rejected,
            },
        )


def _rewrite_function(function) -> bool | None:
    past_key = _find_argument(function, "past_key")
    past_value = _find_argument(function, "past_value")
    if past_key is None and past_value is None:
        return False
    if past_key is None or past_value is None:
        return None
    if not isinstance(past_key.type, TensorType) or not isinstance(
        past_value.type,
        TensorType,
    ):
        return None

    key_cat = _find_cache_cat(function, past_key)
    value_cat = _find_cache_cat(function, past_value)
    if key_cat is None or value_cat is None:
        return None
    current_key = _other_operand(key_cat, past_key)
    current_value = _other_operand(value_cat, past_value)
    if current_key is None or current_value is None:
        return None

    analysis = UseDefAnalysis(function)
    cat_ids = {id(key_cat), id(value_cat)}
    if any(
        use.operation is None or id(use.operation) not in cat_ids
        for argument in (past_key, past_value)
        for use in analysis.uses(argument)
    ):
        return None

    state_type = _infer_state_type(
        past_key.type,
        key_cat.results[0].type,
    )
    if state_type is None:
        return None

    operations = function.block.operations
    positions = {
        id(operation): index
        for index, operation in enumerate(operations)
    }
    producer_positions = []
    for current in (current_key, current_value):
        producer = analysis.producer(current)
        producer_positions.append(
            positions[id(producer)] if producer is not None else -1
        )
    insertion_index = max(
        min(positions[id(key_cat)], positions[id(value_cat)]),
        max(producer_positions) + 1,
    )
    consumer_positions = [
        positions[id(use.operation)]
        for cache_result in (key_cat.results[0], value_cat.results[0])
        for use in analysis.uses(cache_result)
        if use.operation is not None
        and id(use.operation) not in cat_ids
    ]
    if consumer_positions and min(consumer_positions) < insertion_index:
        return None

    used_names = {
        value.name
        for value in function.block.arguments
    }
    used_names.update(
        result.name
        for operation in function.block.operations
        for result in operation.results
    )
    state = Value(_fresh_name("%kv_state", used_names), state_type)
    next_state = Value(
        _fresh_name("%kv_state_next", used_names),
        state_type,
    )
    read_key = Value(
        _fresh_name("%kv_key", used_names),
        key_cat.results[0].type,
    )
    read_value = Value(
        _fresh_name("%kv_value", used_names),
        value_cat.results[0].type,
    )

    resource = state_type.resource
    append = Operation(
        "serve.kv.append",
        [state, current_key, current_value],
        [next_state],
        attributes={"slot": 0, "axis": 2},
        effects=(
            Effect(EffectKind.READ, resource),
            Effect(EffectKind.WRITE, resource),
        ),
    )
    read = Operation(
        "serve.kv.read",
        [next_state],
        [read_key, read_value],
        attributes={"slot": 0},
        effects=(Effect(EffectKind.READ, resource),),
    )

    old_returns = [
        returned
        for returned in function.returns
        if returned not in {key_cat.results[0], value_cat.results[0]}
    ]
    rewriter = IRRewriter(function)
    rewriter.replace_all_uses(key_cat.results[0], read_key)
    rewriter.replace_all_uses(value_cat.results[0], read_value)
    function.returns = [*old_returns, next_state]

    key_index = function.block.arguments.index(past_key)
    function.block.arguments = [
        argument
        for argument in function.block.arguments
        if argument not in {past_key, past_value}
    ]
    function.block.arguments.insert(key_index, state)

    rebuilt = []
    inserted = False
    for index, operation in enumerate(operations):
        if index == insertion_index:
            rebuilt.extend([append, read])
            inserted = True
        if id(operation) not in cat_ids:
            rebuilt.append(operation)
    if not inserted:
        rebuilt.extend([append, read])
    function.block.operations = rebuilt
    return True


def _find_argument(function, name: str) -> Value | None:
    expected = f"%{name}"
    return next(
        (
            argument
            for argument in function.block.arguments
            if argument.name == expected
        ),
        None,
    )


def _find_cache_cat(function, past: Value) -> Operation | None:
    matches = [
        operation
        for operation in function.block.operations
        if operation.name == "aten.cat.default"
        and past in operation.operands
    ]
    return matches[0] if len(matches) == 1 else None


def _other_operand(
    operation: Operation,
    past: Value,
) -> Value | None:
    candidates = [
        operand
        for operand in operation.operands
        if operand is not past
    ]
    return candidates[0] if len(candidates) == 1 else None


def _infer_state_type(
    tensor_type: TensorType,
    present_type,
) -> KVStateType | None:
    if len(tensor_type.shape) != 4:
        return None
    heads = tensor_type.shape[1]
    head_dim = tensor_type.shape[3]
    if not isinstance(heads, StaticDim) or not isinstance(
        head_dim,
        StaticDim,
    ):
        return None
    capacity = None
    if isinstance(present_type, TensorType):
        sequence = present_type.shape[2]
        if isinstance(sequence, StaticDim):
            capacity = sequence.value
        else:
            bounds = getattr(sequence, "bounds", None)
            if bounds:
                match = re.fullmatch(
                    r"VR\[-?\d+,\s*(-?\d+)\]",
                    bounds,
                )
                if match is not None:
                    capacity = int(match.group(1))
    return KVStateType(
        dtype=tensor_type.dtype,
        num_layers=1,
        num_kv_heads=heads.value,
        head_dim=head_dim.value,
        layout="logical_bhsd",
        resource="kv.layer0",
        capacity=capacity,
    )


def _fresh_name(base: str, used_names: set[str]) -> str:
    if base not in used_names:
        used_names.add(base)
        return base
    index = 1
    while f"{base}_{index}" in used_names:
        index += 1
    name = f"{base}_{index}"
    used_names.add(name)
    return name
