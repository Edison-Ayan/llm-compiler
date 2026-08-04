"""把 Tensor 形式的 KV Cache 改写为显式 ServeIR 状态操作。"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    """识别每层 KV `cat`，并改写成共享多 Slot 状态上的追加和读取。"""

    name = "materialize-kv-state"

    def run(self, module: Module) -> PassResult:
        converted = 0
        converted_slots = 0
        rejected = 0
        for function in module.functions:
            result = _rewrite_function(function)
            if isinstance(result, int) and result > 0:
                converted += 1
                converted_slots += result
            rejected += int(result is None)
        return PassResult(
            self.name,
            changed=converted > 0,
            statistics={
                "converted": converted,
                "slots": converted_slots,
                "rejected": rejected,
            },
        )


@dataclass(frozen=True)
class _CachePair:
    """一个 Layer Slot 的历史输入、当前 K/V 和追加操作。"""

    slot: int
    past_key: Value
    past_value: Value
    current_key: Value
    current_value: Value
    key_cat: Operation
    value_cat: Operation
    insertion_index: int


def _rewrite_function(function) -> int | None:
    cache_arguments = _find_cache_arguments(function)
    if cache_arguments == []:
        return 0
    if cache_arguments is None:
        return None

    analysis = UseDefAnalysis(function)
    operations = function.block.operations
    positions = {
        id(operation): index
        for index, operation in enumerate(operations)
    }
    pairs = []
    batch_size_queries: list[tuple[Operation, Value]] = []
    state_type = None
    previous_insertion = -1
    for slot, past_key, past_value in cache_arguments:
        if not isinstance(past_key.type, TensorType) or not isinstance(
            past_value.type,
            TensorType,
        ):
            return None
        key_cat = _find_cache_cat(function, past_key)
        value_cat = _find_cache_cat(function, past_value)
        if key_cat is None or value_cat is None:
            return None
        current_key = _match_cache_cat(key_cat, past_key)
        current_value = _match_cache_cat(value_cat, past_value)
        if current_key is None or current_value is None:
            return None
        if not _compatible_cache_types(
            past_key,
            past_value,
            current_key,
            current_value,
            key_cat.results[0],
            value_cat.results[0],
        ):
            return None

        cat_ids = {id(key_cat), id(value_cat)}
        for argument in (past_key, past_value):
            for use in analysis.uses(argument):
                if use.operation is not None and id(use.operation) in cat_ids:
                    continue
                if use.operation is not None and _is_batch_size_query(
                    use.operation,
                    argument,
                ):
                    batch_size_queries.append((use.operation, argument))
                    continue
                return None
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
        if (
            consumer_positions
            and min(consumer_positions) < insertion_index
        ):
            return None
        # 共享状态必须按照 Layer 数据流顺序推进，避免后一个 Slot 使用未定义状态。
        if insertion_index < previous_insertion:
            return None
        previous_insertion = insertion_index

        inferred = _infer_state_type(
            past_key.type,
            key_cat.results[0].type,
            num_layers=len(cache_arguments),
        )
        if inferred is None or (
            state_type is not None and inferred != state_type
        ):
            return None
        state_type = inferred
        pairs.append(
            _CachePair(
                slot,
                past_key,
                past_value,
                current_key,
                current_value,
                key_cat,
                value_cat,
                insertion_index,
            )
        )

    assert state_type is not None
    cache_inputs = {
        value
        for _, past_key, past_value in cache_arguments
        for value in (past_key, past_value)
    }
    batch_source = _find_batch_source(function, cache_inputs)
    if batch_size_queries and batch_source is None:
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
    current_state = state
    insertions: dict[int, list[Operation]] = {}
    cat_ids: set[int] = set()
    cache_results = {
        result
        for pair in pairs
        for result in (
            pair.key_cat.results[0],
            pair.value_cat.results[0],
        )
    }
    old_returns = [
        returned
        for returned in function.returns
        if returned not in cache_results
    ]
    rewriter = IRRewriter(function)
    resource = state_type.resource

    # Stateful 参数删除后，Batch Shape 查询改从等价的 Hidden State Tensor 读取。
    if batch_source is not None:
        for query, old_cache in batch_size_queries:
            _retarget_batch_size_query(query, old_cache, batch_source)

    for pair in pairs:
        next_state = Value(
            _fresh_name(f"%kv_state_next_{pair.slot}", used_names),
            state_type,
        )
        read_key = Value(
            _fresh_name(f"%kv_key_{pair.slot}", used_names),
            pair.key_cat.results[0].type,
        )
        read_value = Value(
            _fresh_name(f"%kv_value_{pair.slot}", used_names),
            pair.value_cat.results[0].type,
        )
        append = Operation(
            "serve.kv.append",
            [current_state, pair.current_key, pair.current_value],
            [next_state],
            attributes={"slot": pair.slot, "axis": 2},
            effects=(
                Effect(EffectKind.READ, resource),
                Effect(EffectKind.WRITE, resource),
            ),
        )
        read = Operation(
            "serve.kv.read",
            [next_state],
            [read_key, read_value],
            attributes={"slot": pair.slot},
            effects=(Effect(EffectKind.READ, resource),),
        )
        insertions.setdefault(pair.insertion_index, []).extend(
            [append, read]
        )
        cat_ids.update((id(pair.key_cat), id(pair.value_cat)))
        rewriter.replace_all_uses(pair.key_cat.results[0], read_key)
        rewriter.replace_all_uses(pair.value_cat.results[0], read_value)
        current_state = next_state

    # 返回值在替换前已按原 Cat Result 身份过滤，最终只暴露一个多 Slot 状态。
    function.returns = [*old_returns, current_state]

    state_index = min(
        function.block.arguments.index(value)
        for value in cache_inputs
    )
    function.block.arguments = [
        argument
        for argument in function.block.arguments
        if argument not in cache_inputs
    ]
    function.block.arguments.insert(state_index, state)

    rebuilt = []
    for index, operation in enumerate(operations):
        rebuilt.extend(insertions.get(index, ()))
        if id(operation) not in cat_ids:
            rebuilt.append(operation)
    rebuilt.extend(insertions.get(len(operations), ()))
    function.block.operations = rebuilt
    return len(pairs)


def _find_cache_arguments(
    function,
) -> list[tuple[int, Value, Value]] | None:
    """识别旧单层名称和 torch.export 展平后的嵌套多层 Cache 名称。"""

    components: dict[int, dict[str, Value]] = {}
    saw_cache = False
    for argument in function.block.arguments:
        name = argument.name.removeprefix("%")
        component = None
        slot = None
        if name == "past_key":
            slot, component = 0, "key"
        elif name == "past_value":
            slot, component = 0, "value"
        else:
            nested = re.fullmatch(r"past_key_values_(\d+)_(0|1)", name)
            indexed = re.fullmatch(r"past_(key|value)_(\d+)", name)
            if nested is not None:
                slot = int(nested.group(1))
                component = "key" if nested.group(2) == "0" else "value"
            elif indexed is not None:
                component = indexed.group(1)
                slot = int(indexed.group(2))
        if component is None or slot is None:
            continue
        saw_cache = True
        slot_components = components.setdefault(slot, {})
        if component in slot_components:
            return None
        slot_components[component] = argument

    if not saw_cache:
        return []
    if sorted(components) != list(range(len(components))):
        return None
    result = []
    for slot in sorted(components):
        slot_components = components[slot]
        if set(slot_components) != {"key", "value"}:
            return None
        result.append(
            (
                slot,
                slot_components["key"],
                slot_components["value"],
            )
        )
    return result


def _is_batch_size_query(operation: Operation, cache: Value) -> bool:
    """只允许读取 Cache 的第 0 维，其他 Shape 使用仍视为状态逃逸。"""

    if operation.name != "aten.sym_size.int" or operation.operands != [cache]:
        return False
    arguments = operation.attributes.get("args")
    return (
        isinstance(arguments, dict)
        and arguments.get("tuple") == [{"ssa": cache.name}, 0]
    )


def _find_batch_source(function, cache_inputs: set[Value]) -> Value | None:
    """寻找与 Cache 共享符号 Batch 维的非状态 Tensor 参数。"""

    reference = next(iter(cache_inputs))
    assert isinstance(reference.type, TensorType)
    for argument in function.block.arguments:
        if argument in cache_inputs or not isinstance(argument.type, TensorType):
            continue
        if argument.type.shape and argument.type.shape[0] == reference.type.shape[0]:
            return argument
    return None


def _retarget_batch_size_query(
    operation: Operation,
    old_cache: Value,
    batch_source: Value,
) -> None:
    """同步修改 Operand 和 FX 参数树中的 Batch Shape 来源。"""

    operation.operands = [batch_source]
    arguments = operation.attributes.get("args")
    assert isinstance(arguments, dict)
    arguments["tuple"] = [{"ssa": batch_source.name}, 0]


def _find_cache_cat(function, past: Value) -> Operation | None:
    matches = [
        operation
        for operation in function.block.operations
        if operation.name == "aten.cat.default"
        and past in operation.operands
    ]
    return matches[0] if len(matches) == 1 else None


def _match_cache_cat(
    operation: Operation,
    past: Value,
) -> Value | None:
    """验证 `cat([past, current], dim=2)` 并返回当前 Token。"""

    if len(operation.operands) != 2 or operation.operands[0] is not past:
        return None
    current = operation.operands[1]
    arguments = operation.attributes.get("args")
    if not isinstance(arguments, dict):
        return None
    positional = arguments.get("tuple")
    if not isinstance(positional, list) or len(positional) != 2:
        return None
    tensors, axis = positional
    if axis != 2 or not isinstance(tensors, list) or len(tensors) != 2:
        return None
    if tensors != [{"ssa": past.name}, {"ssa": current.name}]:
        return None
    return current


def _compatible_cache_types(
    past_key: Value,
    past_value: Value,
    current_key: Value,
    current_value: Value,
    present_key: Value,
    present_value: Value,
) -> bool:
    """确认 Key/Value 在 DType、设备和非序列维度上完全兼容。"""

    values = (
        past_key,
        past_value,
        current_key,
        current_value,
        present_key,
        present_value,
    )
    if any(not isinstance(value.type, TensorType) for value in values):
        return False
    if past_key.type != past_value.type:
        return False
    if current_key.type != current_value.type:
        return False
    if present_key.type != present_value.type:
        return False
    return all(
        _same_cache_structure(past_key.type, value.type)
        for value in (current_key, present_key)
    )


def _same_cache_structure(left: TensorType, right: TensorType) -> bool:
    """忽略可增长的序列维度，比较 B×H×S×D Cache 契约。"""

    return (
        len(left.shape) == 4
        and len(right.shape) == 4
        and left.dtype == right.dtype
        and left.device == right.device
        and left.shape[0] == right.shape[0]
        and left.shape[1] == right.shape[1]
        and left.shape[3] == right.shape[3]
    )


def _infer_state_type(
    tensor_type: TensorType,
    present_type,
    *,
    num_layers: int,
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
        num_layers=num_layers,
        num_kv_heads=heads.value,
        head_dim=head_dim.value,
        layout="logical_bhsd",
        resource="kv.layer0" if num_layers == 1 else "kv.cache",
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
