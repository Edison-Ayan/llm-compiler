"""把逻辑 KV Append Lower 成预分配 Buffer 的位置写入。"""

from __future__ import annotations

from dataclasses import replace

from ..ir import (
    Effect,
    EffectKind,
    KVStateType,
    Module,
    Operation,
    TensorType,
    Value,
)
from ..pass_manager import CompilerPass, PassResult


class BufferizeKVCachePass(CompilerPass):
    """把 `serve.kv.append` 分解为 Length、Store 和 Advance。"""

    name = "bufferize-kv-cache"

    def __init__(
        self,
        *,
        capacity: int | None = None,
        layout: str = "contiguous_bshd",
    ) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("KV Buffer Capacity 必须为正数")
        self.capacity = capacity
        self.layout = layout

    def run(self, module: Module) -> PassResult:
        bufferized = 0
        missing_capacity = 0
        rejected = 0

        for function in module.functions:
            appends = [
                operation
                for operation in function.block.operations
                if operation.name == "serve.kv.append"
            ]
            if not appends:
                continue
            old_type = appends[0].operands[0].type
            if not isinstance(old_type, KVStateType):
                rejected += len(appends)
                continue
            capacity = self.capacity or old_type.capacity
            if capacity is None:
                missing_capacity += len(appends)
                continue
            lowered_type = replace(
                old_type,
                layout=self.layout,
                capacity=capacity,
            )
            _replace_state_types(function, old_type, lowered_type)

            used_names = _used_names(function)
            rebuilt = []
            for operation in function.block.operations:
                if operation.name != "serve.kv.append":
                    rebuilt.append(operation)
                    continue
                lowered = _lower_append(
                    operation,
                    lowered_type,
                    used_names,
                )
                if lowered is None:
                    rejected += 1
                    rebuilt.append(operation)
                    continue
                rebuilt.extend(lowered)
                bufferized += 1
            function.block.operations = rebuilt

        return PassResult(
            self.name,
            changed=bufferized > 0,
            statistics={
                "bufferized": bufferized,
                "missing_capacity": missing_capacity,
                "rejected": rejected,
                "layout": self.layout,
                "capacity_override": self.capacity,
            },
        )


def _lower_append(
    append: Operation,
    state_type: KVStateType,
    used_names: set[str],
) -> list[Operation] | None:
    if len(append.operands) != 3 or len(append.results) != 1:
        return None
    state, key, value = append.operands
    if not isinstance(key.type, TensorType):
        return None
    if int(append.attributes.get("axis", 2)) not in {2, -2}:
        return None

    batch = key.type.shape[0]
    positions_type = TensorType(
        (batch,),
        "i64",
        key.type.device,
    )
    positions = Value(
        _fresh_name("%kv_positions", used_names),
        positions_type,
    )
    stored = Value(
        _fresh_name("%kv_stored", used_names),
        state_type,
    )
    # 复用原 Append 的结果 Value 作为 Advance 结果，所有既有使用无需重写。
    next_state = append.results[0]
    next_state.type = state_type
    resource = state_type.resource
    read_effect = Effect(EffectKind.READ, resource)
    write_effect = Effect(EffectKind.WRITE, resource)
    slot = int(append.attributes.get("slot", 0))

    length = Operation(
        "serve.kv.length",
        [state],
        [positions],
        attributes={"slot": slot},
        effects=(read_effect,),
    )
    store = Operation(
        "serve.kv.store",
        [state, key, value, positions],
        [stored],
        attributes={
            "slot": slot,
            "layout": state_type.layout,
            "capacity": state_type.capacity,
        },
        effects=(read_effect, write_effect),
    )
    advance = Operation(
        "serve.kv.advance",
        [stored],
        [next_state],
        attributes={"slot": slot, "delta": 1},
        effects=(read_effect, write_effect),
    )
    return [length, store, advance]


def _replace_state_types(
    function,
    old_type: KVStateType,
    new_type: KVStateType,
) -> None:
    values = list(function.block.arguments)
    values.extend(
        result
        for operation in function.block.operations
        for result in operation.results
    )
    for value in values:
        if value.type == old_type:
            value.type = new_type


def _used_names(function) -> set[str]:
    names = {value.name for value in function.block.arguments}
    names.update(
        result.name
        for operation in function.block.operations
        for result in operation.results
    )
    return names


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
