"""把逻辑 KV Append Lower 成预分配 Buffer 的位置写入。"""

from __future__ import annotations

from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class _AppendPlan:
    append: Operation
    old_type: KVStateType
    lowered_type: KVStateType
    delta: int


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
        if layout != "contiguous_bshd":
            raise ValueError(f"暂不支持 KV Buffer Layout：{layout}")
        self.capacity = capacity
        self.layout = layout

    def run(self, module: Module) -> PassResult:
        bufferized = 0
        missing_capacity = 0
        rejected = 0
        transaction_aborted = 0

        for function in module.functions:
            appends = [
                operation
                for operation in function.block.operations
                if operation.name == "serve.kv.append"
            ]
            if not appends:
                continue

            # 先验证函数中的全部 Append，任何一个失败都不允许留下部分 Lower 的混合 IR。
            plans = []
            function_rejected = 0
            function_missing_capacity = 0
            for append in appends:
                old_type = (
                    append.operands[0].type
                    if append.operands
                    else None
                )
                if not isinstance(old_type, KVStateType):
                    function_rejected += 1
                    continue
                capacity = self.capacity or old_type.capacity
                if capacity is None:
                    function_missing_capacity += 1
                    continue
                delta = _append_delta(append, old_type)
                if delta is None:
                    function_rejected += 1
                    continue
                plans.append(
                    _AppendPlan(
                        append=append,
                        old_type=old_type,
                        lowered_type=replace(
                            old_type,
                            layout=self.layout,
                            capacity=capacity,
                        ),
                        delta=delta,
                    )
                )
            if function_rejected or function_missing_capacity:
                rejected += function_rejected
                missing_capacity += function_missing_capacity
                transaction_aborted += 1
                continue

            lowered_types = {
                plan.old_type: plan.lowered_type
                for plan in plans
            }
            for old_type, lowered_type in lowered_types.items():
                _replace_state_types(function, old_type, lowered_type)

            used_names = _used_names(function)
            plans_by_operation = {
                id(plan.append): plan
                for plan in plans
            }
            rebuilt = []
            for operation in function.block.operations:
                plan = plans_by_operation.get(id(operation))
                if plan is None:
                    rebuilt.append(operation)
                    continue
                rebuilt.extend(
                    _lower_append(
                        operation,
                        plan.lowered_type,
                        used_names,
                        delta=plan.delta,
                    )
                )
                bufferized += 1
            function.block.operations = rebuilt

        return PassResult(
            self.name,
            changed=bufferized > 0,
            statistics={
                "bufferized": bufferized,
                "missing_capacity": missing_capacity,
                "rejected": rejected,
                "transaction_aborted": transaction_aborted,
                "layout": self.layout,
                "capacity_override": self.capacity,
            },
        )


def _lower_append(
    append: Operation,
    state_type: KVStateType,
    used_names: set[str],
    *,
    delta: int,
) -> list[Operation]:
    state, key, value = append.operands
    assert isinstance(key.type, TensorType)

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
        attributes={"slot": slot, "delta": delta},
        effects=(read_effect, write_effect),
    )
    return [length, store, advance]


def _append_delta(
    append: Operation,
    state_type: KVStateType,
) -> int | None:
    """验证 Append 契约，并从当前 K/V 的序列维推导推进长度。"""

    if len(append.operands) != 3 or len(append.results) != 1:
        return None
    state, key, value = append.operands
    if state.type != state_type or append.results[0].type != state_type:
        return None
    if not isinstance(key.type, TensorType) or key.type != value.type:
        return None
    if len(key.type.shape) != 4:
        return None
    axis = append.attributes.get("axis", 2)
    slot = append.attributes.get("slot", 0)
    if not isinstance(axis, int) or axis not in {2, -2}:
        return None
    if not isinstance(slot, int) or slot < 0:
        return None
    heads = key.type.shape[1]
    tokens = key.type.shape[2]
    head_dim = key.type.shape[3]
    if (
        not isinstance(heads, StaticDim)
        or heads.value != state_type.num_kv_heads
        or not isinstance(tokens, StaticDim)
        or tokens.value <= 0
        or not isinstance(head_dim, StaticDim)
        or head_dim.value != state_type.head_dim
        or key.type.dtype != state_type.dtype
    ):
        return None
    return tokens.value


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
