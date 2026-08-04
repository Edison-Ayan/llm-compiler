"""把Prefill返回的多层K/V Tensor物化为统一预分配状态。"""

from __future__ import annotations

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


class MaterializePrefillKVStatePass(CompilerPass):
    """创建物理KV状态、批量写入各层Prefill K/V并替换函数返回。"""

    name = "materialize-prefill-kv-state"

    def __init__(self, *, capacity: int | None) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("Prefill KV Capacity必须为正数")
        self.capacity = capacity

    def run(self, module: Module) -> PassResult:
        converted = 0
        slots = 0
        missing_capacity = 0
        rejected = 0
        for function in module.functions:
            if any(
                isinstance(value.type, KVStateType)
                for value in function.returns
            ):
                continue
            cache_pairs = _returned_cache_pairs(function.returns)
            if cache_pairs is None:
                rejected += 1
                continue
            if self.capacity is None:
                missing_capacity += 1
                continue

            state_type = _state_type(cache_pairs, self.capacity)
            if state_type is None:
                rejected += 1
                continue
            used_names = _used_names(function)
            resource = state_type.resource
            read = Effect(EffectKind.READ, resource)
            write = Effect(EffectKind.WRITE, resource)
            allocate = Effect(EffectKind.ALLOCATE, resource)

            initial_state = Value(
                _fresh_name("%prefill_kv", used_names),
                state_type,
            )
            operations = [
                Operation(
                    "serve.kv.init",
                    [cache_pairs[0][0]],
                    [initial_state],
                    attributes={
                        "num_layers": len(cache_pairs),
                        "capacity": self.capacity,
                        "layout": state_type.layout,
                    },
                    effects=(allocate, write),
                )
            ]
            state = initial_state
            for slot, (key, value) in enumerate(cache_pairs):
                next_state = Value(
                    _fresh_name("%prefill_kv_stored", used_names),
                    state_type,
                )
                operations.append(
                    Operation(
                        "serve.kv.prefill_store",
                        [state, key, value],
                        [next_state],
                        attributes={
                            "slot": slot,
                            "layout": state_type.layout,
                            "capacity": self.capacity,
                        },
                        effects=(read, write),
                    )
                )
                state = next_state

            function.block.operations.extend(operations)
            function.returns = [function.returns[0], state]
            converted += 1
            slots += len(cache_pairs)

        return PassResult(
            self.name,
            changed=converted > 0,
            statistics={
                "converted": converted,
                "slots": slots,
                "missing_capacity": missing_capacity,
                "rejected": rejected,
                "capacity": self.capacity,
            },
        )


def _returned_cache_pairs(
    returns: list[Value],
) -> tuple[tuple[Value, Value], ...] | None:
    if len(returns) < 3 or len(returns) % 2 != 1:
        return None
    pairs = []
    for index in range(1, len(returns), 2):
        key = returns[index]
        value = returns[index + 1]
        if (
            not isinstance(key.type, TensorType)
            or key.type != value.type
            or len(key.type.shape) != 4
        ):
            return None
        pairs.append((key, value))
    return tuple(pairs)


def _state_type(
    pairs: tuple[tuple[Value, Value], ...],
    capacity: int,
) -> KVStateType | None:
    first_type = pairs[0][0].type
    assert isinstance(first_type, TensorType)
    heads = first_type.shape[1]
    head_dim = first_type.shape[3]
    if not isinstance(heads, StaticDim) or not isinstance(head_dim, StaticDim):
        return None
    if any(key.type != first_type for key, _ in pairs):
        return None
    return KVStateType(
        dtype=first_type.dtype,
        num_layers=len(pairs),
        num_kv_heads=heads.value,
        head_dim=head_dim.value,
        layout="contiguous_bshd",
        resource="kv",
        capacity=capacity,
    )


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
