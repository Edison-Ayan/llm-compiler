"""Serving 专用操作。

KV 操作显式携带副作用，后续的重排、融合和内存规划 Pass 必须尊重这些效果。
"""

from __future__ import annotations

from ..ir import (
    Effect,
    EffectKind,
    IRBuilder,
    KVStateType,
    Operation,
    TensorType,
    Value,
)


def kv_read(
    builder: IRBuilder,
    state: Value,
    key_type: TensorType,
    value_type: TensorType,
    *,
    slot: int | None = None,
) -> Operation:
    if not isinstance(state.type, KVStateType):
        raise TypeError("kv_read 的 state 必须是 KVStateType")
    attributes = {}
    if slot is not None:
        attributes["slot"] = slot
    return builder.emit(
        "serve.kv.read",
        [state],
        [key_type, value_type],
        attributes=attributes,
        effects=[Effect(EffectKind.READ, state.type.resource)],
    )


def kv_append(
    builder: IRBuilder,
    state: Value,
    key: Value,
    value: Value,
    *,
    slot: int | None = None,
) -> Operation:
    if not isinstance(state.type, KVStateType):
        raise TypeError("kv_append 的 state 必须是 KVStateType")
    if not isinstance(key.type, TensorType) or not isinstance(
        value.type, TensorType
    ):
        raise TypeError("kv_append 的 key/value 必须是 TensorType")
    attributes = {}
    if slot is not None:
        attributes["slot"] = slot
    return builder.emit(
        "serve.kv.append",
        [state, key, value],
        [state.type],
        attributes=attributes,
        effects=[
            Effect(EffectKind.READ, state.type.resource),
            Effect(EffectKind.WRITE, state.type.resource),
        ],
    )

