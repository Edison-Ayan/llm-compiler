"""ServeIR 的核心数据结构。

这一层只负责表达程序，不依赖 PyTorch、MLIR 或具体 GPU 后端。IR 采用 SSA 形式：
每个 Value 只能定义一次，Operation 只能使用函数参数或此前已经定义的 Value。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class IRType:
    """所有 ServeIR 类型的基类。"""

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class StaticDim:
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SymbolicDim:
    name: str
    bounds: str | None = None

    def __str__(self) -> str:
        if self.bounds:
            return f"{self.name}<{self.bounds}>"
        return self.name


Dimension = StaticDim | SymbolicDim


@dataclass(frozen=True)
class TensorType(IRType):
    shape: tuple[Dimension, ...]
    dtype: str
    device: str = "cpu"

    def __str__(self) -> str:
        dims = "x".join(str(dim) for dim in self.shape)
        return f"tensor<{dims}x{self.dtype}, {self.device}>"


@dataclass(frozen=True)
class ScalarType(IRType):
    dtype: str

    def __str__(self) -> str:
        return self.dtype


@dataclass(frozen=True)
class TupleType(IRType):
    elements: tuple[IRType, ...]

    def __str__(self) -> str:
        return "tuple<" + ", ".join(str(item) for item in self.elements) + ">"


@dataclass(frozen=True)
class UnknownType(IRType):
    reason: str = "unknown"

    def __str__(self) -> str:
        return f"!serve.unknown<{self.reason}>"


@dataclass(frozen=True)
class KVStateType(IRType):
    """跨推理迭代存活的 KV Cache 状态句柄。"""

    dtype: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    layout: str = "blocked"
    resource: str = "kv"
    capacity: int | None = None

    def __str__(self) -> str:
        capacity = (
            f", capacity={self.capacity}"
            if self.capacity is not None
            else ""
        )
        return (
            "!serve.kv_state<"
            f"{self.dtype}, layers={self.num_layers}, "
            f"heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"layout={self.layout}, resource={self.resource}"
            f"{capacity}>"
        )


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    ALLOCATE = "allocate"
    FREE = "free"


@dataclass(frozen=True)
class Effect:
    """Operation 对某个逻辑资源产生的副作用。"""

    kind: EffectKind
    resource: str

    def __str__(self) -> str:
        return f"{self.kind.value}({self.resource})"


@dataclass(eq=False)
class Value:
    name: str
    type: IRType

    def __str__(self) -> str:
        return self.name


@dataclass
class Operation:
    name: str
    operands: list[Value]
    results: list[Value]
    attributes: dict[str, Any] = field(default_factory=dict)
    effects: tuple[Effect, ...] = ()


@dataclass
class Block:
    arguments: list[Value] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)


@dataclass
class Function:
    name: str
    block: Block
    returns: list[Value]


@dataclass
class Module:
    functions: list[Function] = field(default_factory=list)


class IRBuilder:
    """集中创建 SSA Value，避免调用方手工维护唯一编号。"""

    def __init__(self, block: Block | None = None) -> None:
        self.block = block or Block()
        self._next_value = 0
        self._used_names: set[str] = set()

    def argument(self, type_: IRType, hint: str = "arg") -> Value:
        name = self._unique_name(f"%{hint}")
        value = Value(name, type_)
        self.block.arguments.append(value)
        return value

    def emit(
        self,
        name: str,
        operands: Iterable[Value],
        result_types: Iterable[IRType],
        *,
        attributes: dict[str, Any] | None = None,
        effects: Iterable[Effect] = (),
    ) -> Operation:
        results = [
            Value(self._unique_name(f"%v{self._next_id()}"), type_)
            for type_ in result_types
        ]
        operation = Operation(
            name=name,
            operands=list(operands),
            results=results,
            attributes=dict(attributes or {}),
            effects=tuple(effects),
        )
        self.block.operations.append(operation)
        return operation

    def _next_id(self) -> int:
        value = self._next_value
        self._next_value += 1
        return value

    def _unique_name(self, base: str) -> str:
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        index = 1
        while f"{base}_{index}" in self._used_names:
            index += 1
        name = f"{base}_{index}"
        self._used_names.add(name)
        return name


class VerificationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


def verify_module(module: Module) -> None:
    """验证 SSA 支配关系、唯一性以及 KV 操作的副作用契约。"""

    errors: list[str] = []
    function_names: set[str] = set()
    for function in module.functions:
        if function.name in function_names:
            errors.append(f"函数重复定义：@{function.name}")
        function_names.add(function.name)
        _verify_function(function, errors)
    if errors:
        raise VerificationError(errors)


def _verify_function(function: Function, errors: list[str]) -> None:
    defined: set[Value] = set()
    names: set[str] = set()

    for argument in function.block.arguments:
        if argument.name in names:
            errors.append(f"@{function.name} 参数名称重复：{argument.name}")
        names.add(argument.name)
        defined.add(argument)

    for index, operation in enumerate(function.block.operations):
        for operand in operation.operands:
            if operand not in defined:
                errors.append(
                    f"@{function.name} 第 {index} 个操作 {operation.name} "
                    f"使用了尚未定义的值 {operand.name}"
                )
        operand_names = {operand.name for operand in operation.operands}
        for reference in _attribute_ssa_references(operation.attributes):
            if reference not in operand_names:
                errors.append(
                    f"@{function.name} 第 {index} 个操作 {operation.name} "
                    f"的 Attribute 引用了非操作数 {reference}"
                )
        for result in operation.results:
            if result.name in names:
                errors.append(
                    f"@{function.name} SSA 名称重复：{result.name}"
                )
            names.add(result.name)
            defined.add(result)
        _verify_kv_operation(operation, errors)

    for returned in function.returns:
        if returned not in defined:
            errors.append(
                f"@{function.name} 返回了尚未定义的值 {returned.name}"
            )


def _verify_kv_operation(operation: Operation, errors: list[str]) -> None:
    if operation.name not in {
        "serve.kv.read",
        "serve.kv.append",
        "serve.kv.length",
        "serve.kv.store",
        "serve.kv.advance",
        "serve.decode_attention",
    }:
        return
    if not operation.operands or not isinstance(
        operation.operands[0].type, KVStateType
    ):
        errors.append(f"{operation.name} 的第一个操作数必须是 KVStateType")
        return

    resource = operation.operands[0].type.resource
    effect_pairs = {(effect.kind, effect.resource) for effect in operation.effects}
    if (EffectKind.READ, resource) not in effect_pairs:
        errors.append(f"{operation.name} 缺少 read({resource}) 副作用")

    if operation.name in {
        "serve.kv.append",
        "serve.kv.store",
        "serve.kv.advance",
    } and (EffectKind.WRITE, resource) not in effect_pairs:
        errors.append(f"{operation.name} 缺少 write({resource}) 副作用")

    if operation.name == "serve.kv.append":
        if len(operation.operands) != 3:
            errors.append("serve.kv.append 必须接收 state、key 和 value")
        else:
            _verify_kv_tensor(
                operation.operands[1].type,
                operation.operands[0].type,
                "serve.kv.append 的 key",
                errors,
            )
            _verify_kv_tensor(
                operation.operands[2].type,
                operation.operands[0].type,
                "serve.kv.append 的 value",
                errors,
            )
        if (
            len(operation.results) != 1
            or operation.results[0].type != operation.operands[0].type
        ):
            errors.append("serve.kv.append 必须返回同类型的新 KV 状态")
    elif operation.name == "serve.kv.read":
        if len(operation.results) != 2:
            errors.append("serve.kv.read 必须返回 key 和 value")
        else:
            _verify_kv_tensor(
                operation.results[0].type,
                operation.operands[0].type,
                "serve.kv.read 的 key",
                errors,
            )
            _verify_kv_tensor(
                operation.results[1].type,
                operation.operands[0].type,
                "serve.kv.read 的 value",
                errors,
            )
    elif operation.name == "serve.kv.length":
        if len(operation.operands) != 1 or len(operation.results) != 1:
            errors.append("serve.kv.length 必须接收一个状态并返回位置 Tensor")
        elif not _is_kv_positions_type(operation.results[0].type):
            errors.append("serve.kv.length 必须返回一维 i64 位置 Tensor")
    elif operation.name == "serve.kv.store":
        if len(operation.operands) != 4:
            errors.append(
                "serve.kv.store 必须接收 state、key、value 和 positions"
            )
        else:
            _verify_kv_tensor(
                operation.operands[1].type,
                operation.operands[0].type,
                "serve.kv.store 的 key",
                errors,
            )
            _verify_kv_tensor(
                operation.operands[2].type,
                operation.operands[0].type,
                "serve.kv.store 的 value",
                errors,
            )
            if not _is_kv_positions_type(operation.operands[3].type):
                errors.append("serve.kv.store 的 positions 必须是一维 i64 Tensor")
        if (
            len(operation.results) != 1
            or operation.results[0].type != operation.operands[0].type
        ):
            errors.append("serve.kv.store 必须返回同类型的别名状态")
    elif operation.name == "serve.kv.advance":
        delta = operation.attributes.get("delta", 1)
        if not isinstance(delta, int) or delta <= 0:
            errors.append("serve.kv.advance 的 delta 必须为正整数")
        if len(operation.operands) != 1:
            errors.append("serve.kv.advance 必须接收一个状态")
        if (
            len(operation.results) != 1
            or operation.results[0].type != operation.operands[0].type
        ):
            errors.append("serve.kv.advance 必须返回同类型的新状态")
    elif operation.name == "serve.decode_attention":
        _verify_decode_attention(operation, errors)


def _verify_decode_attention(
    operation: Operation,
    errors: list[str],
) -> None:
    """验证直接消费物理 KV Buffer 的 Decode Attention 契约。"""

    if len(operation.operands) != 3:
        errors.append(
            "serve.decode_attention 必须接收 state、query 和 mask"
        )
        return
    state_type = operation.operands[0].type
    query_type = operation.operands[1].type
    mask_type = operation.operands[2].type
    if not isinstance(state_type, KVStateType):
        return
    if state_type.layout != "contiguous_bshd":
        errors.append(
            "serve.decode_attention 只支持 contiguous_bshd KV Layout"
        )
    if not isinstance(query_type, TensorType) or len(query_type.shape) != 4:
        errors.append("serve.decode_attention 的 query 必须是四维 Tensor")
        return
    query_tokens = query_type.shape[2]
    if not isinstance(query_tokens, StaticDim) or query_tokens.value != 1:
        errors.append("serve.decode_attention 的 query 必须是单 Token")
    if not isinstance(mask_type, TensorType) or len(mask_type.shape) != 4:
        errors.append("serve.decode_attention 的 mask 必须是四维 Tensor")
    else:
        mask_heads = mask_type.shape[1]
        mask_queries = mask_type.shape[2]
        if not isinstance(mask_heads, StaticDim) or mask_heads.value != 1:
            errors.append("serve.decode_attention 的 mask Head 维必须为 1")
        if not isinstance(mask_queries, StaticDim) or mask_queries.value != 1:
            errors.append("serve.decode_attention 的 mask Query 维必须为 1")
        if mask_type.shape[0] != query_type.shape[0]:
            errors.append("serve.decode_attention 的 mask Batch 与 query 不匹配")
    groups = operation.attributes.get("groups")
    if not isinstance(groups, int) or groups <= 0:
        errors.append("serve.decode_attention 的 groups 必须为正整数")
    elif isinstance(query_type.shape[1], StaticDim):
        expected_heads = state_type.num_kv_heads * groups
        if query_type.shape[1].value != expected_heads:
            errors.append(
                "serve.decode_attention 的 Query Head 数量必须等于 "
                "KV Head 数量乘以 groups"
            )
    if (
        isinstance(query_type.shape[3], StaticDim)
        and query_type.shape[3].value != state_type.head_dim
    ):
        errors.append("serve.decode_attention 的 Head Dim 与 KV 状态不匹配")
    scale = operation.attributes.get("scale")
    if not isinstance(scale, (int, float)) or float(scale) <= 0:
        errors.append("serve.decode_attention 的 scale 必须为正数")
    if (
        len(operation.results) != 1
        or operation.results[0].type != query_type
    ):
        errors.append(
            "serve.decode_attention 必须返回与 query 同类型的 Context"
        )


def _verify_kv_tensor(
    tensor_type: IRType,
    state_type: KVStateType,
    context: str,
    errors: list[str],
) -> None:
    if not isinstance(tensor_type, TensorType):
        errors.append(f"{context} 必须是 TensorType")
        return
    if len(tensor_type.shape) != 4:
        errors.append(f"{context} 必须是四维 B×H×S×D")
        return
    heads = tensor_type.shape[1]
    head_dim = tensor_type.shape[3]
    if (
        not isinstance(heads, StaticDim)
        or heads.value != state_type.num_kv_heads
    ):
        errors.append(
            f"{context} 的 KV Head 数量必须为 {state_type.num_kv_heads}"
        )
    if (
        not isinstance(head_dim, StaticDim)
        or head_dim.value != state_type.head_dim
    ):
        errors.append(
            f"{context} 的 Head Dim 必须为 {state_type.head_dim}"
        )
    if tensor_type.dtype != state_type.dtype:
        errors.append(
            f"{context} 的 DType 必须为 {state_type.dtype}"
        )


def _is_kv_positions_type(type_: IRType) -> bool:
    return (
        isinstance(type_, TensorType)
        and len(type_.shape) == 1
        and type_.dtype == "i64"
    )


def _attribute_ssa_references(value: Any) -> list[str]:
    """提取 FX 参数树 Attribute 中的 SSA 引用。"""

    if isinstance(value, dict):
        if set(value) == {"ssa"} and isinstance(value["ssa"], str):
            return [value["ssa"]]
        references = []
        for item in value.values():
            references.extend(_attribute_ssa_references(item))
        return references
    if isinstance(value, (list, tuple)):
        references = []
        for item in value:
            references.extend(_attribute_ssa_references(item))
        return references
    return []


def format_module(module: Module) -> str:
    """输出便于阅读和做快照测试的 MLIR 风格文本。"""

    lines = ["module {"]
    for function in module.functions:
        args = ", ".join(
            f"{arg.name}: {arg.type}" for arg in function.block.arguments
        )
        result_types = ", ".join(str(value.type) for value in function.returns)
        lines.append(f"  func @{function.name}({args}) -> ({result_types}) {{")
        for operation in function.block.operations:
            results = ", ".join(result.name for result in operation.results)
            assignment = f"{results} = " if results else ""
            operands = ", ".join(value.name for value in operation.operands)
            attrs = ""
            if operation.attributes:
                attrs = " " + json.dumps(
                    operation.attributes,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            effects = ""
            if operation.effects:
                effects = " effects[" + ", ".join(
                    str(effect) for effect in operation.effects
                ) + "]"
            operand_types = ", ".join(
                str(value.type) for value in operation.operands
            )
            produced_types = ", ".join(
                str(value.type) for value in operation.results
            )
            lines.append(
                f'    {assignment}"{operation.name}"({operands})'
                f"{attrs}{effects} : ({operand_types}) -> ({produced_types})"
            )
        returned = ", ".join(value.name for value in function.returns)
        lines.append(f"    return {returned}")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
