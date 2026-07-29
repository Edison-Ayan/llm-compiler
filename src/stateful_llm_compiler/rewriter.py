"""维护 SSA 合法性的 IR 重写工具。"""

from __future__ import annotations

from typing import Any, Iterable

from .analysis import UseDefAnalysis
from .ir import Effect, Function, IRType, Operation, Value


class RewriteError(ValueError):
    pass


class IRRewriter:
    def __init__(self, function: Function) -> None:
        self.function = function
        self.analysis = UseDefAnalysis(function)
        self._used_names = {
            value.name
            for value in function.block.arguments
        }
        for operation in function.block.operations:
            self._used_names.update(result.name for result in operation.results)

    def replace_all_uses(self, old: Value, new: Value) -> int:
        """替换 Operation 操作数和函数返回值中的全部使用。"""

        replacements = 0
        for operation in self.function.block.operations:
            for index, operand in enumerate(operation.operands):
                if operand is old:
                    operation.operands[index] = new
                    replacements += 1
            operation.attributes = _replace_ssa_attribute(
                operation.attributes, old.name, new.name
            )
        for index, returned in enumerate(self.function.returns):
            if returned is old:
                self.function.returns[index] = new
                replacements += 1
        return replacements

    def erase_operations(self, operations: Iterable[Operation]) -> None:
        """只允许删除结果没有被子图外部使用的 Operation 集合。"""

        operations = list(operations)
        operation_ids = {id(operation) for operation in operations}
        for operation in operations:
            for result in operation.results:
                for use in self.analysis.uses(result):
                    if use.is_return:
                        raise RewriteError(
                            f"不能删除仍被函数返回的值 {result.name}"
                        )
                    if (
                        use.operation is not None
                        and id(use.operation) not in operation_ids
                    ):
                        raise RewriteError(
                            f"不能删除仍被 {use.operation.name} 使用的值 "
                            f"{result.name}"
                        )
        self.function.block.operations = [
            operation
            for operation in self.function.block.operations
            if id(operation) not in operation_ids
        ]

    def replace_subgraph(
        self,
        old_operations: Iterable[Operation],
        old_output: Value,
        *,
        name: str,
        operands: Iterable[Value],
        result_type: IRType,
        attributes: dict[str, Any] | None = None,
        effects: Iterable[Effect] = (),
        result_hint: str = "fused",
    ) -> Operation:
        """用一个新 Operation 替换单输出子图。"""

        old_operations = list(old_operations)
        operands = list(operands)
        if not old_operations:
            raise RewriteError("待替换子图不能为空")
        operation_ids = {id(operation) for operation in old_operations}
        positions = [
            index
            for index, operation in enumerate(
                self.function.block.operations
            )
            if id(operation) in operation_ids
        ]
        if len(positions) != len(operation_ids):
            raise RewriteError("待替换子图包含不属于当前函数的 Operation")

        position_by_result = {}
        for index, operation in enumerate(self.function.block.operations):
            for result in operation.results:
                position_by_result[result] = index
        operand_positions = []
        for operand in operands:
            producer = self.analysis.producer(operand)
            if producer is not None and id(producer) in operation_ids:
                raise RewriteError(
                    f"新操作数 {operand.name} 由即将删除的子图产生"
                )
            operand_positions.append(position_by_result.get(operand, -1))

        # 除最终输出外，所有中间值都不得逃逸到匹配子图之外。
        for operation in old_operations:
            for result in operation.results:
                if result is old_output:
                    continue
                if self.analysis.has_use_outside(result, old_operations):
                    raise RewriteError(
                        f"中间值 {result.name} 在匹配子图之外仍有使用"
                    )

        result = Value(self._fresh_name(result_hint), result_type)
        replacement = Operation(
            name=name,
            operands=operands,
            results=[result],
            attributes=dict(attributes or {}),
            effects=tuple(effects),
        )
        # 新操作必须位于全部操作数定义之后，同时尽量靠近原子图的起点。
        latest_operand = max(operand_positions, default=-1)
        insertion_index = max(min(positions), latest_operand + 1)
        self.replace_all_uses(old_output, result)

        rebuilt = []
        inserted = False
        for index, operation in enumerate(self.function.block.operations):
            if index == insertion_index:
                rebuilt.append(replacement)
                inserted = True
            if id(operation) not in operation_ids:
                rebuilt.append(operation)
        if not inserted:
            rebuilt.append(replacement)
        self.function.block.operations = rebuilt
        return replacement

    def _fresh_name(self, hint: str) -> str:
        base = f"%{hint}"
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        index = 1
        while f"{base}_{index}" in self._used_names:
            index += 1
        name = f"{base}_{index}"
        self._used_names.add(name)
        return name


def _replace_ssa_attribute(value: Any, old_name: str, new_name: str) -> Any:
    """同步更新 FX 参数树 Attribute 中保存的 SSA 引用。"""

    if isinstance(value, dict):
        if set(value) == {"ssa"} and value["ssa"] == old_name:
            return {"ssa": new_name}
        return {
            key: _replace_ssa_attribute(item, old_name, new_name)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_ssa_attribute(item, old_name, new_name)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _replace_ssa_attribute(item, old_name, new_name)
            for item in value
        )
    return value
