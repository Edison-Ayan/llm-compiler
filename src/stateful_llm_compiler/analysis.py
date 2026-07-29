"""ServeIR 的 Use-Def 分析。

SSA Value 的定义位置是唯一的，但一个值可以被多个 Operation 或函数返回值使用。
图重写必须先知道这些使用关系，才能安全删除旧子图并替换结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Function, Operation, Value


@dataclass(frozen=True)
class Use:
    """一个 SSA Value 的具体使用位置。"""

    operation: Operation | None
    index: int
    is_return: bool = False


class UseDefAnalysis:
    def __init__(self, function: Function) -> None:
        self.function = function
        self._producers: dict[Value, Operation] = {}
        self._uses: dict[Value, list[Use]] = {}
        self._build()

    def _build(self) -> None:
        for operation in self.function.block.operations:
            for result in operation.results:
                self._producers[result] = operation
                self._uses.setdefault(result, [])
            for index, operand in enumerate(operation.operands):
                self._uses.setdefault(operand, []).append(
                    Use(operation, index)
                )
        for index, returned in enumerate(self.function.returns):
            self._uses.setdefault(returned, []).append(
                Use(None, index, is_return=True)
            )

    def producer(self, value: Value) -> Operation | None:
        return self._producers.get(value)

    def uses(self, value: Value) -> tuple[Use, ...]:
        return tuple(self._uses.get(value, ()))

    def use_count(self, value: Value) -> int:
        return len(self._uses.get(value, ()))

    def has_use_outside(
        self,
        value: Value,
        operations: list[Operation],
        *,
        allow_return: bool = False,
    ) -> bool:
        operation_ids = {id(operation) for operation in operations}
        for use in self.uses(value):
            if use.is_return:
                if not allow_return:
                    return True
                continue
            if use.operation is not None and id(use.operation) not in operation_ids:
                return True
        return False

