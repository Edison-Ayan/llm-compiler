"""编译 Pass 接口与 PassManager。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from .ir import Module, verify_module


@dataclass
class PassResult:
    name: str
    changed: bool
    statistics: dict[str, Any] = field(default_factory=dict)
    operations_before: int = 0
    operations_after: int = 0


class CompilerPass(ABC):
    name: str

    @abstractmethod
    def run(self, module: Module) -> PassResult:
        raise NotImplementedError


class PassManager:
    """在每个 Pass 前后校验 IR，尽早暴露非法重写。"""

    def __init__(
        self,
        passes: Iterable[CompilerPass],
        *,
        verify_each: bool = True,
    ) -> None:
        self.passes = list(passes)
        self.verify_each = verify_each

    def run(self, module: Module) -> list[PassResult]:
        if self.verify_each:
            verify_module(module)
        results = []
        for compiler_pass in self.passes:
            before = _operation_count(module)
            result = compiler_pass.run(module)
            result.operations_before = before
            result.operations_after = _operation_count(module)
            if self.verify_each:
                verify_module(module)
            results.append(result)
        return results


def _operation_count(module: Module) -> int:
    return sum(
        len(function.block.operations)
        for function in module.functions
    )

