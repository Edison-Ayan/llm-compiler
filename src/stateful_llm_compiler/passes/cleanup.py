"""导出图清理 Pass。"""

from __future__ import annotations

from ..analysis import UseDefAnalysis
from ..ir import Module
from ..pass_manager import CompilerPass, PassResult


class RemoveExportAssertionsPass(CompilerPass):
    """删除已由 ServeIR 类型和入口 Guard 覆盖的元数据断言。

    只有断言结果完全未被使用时才删除；如果未来 PyTorch 改变导出语义，让断言结果参与
    计算，本 Pass 会保守地保留该操作。
    """

    name = "remove-export-assertions"
    target = "aten._assert_tensor_metadata.default"

    def run(self, module: Module) -> PassResult:
        removed = 0
        skipped = 0
        for function in module.functions:
            analysis = UseDefAnalysis(function)
            kept = []
            for operation in function.block.operations:
                if operation.name != self.target:
                    kept.append(operation)
                    continue
                if any(
                    analysis.use_count(result) > 0
                    for result in operation.results
                ):
                    kept.append(operation)
                    skipped += 1
                    continue
                removed += 1
            function.block.operations = kept
        return PassResult(
            self.name,
            changed=removed > 0,
            statistics={"removed": removed, "skipped": skipped},
        )

