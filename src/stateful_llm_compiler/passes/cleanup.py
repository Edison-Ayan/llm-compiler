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
    targets = {
        "aten._assert_tensor_metadata.default",
        "aten._assert_scalar.default",
    }
    dead_metadata_operations = {
        "aten.sym_size.int",
        "builtin.add",
        "builtin.eq",
        "serve.external",
    }

    def run(self, module: Module) -> PassResult:
        removed = 0
        skipped = 0
        dead_metadata = 0
        for function in module.functions:
            analysis = UseDefAnalysis(function)
            kept = []
            for operation in function.block.operations:
                if operation.name not in self.targets:
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

            # PyTorch 2.8 会用 sym_size + 标量比较构造动态 Shape 断言。
            # 删除断言后递归清理只为断言服务的纯元数据操作，避免它们继续引用
            # 即将被 KV 状态句柄替换的 past_key/past_value 参数。
            while True:
                analysis = UseDefAnalysis(function)
                removable = {
                    id(operation)
                    for operation in function.block.operations
                    if operation.name in self.dead_metadata_operations
                    and not operation.effects
                    and all(
                        analysis.use_count(result) == 0
                        for result in operation.results
                    )
                }
                if not removable:
                    break
                dead_metadata += len(removable)
                function.block.operations = [
                    operation
                    for operation in function.block.operations
                    if id(operation) not in removable
                ]
        return PassResult(
            self.name,
            changed=(removed + dead_metadata) > 0,
            statistics={
                "removed": removed,
                "skipped": skipped,
                "dead_metadata": dead_metadata,
            },
        )
