"""面向 ExportedProgram 的统一编译入口。

调用关系与 ``torch.compile`` 类似：先接收已经捕获的整图，再依次完成导入、
高层优化和后端 Lowering。当前阶段仍保留不完整 KernelIR，便于逐项补齐算子；
严格模式会要求编译结果完全没有 PyTorch 回退。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cost_model import RMSNormCostModel
from .importer import import_exported_program
from .ir import Module
from .lowering import (
    LoweringCoverage,
    analyze_lowering_coverage,
    require_full_lowering,
)
from .pass_manager import PassResult


@dataclass(frozen=True)
class CompileOptions:
    """控制整图优化和后端 Lowering 的编译选项。"""

    function_name: str = "main"
    stateful_decode: bool = False
    preallocate_kv: bool = False
    kv_capacity: int | None = None
    require_full_lowering: bool = False
    prefill_kv_state: bool = False


@dataclass(frozen=True)
class CompilationArtifact:
    """一次编译产生的 KernelIR、Pass 记录和覆盖率报告。"""

    module: Module
    pass_results: tuple[PassResult, ...]
    coverage: LoweringCoverage


def compile_exported_program(
    program: torch.export.ExportedProgram,
    *,
    options: CompileOptions | None = None,
    cost_model: RMSNormCostModel | None = None,
) -> CompilationArtifact:
    """把 PyTorch ExportedProgram 编译为当前可用的 KernelIR。"""

    # 延迟导入避免包初始化时提前加载命令行模块。
    from .optimizer import default_pass_manager

    options = options or CompileOptions()
    module = import_exported_program(
        program,
        function_name=options.function_name,
    )
    pass_results = default_pass_manager(
        cost_model,
        stateful_decode=options.stateful_decode,
        preallocate_kv=options.preallocate_kv,
        kv_capacity=options.kv_capacity,
        lower_to_kernel_ir=True,
        prefill_kv_state=options.prefill_kv_state,
    ).run(module)
    coverage = analyze_lowering_coverage(module)
    if options.require_full_lowering:
        require_full_lowering(module)
    return CompilationArtifact(
        module=module,
        pass_results=tuple(pass_results),
        coverage=coverage,
    )
