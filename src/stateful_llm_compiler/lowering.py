"""从 ServeIR 到可执行 KernelIR 的 Lowering 与覆盖率检查。

这一层刻意不执行算子。它只负责把已经完成图级优化的高层操作改写为明确的
后端操作，并检查图中是否还存在会回退到 PyTorch eager 的节点。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .ir import Module
from .pass_manager import CompilerPass, PassResult


# KernelIR 目前复用 ServeIR 的 SSA 容器，但通过 Dialect 前缀明确后端边界。
# 后续可以在不改变前端和高层优化 Pass 的前提下，为这些操作增加 Launch 配置、
# Workspace 和内存布局等更底层的信息。
_KERNEL_LOWERINGS = {
    "serve.linear": "kernel.triton.linear",
    "serve.prefill_attention": "kernel.triton.prefill_attention",
    "serve.rms_norm": "kernel.triton.rms_norm",
    "serve.rope": "kernel.triton.rope",
    "serve.kv.store": "kernel.triton.kv_store",
    "serve.kv.prefill_store": "kernel.triton.kv_prefill_store",
    "serve.decode_attention": "kernel.triton.decode_attention",
}

_RUNTIME_LOWERINGS = {
    "serve.kv.init": "runtime.kv.init",
    "serve.kv.length": "runtime.kv.length",
    "serve.kv.advance": "runtime.kv.advance",
}

_EXECUTABLE_OPERATIONS = frozenset(
    _KERNEL_LOWERINGS.values()
) | frozenset(_RUNTIME_LOWERINGS.values())


@dataclass(frozen=True)
class LoweringCoverage:
    """描述一张图离“零 PyTorch 回退”还有多远。"""

    total_operations: int
    lowered_operations: int
    unlowered_operations: int
    coverage: float
    lowered_by_name: dict[str, int]
    unlowered_by_name: dict[str, int]

    @property
    def is_complete(self) -> bool:
        return self.unlowered_operations == 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LoweringCoverageError(RuntimeError):
    """严格编译模式发现未 Lower 操作。"""

    def __init__(self, coverage: LoweringCoverage) -> None:
        self.coverage = coverage
        details = ", ".join(
            f"{name}×{count}"
            for name, count in coverage.unlowered_by_name.items()
        )
        super().__init__(
            "KernelIR 尚未完成零回退 Lowering："
            f"{coverage.unlowered_operations}/{coverage.total_operations} "
            f"个操作未 Lower（{details}）"
        )


class LowerToKernelIRPass(CompilerPass):
    """把已支持的 ServeIR 操作改写为显式 Kernel/Runtime Dialect。"""

    name = "lower-to-kernel-ir"

    def run(self, module: Module) -> PassResult:
        lowered = Counter()
        for function in module.functions:
            for operation in function.block.operations:
                source_name = operation.name
                target_name = _KERNEL_LOWERINGS.get(source_name)
                if target_name is None:
                    target_name = _RUNTIME_LOWERINGS.get(source_name)
                if target_name is None:
                    continue

                operation.name = target_name
                # 保留来源便于调试 KernelIR，同时不影响 SSA 和副作用信息。
                operation.attributes["lowered_from"] = source_name
                lowered[f"{source_name}->{target_name}"] += 1

        coverage = analyze_lowering_coverage(module)
        return PassResult(
            name=self.name,
            changed=bool(lowered),
            statistics={
                "lowered": sum(lowered.values()),
                "lowered_rules": dict(sorted(lowered.items())),
                "coverage": coverage.to_dict(),
            },
        )


def analyze_lowering_coverage(module: Module) -> LoweringCoverage:
    """统计 KernelIR 中已下沉和未下沉的操作。"""

    lowered = Counter()
    unlowered = Counter()
    for function in module.functions:
        for operation in function.block.operations:
            destination = (
                lowered
                if operation.name in _EXECUTABLE_OPERATIONS
                else unlowered
            )
            destination[operation.name] += 1

    lowered_count = sum(lowered.values())
    unlowered_count = sum(unlowered.values())
    total = lowered_count + unlowered_count
    coverage = 1.0 if total == 0 else lowered_count / total
    return LoweringCoverage(
        total_operations=total,
        lowered_operations=lowered_count,
        unlowered_operations=unlowered_count,
        coverage=coverage,
        lowered_by_name=dict(sorted(lowered.items())),
        unlowered_by_name=dict(sorted(unlowered.items())),
    )


def require_full_lowering(module: Module) -> LoweringCoverage:
    """要求整张图全部进入后端 Dialect，否则拒绝执行。"""

    coverage = analyze_lowering_coverage(module)
    if not coverage.is_complete:
        raise LoweringCoverageError(coverage)
    return coverage
