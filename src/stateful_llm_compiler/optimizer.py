"""ServeIR 优化流水线命令行入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .cost_model import RMSNormCostModel
from .importer import import_exported_program
from .ir import format_module
from .lowering import require_full_lowering
from .pass_manager import PassManager
from .passes import (
    BufferizeKVCachePass,
    FuseDecodeAttentionPass,
    FusePrefillAttentionPass,
    FuseRoPEPass,
    FuseRMSNormPass,
    LowerToKernelIRPass,
    MaterializeKVStatePass,
    MaterializePrefillKVStatePass,
    NormalizeLinearPass,
    RemoveExportAssertionsPass,
    SelectRMSNormLoweringPass,
)


def default_pass_manager(
    cost_model: RMSNormCostModel | None = None,
    *,
    stateful_decode: bool = False,
    preallocate_kv: bool = False,
    kv_capacity: int | None = None,
    lower_to_kernel_ir: bool = False,
    prefill_kv_state: bool = False,
    numerical_mode: str = "fast",
) -> PassManager:
    passes = [
        RemoveExportAssertionsPass(),
        NormalizeLinearPass(),
        FuseRMSNormPass(),
        FuseRoPEPass(),
        FusePrefillAttentionPass(),
    ]
    if prefill_kv_state:
        passes.append(
            MaterializePrefillKVStatePass(capacity=kv_capacity)
        )
    if stateful_decode or preallocate_kv:
        passes.append(MaterializeKVStatePass())
    if preallocate_kv:
        passes.append(BufferizeKVCachePass(capacity=kv_capacity))
        passes.append(FuseDecodeAttentionPass())
    if cost_model is not None:
        passes.append(SelectRMSNormLoweringPass(cost_model))
    if lower_to_kernel_ir:
        passes.append(
            LowerToKernelIRPass(numerical_mode=numerical_mode)
        )
    return PassManager(passes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导入并优化 StatefulLLM-Compiler 的 ServeIR"
    )
    parser.add_argument("program", type=Path, help="输入的 .pt2 文件")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before-out", type=Path)
    parser.add_argument("--stats-out", type=Path)
    parser.add_argument(
        "--profile",
        type=Path,
        help="RMSNorm GPU Benchmark 生成的 Profile JSON",
    )
    parser.add_argument(
        "--stateful-decode",
        action="store_true",
        help="把 Tensor KV Cache 改写为显式 ServeIR 状态",
    )
    parser.add_argument(
        "--preallocate-kv",
        action="store_true",
        help="把 KV Append Lower 成预分配 Buffer 的位置写入",
    )
    parser.add_argument(
        "--kv-capacity",
        type=int,
        help="覆盖导出动态上界推导出的 KV Buffer Capacity",
    )
    parser.add_argument(
        "--prefill-kv-state",
        action="store_true",
        help="把Prefill返回的多层K/V物化为预分配状态",
    )
    parser.add_argument(
        "--lower-kernel-ir",
        action="store_true",
        help="把已支持的 ServeIR 操作 Lower 为 KernelIR",
    )
    parser.add_argument(
        "--require-full-lowering",
        action="store_true",
        help="要求 KernelIR 零 ATen/参考执行器回退",
    )
    parser.add_argument(
        "--numerical-mode",
        choices=("fast", "pytorch_compatible"),
        default="fast",
        help="选择融合快速路径或保留PyTorch BF16舍入边界的兼容路径",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exported = torch.export.load(args.program)
    module = import_exported_program(exported)
    before = format_module(module)
    cost_model = (
        RMSNormCostModel.load(args.profile) if args.profile else None
    )
    results = default_pass_manager(
        cost_model,
        stateful_decode=args.stateful_decode,
        preallocate_kv=args.preallocate_kv,
        kv_capacity=args.kv_capacity,
        prefill_kv_state=args.prefill_kv_state,
        lower_to_kernel_ir=(
            args.lower_kernel_ir or args.require_full_lowering
        ),
        numerical_mode=args.numerical_mode,
    ).run(module)
    if args.require_full_lowering:
        require_full_lowering(module)
    after = format_module(module)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(after + "\n", encoding="utf-8")
    if args.before_out:
        args.before_out.parent.mkdir(parents=True, exist_ok=True)
        args.before_out.write_text(before + "\n", encoding="utf-8")
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(
            json.dumps(
                [asdict(result) for result in results],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for result in results:
        print(
            f"{result.name}: changed={result.changed}, "
            f"ops={result.operations_before}->{result.operations_after}, "
            f"stats={result.statistics}"
        )
    print(f"优化后 ServeIR：{args.out}")


if __name__ == "__main__":
    main()
