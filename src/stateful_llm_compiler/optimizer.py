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
from .pass_manager import PassManager
from .passes import (
    FuseRMSNormPass,
    MaterializeKVStatePass,
    RemoveExportAssertionsPass,
    SelectRMSNormLoweringPass,
)


def default_pass_manager(
    cost_model: RMSNormCostModel | None = None,
    *,
    stateful_decode: bool = False,
) -> PassManager:
    passes = [
        RemoveExportAssertionsPass(),
        FuseRMSNormPass(),
    ]
    if stateful_decode:
        passes.append(MaterializeKVStatePass())
    if cost_model is not None:
        passes.append(SelectRMSNormLoweringPass(cost_model))
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
    ).run(module)
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
