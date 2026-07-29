"""ServeIR 优化流水线命令行入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .importer import import_exported_program
from .ir import format_module
from .pass_manager import PassManager
from .passes import FuseRMSNormPass, RemoveExportAssertionsPass


def default_pass_manager() -> PassManager:
    return PassManager(
        [
            RemoveExportAssertionsPass(),
            FuseRMSNormPass(),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导入并优化 StatefulLLM-Compiler 的 ServeIR"
    )
    parser.add_argument("program", type=Path, help="输入的 .pt2 文件")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before-out", type=Path)
    parser.add_argument("--stats-out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exported = torch.export.load(args.program)
    module = import_exported_program(exported)
    before = format_module(module)
    results = default_pass_manager().run(module)
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

