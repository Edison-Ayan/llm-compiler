"""Stateful 单 Token Decode 的导出命令行。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .frontend import export_multilayer_stateful_decode
from .graph_summary import summarize_exported_program
from .model import (
    DecoderConfig,
    StatefulTinyDecoder,
    make_multilayer_decode_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="导出带动态 KV Cache 的单 Token Decode 图"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph-out", type=Path)
    parser.add_argument("--program-out", type=Path, required=True)
    parser.add_argument("--example-batch", type=int, default=2)
    parser.add_argument("--example-past-length", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-cache-length", type=int, default=128)
    parser.add_argument("--verify-batch", type=int, default=3)
    parser.add_argument("--verify-past-length", type=int, default=13)
    parser.add_argument("--num-layers", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(0)
    config = DecoderConfig(num_layers=args.num_layers)
    model = StatefulTinyDecoder(config).eval()
    example_inputs = make_multilayer_decode_inputs(
        config,
        args.example_batch,
        args.example_past_length,
    )
    program = export_multilayer_stateful_decode(
        model,
        example_inputs,
        max_batch=args.max_batch,
        max_cache_length=args.max_cache_length,
    )

    verify_inputs = make_multilayer_decode_inputs(
        config,
        args.verify_batch,
        args.verify_past_length,
        seed=17,
    )
    with torch.no_grad():
        expected = model(*verify_inputs)
        actual = program.module()(*verify_inputs)
    errors = [float((expected[0] - actual[0]).abs().max())]
    for expected_slot, actual_slot in zip(expected[1], actual[1]):
        errors.extend(
            float((left - right).abs().max())
            for left, right in zip(expected_slot, actual_slot)
        )

    summary = summarize_exported_program(program)
    summary["frontend"] = {
        "model": "StatefulTinyDecoder",
        "mode": "single_token_decode",
        "num_layers": config.num_layers,
        "example_batch": args.example_batch,
        "example_past_length": args.example_past_length,
        "verification_batch": args.verify_batch,
        "verification_past_length": args.verify_past_length,
        "verification_max_abs_errors": errors,
        "cache_layout": "batch,kv_heads,sequence,head_dim",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.graph_out:
        args.graph_out.parent.mkdir(parents=True, exist_ok=True)
        args.graph_out.write_text(
            str(program.graph_module.graph) + "\n",
            encoding="utf-8",
        )
    args.program_out.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, args.program_out)

    print(
        f"导出 {len(summary['nodes'])} 个节点；"
        f"动态约束={summary['range_constraints']}；"
        f"验证最大绝对误差={max(errors):.3e}"
    )
    print(f"摘要：{args.out}")
    print(f"程序：{args.program_out}")


if __name__ == "__main__":
    main()
