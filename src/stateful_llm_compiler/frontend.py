"""Milestone-1 frontend: capture a dynamic decoder graph with torch.export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .graph_summary import summarize_exported_program
from .model import DecoderConfig, TinyDecoderBlock, make_inputs


def export_decoder(
    model: TinyDecoderBlock,
    example_inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    max_batch: int = 8,
    max_sequence: int = 128,
) -> torch.export.ExportedProgram:
    """Export a decoder with shared symbolic batch and sequence dimensions."""

    batch = torch.export.Dim("batch", min=1, max=max_batch)
    sequence = torch.export.Dim("sequence", min=1, max=max_sequence)
    dynamic_shapes = {
        "hidden_states": {0: batch, 1: sequence},
        "attention_mask": {0: batch, 2: sequence, 3: sequence},
    }
    return torch.export.export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def verify_export(
    model: TinyDecoderBlock,
    program: torch.export.ExportedProgram,
    *,
    batch: int,
    sequence: int,
) -> float:
    """Run a non-example shape and return the maximum absolute error."""

    inputs = make_inputs(model.config, batch, sequence, seed=17)
    with torch.no_grad():
        reference = model(*inputs)
        actual = program.module()(*inputs)
    return float((reference - actual).abs().max())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a dynamic Qwen-style decoder graph"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph-out", type=Path)
    parser.add_argument("--program-out", type=Path)
    parser.add_argument("--example-batch", type=int, default=2)
    parser.add_argument("--example-sequence", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-sequence", type=int, default=128)
    parser.add_argument("--verify-batch", type=int, default=3)
    parser.add_argument("--verify-sequence", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(0)
    config = DecoderConfig()
    model = TinyDecoderBlock(config).eval()
    example_inputs = make_inputs(
        config, args.example_batch, args.example_sequence
    )
    program = export_decoder(
        model,
        example_inputs,
        max_batch=args.max_batch,
        max_sequence=args.max_sequence,
    )
    max_error = verify_export(
        model,
        program,
        batch=args.verify_batch,
        sequence=args.verify_sequence,
    )

    summary = summarize_exported_program(program)
    summary["frontend"] = {
        "model": "TinyDecoderBlock",
        "example_shape": [
            args.example_batch,
            args.example_sequence,
            config.hidden_size,
        ],
        "verification_shape": [
            args.verify_batch,
            args.verify_sequence,
            config.hidden_size,
        ],
        "verification_max_abs_error": max_error,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.graph_out:
        args.graph_out.parent.mkdir(parents=True, exist_ok=True)
        args.graph_out.write_text(
            str(program.graph_module.graph), encoding="utf-8"
        )
    if args.program_out:
        args.program_out.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(program, args.program_out)

    print(
        f"exported {len(summary['nodes'])} nodes; "
        f"range constraints={summary['range_constraints']}; "
        f"verification max abs error={max_error:.3e}"
    )
    print(f"summary: {args.out}")


if __name__ == "__main__":
    main()

