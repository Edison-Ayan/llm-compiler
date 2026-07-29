"""优化后 ServeIR 与 ExportedProgram 的数值差分验证入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .execution import ReferenceExecutor, bind_exported_program_arguments
from .importer import import_exported_program
from .ir import StaticDim, TensorType
from .optimizer import default_pass_manager


@dataclass
class DifferentialRow:
    batch: int
    sequence: int
    dtype: str
    max_abs_error: float
    relative_l2_error: float
    operations: int


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"非法 Shape：{item}，应使用 BxS")
        shapes.append((int(parts[0]), int(parts[1])))
    return shapes


def run_differential(
    program: torch.export.ExportedProgram,
    shapes: list[tuple[int, int]],
    *,
    seed: int = 0,
) -> list[DifferentialRow]:
    module = import_exported_program(program)
    default_pass_manager().run(module)
    function = module.functions[0]
    hidden_argument = next(
        argument
        for argument in function.block.arguments
        if "hidden_states" in argument.name
    )
    if not isinstance(hidden_argument.type, TensorType):
        raise TypeError("hidden_states 不是 TensorType")
    hidden_dimension = hidden_argument.type.shape[-1]
    if not isinstance(hidden_dimension, StaticDim):
        raise TypeError("Hidden Size 必须是静态维度")

    dtype = _torch_dtype(hidden_argument.type.dtype)
    executor = ReferenceExecutor()
    rows = []
    for index, (batch, sequence) in enumerate(shapes):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        hidden = torch.randn(
            batch,
            sequence,
            hidden_dimension.value,
            generator=generator,
            dtype=dtype,
        )
        causal = torch.triu(
            torch.full(
                (sequence, sequence),
                float("-inf"),
                dtype=dtype,
            ),
            diagonal=1,
        )
        mask = causal.view(1, 1, sequence, sequence).expand(
            batch, 1, sequence, sequence
        ).clone()
        inputs = (hidden, mask)
        arguments = bind_exported_program_arguments(program, inputs)
        with torch.no_grad():
            expected = program.module()(*inputs)
            result = executor.run(module, arguments)
        actual = result.outputs[0]
        difference = actual.float() - expected.float()
        max_abs = float(difference.abs().max())
        relative_l2 = float(
            difference.norm()
            / expected.float().norm().clamp(min=1e-12)
        )
        rows.append(
            DifferentialRow(
                batch=batch,
                sequence=sequence,
                dtype=hidden_argument.type.dtype,
                max_abs_error=max_abs,
                relative_l2_error=relative_l2,
                operations=len(result.executed_operations),
            )
        )
    return rows


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise TypeError(f"不支持的输入 DType：{name}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证优化后 ServeIR 与原始 ExportedProgram 数值等价"
    )
    parser.add_argument("program", type=Path)
    parser.add_argument(
        "--shapes",
        default="1x1,2x8,3x13,4x17",
        help="逗号分隔的 Batch×Sequence 列表",
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    program = torch.export.load(args.program)
    rows = run_differential(program, parse_shapes(args.shapes))
    for row in rows:
        print(
            f"B={row.batch:>2} S={row.sequence:>3} "
            f"dtype={row.dtype:>3} ops={row.operations:>2} "
            f"max_abs={row.max_abs_error:.3e} "
            f"rel_l2={row.relative_l2_error:.3e}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                [asdict(row) for row in rows],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"差分结果：{args.out}")


if __name__ == "__main__":
    main()

