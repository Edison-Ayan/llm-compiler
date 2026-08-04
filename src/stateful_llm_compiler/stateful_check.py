"""Stateful Decode 的多轮数值差分验证命令。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .execution import (
    KVCacheState,
    PreallocatedKVCacheState,
    ReferenceExecutor,
    bind_stateful_decode_arguments,
)
from .importer import import_exported_program
from .ir import KVStateType, StaticDim, TensorType
from .optimizer import default_pass_manager


@dataclass
class StatefulDifferentialRow:
    step: int
    cache_length_before: int
    cache_length_after: int
    state_generation: int
    output_max_abs_error: float
    key_max_abs_error: float
    value_max_abs_error: float
    operations: int


def run_stateful_differential(
    program: torch.export.ExportedProgram,
    *,
    batch: int,
    past_length: int,
    steps: int,
    seed: int = 0,
    preallocate_kv: bool = False,
    kv_capacity: int | None = None,
) -> list[StatefulDifferentialRow]:
    module = import_exported_program(
        program,
        function_name="decode",
    )
    default_pass_manager(
        stateful_decode=True,
        preallocate_kv=preallocate_kv,
        kv_capacity=kv_capacity,
    ).run(module)
    function = module.functions[0]
    state_argument = next(
        argument
        for argument in function.block.arguments
        if isinstance(argument.type, KVStateType)
    )
    hidden_argument = next(
        argument
        for argument in function.block.arguments
        if "hidden_states" in argument.name
    )
    if not isinstance(hidden_argument.type, TensorType):
        raise TypeError("hidden_states 必须是 TensorType")
    hidden_size = hidden_argument.type.shape[-1]
    if not isinstance(hidden_size, StaticDim):
        raise TypeError("Hidden Size 必须是静态维度")

    state_type = state_argument.type
    dtype = _torch_dtype(state_type.dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    past_key_values = tuple(
        (
            torch.randn(
                batch,
                state_type.num_kv_heads,
                past_length,
                state_type.head_dim,
                generator=generator,
                dtype=dtype,
            ),
            torch.randn(
                batch,
                state_type.num_kv_heads,
                past_length,
                state_type.head_dim,
                generator=generator,
                dtype=dtype,
            ),
        )
        for _ in range(state_type.num_layers)
    )
    if preallocate_kv:
        capacity = state_type.capacity
        if capacity is None:
            raise TypeError("Bufferized KVState 缺少 Capacity")
        state = PreallocatedKVCacheState.from_layer_tensors(
            past_key_values,
            capacity=capacity,
        )
    else:
        state = KVCacheState.from_layer_tensors(past_key_values)
    executor = ReferenceExecutor()
    rows = []
    user_names = {
        spec.arg.name
        for spec in program.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT"
    }

    with torch.no_grad():
        for step in range(steps):
            logical_cache = tuple(
                state.read(slot)
                for slot in range(state_type.num_layers)
            )
            cache_before = logical_cache[0][0].shape[2]
            hidden = torch.randn(
                batch,
                1,
                hidden_size.value,
                generator=generator,
                dtype=dtype,
            )
            mask = torch.zeros(
                batch,
                1,
                1,
                cache_before + 1,
                dtype=dtype,
            )
            extra_user_inputs = {}
            if "position_ids" in user_names:
                extra_user_inputs["position_ids"] = torch.full(
                    (batch, 1),
                    cache_before,
                    dtype=torch.int64,
                )
            expected_output, expected_cache = _run_exported_decode(
                program,
                hidden,
                mask,
                logical_cache,
                extra_user_inputs,
            )
            result = executor.run(
                module,
                bind_stateful_decode_arguments(
                    program,
                    hidden,
                    mask,
                    state,
                    extra_user_inputs=extra_user_inputs,
                ),
            )
            output, state = result.outputs
            actual_cache = tuple(
                state.read(slot)
                for slot in range(state_type.num_layers)
            )
            key_error = max(
                _max_abs(actual[0], expected[0])
                for actual, expected in zip(actual_cache, expected_cache)
            )
            value_error = max(
                _max_abs(actual[1], expected[1])
                for actual, expected in zip(actual_cache, expected_cache)
            )
            rows.append(
                StatefulDifferentialRow(
                    step=step,
                    cache_length_before=cache_before,
                    cache_length_after=actual_cache[0][0].shape[2],
                    state_generation=state.generation,
                    output_max_abs_error=_max_abs(output, expected_output),
                    key_max_abs_error=key_error,
                    value_max_abs_error=value_error,
                    operations=len(result.executed_operations),
                )
            )
    return rows


def _run_exported_decode(
    program: torch.export.ExportedProgram,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    logical_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    extra_user_inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
    """统一旧单层签名和 Hugging Face 风格多层嵌套 Cache 的输出。"""

    user_names = {
        spec.arg.name
        for spec in program.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT"
    }
    if "past_key" in user_names:
        output, key, value = program.module()(
            hidden,
            mask,
            *logical_cache[0],
        )
        return output, ((key, value),)
    if "position_ids" in user_names:
        output, present_key_values = program.module()(
            hidden,
            mask,
            extra_user_inputs["position_ids"],
            logical_cache,
        )
    else:
        output, present_key_values = program.module()(
            hidden,
            mask,
            logical_cache,
        )
    return output, present_key_values


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise TypeError(f"不支持的 KV Cache DType：{name}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="验证 Stateful ServeIR 的多轮 Decode 状态传递"
    )
    parser.add_argument("program", type=Path)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--past-length", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preallocate-kv",
        action="store_true",
        help="验证预分配 KV Bufferization 路径",
    )
    parser.add_argument("--kv-capacity", type=int)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    program = torch.export.load(args.program)
    rows = run_stateful_differential(
        program,
        batch=args.batch,
        past_length=args.past_length,
        steps=args.steps,
        seed=args.seed,
        preallocate_kv=args.preallocate_kv,
        kv_capacity=args.kv_capacity,
    )
    for row in rows:
        print(
            f"step={row.step} "
            f"cache={row.cache_length_before}->{row.cache_length_after} "
            f"generation={row.state_generation} "
            f"ops={row.operations} "
            f"output_max_abs={row.output_max_abs_error:.3e} "
            f"key_max_abs={row.key_max_abs_error:.3e} "
            f"value_max_abs={row.value_max_abs_error:.3e}"
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
