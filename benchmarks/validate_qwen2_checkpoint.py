"""验证真实Qwen2 Checkpoint的转换、编译和有状态推理链路。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateful_llm_compiler.backends import TritonExecutor
from stateful_llm_compiler.compiler import (
    CompileOptions,
    compile_exported_program,
)
from stateful_llm_compiler.execution import (
    bind_exported_program_arguments,
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import (
    export_qwen2_causal_lm_decode,
    export_qwen2_causal_lm_prefill,
)
from stateful_llm_compiler.qwen2 import (
    StatefulQwen2ForCausalLM,
    make_qwen2_prefill_inputs,
)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return mapping[name]


def _timed(call: Callable[[], Any], *, synchronize: bool = False):
    if synchronize:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    if synchronize:
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _copy_official_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (layer.keys.detach().cpu(), layer.values.detach().cpu())
        for layer in cache.layers
    )


def _copy_project_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (key.detach().cpu(), value.detach().cpu())
        for key, value in cache
    )


def _cache_max_error(actual, expected) -> float:
    error = 0.0
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        actual,
        expected,
    ):
        error = max(
            error,
            float(
                (
                    actual_key.float().cpu()
                    - expected_key.float().cpu()
                )
                .abs()
                .max()
            ),
            float(
                (
                    actual_value.float().cpu()
                    - expected_value.float().cpu()
                )
                .abs()
                .max()
            ),
        )
    return error


def _state_max_error(state, expected) -> float:
    actual = tuple(state.read(slot) for slot in range(len(state.keys)))
    return _cache_max_error(actual, expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--capture-batch", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/qwen2_checkpoint_validation.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("真实Checkpoint验证需要CUDA GPU")
    if args.capture_batch < 2:
        raise SystemExit("动态导出样例的capture-batch必须至少为2")
    if (
        args.prompt_tokens < 2
        or args.decode_steps < 1
        or args.capacity < args.prompt_tokens + args.decode_steps
    ):
        raise SystemExit(
            "prompt-tokens必须至少为2，decode-steps必须为正，且总长度不能超过capacity"
        )

    try:
        from transformers import Qwen2ForCausalLM
    except ImportError as error:
        raise SystemExit("需要安装transformers可选依赖") from error

    torch_dtype = _dtype(args.dtype)
    official, load_seconds = _timed(
        lambda: Qwen2ForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            attn_implementation="eager",
            local_files_only=args.local_files_only,
        ).eval()
    )
    project, convert_seconds = _timed(
        lambda: StatefulQwen2ForCausalLM.from_huggingface(official).eval()
    )
    parameter_count = sum(parameter.numel() for parameter in project.parameters())

    cpu_inputs = make_qwen2_prefill_inputs(
        project.config,
        batch=args.capture_batch,
        tokens=args.prompt_tokens,
        seed=args.seed,
    )
    input_ids, additive_mask, position_ids = cpu_inputs
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    decode_ids = torch.randint(
        0,
        project.config.vocab_size,
        (args.decode_steps, args.capture_batch, 1),
        generator=generator,
    )

    official = official.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        official_prefill = official(
            input_ids=input_ids.cuda(),
            attention_mask=torch.ones_like(input_ids).cuda(),
            position_ids=position_ids.cuda(),
            use_cache=True,
        )
        official_prefill_logits = official_prefill.logits.detach().cpu()
        official_prefill_cache = _copy_official_cache(
            official_prefill.past_key_values
        )
        official_cache = official_prefill.past_key_values
        official_decode_logits_by_step = []
        for step in range(args.decode_steps):
            current_length = args.prompt_tokens + step
            official_decode = official(
                input_ids=decode_ids[step].cuda(),
                attention_mask=torch.ones(
                    args.capture_batch,
                    current_length + 1,
                    dtype=torch.int64,
                    device="cuda",
                ),
                position_ids=torch.full(
                    (args.capture_batch, 1),
                    current_length,
                    dtype=torch.int64,
                    device="cuda",
                ),
                past_key_values=official_cache,
                use_cache=True,
            )
            official_cache = official_decode.past_key_values
            official_decode_logits_by_step.append(
                official_decode.logits.detach().cpu()
            )
        official_decode_cache = _copy_official_cache(
            official_cache
        )
    official_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    del official_prefill, official_decode, official
    gc.collect()
    torch.cuda.empty_cache()

    project = project.cuda()
    gpu_inputs = tuple(tensor.cuda() for tensor in cpu_inputs)
    decode_ids_gpu = decode_ids.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        project_prefill_logits, project_prefill_cache = project(*gpu_inputs)
        project_decode_cache = project_prefill_cache
        project_decode_logits_by_step = []
        for step in range(args.decode_steps):
            current_length = args.prompt_tokens + step
            project_decode_logits, project_decode_cache = project.decode(
                decode_ids_gpu[step],
                torch.zeros(
                    args.capture_batch,
                    1,
                    1,
                    current_length + 1,
                    device="cuda",
                ),
                torch.full(
                    (args.capture_batch, 1),
                    current_length,
                    dtype=torch.int64,
                    device="cuda",
                ),
                project_decode_cache,
            )
            project_decode_logits_by_step.append(
                project_decode_logits.detach().cpu()
            )
    project_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    eager_prefill_logit_error = float(
        (
            project_prefill_logits.float().cpu()
            - official_prefill_logits.float()
        )
        .abs()
        .max()
    )
    eager_decode_step_errors = [
        float((actual.float() - expected.float()).abs().max())
        for actual, expected in zip(
            project_decode_logits_by_step,
            official_decode_logits_by_step,
        )
    ]
    eager_decode_logit_error = max(eager_decode_step_errors)
    eager_prefill_cache_error = _cache_max_error(
        _copy_project_cache(project_prefill_cache),
        official_prefill_cache,
    )
    eager_decode_cache_error = _cache_max_error(
        _copy_project_cache(project_decode_cache),
        official_decode_cache,
    )

    prefill_program, prefill_export_seconds = _timed(
        lambda: export_qwen2_causal_lm_prefill(
            project,
            gpu_inputs,
            max_batch=args.max_batch,
            max_prompt_length=args.capacity,
        )
    )
    prefill_artifact, prefill_compile_seconds = _timed(
        lambda: compile_exported_program(
            prefill_program,
            options=CompileOptions(
                function_name="prefill",
                prefill_kv_state=True,
                kv_capacity=args.capacity,
            ),
        )
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prefill_execution, prefill_execute_seconds = _timed(
            lambda: TritonExecutor().run(
                prefill_artifact.module,
                bind_exported_program_arguments(prefill_program, gpu_inputs),
            ),
            synchronize=True,
        )
    compiled_prefill_logits, state = prefill_execution.outputs
    prefill_peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    compiled_prefill_cache_error = _state_max_error(
        state,
        _copy_project_cache(project_prefill_cache),
    )
    key_addresses = tuple(tensor.data_ptr() for tensor in state.keys)
    value_addresses = tuple(tensor.data_ptr() for tensor in state.values)

    first_decode_position = torch.full(
        (args.capture_batch, 1),
        args.prompt_tokens,
        dtype=torch.int64,
        device="cuda",
    )
    first_decode_mask = torch.zeros(
        args.capture_batch,
        1,
        1,
        args.prompt_tokens + 1,
        device="cuda",
    )
    decode_program, decode_export_seconds = _timed(
        lambda: export_qwen2_causal_lm_decode(
            project,
            (
                decode_ids_gpu[0],
                first_decode_mask,
                first_decode_position,
                project_prefill_cache,
            ),
            max_batch=args.max_batch,
            max_cache_length=args.capacity,
        )
    )
    decode_artifact, decode_compile_seconds = _timed(
        lambda: compile_exported_program(
            decode_program,
            options=CompileOptions(
                function_name="decode",
                preallocate_kv=True,
                kv_capacity=args.capacity,
            ),
        )
    )
    torch.cuda.reset_peak_memory_stats()
    decode_step_seconds = []
    compiled_decode_logits_by_step = []
    with torch.no_grad():
        for step in range(args.decode_steps):
            current_length = args.prompt_tokens + step
            current_position = torch.full(
                (args.capture_batch, 1),
                current_length,
                dtype=torch.int64,
                device="cuda",
            )
            current_mask = torch.zeros(
                args.capture_batch,
                1,
                1,
                current_length + 1,
                device="cuda",
            )
            decode_execution, step_seconds = _timed(
                lambda: TritonExecutor().run(
                    decode_artifact.module,
                    bind_stateful_decode_arguments(
                        decode_program,
                        decode_ids_gpu[step],
                        current_mask,
                        state,
                        extra_user_inputs={
                            "position_ids": current_position
                        },
                        primary_input_name="input_ids",
                    ),
                ),
                synchronize=True,
            )
            compiled_decode_logits, state = decode_execution.outputs
            decode_step_seconds.append(step_seconds)
            compiled_decode_logits_by_step.append(
                compiled_decode_logits.detach().cpu()
            )
    decode_peak_mib = torch.cuda.max_memory_allocated() / 1024**2

    compiled_prefill_diff = (
        compiled_prefill_logits.float() - project_prefill_logits.float()
    ).abs()
    compiled_decode_diffs = [
        (actual.float() - expected.float()).abs()
        for actual, expected in zip(
            compiled_decode_logits_by_step,
            project_decode_logits_by_step,
        )
    ]
    compiled_decode_step_top1 = [
        bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))
        for actual, expected in zip(
            compiled_decode_logits_by_step,
            project_decode_logits_by_step,
        )
    ]
    buffer_addresses_stable = (
        key_addresses == tuple(tensor.data_ptr() for tensor in state.keys)
        and value_addresses
        == tuple(tensor.data_ptr() for tensor in state.values)
    )
    expected_length = args.prompt_tokens + args.decode_steps
    lengths_correct = all(
        torch.equal(length, torch.full_like(length, expected_length))
        for length in state.lengths
    )

    payload = {
        "schema_version": 1,
        "model": args.model,
        "dtype": args.dtype,
        "parameter_count": parameter_count,
        "config": {
            "layers": project.config.num_layers,
            "hidden_size": project.config.hidden_size,
            "intermediate_size": project.config.intermediate_size,
            "query_heads": project.config.num_heads,
            "kv_heads": project.config.num_kv_heads,
            "head_dim": project.config.head_dim,
            "vocab_size": project.config.vocab_size,
        },
        "input": {
            "capture_batch": args.capture_batch,
            "prompt_tokens": args.prompt_tokens,
            "decode_steps": args.decode_steps,
            "capacity": args.capacity,
            "seed": args.seed,
        },
        "timing_seconds": {
            "load": load_seconds,
            "convert": convert_seconds,
            "prefill_export": prefill_export_seconds,
            "prefill_compile": prefill_compile_seconds,
            "prefill_first_execute": prefill_execute_seconds,
            "decode_export": decode_export_seconds,
            "decode_compile": decode_compile_seconds,
            "decode_first_execute": decode_step_seconds[0],
            "decode_total_execute": sum(decode_step_seconds),
            "decode_step_execute": decode_step_seconds,
        },
        "peak_allocated_mib": {
            "official_eager": official_peak_mib,
            "project_eager": project_peak_mib,
            "compiled_prefill": prefill_peak_mib,
            "compiled_decode": decode_peak_mib,
        },
        "eager_compatibility": {
            "prefill_logit_max_abs_error": eager_prefill_logit_error,
            "prefill_cache_max_abs_error": eager_prefill_cache_error,
            "decode_logit_max_abs_error": eager_decode_logit_error,
            "decode_step_logit_max_abs_error": eager_decode_step_errors,
            "decode_cache_max_abs_error": eager_decode_cache_error,
        },
        "compiled_prefill": {
            "coverage": prefill_artifact.coverage.to_dict(),
            "logit_max_abs_error": float(compiled_prefill_diff.max()),
            "logit_mean_abs_error": float(compiled_prefill_diff.mean()),
            "cache_max_abs_error": compiled_prefill_cache_error,
            "top1_equal": bool(
                torch.equal(
                    compiled_prefill_logits.argmax(-1),
                    project_prefill_logits.argmax(-1),
                )
            ),
        },
        "compiled_decode": {
            "coverage": decode_artifact.coverage.to_dict(),
            "logit_max_abs_error": max(
                float(difference.max())
                for difference in compiled_decode_diffs
            ),
            "logit_mean_abs_error": float(
                torch.cat(
                    [difference.reshape(-1) for difference in compiled_decode_diffs]
                ).mean()
            ),
            "step_logit_max_abs_error": [
                float(difference.max())
                for difference in compiled_decode_diffs
            ],
            "step_logit_mean_abs_error": [
                float(difference.mean())
                for difference in compiled_decode_diffs
            ],
            "step_top1_equal": compiled_decode_step_top1,
            "top1_equal": all(compiled_decode_step_top1),
            "cache_max_abs_error": _state_max_error(
                state,
                project_decode_cache,
            ),
            "buffer_addresses_stable": buffer_addresses_stable,
            "lengths_correct": lengths_correct,
        },
    }
    payload["passed"] = bool(
        eager_prefill_logit_error == 0.0
        and eager_prefill_cache_error == 0.0
        and eager_decode_logit_error == 0.0
        and eager_decode_cache_error == 0.0
        and payload["compiled_prefill"]["top1_equal"]
        and payload["compiled_decode"]["top1_equal"]
        and buffer_addresses_stable
        and lengths_correct
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"结果：{args.out}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
