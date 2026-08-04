"""使用 torch.export 捕获动态 Decoder 图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .graph_summary import summarize_exported_program
from .model import (
    DecoderConfig,
    StatefulTinyDecoder,
    StatefulTinyDecoderBlock,
    TinyDecoderBlock,
    make_inputs,
)
from .qwen2 import StatefulQwen2ForCausalLM, StatefulQwen2Model


class _Qwen2CausalLMDecodeWrapper(nn.Module):
    """把完整CausalLM的Decode方法暴露为可导出的forward。"""

    def __init__(self, model: StatefulQwen2ForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values,
    ):
        return self.model.decode(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )


def export_decoder(
    model: TinyDecoderBlock,
    example_inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    max_batch: int = 8,
    max_sequence: int = 128,
) -> torch.export.ExportedProgram:
    """导出共享 Batch 和 Sequence 符号维度的 Decoder。"""

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


def export_stateful_decode(
    model: StatefulTinyDecoderBlock,
    example_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    max_batch: int = 8,
    max_cache_length: int = 128,
) -> torch.export.ExportedProgram:
    """导出 Batch 和历史 KV 长度动态的单 Token Decode。"""

    batch = torch.export.Dim("batch", min=1, max=max_batch)
    past = torch.export.Dim(
        "past_sequence",
        min=1,
        max=max_cache_length,
    )
    dynamic_shapes = {
        "hidden_states": {0: batch},
        "attention_mask": {0: batch, 3: past + 1},
        "past_key": {0: batch, 2: past},
        "past_value": {0: batch, 2: past},
    }
    return torch.export.export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def export_multilayer_stateful_decode(
    model: StatefulTinyDecoder,
    example_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ],
    *,
    max_batch: int = 8,
    max_cache_length: int = 128,
) -> torch.export.ExportedProgram:
    """导出共享动态 Batch、历史长度和多层 KV Cache 的 Decode。"""

    past_key_values = example_inputs[2]
    if len(past_key_values) != model.config.num_layers:
        raise ValueError("导出输入的 KV Slot 数量与模型层数不一致")
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    past = torch.export.Dim(
        "past_sequence",
        min=1,
        max=max_cache_length,
    )
    cache_shapes = tuple(
        (
            {0: batch, 2: past},
            {0: batch, 2: past},
        )
        for _ in past_key_values
    )
    dynamic_shapes = {
        "hidden_states": {0: batch},
        "attention_mask": {0: batch, 3: past + 1},
        "past_key_values": cache_shapes,
    }
    return torch.export.export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def export_qwen2_stateful_decode(
    model: StatefulQwen2Model,
    example_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ],
    *,
    max_batch: int = 8,
    max_cache_length: int = 128,
) -> torch.export.ExportedProgram:
    """导出带 Position ID、RoPE 和多层 Cache 的 Qwen2 单 Token Decode。"""

    _validate_qwen2_dynamic_batch_example(
        example_inputs[0],
        max_batch,
        "Qwen2 Decode",
    )
    past_key_values = example_inputs[3]
    if len(past_key_values) != model.config.num_layers:
        raise ValueError("导出输入的 KV Slot 数量与 Qwen2 模型层数不一致")
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    past = torch.export.Dim(
        "past_sequence",
        min=1,
        max=max_cache_length,
    )
    cache_shapes = tuple(
        (
            {0: batch, 2: past},
            {0: batch, 2: past},
        )
        for _ in past_key_values
    )
    dynamic_shapes = {
        "hidden_states": {0: batch},
        "attention_mask": {0: batch, 3: past + 1},
        "position_ids": {0: batch},
        "past_key_values": cache_shapes,
    }
    return torch.export.export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def export_qwen2_causal_lm_prefill(
    model: StatefulQwen2ForCausalLM,
    example_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    max_batch: int = 8,
    max_prompt_length: int = 128,
) -> torch.export.ExportedProgram:
    """捕获从Input IDs到Logits和多层KV Cache的动态Prefill整图。"""

    _validate_qwen2_dynamic_batch_example(
        example_inputs[0],
        max_batch,
        "Qwen2 Prefill",
    )
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    tokens = torch.export.Dim(
        "prompt_tokens",
        min=2,
        max=max_prompt_length,
    )
    dynamic_shapes = {
        "input_ids": {0: batch, 1: tokens},
        "attention_mask": {0: batch, 2: tokens, 3: tokens},
        "position_ids": {0: batch, 1: tokens},
    }
    return torch.export.export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def export_qwen2_causal_lm_decode(
    model: StatefulQwen2ForCausalLM,
    example_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ],
    *,
    max_batch: int = 8,
    max_cache_length: int = 128,
) -> torch.export.ExportedProgram:
    """捕获Input IDs、状态化Decoder和LM Head组成的单Token整图。"""

    _validate_qwen2_dynamic_batch_example(
        example_inputs[0],
        max_batch,
        "Qwen2 CausalLM Decode",
    )
    past_key_values = example_inputs[3]
    if len(past_key_values) != model.config.num_layers:
        raise ValueError("Decode输入的KV Slot数量与CausalLM层数不一致")
    batch = torch.export.Dim("batch", min=1, max=max_batch)
    past = torch.export.Dim(
        "past_sequence",
        min=1,
        max=max_cache_length,
    )
    cache_shapes = tuple(
        (
            {0: batch, 2: past},
            {0: batch, 2: past},
        )
        for _ in past_key_values
    )
    dynamic_shapes = {
        "input_ids": {0: batch},
        "attention_mask": {0: batch, 3: past + 1},
        "position_ids": {0: batch},
        "past_key_values": cache_shapes,
    }
    return torch.export.export(
        _Qwen2CausalLMDecodeWrapper(model).eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )


def _validate_qwen2_dynamic_batch_example(
    primary_input: torch.Tensor,
    max_batch: int,
    context: str,
) -> None:
    """避免PyTorch把Batch=1样例特化后再报告难以理解的约束错误。"""

    sample_batch = int(primary_input.shape[0])
    if sample_batch == 1 and max_batch > 1:
        raise ValueError(
            f"{context}声明动态Batch时，导出样例Batch必须至少为2；"
            "Batch=1会被PyTorch 2.8特化为静态维。导出后运行时仍可使用Batch=1。"
        )


def verify_export(
    model: TinyDecoderBlock,
    program: torch.export.ExportedProgram,
    *,
    batch: int,
    sequence: int,
) -> float:
    """运行非样例 Shape，并返回最大绝对误差。"""

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
