from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.backends import TritonExecutor, triton_rope
from stateful_llm_compiler.frontend import export_qwen2_causal_lm_prefill
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.ir import (
    Function,
    IRBuilder,
    Module,
    StaticDim,
    TensorType,
    VerificationError,
    verify_module,
)
from stateful_llm_compiler.lowering import LowerToKernelIRPass
from stateful_llm_compiler.passes import FuseRoPEPass
from stateful_llm_compiler.qwen2 import (
    Qwen2CompatConfig,
    StatefulQwen2ForCausalLM,
    make_qwen2_prefill_inputs,
)


def _reference_rope(tensor, cosine, sine):
    half = tensor.shape[-1] // 2
    rotated = torch.cat(
        (-tensor[..., half:], tensor[..., :half]),
        dim=-1,
    )
    return tensor * cosine.unsqueeze(1) + rotated * sine.unsqueeze(1)


class RoPEFusionTest(unittest.TestCase):
    def test_fuses_each_layer_into_one_dual_result_operation(self) -> None:
        config = Qwen2CompatConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_position_embeddings=64,
        )
        model = StatefulQwen2ForCausalLM(config).eval()
        example = make_qwen2_prefill_inputs(config, batch=2, tokens=4)
        program = export_qwen2_causal_lm_prefill(
            model,
            example,
            max_batch=8,
            max_prompt_length=16,
        )
        module = import_exported_program(program, function_name="prefill")
        before = len(module.functions[0].block.operations)

        first = FuseRoPEPass().run(module)
        second = FuseRoPEPass().run(module)
        verify_module(module)

        ropes = [
            operation
            for operation in module.functions[0].block.operations
            if operation.name == "serve.rope"
        ]
        self.assertEqual(first.statistics["fused"], 2)
        self.assertEqual(before - len(module.functions[0].block.operations), 30)
        self.assertEqual(len(ropes), 2)
        self.assertTrue(all(len(operation.results) == 2 for operation in ropes))
        self.assertIs(ropes[0].operands[2], ropes[1].operands[2])
        self.assertIs(ropes[0].operands[3], ropes[1].operands[3])
        self.assertFalse(second.changed)

    def test_verifier_rejects_odd_head_dimension(self) -> None:
        builder = IRBuilder()
        tensor_type = TensorType(
            (StaticDim(1), StaticDim(4), StaticDim(3), StaticDim(7)),
            "f32",
        )
        position_type = TensorType(
            (StaticDim(1), StaticDim(3), StaticDim(7)),
            "f32",
        )
        query = builder.argument(tensor_type, "query")
        key = builder.argument(tensor_type, "key")
        cosine = builder.argument(position_type, "cosine")
        sine = builder.argument(position_type, "sine")
        rope = builder.emit(
            "serve.rope",
            [query, key, cosine, sine],
            [tensor_type, tensor_type],
            attributes={
                "head_dim": 7,
                "variant": "qwen2_half_rotation",
            },
        )
        module = Module([Function("main", builder.block, rope.results)])

        with self.assertRaisesRegex(VerificationError, "偶数静态Head Dim"):
            verify_module(module)


@unittest.skipUnless(torch.cuda.is_available(), "需要CUDA GPU")
class TritonRoPETest(unittest.TestCase):
    def test_kernel_matches_dynamic_prefill_and_decode_shapes(self) -> None:
        configurations = (
            (1, 1, 8),
            (2, 7, 16),
            (1, 33, 32),
        )
        for dtype in (torch.float16, torch.float32):
            for batch, tokens, head_dim in configurations:
                with self.subTest(
                    dtype=dtype,
                    batch=batch,
                    tokens=tokens,
                    head_dim=head_dim,
                ):
                    torch.manual_seed(batch * 1000 + tokens * 10 + head_dim)
                    query = torch.randn(
                        batch,
                        4,
                        tokens,
                        head_dim,
                        device="cuda",
                        dtype=dtype,
                    )
                    key = torch.randn(
                        batch,
                        2,
                        tokens,
                        head_dim,
                        device="cuda",
                        dtype=dtype,
                    )
                    angles = torch.randn(
                        batch,
                        tokens,
                        head_dim,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    cosine = angles.cos().to(dtype)
                    sine = angles.sin().to(dtype)

                    actual_query, actual_key = triton_rope(
                        query,
                        key,
                        cosine,
                        sine,
                    )
                    expected_query = _reference_rope(query, cosine, sine)
                    expected_key = _reference_rope(key, cosine, sine)

                    tolerance = 3e-3 if dtype == torch.float16 else 2e-6
                    torch.testing.assert_close(
                        actual_query,
                        expected_query,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    torch.testing.assert_close(
                        actual_key,
                        expected_key,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_strict_executor_runs_fully_lowered_rope(self) -> None:
        device = str(torch.device("cuda", torch.cuda.current_device()))
        builder = IRBuilder()
        query_type = TensorType(
            (StaticDim(2), StaticDim(4), StaticDim(5), StaticDim(16)),
            "f32",
            device,
        )
        key_type = TensorType(
            (StaticDim(2), StaticDim(2), StaticDim(5), StaticDim(16)),
            "f32",
            device,
        )
        position_type = TensorType(
            (StaticDim(2), StaticDim(5), StaticDim(16)),
            "f32",
            device,
        )
        query = builder.argument(query_type, "query")
        key = builder.argument(key_type, "key")
        cosine = builder.argument(position_type, "cosine")
        sine = builder.argument(position_type, "sine")
        rope = builder.emit(
            "serve.rope",
            [query, key, cosine, sine],
            [query_type, key_type],
            attributes={
                "head_dim": 16,
                "variant": "qwen2_half_rotation",
            },
        )
        module = Module([Function("main", builder.block, rope.results)])
        LowerToKernelIRPass().run(module)
        query_tensor = torch.randn(2, 4, 5, 16, device=device)
        key_tensor = torch.randn(2, 2, 5, 16, device=device)
        angles = torch.randn(2, 5, 16, device=device)
        cosine_tensor = angles.cos()
        sine_tensor = angles.sin()

        actual_query, actual_key = TritonExecutor(strict=True).run(
            module,
            [query_tensor, key_tensor, cosine_tensor, sine_tensor],
        ).outputs

        torch.testing.assert_close(
            actual_query,
            _reference_rope(query_tensor, cosine_tensor, sine_tensor),
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(
            actual_key,
            _reference_rope(key_tensor, cosine_tensor, sine_tensor),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_pytorch_compatible_mode_is_bf16_bitwise_equal(self) -> None:
        device = str(torch.device("cuda", torch.cuda.current_device()))
        builder = IRBuilder()
        query_type = TensorType(
            (StaticDim(2), StaticDim(14), StaticDim(7), StaticDim(64)),
            "bf16",
            device,
        )
        key_type = TensorType(
            (StaticDim(2), StaticDim(2), StaticDim(7), StaticDim(64)),
            "bf16",
            device,
        )
        position_type = TensorType(
            (StaticDim(2), StaticDim(7), StaticDim(64)),
            "bf16",
            device,
        )
        query = builder.argument(query_type, "query")
        key = builder.argument(key_type, "key")
        cosine = builder.argument(position_type, "cosine")
        sine = builder.argument(position_type, "sine")
        rope = builder.emit(
            "serve.rope",
            [query, key, cosine, sine],
            [query_type, key_type],
            attributes={
                "head_dim": 64,
                "variant": "qwen2_half_rotation",
            },
        )
        module = Module([Function("main", builder.block, rope.results)])
        LowerToKernelIRPass(
            numerical_mode="pytorch_compatible"
        ).run(module)
        torch.manual_seed(2026)
        query_tensor = torch.randn(
            2, 14, 7, 64, device=device, dtype=torch.bfloat16
        )
        key_tensor = torch.randn(
            2, 2, 7, 64, device=device, dtype=torch.bfloat16
        )
        angles = torch.randn(2, 7, 64, device=device)
        cosine_tensor = angles.cos().bfloat16()
        sine_tensor = angles.sin().bfloat16()

        actual_query, actual_key = TritonExecutor(strict=True).run(
            module,
            [query_tensor, key_tensor, cosine_tensor, sine_tensor],
        ).outputs

        self.assertTrue(
            torch.equal(
                actual_query,
                _reference_rope(query_tensor, cosine_tensor, sine_tensor),
            )
        )
        self.assertTrue(
            torch.equal(
                actual_key,
                _reference_rope(key_tensor, cosine_tensor, sine_tensor),
            )
        )


if __name__ == "__main__":
    unittest.main()
