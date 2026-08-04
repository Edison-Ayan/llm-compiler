from __future__ import annotations

import math
import unittest

import torch

from stateful_llm_compiler.backends import (
    TritonExecutor,
    triton_prefill_attention,
)
from stateful_llm_compiler.ir import (
    Function,
    IRBuilder,
    Module,
    StaticDim,
    TensorType,
)
from stateful_llm_compiler.lowering import LowerToKernelIRPass


def _reference_attention(query, key, value, mask, scale):
    groups = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(groups, dim=1)
    expanded_value = value.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query, expanded_key.transpose(-2, -1)) * scale
    probabilities = torch.softmax(scores.float() + mask.float(), dim=-1)
    return torch.matmul(probabilities.to(query.dtype), expanded_value)


@unittest.skipUnless(torch.cuda.is_available(), "需要CUDA GPU")
class TritonPrefillAttentionTest(unittest.TestCase):
    def test_kernel_matches_dynamic_causal_gqa(self) -> None:
        configurations = ((1, 2), (2, 7), (1, 33))
        for dtype in (torch.float16, torch.float32):
            for batch, tokens in configurations:
                with self.subTest(dtype=dtype, batch=batch, tokens=tokens):
                    torch.manual_seed(batch * 100 + tokens)
                    query = torch.randn(
                        batch,
                        4,
                        tokens,
                        16,
                        device="cuda",
                        dtype=dtype,
                    )
                    key = torch.randn(
                        batch,
                        2,
                        tokens,
                        16,
                        device="cuda",
                        dtype=dtype,
                    )
                    value = torch.randn_like(key)
                    future = torch.triu(
                        torch.ones(
                            tokens,
                            tokens,
                            device="cuda",
                            dtype=torch.bool,
                        ),
                        diagonal=1,
                    )
                    mask = torch.zeros(
                        batch,
                        1,
                        tokens,
                        tokens,
                        device="cuda",
                    ).masked_fill(future, float("-inf"))
                    scale = 1.0 / math.sqrt(16)

                    actual = triton_prefill_attention(
                        query,
                        key,
                        value,
                        mask,
                        scale=scale,
                    )
                    expected = _reference_attention(
                        query,
                        key,
                        value,
                        mask,
                        scale,
                    )

                    tolerance = 2e-3 if dtype == torch.float16 else 2e-5
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_kernel_respects_mask_instead_of_hardcoding_causal(self) -> None:
        query = torch.randn(1, 4, 4, 16, device="cuda")
        key = torch.randn(1, 2, 4, 16, device="cuda")
        value = torch.randn_like(key)
        # 全零Mask表示双向Attention，用来防止融合错误地改变通用Mask语义。
        mask = torch.zeros(1, 1, 4, 4, device="cuda")
        scale = 0.25

        actual = triton_prefill_attention(
            query,
            key,
            value,
            mask,
            scale=scale,
        )
        expected = _reference_attention(query, key, value, mask, scale)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_strict_executor_runs_fully_lowered_attention(self) -> None:
        device = str(torch.device("cuda", torch.cuda.current_device()))
        builder = IRBuilder()
        query_type = TensorType(
            (
                StaticDim(1),
                StaticDim(4),
                StaticDim(5),
                StaticDim(16),
            ),
            "f32",
            device,
        )
        kv_type = TensorType(
            (
                StaticDim(1),
                StaticDim(2),
                StaticDim(5),
                StaticDim(16),
            ),
            "f32",
            device,
        )
        mask_type = TensorType(
            (
                StaticDim(1),
                StaticDim(1),
                StaticDim(5),
                StaticDim(5),
            ),
            "f32",
            device,
        )
        query = builder.argument(query_type, "query")
        key = builder.argument(kv_type, "key")
        value = builder.argument(kv_type, "value")
        mask = builder.argument(mask_type, "mask")
        attention = builder.emit(
            "serve.prefill_attention",
            [query, key, value, mask],
            [query_type],
            attributes={
                "groups": 2,
                "scale": 0.25,
                "causal": "mask",
            },
        )
        module = Module(
            [Function("main", builder.block, attention.results)]
        )
        LowerToKernelIRPass().run(module)
        query_tensor = torch.randn(1, 4, 5, 16, device=device)
        key_tensor = torch.randn(1, 2, 5, 16, device=device)
        value_tensor = torch.randn_like(key_tensor)
        mask_tensor = torch.zeros(1, 1, 5, 5, device=device)

        actual = TritonExecutor(strict=True).run(
            module,
            [query_tensor, key_tensor, value_tensor, mask_tensor],
        ).outputs[0]
        expected = _reference_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            mask_tensor,
            0.25,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_pytorch_compatible_mode_is_bf16_bitwise_equal(self) -> None:
        device = str(torch.device("cuda", torch.cuda.current_device()))
        builder = IRBuilder()
        query_type = TensorType(
            (StaticDim(2), StaticDim(14), StaticDim(7), StaticDim(64)),
            "bf16",
            device,
        )
        kv_type = TensorType(
            (StaticDim(2), StaticDim(2), StaticDim(7), StaticDim(64)),
            "bf16",
            device,
        )
        mask_type = TensorType(
            (StaticDim(2), StaticDim(1), StaticDim(7), StaticDim(7)),
            "f32",
            device,
        )
        query = builder.argument(query_type, "query")
        key = builder.argument(kv_type, "key")
        value = builder.argument(kv_type, "value")
        mask = builder.argument(mask_type, "mask")
        attention = builder.emit(
            "serve.prefill_attention",
            [query, key, value, mask],
            [query_type],
            attributes={
                "groups": 7,
                "scale": 64**-0.5,
                "causal": "mask",
            },
        )
        module = Module(
            [Function("main", builder.block, attention.results)]
        )
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
        value_tensor = torch.randn_like(key_tensor)
        future = torch.triu(
            torch.ones(7, 7, device=device, dtype=torch.bool),
            diagonal=1,
        )
        mask_tensor = torch.zeros(2, 1, 7, 7, device=device).masked_fill(
            future,
            float("-inf"),
        )

        actual = TritonExecutor(strict=True).run(
            module,
            [query_tensor, key_tensor, value_tensor, mask_tensor],
        ).outputs[0]
        expected = _reference_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            mask_tensor,
            64**-0.5,
        )

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
