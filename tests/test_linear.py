from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.backends import TritonExecutor, triton_linear
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
from stateful_llm_compiler.passes import NormalizeLinearPass


def _make_linear_module(
    *,
    batch: int = 2,
    tokens: int = 3,
    input_features: int = 32,
    output_features: int = 48,
    bias: bool = True,
    device: str = "cpu",
) -> Module:
    builder = IRBuilder()
    input_type = TensorType(
        (
            StaticDim(batch),
            StaticDim(tokens),
            StaticDim(input_features),
        ),
        "f32",
        device,
    )
    weight_type = TensorType(
        (StaticDim(output_features), StaticDim(input_features)),
        "f32",
        device,
    )
    output_type = TensorType(
        (
            StaticDim(batch),
            StaticDim(tokens),
            StaticDim(output_features),
        ),
        "f32",
        device,
    )
    tensor = builder.argument(input_type, "input")
    weight = builder.argument(weight_type, "weight")
    operands = [tensor, weight]
    if bias:
        operands.append(
            builder.argument(
                TensorType((StaticDim(output_features),), "f32", device),
                "bias",
            )
        )
    output = builder.emit(
        "aten.linear.default",
        operands,
        [output_type],
    )
    return Module([Function("main", builder.block, output.results)])


class LinearIRTest(unittest.TestCase):
    def test_normalize_linear_builds_high_level_contract(self) -> None:
        module = _make_linear_module()

        result = NormalizeLinearPass().run(module)
        operation = module.functions[0].block.operations[0]

        self.assertTrue(result.changed)
        self.assertEqual(result.statistics["normalized"], 1)
        self.assertEqual(operation.name, "serve.linear")
        self.assertEqual(operation.attributes["input_features"], 32)
        self.assertEqual(operation.attributes["output_features"], 48)
        self.assertTrue(operation.attributes["has_bias"])
        verify_module(module)

    def test_normalize_and_lower_are_idempotent(self) -> None:
        module = _make_linear_module(bias=False)

        first_normalize = NormalizeLinearPass().run(module)
        second_normalize = NormalizeLinearPass().run(module)
        first_lower = LowerToKernelIRPass().run(module)
        second_lower = LowerToKernelIRPass().run(module)

        self.assertTrue(first_normalize.changed)
        self.assertFalse(second_normalize.changed)
        self.assertTrue(first_lower.changed)
        self.assertFalse(second_lower.changed)
        self.assertEqual(
            module.functions[0].block.operations[0].name,
            "kernel.triton.linear",
        )
        verify_module(module)

    def test_verifier_rejects_wrong_linear_result_shape(self) -> None:
        module = _make_linear_module()
        NormalizeLinearPass().run(module)
        operation = module.functions[0].block.operations[0]
        operation.results[0].type = TensorType(
            (StaticDim(2), StaticDim(3), StaticDim(47)),
            "f32",
        )

        with self.assertRaisesRegex(
            VerificationError,
            "结果最后一维必须等于 weight 的 N",
        ):
            verify_module(module)


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA GPU")
class TritonLinearTest(unittest.TestCase):
    def test_kernel_matches_pytorch_for_dynamic_shapes_and_bias(self) -> None:
        configurations = [
            (1, 1, 32, 48, True),
            (2, 7, 32, 64, False),
            (3, 11, 64, 32, True),
        ]
        for dtype in (torch.float16, torch.float32):
            for batch, tokens, k, n, has_bias in configurations:
                with self.subTest(
                    dtype=dtype,
                    batch=batch,
                    tokens=tokens,
                    k=k,
                    n=n,
                    bias=has_bias,
                ):
                    torch.manual_seed(batch * 1000 + tokens * 100 + k + n)
                    tensor = torch.randn(
                        batch,
                        tokens,
                        k,
                        device="cuda",
                        dtype=dtype,
                    )
                    weight = torch.randn(n, k, device="cuda", dtype=dtype)
                    bias = (
                        torch.randn(n, device="cuda", dtype=dtype)
                        if has_bias
                        else None
                    )

                    actual = triton_linear(tensor, weight, bias)
                    expected = torch.nn.functional.linear(tensor, weight, bias)

                    tolerance = 3e-2 if dtype == torch.float16 else 3e-5
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_strict_executor_runs_linear_without_fallback(self) -> None:
        device = str(torch.device("cuda", torch.cuda.current_device()))
        module = _make_linear_module(device=device)
        NormalizeLinearPass().run(module)
        LowerToKernelIRPass().run(module)
        tensor = torch.randn(2, 3, 32, device=device)
        weight = torch.randn(48, 32, device=device)
        bias = torch.randn(48, device=device)

        actual = TritonExecutor(strict=True).run(
            module,
            [tensor, weight, bias],
        ).outputs[0]
        expected = torch.nn.functional.linear(tensor, weight, bias)

        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
