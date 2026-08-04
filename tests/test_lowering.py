from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.backends import TritonExecutor
from stateful_llm_compiler.compiler import (
    CompileOptions,
    compile_exported_program,
)
from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.ir import (
    Block,
    Function,
    IRBuilder,
    Module,
    StaticDim,
    TensorType,
)
from stateful_llm_compiler.lowering import (
    LoweringCoverageError,
    LowerToKernelIRPass,
    analyze_lowering_coverage,
)
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)


class KernelIRLoweringTest(unittest.TestCase):
    def test_lowering_rewrites_supported_ops_and_reports_gap(self) -> None:
        tensor_type = TensorType((StaticDim(2), StaticDim(8)), "f32")
        builder = IRBuilder()
        tensor = builder.argument(tensor_type, "input")
        weight = builder.argument(
            TensorType((StaticDim(8),), "f32"),
            "weight",
        )
        norm = builder.emit(
            "serve.rms_norm",
            [tensor, weight],
            [tensor_type],
            attributes={"epsilon": 1e-6, "output_dtype": "f32"},
        )
        output = builder.emit(
            "aten.neg.default",
            [norm.results[0]],
            [tensor_type],
        )
        module = Module(
            [Function("main", builder.block, [output.results[0]])]
        )

        result = LowerToKernelIRPass().run(module)
        coverage = analyze_lowering_coverage(module)

        self.assertTrue(result.changed)
        self.assertEqual(
            [operation.name for operation in builder.block.operations],
            ["kernel.triton.rms_norm", "aten.neg.default"],
        )
        self.assertEqual(
            builder.block.operations[0].attributes["lowered_from"],
            "serve.rms_norm",
        )
        self.assertEqual(coverage.total_operations, 2)
        self.assertEqual(coverage.lowered_operations, 1)
        self.assertEqual(coverage.unlowered_by_name, {"aten.neg.default": 1})
        self.assertEqual(coverage.coverage, 0.5)

    def test_lowering_is_idempotent(self) -> None:
        tensor_type = TensorType((StaticDim(1), StaticDim(8)), "f32")
        builder = IRBuilder()
        tensor = builder.argument(tensor_type, "input")
        weight = builder.argument(
            TensorType((StaticDim(8),), "f32"),
            "weight",
        )
        result = builder.emit(
            "serve.rms_norm",
            [tensor, weight],
            [tensor_type],
            attributes={"epsilon": 1e-6},
        )
        module = Module([Function("main", builder.block, result.results)])

        first = LowerToKernelIRPass().run(module)
        second = LowerToKernelIRPass().run(module)

        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(second.statistics["lowered"], 0)

    def test_unknown_kernel_name_is_not_counted_as_executable(self) -> None:
        tensor_type = TensorType((StaticDim(2),), "f32")
        builder = IRBuilder()
        tensor = builder.argument(tensor_type, "input")
        output = builder.emit(
            "kernel.triton.not_registered",
            [tensor],
            [tensor_type],
        )
        module = Module([Function("main", builder.block, output.results)])

        coverage = analyze_lowering_coverage(module)

        self.assertEqual(coverage.lowered_operations, 0)
        self.assertEqual(
            coverage.unlowered_by_name,
            {"kernel.triton.not_registered": 1},
        )

    def test_large_static_linear_selects_cublas_backend(self) -> None:
        input_type = TensorType(
            (StaticDim(2), StaticDim(2), StaticDim(4864)),
            "bf16",
        )
        weight_type = TensorType(
            (StaticDim(896), StaticDim(4864)),
            "bf16",
        )
        output_type = TensorType(
            (StaticDim(2), StaticDim(2), StaticDim(896)),
            "bf16",
        )
        builder = IRBuilder()
        tensor = builder.argument(input_type, "input")
        weight = builder.argument(weight_type, "weight")
        output = builder.emit(
            "serve.linear",
            [tensor, weight],
            [output_type],
            attributes={
                "input_features": 4864,
                "output_features": 896,
                "has_bias": False,
            },
        )
        module = Module([Function("main", builder.block, output.results)])

        LowerToKernelIRPass().run(module)
        coverage = analyze_lowering_coverage(module)

        self.assertEqual(output.name, "kernel.cublas.linear")
        self.assertEqual(
            output.attributes["backend_selection"],
            "large_static_gemm",
        )
        self.assertEqual(
            coverage.lowered_by_name,
            {"kernel.cublas.linear": 1},
        )

    def test_strict_executor_rejects_aten_before_execution(self) -> None:
        tensor_type = TensorType((StaticDim(2),), "f32")
        builder = IRBuilder()
        tensor = builder.argument(tensor_type, "input")
        output = builder.emit(
            "aten.neg.default",
            [tensor],
            [tensor_type],
        )
        module = Module([Function("main", builder.block, output.results)])

        with self.assertRaisesRegex(
            LoweringCoverageError,
            r"aten\.neg\.default×1",
        ):
            TritonExecutor(strict=True).run(module, [torch.ones(2)])

    def test_strict_executor_accepts_zero_operation_graph(self) -> None:
        tensor_type = TensorType((StaticDim(2),), "f32")
        argument = IRBuilder(Block()).argument(tensor_type, "input")
        block = Block(arguments=[argument])
        module = Module([Function("main", block, [argument])])
        tensor = torch.ones(2)

        result = TritonExecutor(strict=True).run(module, [tensor])

        self.assertIs(result.outputs[0], tensor)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA GPU")
    def test_strict_executor_runs_fully_lowered_triton_graph(self) -> None:
        device = str(torch.device("cuda", torch.cuda.current_device()))
        tensor_type = TensorType(
            (StaticDim(4), StaticDim(32)),
            "f32",
            device=device,
        )
        weight_type = TensorType(
            (StaticDim(32),),
            "f32",
            device=device,
        )
        builder = IRBuilder()
        tensor = builder.argument(tensor_type, "input")
        weight = builder.argument(weight_type, "weight")
        output = builder.emit(
            "serve.rms_norm",
            [tensor, weight],
            [tensor_type],
            attributes={"epsilon": 1e-6, "output_dtype": "f32"},
        )
        module = Module([Function("main", builder.block, output.results)])
        LowerToKernelIRPass().run(module)
        input_tensor = torch.randn(4, 32, device="cuda")
        norm_weight = torch.randn(32, device="cuda")

        actual = TritonExecutor(strict=True).run(
            module,
            [input_tensor, norm_weight],
        ).outputs[0]
        expected = torch.nn.functional.rms_norm(
            input_tensor,
            (32,),
            norm_weight,
            1e-6,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


class CompilerEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )
        model = TinyDecoderBlock(config).eval()
        cls.program = export_decoder(
            model,
            make_inputs(config, batch=2, sequence=8),
            max_batch=8,
            max_sequence=64,
        )

    def test_compile_entry_runs_graph_optimization_then_lowering(self) -> None:
        artifact = compile_exported_program(self.program)
        operation_names = [
            operation.name
            for operation in artifact.module.functions[0].block.operations
        ]

        self.assertEqual(
            [result.name for result in artifact.pass_results],
            [
                "remove-export-assertions",
                "normalize-linear",
                "fuse-rmsnorm",
                "fuse-rope",
                "fuse-prefill-attention",
                "lower-to-kernel-ir",
            ],
        )
        self.assertEqual(operation_names.count("kernel.triton.rms_norm"), 2)
        self.assertNotIn("serve.rms_norm", operation_names)
        self.assertGreater(artifact.coverage.lowered_operations, 0)
        self.assertGreater(artifact.coverage.unlowered_operations, 0)

    def test_compile_entry_strict_mode_rejects_incomplete_backend(self) -> None:
        with self.assertRaises(LoweringCoverageError) as context:
            compile_exported_program(
                self.program,
                options=CompileOptions(require_full_lowering=True),
            )

        self.assertIn(
            "aten.transpose.int",
            context.exception.coverage.unlowered_by_name,
        )


if __name__ == "__main__":
    unittest.main()
