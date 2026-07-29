from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.backends import (
    TritonExecutor,
    triton_rms_norm,
)
from stateful_llm_compiler.execution import bind_exported_program_arguments
from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)
from stateful_llm_compiler.optimizer import default_pass_manager


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA GPU")
class TritonRMSNormTest(unittest.TestCase):
    def assert_kernel_matches(
        self,
        rows: int,
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(rows + hidden_size)
        tensor = torch.randn(
            rows, hidden_size, device="cuda", dtype=dtype
        )
        weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
        expected = torch.nn.functional.rms_norm(
            tensor.float(),
            (hidden_size,),
            weight.float(),
            1e-6,
        ).to(dtype)
        actual = triton_rms_norm(
            tensor, weight, epsilon=1e-6, output_dtype=dtype
        )
        tolerance = 2e-3 if dtype == torch.float16 else 2e-5
        torch.testing.assert_close(
            actual, expected, rtol=tolerance, atol=tolerance
        )

    def test_kernel_matches_fp16_dynamic_rows(self) -> None:
        for rows in (1, 7, 64, 257):
            self.assert_kernel_matches(rows, 1536, torch.float16)

    def test_kernel_matches_fp32_dynamic_rows(self) -> None:
        for rows in (1, 7, 64):
            self.assert_kernel_matches(rows, 1536, torch.float32)

    def test_kernel_supports_fp32_compute_to_fp16_output(self) -> None:
        tensor = torch.randn(13, 64, device="cuda")
        weight = torch.randn(64, device="cuda")
        output = triton_rms_norm(
            tensor, weight, epsilon=1e-6, output_dtype="f16"
        )
        self.assertEqual(output.dtype, torch.float16)

    def test_full_optimized_graph_matches_exported_program(self) -> None:
        for dtype in (torch.float16, torch.float32):
            torch.manual_seed(0)
            config = DecoderConfig(
                hidden_size=32,
                num_heads=4,
                num_kv_heads=2,
                intermediate_size=64,
            )
            model = TinyDecoderBlock(config).eval().to(
                device="cuda", dtype=dtype
            )
            example = tuple(
                tensor.to(device="cuda", dtype=dtype)
                for tensor in make_inputs(config, 2, 8)
            )
            program = export_decoder(
                model,
                example,
                max_batch=8,
                max_sequence=64,
            )
            module = import_exported_program(program)
            default_pass_manager().run(module)
            executor = TritonExecutor()

            for batch, sequence in ((1, 1), (2, 8), (3, 11)):
                inputs = tuple(
                    tensor.to(device="cuda", dtype=dtype)
                    for tensor in make_inputs(
                        config, batch, sequence, seed=sequence
                    )
                )
                arguments = bind_exported_program_arguments(
                    program, inputs
                )
                with torch.no_grad():
                    expected = program.module()(*inputs)
                    actual = executor.run(module, arguments).outputs[0]
                tolerance = 3e-3 if dtype == torch.float16 else 3e-5
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=tolerance,
                    atol=tolerance,
                )


if __name__ == "__main__":
    unittest.main()

