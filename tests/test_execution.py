from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.execution import (
    ExecutionError,
    ReferenceExecutor,
    bind_exported_program_arguments,
)
from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)
from stateful_llm_compiler.optimizer import default_pass_manager


def build_program_and_ir(dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    config = DecoderConfig(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
    )
    model = TinyDecoderBlock(config).eval().to(dtype)
    inputs = tuple(
        tensor.to(dtype) for tensor in make_inputs(config, 2, 8)
    )
    program = export_decoder(
        model,
        inputs,
        max_batch=8,
        max_sequence=64,
    )
    module = import_exported_program(program)
    default_pass_manager().run(module)
    return config, program, module


class ReferenceExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.program, cls.module = build_program_and_ir()
        cls.executor = ReferenceExecutor()

    def assert_matches_export(
        self,
        batch: int,
        sequence: int,
        *,
        dtype: torch.dtype = torch.float32,
        program=None,
        module=None,
    ) -> float:
        program = program or self.program
        module = module or self.module
        inputs = tuple(
            tensor.to(dtype)
            for tensor in make_inputs(
                self.config, batch, sequence, seed=batch * 100 + sequence
            )
        )
        arguments = bind_exported_program_arguments(program, inputs)
        with torch.no_grad():
            expected = program.module()(*inputs)
            result = self.executor.run(module, arguments)
        actual = result.outputs[0]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(result.executed_operations), 33)
        return float((actual.float() - expected.float()).abs().max())

    def test_optimized_ir_matches_multiple_dynamic_shapes(self) -> None:
        errors = [
            self.assert_matches_export(batch, sequence)
            for batch, sequence in [(1, 1), (2, 8), (3, 11), (4, 17)]
        ]
        self.assertLessEqual(max(errors), 1e-6)

    def test_symbolic_dimensions_are_bound_once(self) -> None:
        inputs = make_inputs(self.config, 3, 7)
        arguments = bind_exported_program_arguments(self.program, inputs)
        result = self.executor.run(self.module, arguments)
        self.assertEqual(sorted(result.symbolic_dimensions.values()), [3, 7])

    def test_shape_guard_rejects_batch_above_upper_bound(self) -> None:
        inputs = make_inputs(self.config, 9, 8)
        arguments = bind_exported_program_arguments(self.program, inputs)
        with self.assertRaisesRegex(ExecutionError, "大于上界 8"):
            self.executor.run(self.module, arguments)

    def test_shape_guard_rejects_inconsistent_mask_sequence(self) -> None:
        hidden, _ = make_inputs(self.config, 2, 8)
        _, wrong_mask = make_inputs(self.config, 2, 7)
        arguments = bind_exported_program_arguments(
            self.program, (hidden, wrong_mask)
        )
        with self.assertRaisesRegex(ExecutionError, "相等约束"):
            self.executor.run(self.module, arguments)

    def test_fp16_optimized_ir_matches_export(self) -> None:
        config, program, module = build_program_and_ir(torch.float16)
        original_config = self.config
        try:
            self.config = config
            error = self.assert_matches_export(
                2,
                5,
                dtype=torch.float16,
                program=program,
                module=module,
            )
        finally:
            self.config = original_config
        self.assertLessEqual(error, 1e-3)


if __name__ == "__main__":
    unittest.main()
