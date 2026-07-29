from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.ir import Operation, UnknownType, Value, verify_module
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)
from stateful_llm_compiler.pass_manager import PassManager
from stateful_llm_compiler.passes import (
    FuseRMSNormPass,
    RemoveExportAssertionsPass,
)


class OptimizationPassTest(unittest.TestCase):
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

    def make_module(self):
        return import_exported_program(self.program)

    def test_pipeline_removes_assertions_and_fuses_two_rmsnorms(self) -> None:
        module = self.make_module()
        manager = PassManager(
            [RemoveExportAssertionsPass(), FuseRMSNormPass()]
        )
        results = manager.run(module)
        operations = module.functions[0].block.operations
        names = [operation.name for operation in operations]

        self.assertGreater(results[0].statistics["removed"], 0)
        self.assertEqual(results[1].statistics["fused"], 2)
        self.assertEqual(names.count("serve.rms_norm"), 2)
        self.assertNotIn(
            "aten._assert_tensor_metadata.default", names
        )
        optimized_text = str(
            [
                operation.attributes
                for operation in operations
            ]
        )
        self.assertNotIn("'ssa': '%v15'", optimized_text)
        self.assertNotIn("'ssa': '%v58'", optimized_text)
        self.assertLess(
            results[-1].operations_after,
            results[0].operations_before,
        )
        verify_module(module)

    def test_pipeline_is_idempotent(self) -> None:
        module = self.make_module()
        manager = PassManager(
            [RemoveExportAssertionsPass(), FuseRMSNormPass()]
        )
        manager.run(module)
        second = manager.run(module)

        self.assertFalse(second[0].changed)
        self.assertFalse(second[1].changed)
        self.assertEqual(second[0].statistics["removed"], 0)
        self.assertEqual(second[1].statistics["fused"], 0)

    def test_fusion_rejects_intermediate_value_escape(self) -> None:
        module = self.make_module()
        RemoveExportAssertionsPass().run(module)
        function = module.functions[0]
        power = next(
            operation
            for operation in function.block.operations
            if operation.name == "aten.pow.Tensor_Scalar"
        )
        escaped = Operation(
            "test.observe",
            [power.results[0]],
            [Value("%observed", UnknownType("test"))],
        )
        function.block.operations.append(escaped)

        result = FuseRMSNormPass().run(module)

        self.assertEqual(result.statistics["fused"], 0)
        self.assertEqual(result.statistics["rejected_escape"], 1)
        self.assertNotIn(
            "serve.rms_norm",
            [operation.name for operation in function.block.operations],
        )
        verify_module(module)

    def test_cleanup_keeps_assertion_when_result_is_used(self) -> None:
        module = self.make_module()
        function = module.functions[0]
        assertion = next(
            operation
            for operation in function.block.operations
            if operation.name
            == "aten._assert_tensor_metadata.default"
        )
        observer = Operation(
            "test.observe",
            [assertion.results[0]],
            [Value("%assert_observed", UnknownType("test"))],
        )
        function.block.operations.append(observer)

        result = RemoveExportAssertionsPass().run(module)

        self.assertEqual(result.statistics["skipped"], 1)
        self.assertIn(assertion, function.block.operations)
        verify_module(module)


if __name__ == "__main__":
    unittest.main()
