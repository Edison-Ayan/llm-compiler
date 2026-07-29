from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.ir import (
    SymbolicDim,
    TensorType,
    format_module,
    verify_module,
)
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)


class ExportImporterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )
        model = TinyDecoderBlock(cls.config).eval()
        cls.program = export_decoder(
            model,
            make_inputs(cls.config, batch=2, sequence=8),
            max_batch=8,
            max_sequence=64,
        )
        cls.module = import_exported_program(cls.program)

    def test_imported_module_passes_verification(self) -> None:
        verify_module(self.module)
        function = self.module.functions[0]
        self.assertEqual(function.name, "decoder")
        self.assertEqual(len(function.returns), 1)
        self.assertGreater(len(function.block.operations), 50)

    def test_importer_preserves_main_aten_patterns(self) -> None:
        names = {
            operation.name
            for operation in self.module.functions[0].block.operations
        }
        self.assertIn("aten.linear.default", names)
        self.assertIn("aten.softmax.int", names)
        self.assertIn("aten.silu.default", names)
        self.assertIn("builtin.getitem", names)
        self.assertNotIn("serve.external", names)

    def test_importer_preserves_dynamic_dimensions_and_bounds(self) -> None:
        function = self.module.functions[0]
        hidden = next(
            argument
            for argument in function.block.arguments
            if "hidden_states" in argument.name
        )
        self.assertIsInstance(hidden.type, TensorType)
        dimensions = hidden.type.shape
        self.assertIsInstance(dimensions[0], SymbolicDim)
        self.assertIsInstance(dimensions[1], SymbolicDim)
        self.assertIn("8", dimensions[0].bounds or "")
        self.assertIn("64", dimensions[1].bounds or "")

    def test_text_ir_is_deterministic_and_inspectable(self) -> None:
        first = format_module(self.module)
        second = format_module(self.module)
        self.assertEqual(first, second)
        self.assertIn("sym_size", first)
        self.assertIn("tensor<", first)
        self.assertIn("return", first)


if __name__ == "__main__":
    unittest.main()

