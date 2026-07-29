from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.graph_summary import summarize_exported_program
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)


class ExportFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )
        cls.model = TinyDecoderBlock(cls.config).eval()
        cls.program = export_decoder(
            cls.model,
            make_inputs(cls.config, batch=2, sequence=8),
            max_batch=8,
            max_sequence=64,
        )

    def assert_export_matches(self, batch: int, sequence: int) -> None:
        inputs = make_inputs(
            self.config, batch=batch, sequence=sequence, seed=sequence
        )
        with torch.no_grad():
            expected = self.model(*inputs)
            actual = self.program.module()(*inputs)
        torch.testing.assert_close(actual, expected)

    def test_export_matches_capture_shape(self) -> None:
        self.assert_export_matches(batch=2, sequence=8)

    def test_export_matches_different_dynamic_shape(self) -> None:
        self.assert_export_matches(batch=3, sequence=11)

    def test_export_records_symbolic_ranges(self) -> None:
        constraints = {
            str(symbol): str(value)
            for symbol, value in self.program.range_constraints.items()
        }
        self.assertEqual(len(constraints), 2)
        joined = " ".join(constraints.values())
        self.assertIn("8", joined)
        self.assertIn("64", joined)

    def test_graph_summary_exposes_compiler_patterns(self) -> None:
        summary = summarize_exported_program(self.program)
        targets = " ".join(summary["target_counts"])
        self.assertEqual(summary["dialect"], "functional_aten")
        self.assertIn("aten::linear", targets)
        self.assertIn("aten::softmax", targets)
        self.assertIn("aten::silu", targets)
        self.assertGreater(len(summary["nodes"]), 20)

    def test_saved_program_preserves_dynamic_shape_execution(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.pt2"
            torch.export.save(self.program, path)
            loaded = torch.export.load(path)
            inputs = make_inputs(self.config, batch=4, sequence=7, seed=9)
            with torch.no_grad():
                expected = self.model(*inputs)
                actual = loaded.module()(*inputs)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
