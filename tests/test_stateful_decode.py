from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.execution import (
    ExecutionError,
    KVCacheState,
    ReferenceExecutor,
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import export_stateful_decode
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.ir import (
    EffectKind,
    KVStateType,
    format_module,
    verify_module,
)
from stateful_llm_compiler.model import (
    DecoderConfig,
    StatefulTinyDecoderBlock,
    TinyDecoderBlock,
    make_decode_inputs,
    make_inputs,
)
from stateful_llm_compiler.optimizer import default_pass_manager


class StatefulDecodeFrontendTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )

    def test_incremental_decode_matches_full_causal_attention(self) -> None:
        full_model = TinyDecoderBlock(self.config).eval()
        decode_model = StatefulTinyDecoderBlock(self.config).eval()
        decode_model.load_state_dict(full_model.state_dict())
        hidden, causal_mask = make_inputs(
            self.config,
            batch=2,
            sequence=6,
            seed=9,
        )

        with torch.no_grad():
            expected = full_model(hidden, causal_mask)
            key = torch.empty(
                2,
                self.config.num_kv_heads,
                0,
                self.config.head_dim,
            )
            value = torch.empty_like(key)
            outputs = []
            for token in range(hidden.shape[1]):
                mask = torch.zeros(2, 1, 1, token + 1)
                output, key, value = decode_model(
                    hidden[:, token : token + 1],
                    mask,
                    key,
                    value,
                )
                outputs.append(output)
            actual = torch.cat(outputs, dim=1)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(key.shape[2], hidden.shape[1])

    def test_export_supports_dynamic_batch_and_past_length(self) -> None:
        model = StatefulTinyDecoderBlock(self.config).eval()
        program = export_stateful_decode(
            model,
            make_decode_inputs(self.config, 2, 4),
            max_batch=8,
            max_cache_length=64,
        )

        with torch.no_grad():
            output, key, value = program.module()(
                *make_decode_inputs(self.config, 3, 9, seed=5)
            )

        self.assertEqual(output.shape, (3, 1, 32))
        self.assertEqual(key.shape, (3, 2, 10, 8))
        self.assertEqual(value.shape, key.shape)
        constraints = {
            str(symbol): str(bounds)
            for symbol, bounds in program.range_constraints.items()
        }
        self.assertTrue(any("64" in bounds for bounds in constraints.values()))


class KVStateCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )
        cls.model = StatefulTinyDecoderBlock(cls.config).eval()
        cls.program = export_stateful_decode(
            cls.model,
            make_decode_inputs(cls.config, 2, 4),
            max_batch=8,
            max_cache_length=64,
        )

    def make_module(self):
        return import_exported_program(
            self.program,
            function_name="decode",
        )

    def test_pass_materializes_state_and_effects(self) -> None:
        module = self.make_module()
        results = default_pass_manager(stateful_decode=True).run(module)
        function = module.functions[0]
        names = [
            operation.name
            for operation in function.block.operations
        ]

        self.assertEqual(results[-1].statistics["converted"], 1)
        self.assertNotIn("aten.cat.default", names)
        self.assertEqual(names.count("serve.kv.append"), 1)
        self.assertEqual(names.count("serve.kv.read"), 1)
        self.assertIsInstance(function.block.arguments[-1].type, KVStateType)
        self.assertIsInstance(function.returns[-1].type, KVStateType)
        self.assertFalse(
            any("past_key" in argument.name for argument in function.block.arguments)
        )

        append = next(
            operation
            for operation in function.block.operations
            if operation.name == "serve.kv.append"
        )
        effects = {effect.kind for effect in append.effects}
        self.assertEqual(
            effects,
            {EffectKind.READ, EffectKind.WRITE},
        )
        verify_module(module)
        text = format_module(module)
        self.assertIn("effects[read(kv.layer0), write(kv.layer0)]", text)

    def test_stateful_pipeline_is_idempotent(self) -> None:
        module = self.make_module()
        manager = default_pass_manager(stateful_decode=True)
        manager.run(module)
        second = manager.run(module)

        self.assertTrue(all(not result.changed for result in second))
        self.assertEqual(second[-1].statistics["converted"], 0)

    def test_reference_executor_preserves_four_decode_rounds(self) -> None:
        module = self.make_module()
        default_pass_manager(stateful_decode=True).run(module)
        executor = ReferenceExecutor()
        _, _, past_key, past_value = make_decode_inputs(
            self.config,
            batch=2,
            past_length=3,
            seed=7,
        )
        state = KVCacheState.from_tensors(past_key, past_value)
        original_key = state.keys[0]

        errors = []
        lengths = []
        with torch.no_grad():
            for step in range(4):
                generator = torch.Generator().manual_seed(100 + step)
                hidden = torch.randn(
                    2,
                    1,
                    self.config.hidden_size,
                    generator=generator,
                )
                mask = torch.zeros(
                    2,
                    1,
                    1,
                    state.keys[0].shape[2] + 1,
                )
                expected = self.program.module()(
                    hidden,
                    mask,
                    *state.read(0),
                )
                result = executor.run(
                    module,
                    bind_stateful_decode_arguments(
                        self.program,
                        hidden,
                        mask,
                        state,
                    ),
                )
                actual, state = result.outputs
                errors.append(float((actual - expected[0]).abs().max()))
                lengths.append(state.keys[0].shape[2])
                torch.testing.assert_close(state.keys[0], expected[1])
                torch.testing.assert_close(state.values[0], expected[2])

        self.assertEqual(errors, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(lengths, [4, 5, 6, 7])
        self.assertEqual(state.generation, 4)
        self.assertEqual(original_key.shape[2], 3)
        self.assertEqual(len(result.executed_operations), 45)

    def test_kv_runtime_rejects_incompatible_append(self) -> None:
        _, _, key, value = make_decode_inputs(
            self.config,
            batch=2,
            past_length=3,
        )
        state = KVCacheState.from_tensors(key, value)
        wrong_batch = torch.randn(3, 2, 1, 8)

        with self.assertRaisesRegex(ExecutionError, "第 0 维不匹配"):
            state.append(0, wrong_batch, wrong_batch)


if __name__ == "__main__":
    unittest.main()
