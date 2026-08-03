from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.analysis import UseDefAnalysis
from stateful_llm_compiler.execution import (
    ExecutionError,
    KVCacheState,
    PreallocatedKVCacheState,
    ReferenceExecutor,
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import export_stateful_decode
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.ir import (
    EffectKind,
    KVStateType,
    Operation,
    StaticDim,
    TensorType,
    Value,
    VerificationError,
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
from stateful_llm_compiler.passes import (
    BufferizeKVCachePass,
    FuseDecodeAttentionPass,
    MaterializeKVStatePass,
)
from stateful_llm_compiler.passes.decode_attention import _match_attention


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

    def test_materialization_rejects_wrong_cat_axis(self) -> None:
        module = self.make_module()
        key_cat = self._cache_cat(module, "%past_key")
        key_cat.attributes["args"]["tuple"][1] = 1

        self._assert_materialization_rejected(module)

    def test_materialization_rejects_reversed_cat_order(self) -> None:
        module = self.make_module()
        key_cat = self._cache_cat(module, "%past_key")
        key_cat.operands.reverse()
        key_cat.attributes["args"]["tuple"][0].reverse()

        self._assert_materialization_rejected(module)

    def test_materialization_rejects_incompatible_cache_types(self) -> None:
        module = self.make_module()
        function = module.functions[0]
        past_value = next(
            argument
            for argument in function.block.arguments
            if argument.name == "%past_value"
        )
        assert isinstance(past_value.type, TensorType)
        past_value.type = TensorType(
            past_value.type.shape,
            "f16",
            past_value.type.device,
        )

        self._assert_materialization_rejected(module)

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

    def test_bufferization_generates_position_store_and_advance(self) -> None:
        module = self.make_module()
        results = default_pass_manager(preallocate_kv=True).run(module)
        function = module.functions[0]
        names = [
            operation.name
            for operation in function.block.operations
        ]
        state_type = function.block.arguments[-1].type

        self.assertEqual(results[-2].statistics["bufferized"], 1)
        self.assertEqual(results[-1].statistics["fused"], 1)
        self.assertNotIn("serve.kv.append", names)
        self.assertNotIn("serve.kv.read", names)
        self.assertEqual(names.count("serve.kv.length"), 1)
        self.assertEqual(names.count("serve.kv.store"), 1)
        self.assertEqual(names.count("serve.kv.advance"), 1)
        self.assertEqual(names.count("serve.decode_attention"), 1)
        self.assertEqual(len(names), 37)
        advance = next(
            operation
            for operation in function.block.operations
            if operation.name == "serve.kv.advance"
        )
        self.assertEqual(advance.attributes["delta"], 1)
        self.assertIsInstance(state_type, KVStateType)
        self.assertEqual(state_type.layout, "contiguous_bshd")
        self.assertEqual(state_type.capacity, 65)
        verify_module(module)

    def test_bufferization_derives_multi_token_advance_delta(self) -> None:
        module = self.make_module()
        default_pass_manager(stateful_decode=True).run(module)
        append = next(
            operation
            for operation in module.functions[0].block.operations
            if operation.name == "serve.kv.append"
        )
        for current in append.operands[1:]:
            assert isinstance(current.type, TensorType)
            shape = list(current.type.shape)
            shape[2] = StaticDim(2)
            current.type = TensorType(
                tuple(shape),
                current.type.dtype,
                current.type.device,
            )

        result = BufferizeKVCachePass().run(module)
        advance = next(
            operation
            for operation in module.functions[0].block.operations
            if operation.name == "serve.kv.advance"
        )

        self.assertTrue(result.changed)
        self.assertEqual(advance.attributes["delta"], 2)

    def test_bufferization_is_transactional(self) -> None:
        module = self.make_module()
        default_pass_manager(stateful_decode=True).run(module)
        function = module.functions[0]
        append = next(
            operation
            for operation in function.block.operations
            if operation.name == "serve.kv.append"
        )
        bad_append = Operation(
            "serve.kv.append",
            list(append.operands),
            [Value("%bad_kv_state", append.results[0].type)],
            attributes={**append.attributes, "axis": 1},
            effects=append.effects,
        )
        function.block.operations.insert(
            function.block.operations.index(append) + 1,
            bad_append,
        )
        old_type = append.operands[0].type

        result = BufferizeKVCachePass().run(module)
        names = [operation.name for operation in function.block.operations]

        self.assertFalse(result.changed)
        self.assertEqual(result.statistics["rejected"], 1)
        self.assertEqual(result.statistics["transaction_aborted"], 1)
        self.assertEqual(names.count("serve.kv.append"), 2)
        self.assertNotIn("serve.kv.store", names)
        self.assertEqual(append.operands[0].type, old_type)

    def test_bufferization_rejects_unknown_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "暂不支持"):
            BufferizeKVCachePass(layout="paged")

    def test_attention_fusion_rejects_reversed_score_matmul(self) -> None:
        module = self.make_module()
        default_pass_manager(stateful_decode=True).run(module)
        BufferizeKVCachePass().run(module)
        function = module.functions[0]
        read = next(
            operation
            for operation in function.block.operations
            if operation.name == "serve.kv.read"
        )
        match = _match_attention(read, UseDefAnalysis(function))
        assert match is not None
        score_matmul = next(
            operation
            for operation in match.operations
            if operation.name == "aten.matmul.default"
            and operation is not match.final_matmul
        )
        score_matmul.operands.reverse()
        score_matmul.attributes["args"]["tuple"].reverse()

        result = FuseDecodeAttentionPass().run(module)

        self.assertFalse(result.changed)
        self.assertEqual(result.statistics["rejected"], 1)
        self.assertIn(read, function.block.operations)

    def test_attention_fusion_rejects_reversed_context_matmul(self) -> None:
        module = self.make_module()
        default_pass_manager(stateful_decode=True).run(module)
        BufferizeKVCachePass().run(module)
        function = module.functions[0]
        read = next(
            operation
            for operation in function.block.operations
            if operation.name == "serve.kv.read"
        )
        match = _match_attention(read, UseDefAnalysis(function))
        assert match is not None
        match.final_matmul.operands.reverse()
        match.final_matmul.attributes["args"]["tuple"].reverse()

        result = FuseDecodeAttentionPass().run(module)

        self.assertFalse(result.changed)
        self.assertEqual(result.statistics["rejected"], 1)
        self.assertIn(read, function.block.operations)

    def test_attention_fusion_rejects_multi_token_query(self) -> None:
        module = self.make_module()
        default_pass_manager(stateful_decode=True).run(module)
        BufferizeKVCachePass().run(module)
        function = module.functions[0]
        read = next(
            operation
            for operation in function.block.operations
            if operation.name == "serve.kv.read"
        )
        match = _match_attention(read, UseDefAnalysis(function))
        assert match is not None
        assert isinstance(match.query.type, TensorType)
        shape = list(match.query.type.shape)
        shape[2] = StaticDim(2)
        match.query.type = TensorType(
            tuple(shape),
            match.query.type.dtype,
            match.query.type.device,
        )

        result = FuseDecodeAttentionPass().run(module)

        self.assertFalse(result.changed)
        self.assertEqual(result.statistics["rejected"], 1)

    def test_verifier_rejects_multi_token_decode_attention(self) -> None:
        module = self.make_module()
        default_pass_manager(preallocate_kv=True).run(module)
        operation = next(
            operation
            for operation in module.functions[0].block.operations
            if operation.name == "serve.decode_attention"
        )
        query = operation.operands[1]
        assert isinstance(query.type, TensorType)
        shape = list(query.type.shape)
        shape[2] = StaticDim(2)
        query.type = TensorType(
            tuple(shape),
            query.type.dtype,
            query.type.device,
        )
        operation.results[0].type = query.type

        with self.assertRaisesRegex(VerificationError, "必须是单 Token"):
            verify_module(module)

    def test_preallocated_runtime_preserves_values_and_addresses(self) -> None:
        module = self.make_module()
        default_pass_manager(preallocate_kv=True).run(module)
        executor = ReferenceExecutor()
        _, _, past_key, past_value = make_decode_inputs(
            self.config,
            batch=2,
            past_length=3,
            seed=11,
        )
        state = PreallocatedKVCacheState.from_tensors(
            past_key,
            past_value,
            capacity=65,
        )
        pointers = (
            state.keys[0].data_ptr(),
            state.values[0].data_ptr(),
        )

        with torch.no_grad():
            for step in range(4):
                hidden = torch.randn(
                    2,
                    1,
                    self.config.hidden_size,
                    generator=torch.Generator().manual_seed(200 + step),
                )
                mask = torch.zeros(
                    2,
                    1,
                    1,
                    int(state.lengths[0].max()) + 1,
                )
                expected = self.program.module()(
                    hidden,
                    mask,
                    *state.read(0),
                )
                actual, state = executor.run(
                    module,
                    bind_stateful_decode_arguments(
                        self.program,
                        hidden,
                        mask,
                        state,
                    ),
                ).outputs
                torch.testing.assert_close(actual, expected[0])
                actual_key, actual_value = state.read(0)
                torch.testing.assert_close(actual_key, expected[1])
                torch.testing.assert_close(actual_value, expected[2])

        self.assertEqual(state.lengths[0].tolist(), [7, 7])
        self.assertEqual(state.generation, 4)
        self.assertEqual(
            pointers,
            (
                state.keys[0].data_ptr(),
                state.values[0].data_ptr(),
            ),
        )

    def test_preallocated_runtime_rejects_capacity_overflow(self) -> None:
        _, _, key, value = make_decode_inputs(
            self.config,
            batch=2,
            past_length=3,
        )
        state = PreallocatedKVCacheState.from_tensors(
            key,
            value,
            capacity=3,
        )
        current = torch.randn(2, 2, 1, 8)

        with self.assertRaisesRegex(ExecutionError, "超过 Capacity 3"):
            state.store(
                0,
                current,
                current,
                state.positions(0),
            )

    def test_bufferized_pipeline_is_idempotent_and_can_override_capacity(
        self,
    ) -> None:
        module = self.make_module()
        manager = default_pass_manager(
            preallocate_kv=True,
            kv_capacity=128,
        )
        first = manager.run(module)
        second = manager.run(module)
        state_type = module.functions[0].block.arguments[-1].type

        self.assertEqual(first[-2].statistics["bufferized"], 1)
        self.assertEqual(first[-1].statistics["fused"], 1)
        self.assertTrue(all(not result.changed for result in second))
        self.assertEqual(second[-2].statistics["bufferized"], 0)
        self.assertEqual(second[-1].statistics["fused"], 0)
        self.assertIsInstance(state_type, KVStateType)
        self.assertEqual(state_type.capacity, 128)

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

    @staticmethod
    def _cache_cat(module, argument_name: str):
        function = module.functions[0]
        argument = next(
            value
            for value in function.block.arguments
            if value.name == argument_name
        )
        return next(
            operation
            for operation in function.block.operations
            if operation.name == "aten.cat.default"
            and argument in operation.operands
        )

    def _assert_materialization_rejected(self, module) -> None:
        result = MaterializeKVStatePass().run(module)
        names = [
            operation.name
            for operation in module.functions[0].block.operations
        ]

        self.assertFalse(result.changed)
        self.assertEqual(result.statistics["converted"], 0)
        self.assertEqual(result.statistics["rejected"], 1)
        self.assertEqual(names.count("aten.cat.default"), 2)
        self.assertNotIn("serve.kv.append", names)


if __name__ == "__main__":
    unittest.main()
