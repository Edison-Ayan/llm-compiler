from __future__ import annotations

import unittest

import torch

from stateful_llm_compiler.backends import TritonExecutor
from stateful_llm_compiler.execution import (
    PreallocatedKVCacheState,
    ReferenceExecutor,
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import export_multilayer_stateful_decode
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.ir import KVStateType, VerificationError, verify_module
from stateful_llm_compiler.model import (
    DecoderConfig,
    StatefulTinyDecoder,
    make_multilayer_decode_inputs,
)
from stateful_llm_compiler.optimizer import default_pass_manager


class MultiLayerStatefulDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            num_layers=2,
        )
        cls.model = StatefulTinyDecoder(cls.config).eval()
        cls.example = make_multilayer_decode_inputs(
            cls.config,
            batch=2,
            past_length=4,
        )
        cls.program = export_multilayer_stateful_decode(
            cls.model,
            cls.example,
            max_batch=8,
            max_cache_length=16,
        )

    def make_module(self):
        return import_exported_program(
            self.program,
            function_name="decode",
        )

    def test_export_flattens_two_layer_cache_with_shared_symbols(self) -> None:
        user_inputs = [
            spec.arg.name
            for spec in self.program.graph_signature.input_specs
            if spec.kind.name == "USER_INPUT"
        ]
        self.assertEqual(
            user_inputs,
            [
                "hidden_states",
                "attention_mask",
                "past_key_values_0_0",
                "past_key_values_0_1",
                "past_key_values_1_0",
                "past_key_values_1_1",
            ],
        )

        inputs = make_multilayer_decode_inputs(
            self.config,
            batch=3,
            past_length=7,
            seed=11,
        )
        with torch.no_grad():
            hidden, present = self.program.module()(*inputs)
        self.assertEqual(hidden.shape, (3, 1, self.config.hidden_size))
        self.assertEqual(len(present), 2)
        self.assertTrue(
            all(key.shape[2] == 8 and value.shape == key.shape for key, value in present)
        )

    def test_pipeline_materializes_and_bufferizes_two_slots(self) -> None:
        module = self.make_module()
        results = default_pass_manager(preallocate_kv=True).run(module)
        function = module.functions[0]
        state_type = next(
            value.type
            for value in function.block.arguments
            if isinstance(value.type, KVStateType)
        )

        self.assertEqual(results[-3].statistics["converted"], 1)
        self.assertEqual(results[-3].statistics["slots"], 2)
        self.assertEqual(results[-2].statistics["bufferized"], 2)
        self.assertEqual(results[-1].statistics["fused"], 2)
        self.assertEqual(state_type.num_layers, 2)
        self.assertEqual(state_type.layout, "contiguous_bshd")
        self.assertEqual(state_type.resource, "kv.cache")
        self.assertEqual(state_type.capacity, 17)
        for name in (
            "serve.kv.store",
            "serve.kv.advance",
            "serve.decode_attention",
        ):
            operations = [
                operation
                for operation in function.block.operations
                if operation.name == name
            ]
            self.assertEqual(
                [operation.attributes["slot"] for operation in operations],
                [0, 1],
            )
        self.assertNotIn(
            "serve.kv.read",
            [operation.name for operation in function.block.operations],
        )
        verify_module(module)

    def test_preallocated_two_layer_decode_matches_export_for_three_steps(
        self,
    ) -> None:
        module = self.make_module()
        default_pass_manager(preallocate_kv=True).run(module)
        state = PreallocatedKVCacheState.from_layer_tensors(
            self.example[2],
            capacity=17,
        )
        pointers = tuple(
            (key.data_ptr(), value.data_ptr())
            for key, value in zip(state.keys, state.values)
        )
        executor = ReferenceExecutor()

        with torch.no_grad():
            for step in range(3):
                hidden = torch.randn(
                    2,
                    1,
                    self.config.hidden_size,
                    generator=torch.Generator().manual_seed(100 + step),
                )
                mask = torch.zeros(
                    2,
                    1,
                    1,
                    int(state.lengths[0].max()) + 1,
                )
                logical_cache = tuple(
                    state.read(slot)
                    for slot in range(self.config.num_layers)
                )
                expected_hidden, expected_cache = self.program.module()(
                    hidden,
                    mask,
                    logical_cache,
                )
                actual_hidden, state = executor.run(
                    module,
                    bind_stateful_decode_arguments(
                        self.program,
                        hidden,
                        mask,
                        state,
                    ),
                ).outputs
                torch.testing.assert_close(
                    actual_hidden,
                    expected_hidden,
                    rtol=1e-5,
                    atol=1e-6,
                )
                for slot, (expected_key, expected_value) in enumerate(
                    expected_cache
                ):
                    actual_key, actual_value = state.read(slot)
                    torch.testing.assert_close(actual_key, expected_key)
                    torch.testing.assert_close(actual_value, expected_value)

        self.assertEqual(
            [lengths.tolist() for lengths in state.lengths],
            [[7, 7], [7, 7]],
        )
        # 每轮两个 Layer Slot 各推进一次，因此三轮共有六个状态版本。
        self.assertEqual(state.generation, 6)
        self.assertEqual(
            pointers,
            tuple(
                (key.data_ptr(), value.data_ptr())
                for key, value in zip(state.keys, state.values)
            ),
        )

    def test_verifier_rejects_slot_outside_layer_range(self) -> None:
        module = self.make_module()
        default_pass_manager(preallocate_kv=True).run(module)
        operation = next(
            operation
            for operation in module.functions[0].block.operations
            if operation.name == "serve.decode_attention"
        )
        operation.attributes["slot"] = 2

        with self.assertRaisesRegex(VerificationError, r"slot 必须位于 \[0, 2\)"):
            verify_module(module)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA GPU")
    def test_two_layer_bufferized_decode_runs_triton_on_gpu(self) -> None:
        model = StatefulTinyDecoder(self.config).eval().cuda()
        cpu_inputs = make_multilayer_decode_inputs(
            self.config,
            batch=2,
            past_length=4,
            seed=23,
        )
        inputs = (
            cpu_inputs[0].cuda(),
            cpu_inputs[1].cuda(),
            tuple(
                (key.cuda(), value.cuda())
                for key, value in cpu_inputs[2]
            ),
        )
        program = export_multilayer_stateful_decode(
            model,
            inputs,
            max_cache_length=16,
        )
        module = import_exported_program(program, function_name="decode")
        default_pass_manager(preallocate_kv=True).run(module)
        state = PreallocatedKVCacheState.from_layer_tensors(
            inputs[2],
            capacity=17,
        )
        hidden = torch.randn(2, 1, self.config.hidden_size, device="cuda")
        mask = torch.zeros(2, 1, 1, 5, device="cuda")

        with torch.no_grad():
            expected_hidden, expected_cache = program.module()(
                hidden,
                mask,
                tuple(state.read(slot) for slot in range(2)),
            )
            actual_hidden, state = TritonExecutor().run(
                module,
                bind_stateful_decode_arguments(
                    program,
                    hidden,
                    mask,
                    state,
                ),
            ).outputs

        torch.testing.assert_close(
            actual_hidden,
            expected_hidden,
            rtol=3e-5,
            atol=3e-5,
        )
        for slot, (expected_key, expected_value) in enumerate(expected_cache):
            actual_key, actual_value = state.read(slot)
            torch.testing.assert_close(
                actual_key,
                expected_key,
                rtol=3e-5,
                atol=3e-5,
            )
            torch.testing.assert_close(
                actual_value,
                expected_value,
                rtol=3e-5,
                atol=3e-5,
            )


if __name__ == "__main__":
    unittest.main()
