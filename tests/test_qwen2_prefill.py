from __future__ import annotations

import copy
import unittest

import torch

from stateful_llm_compiler.backends import TritonExecutor
from stateful_llm_compiler.compiler import (
    CompileOptions,
    compile_exported_program,
)
from stateful_llm_compiler.execution import (
    PreallocatedKVCacheState,
    ReferenceExecutor,
    bind_exported_program_arguments,
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import (
    export_qwen2_causal_lm_decode,
    export_qwen2_causal_lm_prefill,
)
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.optimizer import default_pass_manager
from stateful_llm_compiler.qwen2 import (
    StatefulQwen2ForCausalLM,
    make_qwen2_prefill_inputs,
)

try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    Qwen2Config = None
    Qwen2ForCausalLM = None


@unittest.skipIf(Qwen2ForCausalLM is None, "需要可选的transformers Qwen2")
class Qwen2PrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.hf_config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            use_cache=True,
            attention_dropout=0.0,
        )
        cls.hf_config._attn_implementation = "eager"
        cls.hf_model = Qwen2ForCausalLM(cls.hf_config).eval()
        cls.model = StatefulQwen2ForCausalLM.from_huggingface(
            cls.hf_model
        ).eval()
        cls.example = make_qwen2_prefill_inputs(
            cls.model.config,
            batch=2,
            tokens=4,
        )
        cls.program = export_qwen2_causal_lm_prefill(
            cls.model,
            cls.example,
            max_batch=8,
            max_prompt_length=16,
        )

    def test_full_causal_lm_matches_official_logits_and_cache(self) -> None:
        input_ids, additive_mask, position_ids = self.example
        with torch.no_grad():
            actual_logits, actual_cache = self.model(
                input_ids,
                additive_mask,
                position_ids,
            )
            expected = self.hf_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=position_ids,
                use_cache=True,
            )

        torch.testing.assert_close(actual_logits, expected.logits, rtol=0, atol=0)
        for (actual_key, actual_value), layer in zip(
            actual_cache,
            expected.past_key_values.layers,
        ):
            torch.testing.assert_close(actual_key, layer.keys, rtol=0, atol=0)
            torch.testing.assert_close(actual_value, layer.values, rtol=0, atol=0)

    def test_prefill_cache_can_feed_next_decode_token(self) -> None:
        input_ids, additive_mask, position_ids = self.example
        next_ids = torch.tensor([[3], [7]], dtype=torch.int64)
        next_position = torch.full((2, 1), 4, dtype=torch.int64)
        decode_mask = torch.zeros(2, 1, 1, 5)

        with torch.no_grad():
            _, project_cache = self.model(
                input_ids,
                additive_mask,
                position_ids,
            )
            actual_logits, _ = self.model.decode(
                next_ids,
                decode_mask,
                next_position,
                project_cache,
            )
            official_prefill = self.hf_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=position_ids,
                use_cache=True,
            )
            expected = self.hf_model(
                input_ids=next_ids,
                attention_mask=torch.ones(2, 5, dtype=torch.int64),
                position_ids=next_position,
                past_key_values=official_prefill.past_key_values,
                use_cache=True,
            )

        torch.testing.assert_close(actual_logits, expected.logits, rtol=0, atol=0)

    def test_export_has_dynamic_full_boundary_without_empty_cache_ops(self) -> None:
        module = import_exported_program(
            self.program,
            function_name="prefill",
        )
        imported_names = [
            operation.name
            for operation in module.functions[0].block.operations
        ]
        results = default_pass_manager().run(module)
        optimized_names = [
            operation.name
            for operation in module.functions[0].block.operations
        ]

        self.assertEqual(len(imported_names), 203)
        self.assertNotIn("serve.external", imported_names)
        self.assertFalse(any("empty" in name for name in imported_names))
        self.assertEqual(imported_names.count("aten.embedding.default"), 1)
        self.assertEqual(results[1].statistics["normalized"], 15)
        self.assertEqual(results[2].statistics["fused"], 5)
        self.assertEqual(results[3].statistics["fused"], 2)
        self.assertEqual(results[4].statistics["fused"], 2)
        self.assertEqual(len(optimized_names), 80)
        self.assertEqual(optimized_names.count("serve.linear"), 15)
        self.assertEqual(optimized_names.count("serve.rope"), 2)
        self.assertEqual(
            optimized_names.count("serve.prefill_attention"),
            2,
        )

    def test_optimized_prefill_matches_multiple_dynamic_prompts(self) -> None:
        module = import_exported_program(
            self.program,
            function_name="prefill",
        )
        default_pass_manager().run(module)
        executor = ReferenceExecutor()

        for batch, tokens in ((1, 2), (2, 4), (3, 7)):
            inputs = make_qwen2_prefill_inputs(
                self.model.config,
                batch,
                tokens,
                seed=batch * 100 + tokens,
            )
            with torch.no_grad():
                expected_logits, expected_cache = self.program.module()(*inputs)
                actual = executor.run(
                    module,
                    bind_exported_program_arguments(self.program, inputs),
                ).outputs
            expected_flat = (expected_logits,) + tuple(
                tensor
                for key_value in expected_cache
                for tensor in key_value
            )
            for actual_tensor, expected_tensor in zip(actual, expected_flat):
                torch.testing.assert_close(
                    actual_tensor,
                    expected_tensor,
                    rtol=0,
                    atol=0,
                )

    def test_prefill_kernel_ir_reports_current_backend_coverage(self) -> None:
        artifact = compile_exported_program(
            self.program,
            options=CompileOptions(function_name="prefill"),
        )

        self.assertEqual(artifact.coverage.total_operations, 80)
        self.assertEqual(artifact.coverage.lowered_operations, 24)
        self.assertEqual(artifact.coverage.unlowered_operations, 56)
        self.assertEqual(
            artifact.coverage.lowered_by_name,
            {
                "kernel.triton.linear": 15,
                "kernel.triton.prefill_attention": 2,
                "kernel.triton.rms_norm": 5,
                "kernel.triton.rope": 2,
            },
        )

    def test_prefill_materializes_exact_preallocated_state(self) -> None:
        module = import_exported_program(
            self.program,
            function_name="prefill",
        )
        results = default_pass_manager(
            prefill_kv_state=True,
            kv_capacity=16,
        ).run(module)

        with torch.no_grad():
            expected_logits, expected_cache = self.program.module()(
                *self.example
            )
            actual_logits, state = ReferenceExecutor().run(
                module,
                bind_exported_program_arguments(
                    self.program,
                    self.example,
                ),
            ).outputs

        self.assertEqual(results[-1].statistics["slots"], 2)
        self.assertIsInstance(state, PreallocatedKVCacheState)
        self.assertEqual(state.capacity, 16)
        self.assertTrue(
            all(torch.equal(lengths, torch.full((2,), 4)) for lengths in state.lengths)
        )
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        for slot, expected_pair in enumerate(expected_cache):
            actual_pair = state.read(slot)
            for actual_tensor, expected_tensor in zip(
                actual_pair,
                expected_pair,
            ):
                torch.testing.assert_close(
                    actual_tensor,
                    expected_tensor,
                    rtol=0,
                    atol=0,
                )

    def test_stateful_prefill_kernel_ir_reports_init_and_store(self) -> None:
        artifact = compile_exported_program(
            self.program,
            options=CompileOptions(
                function_name="prefill",
                prefill_kv_state=True,
                kv_capacity=16,
            ),
        )

        self.assertEqual(artifact.coverage.total_operations, 83)
        self.assertEqual(artifact.coverage.lowered_operations, 27)
        self.assertEqual(artifact.coverage.unlowered_operations, 56)
        self.assertEqual(
            artifact.coverage.lowered_by_name[
                "kernel.triton.kv_prefill_store"
            ],
            2,
        )
        self.assertEqual(
            artifact.coverage.lowered_by_name["runtime.kv.init"],
            1,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要CUDA GPU")
    def test_prefill_kernel_ir_executes_all_triton_linears_on_gpu(self) -> None:
        model = copy.deepcopy(self.model).cuda()
        example = tuple(tensor.cuda() for tensor in self.example)
        program = export_qwen2_causal_lm_prefill(
            model,
            example,
            max_batch=8,
            max_prompt_length=16,
        )
        artifact = compile_exported_program(
            program,
            options=CompileOptions(function_name="prefill"),
        )

        with torch.no_grad():
            expected_logits, expected_cache = program.module()(*example)
            execution = TritonExecutor().run(
                artifact.module,
                bind_exported_program_arguments(program, example),
            )
        expected_flat = (expected_logits,) + tuple(
            tensor
            for key_value in expected_cache
            for tensor in key_value
        )
        for actual_tensor, expected_tensor in zip(
            execution.outputs,
            expected_flat,
        ):
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=3e-5,
                atol=3e-5,
            )
        self.assertEqual(
            execution.executed_operations.count("kernel.triton.linear"),
            15,
        )
        self.assertEqual(
            execution.executed_operations.count(
                "kernel.triton.prefill_attention"
            ),
            2,
        )
        self.assertEqual(
            execution.executed_operations.count("kernel.triton.rope"),
            2,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要CUDA GPU")
    def test_compiled_prefill_state_directly_feeds_compiled_decode(self) -> None:
        model = copy.deepcopy(self.model).cuda()
        prefill_inputs = tuple(tensor.cuda() for tensor in self.example)
        prefill_program = export_qwen2_causal_lm_prefill(
            model,
            prefill_inputs,
            max_batch=8,
            max_prompt_length=16,
        )
        prefill_artifact = compile_exported_program(
            prefill_program,
            options=CompileOptions(
                function_name="prefill",
                prefill_kv_state=True,
                kv_capacity=16,
            ),
        )
        next_ids = torch.tensor([[3], [7]], device="cuda")
        next_position = torch.full(
            (2, 1),
            4,
            dtype=torch.int64,
            device="cuda",
        )
        decode_mask = torch.zeros(2, 1, 1, 5, device="cuda")

        with torch.no_grad():
            expected_prefill_logits, tensor_cache = model(*prefill_inputs)
            expected_decode_logits, expected_cache = model.decode(
                next_ids,
                decode_mask,
                next_position,
                tensor_cache,
            )
        decode_program = export_qwen2_causal_lm_decode(
            model,
            (
                next_ids,
                decode_mask,
                next_position,
                tensor_cache,
            ),
            max_batch=8,
            max_cache_length=16,
        )
        decode_artifact = compile_exported_program(
            decode_program,
            options=CompileOptions(
                function_name="decode",
                preallocate_kv=True,
                kv_capacity=16,
            ),
        )

        with torch.no_grad():
            actual_prefill_logits, state = TritonExecutor().run(
                prefill_artifact.module,
                bind_exported_program_arguments(
                    prefill_program,
                    prefill_inputs,
                ),
            ).outputs
            key_addresses = tuple(tensor.data_ptr() for tensor in state.keys)
            value_addresses = tuple(
                tensor.data_ptr() for tensor in state.values
            )
            actual_decode_logits, state = TritonExecutor().run(
                decode_artifact.module,
                bind_stateful_decode_arguments(
                    decode_program,
                    next_ids,
                    decode_mask,
                    state,
                    extra_user_inputs={"position_ids": next_position},
                    primary_input_name="input_ids",
                ),
            ).outputs

        torch.testing.assert_close(
            actual_prefill_logits,
            expected_prefill_logits,
            rtol=3e-5,
            atol=3e-5,
        )
        torch.testing.assert_close(
            actual_decode_logits,
            expected_decode_logits,
            rtol=3e-5,
            atol=3e-5,
        )
        self.assertEqual(
            tuple(tensor.data_ptr() for tensor in state.keys),
            key_addresses,
        )
        self.assertEqual(
            tuple(tensor.data_ptr() for tensor in state.values),
            value_addresses,
        )
        self.assertTrue(
            all(torch.equal(lengths, torch.full((2,), 5, device="cuda")) for lengths in state.lengths)
        )
        for slot, expected_pair in enumerate(expected_cache):
            for actual_tensor, expected_tensor in zip(
                state.read(slot),
                expected_pair,
            ):
                torch.testing.assert_close(
                    actual_tensor,
                    expected_tensor,
                    rtol=3e-5,
                    atol=3e-5,
                )


if __name__ == "__main__":
    unittest.main()
