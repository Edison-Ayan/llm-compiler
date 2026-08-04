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
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import export_qwen2_stateful_decode
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.optimizer import default_pass_manager
from stateful_llm_compiler.qwen2 import StatefulQwen2Model

try:
    from transformers import DynamicCache, Qwen2Config, Qwen2Model
except ImportError:
    DynamicCache = None
    Qwen2Config = None
    Qwen2Model = None


@unittest.skipIf(Qwen2Model is None, "需要可选的 transformers Qwen2")
class Qwen2CompatibilityTest(unittest.TestCase):
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
        # 固定 Eager Attention，避免测试结果依赖本机可选 Attention Backend。
        cls.hf_config._attn_implementation = "eager"
        cls.hf_model = Qwen2Model(cls.hf_config).eval()
        cls.model = StatefulQwen2Model.from_huggingface(cls.hf_model).eval()

        cls.batch = 2
        cls.past_length = 4
        prefill = torch.randn(cls.batch, cls.past_length, 32)
        prefill_cache = DynamicCache(config=cls.hf_config)
        with torch.no_grad():
            cls.hf_model(
                inputs_embeds=prefill,
                attention_mask=torch.ones(cls.batch, cls.past_length),
                position_ids=torch.arange(cls.past_length)
                .view(1, -1)
                .expand(cls.batch, -1),
                past_key_values=prefill_cache,
                use_cache=True,
            )
        cls.past_key_values = tuple(
            (layer.keys.clone(), layer.values.clone())
            for layer in prefill_cache.layers
        )
        cls.hidden = torch.randn(cls.batch, 1, 32)
        cls.position_ids = torch.full(
            (cls.batch, 1),
            cls.past_length,
            dtype=torch.int64,
        )
        cls.additive_mask = torch.zeros(
            cls.batch,
            1,
            1,
            cls.past_length + 1,
        )
        cls.program = export_qwen2_stateful_decode(
            cls.model,
            (
                cls.hidden,
                cls.additive_mask,
                cls.position_ids,
                cls.past_key_values,
            ),
            max_cache_length=16,
        )

    def make_module(self):
        return import_exported_program(
            self.program,
            function_name="decode",
        )

    def test_huggingface_weights_match_every_layer_and_cache(self) -> None:
        official_cache = DynamicCache(
            (
                (key.clone(), value.clone())
                for key, value in self.past_key_values
            ),
            config=self.hf_config,
        )
        official_layers = []
        converted_layers = []
        handles = []
        for layer in self.hf_model.layers:
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output: official_layers.append(
                        output.detach().clone()
                    )
                )
            )
        for layer in self.model.layers:
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output: converted_layers.append(
                        output[0].detach().clone()
                    )
                )
            )

        try:
            with torch.no_grad():
                expected = self.hf_model(
                    inputs_embeds=self.hidden,
                    attention_mask=torch.ones(
                        self.batch,
                        self.past_length + 1,
                    ),
                    position_ids=self.position_ids,
                    past_key_values=official_cache,
                    use_cache=True,
                ).last_hidden_state
                actual, present = self.model(
                    self.hidden,
                    self.additive_mask,
                    self.position_ids,
                    self.past_key_values,
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(len(official_layers), 2)
        self.assertEqual(len(converted_layers), 2)
        for official, converted in zip(official_layers, converted_layers):
            torch.testing.assert_close(converted, official, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for (key, value), layer in zip(present, official_cache.layers):
            torch.testing.assert_close(key, layer.keys, rtol=0, atol=0)
            torch.testing.assert_close(value, layer.values, rtol=0, atol=0)

    def test_qwen2_export_imports_without_fallback_and_fuses_attention(
        self,
    ) -> None:
        module = self.make_module()
        imported_operations = len(module.functions[0].block.operations)
        results = default_pass_manager(preallocate_kv=True).run(module)
        function = module.functions[0]
        names = [operation.name for operation in function.block.operations]
        attentions = [
            operation
            for operation in function.block.operations
            if operation.name == "serve.decode_attention"
        ]

        self.assertEqual(imported_operations, 219)
        self.assertNotIn("serve.external", names)
        self.assertEqual(results[-3].statistics["slots"], 2)
        self.assertEqual(results[-2].statistics["bufferized"], 2)
        self.assertEqual(results[-1].statistics["fused"], 2)
        self.assertEqual(len(attentions), 2)
        self.assertEqual(
            [operation.attributes["slot"] for operation in attentions],
            [0, 1],
        )
        self.assertTrue(
            all(
                operation.attributes["scale"] == 8**-0.5
                for operation in attentions
            )
        )

    def test_qwen2_serveir_matches_exported_program(self) -> None:
        module = self.make_module()
        default_pass_manager(preallocate_kv=True).run(module)
        state = PreallocatedKVCacheState.from_layer_tensors(
            self.past_key_values,
            capacity=17,
        )

        with torch.no_grad():
            expected_hidden, expected_cache = self.program.module()(
                self.hidden,
                self.additive_mask,
                self.position_ids,
                self.past_key_values,
            )
            actual_hidden, state = ReferenceExecutor().run(
                module,
                bind_stateful_decode_arguments(
                    self.program,
                    self.hidden,
                    self.additive_mask,
                    state,
                    extra_user_inputs={"position_ids": self.position_ids},
                ),
            ).outputs

        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=0, atol=0)
        for slot, (expected_key, expected_value) in enumerate(expected_cache):
            actual_key, actual_value = state.read(slot)
            torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    def test_qwen2_kernel_ir_reports_exact_backend_gap(self) -> None:
        artifact = compile_exported_program(
            self.program,
            options=CompileOptions(
                function_name="decode",
                preallocate_kv=True,
            ),
        )

        self.assertEqual(artifact.coverage.total_operations, 81)
        self.assertEqual(artifact.coverage.lowered_operations, 29)
        self.assertEqual(artifact.coverage.unlowered_operations, 52)
        self.assertEqual(
            artifact.coverage.lowered_by_name,
            {
                "kernel.triton.decode_attention": 2,
                "kernel.triton.kv_store": 2,
                "kernel.triton.linear": 14,
                "kernel.triton.rms_norm": 5,
                "kernel.triton.rope": 2,
                "runtime.kv.advance": 2,
                "runtime.kv.length": 2,
            },
        )
        self.assertNotIn(
            "aten.linear.default",
            artifact.coverage.unlowered_by_name,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA GPU")
    def test_qwen2_bufferized_decode_runs_triton_on_gpu(self) -> None:
        model = copy.deepcopy(self.model).cuda()
        hidden = self.hidden.cuda()
        additive_mask = self.additive_mask.cuda()
        position_ids = self.position_ids.cuda()
        past_key_values = tuple(
            (key.cuda(), value.cuda())
            for key, value in self.past_key_values
        )
        program = export_qwen2_stateful_decode(
            model,
            (hidden, additive_mask, position_ids, past_key_values),
            max_cache_length=16,
        )
        artifact = compile_exported_program(
            program,
            options=CompileOptions(
                function_name="decode",
                preallocate_kv=True,
            ),
        )
        state = PreallocatedKVCacheState.from_layer_tensors(
            past_key_values,
            capacity=17,
        )

        with torch.no_grad():
            expected_hidden, expected_cache = program.module()(
                hidden,
                additive_mask,
                position_ids,
                past_key_values,
            )
            execution = TritonExecutor().run(
                artifact.module,
                bind_stateful_decode_arguments(
                    program,
                    hidden,
                    additive_mask,
                    state,
                    extra_user_inputs={"position_ids": position_ids},
                ),
            )
            actual_hidden, state = execution.outputs

        self.assertEqual(
            execution.executed_operations.count("kernel.triton.linear"),
            14,
        )
        self.assertEqual(
            execution.executed_operations.count("kernel.triton.rope"),
            2,
        )

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
