from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from stateful_llm_compiler.backends import (
    TritonExecutor,
    triton_decode_attention,
    triton_kv_store,
    triton_rms_norm,
)
from stateful_llm_compiler.execution import (
    KVCacheState,
    PreallocatedKVCacheState,
    bind_exported_program_arguments,
    bind_stateful_decode_arguments,
)
from stateful_llm_compiler.frontend import (
    export_decoder,
    export_stateful_decode,
)
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.model import (
    DecoderConfig,
    StatefulTinyDecoderBlock,
    TinyDecoderBlock,
    make_decode_inputs,
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

    def test_kernel_preserves_qwen2_bf16_rounding_order(self) -> None:
        tensor = torch.randn(7, 896, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(896, device="cuda", dtype=torch.bfloat16)
        variance = tensor.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = tensor.float() * torch.rsqrt(variance + 1e-6)
        expected = weight * normalized.to(torch.bfloat16)

        actual = triton_rms_norm(
            tensor,
            weight,
            epsilon=1e-6,
            output_dtype="bf16",
            round_before_weight=True,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_executor_dispatches_profile_variant(self) -> None:
        tensor = torch.randn(8, 64, device="cuda")
        weight = torch.randn(64, device="cuda")
        attributes = {
            "epsilon": 1e-6,
            "output_dtype": "f32",
            "lowering_plan": {
                "fallback": "inductor",
                "variants": [
                    {
                        "min_rows": 1,
                        "max_rows": 16,
                        "profile_rows": 8,
                        "backend": "triton",
                        "estimated_us": 4.0,
                        "num_warps": 4,
                    }
                ],
            },
        }
        executor = TritonExecutor()

        with patch(
            "stateful_llm_compiler.backends.triton_executor.triton_rms_norm",
            wraps=triton_rms_norm,
        ) as lowering:
            actual = executor._serve_rms_norm(
                [tensor, weight],
                attributes,
            )

        self.assertEqual(actual.shape, tensor.shape)
        self.assertEqual(executor.lowering_trace[-1]["backend"], "triton")
        self.assertEqual(executor.lowering_trace[-1]["profile_rows"], 8)
        self.assertEqual(lowering.call_args.kwargs["num_warps"], 4)

    def test_executor_dispatches_missing_profile_to_inductor(self) -> None:
        tensor = torch.randn(32, 64, device="cuda")
        weight = torch.randn(64, device="cuda")
        attributes = {
            "epsilon": 1e-6,
            "output_dtype": "f32",
            "lowering_plan": {
                "fallback": "inductor",
                "variants": [],
            },
        }
        executor = TritonExecutor()

        with patch(
            "stateful_llm_compiler.backends.triton_executor.torch.compile",
            side_effect=lambda function, **_: function,
        ) as compile_function:
            actual = executor._serve_rms_norm(
                [tensor, weight],
                attributes,
            )
        expected = torch.nn.functional.rms_norm(
            tensor,
            (64,),
            weight,
            1e-6,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-5,
            atol=2e-5,
        )
        compile_function.assert_called_once()
        self.assertEqual(
            executor.lowering_trace[-1],
            {
                "backend": "inductor",
                "source": "fallback",
                "rows": 32,
                "profile_rows": None,
                "estimated_us": None,
                "num_warps": None,
            },
        )

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

    def test_stateful_decode_runs_kv_state_and_triton_on_gpu(self) -> None:
        torch.manual_seed(0)
        config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )
        model = StatefulTinyDecoderBlock(config).eval().cuda()
        example = tuple(
            tensor.cuda()
            for tensor in make_decode_inputs(config, 2, 4)
        )
        program = export_stateful_decode(
            model,
            example,
            max_batch=8,
            max_cache_length=64,
        )
        module = import_exported_program(
            program,
            function_name="decode",
        )
        default_pass_manager(stateful_decode=True).run(module)
        executor = TritonExecutor()
        state = KVCacheState.from_tensors(example[2], example[3])

        with torch.no_grad():
            for step in range(2):
                hidden = torch.randn(
                    2,
                    1,
                    config.hidden_size,
                    device="cuda",
                )
                mask = torch.zeros(
                    2,
                    1,
                    1,
                    state.keys[0].shape[2] + 1,
                    device="cuda",
                )
                expected = program.module()(
                    hidden,
                    mask,
                    *state.read(0),
                )
                actual, state = executor.run(
                    module,
                    bind_stateful_decode_arguments(
                        program,
                        hidden,
                        mask,
                        state,
                    ),
                ).outputs
                torch.testing.assert_close(
                    actual,
                    expected[0],
                    rtol=3e-5,
                    atol=3e-5,
                )
                torch.testing.assert_close(state.keys[0], expected[1])
                torch.testing.assert_close(state.values[0], expected[2])

        self.assertEqual(state.keys[0].shape[2], 6)
        self.assertEqual(state.generation, 2)
        self.assertEqual(
            [item["backend"] for item in executor.lowering_trace],
            ["triton", "triton", "triton", "triton"],
        )

    def test_bufferized_decode_runs_triton_kv_store_on_gpu(self) -> None:
        torch.manual_seed(0)
        config = DecoderConfig(
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
        )
        model = StatefulTinyDecoderBlock(config).eval().cuda()
        example = tuple(
            tensor.cuda()
            for tensor in make_decode_inputs(config, 2, 4)
        )
        program = export_stateful_decode(
            model,
            example,
            max_cache_length=64,
        )
        module = import_exported_program(
            program,
            function_name="decode",
        )
        default_pass_manager(preallocate_kv=True).run(module)
        operation_names = [
            operation.name
            for operation in module.functions[0].block.operations
        ]
        self.assertIn("serve.decode_attention", operation_names)
        self.assertNotIn("serve.kv.read", operation_names)
        state = PreallocatedKVCacheState.from_tensors(
            example[2],
            example[3],
            capacity=65,
        )
        pointers = (
            state.keys[0].data_ptr(),
            state.values[0].data_ptr(),
        )
        executor = TritonExecutor()

        with torch.no_grad():
            for _ in range(2):
                hidden = torch.randn(2, 1, 32, device="cuda")
                mask = torch.zeros(
                    2,
                    1,
                    1,
                    int(state.lengths[0].max()) + 1,
                    device="cuda",
                )
                expected = program.module()(
                    hidden,
                    mask,
                    *state.read(0),
                )
                actual, state = executor.run(
                    module,
                    bind_stateful_decode_arguments(
                        program,
                        hidden,
                        mask,
                        state,
                    ),
                ).outputs
                torch.testing.assert_close(
                    actual,
                    expected[0],
                    rtol=3e-5,
                    atol=3e-5,
                )
                actual_key, actual_value = state.read(0)
                torch.testing.assert_close(actual_key, expected[1])
                torch.testing.assert_close(actual_value, expected[2])

        self.assertEqual(state.lengths[0].tolist(), [6, 6])
        self.assertEqual(
            pointers,
            (
                state.keys[0].data_ptr(),
                state.values[0].data_ptr(),
            ),
        )

    def test_triton_decode_attention_supports_per_batch_lengths(
        self,
    ) -> None:
        batch, capacity, query_heads, kv_heads, head_dim = 3, 65, 4, 2, 16
        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            device="cuda",
        )
        key_buffer = torch.randn(
            batch,
            capacity,
            kv_heads,
            head_dim,
            device="cuda",
        )
        value_buffer = torch.randn_like(key_buffer)
        lengths = torch.tensor([7, 31, 65], device="cuda")
        attention_mask = torch.zeros(
            batch,
            1,
            1,
            capacity,
            device="cuda",
        )
        state = PreallocatedKVCacheState(
            (key_buffer,),
            (value_buffer,),
            (lengths,),
            capacity,
        )
        expected = state.decode_attention(
            0,
            query,
            attention_mask,
            groups=query_heads // kv_heads,
            scale=head_dim**-0.5,
        )
        actual = triton_decode_attention(
            query,
            key_buffer,
            value_buffer,
            lengths,
            attention_mask,
            scale=head_dim**-0.5,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=3e-5,
            atol=3e-5,
        )

    def test_triton_kv_store_supports_per_batch_positions(self) -> None:
        batch, capacity, heads, head_dim = 3, 16, 2, 8
        key_buffer = torch.zeros(
            batch,
            capacity,
            heads,
            head_dim,
            device="cuda",
        )
        value_buffer = torch.zeros_like(key_buffer)
        key = torch.randn(batch, heads, 1, head_dim, device="cuda")
        value = torch.randn_like(key)
        positions = torch.tensor([1, 5, 9], device="cuda")

        triton_kv_store(
            key_buffer,
            value_buffer,
            key,
            value,
            positions,
        )

        for item, position in enumerate(positions.tolist()):
            torch.testing.assert_close(
                key_buffer[item, position],
                key[item, :, 0],
            )
            torch.testing.assert_close(
                value_buffer[item, position],
                value[item, :, 0],
            )


if __name__ == "__main__":
    unittest.main()
