from __future__ import annotations

import math
import unittest

import torch

from stateful_llm_compiler.cost_model import (
    ProfileError,
    RMSNormCostModel,
    normalize_dtype,
    resolve_lowering_plan,
    select_variant,
)
from stateful_llm_compiler.frontend import export_decoder
from stateful_llm_compiler.importer import import_exported_program
from stateful_llm_compiler.model import (
    DecoderConfig,
    TinyDecoderBlock,
    make_inputs,
)
from stateful_llm_compiler.optimizer import default_pass_manager


def make_profile(
    *,
    hidden_size: int = 32,
    dtype: str = "fp32",
) -> dict:
    # 人工构造交叉曲线，确保测试能观察到 Shape 相关的后端切换。
    timings = [
        (1, 6.0, 5.0, 4.0),
        (8, 5.0, 3.0, 4.0),
        (32, 5.0, 4.5, 3.5),
        (128, 5.0, 3.0, 4.0),
    ]
    return {
        "schema_version": 1,
        "target": {
            "device_name": "测试 GPU",
            "compute_capability": "8.9",
            "torch_version": "test",
            "triton_version": "test",
        },
        "results": [
            {
                "rows": rows,
                "hidden_size": hidden_size,
                "dtype": dtype,
                "native_eager_us": native,
                "triton_us": triton,
                "inductor_us": inductor,
            }
            for rows, native, triton, inductor in timings
        ],
    }


class RMSNormCostModelTest(unittest.TestCase):
    def test_dtype_aliases(self) -> None:
        self.assertEqual(normalize_dtype("f16"), "fp16")
        self.assertEqual(normalize_dtype("float32"), "fp32")
        self.assertEqual(normalize_dtype("BF16"), "bf16")
        with self.assertRaises(ProfileError):
            normalize_dtype("int8")

    def test_builds_geometric_buckets_and_selects_backend(self) -> None:
        model = RMSNormCostModel.from_dict(make_profile())
        variants = model.variants(32, "f32")

        self.assertEqual(
            [(item.min_rows, item.max_rows) for item in variants],
            [(1, 2), (3, 16), (17, 64), (65, None)],
        )
        self.assertEqual(
            [item.backend for item in variants],
            ["inductor", "triton", "inductor", "triton"],
        )
        self.assertEqual(
            [select_variant(variants, rows).profile_rows for rows in (1, 3, 17, 65, 4096)],
            [1, 8, 32, 128, 128],
        )
        self.assertEqual(variants[1].num_warps, 4)

        plan = model.plan_attribute(32, "fp32")
        self.assertEqual(
            resolve_lowering_plan(plan, 1).backend,
            "inductor",
        )
        decision = resolve_lowering_plan(plan, 9)
        self.assertEqual(decision.backend, "triton")
        self.assertEqual(decision.profile_rows, 8)

    def test_missing_shape_uses_empty_plan_and_fallback(self) -> None:
        model = RMSNormCostModel.from_dict(make_profile())
        plan = model.plan_attribute(64, "fp16")

        self.assertEqual(plan["variants"], [])
        self.assertEqual(plan["fallback"], "inductor")
        self.assertIsNone(select_variant(plan["variants"], 8))
        self.assertEqual(
            resolve_lowering_plan(plan, 8).source,
            "fallback",
        )

    def test_rejects_duplicate_and_invalid_latency(self) -> None:
        duplicate = make_profile()
        duplicate["results"].append(dict(duplicate["results"][0]))
        with self.assertRaises(ProfileError):
            RMSNormCostModel.from_dict(duplicate)

        invalid = make_profile()
        invalid["results"][0]["triton_us"] = math.nan
        with self.assertRaises(ProfileError):
            RMSNormCostModel.from_dict(invalid)


class RMSNormLoweringSelectionPassTest(unittest.TestCase):
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

    def make_module(self):
        return import_exported_program(self.program)

    def test_pipeline_attaches_plan_to_both_rmsnorms(self) -> None:
        module = self.make_module()
        model = RMSNormCostModel.from_dict(make_profile())

        results = default_pass_manager(model).run(module)
        operations = [
            operation
            for operation in module.functions[0].block.operations
            if operation.name == "serve.rms_norm"
        ]

        self.assertEqual(len(operations), 2)
        self.assertEqual(results[-1].statistics["planned_operations"], 2)
        self.assertEqual(
            results[-1].statistics["variant_backends"],
            {"inductor": 4, "triton": 4},
        )
        for operation in operations:
            plan = operation.attributes["lowering_plan"]
            self.assertEqual(plan["hidden_size"], 32)
            self.assertEqual(plan["dtype"], "fp32")
            self.assertEqual(len(plan["variants"]), 4)

    def test_selection_pass_is_idempotent(self) -> None:
        module = self.make_module()
        manager = default_pass_manager(
            RMSNormCostModel.from_dict(make_profile())
        )
        manager.run(module)
        second = manager.run(module)

        self.assertFalse(second[-1].changed)
        self.assertEqual(second[-1].statistics["planned_operations"], 2)

    def test_pipeline_records_missing_profile(self) -> None:
        module = self.make_module()
        model = RMSNormCostModel.from_dict(
            make_profile(hidden_size=64, dtype="fp16")
        )

        result = default_pass_manager(model).run(module)[-1]
        plans = [
            operation.attributes["lowering_plan"]
            for operation in module.functions[0].block.operations
            if operation.name == "serve.rms_norm"
        ]

        self.assertEqual(result.statistics["missing_profile"], 2)
        self.assertTrue(all(not plan["variants"] for plan in plans))
        self.assertTrue(
            all(plan["fallback"] == "inductor" for plan in plans)
        )


if __name__ == "__main__":
    unittest.main()
