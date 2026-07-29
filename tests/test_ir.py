from __future__ import annotations

import unittest

from stateful_llm_compiler.dialects import kv_append, kv_read
from stateful_llm_compiler.ir import (
    Block,
    Function,
    IRBuilder,
    KVStateType,
    Module,
    StaticDim,
    TensorType,
    Value,
    VerificationError,
    format_module,
    verify_module,
)


class ServeIRTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tensor_type = TensorType(
            (StaticDim(2), StaticDim(4), StaticDim(16)),
            "f16",
            "cuda",
        )
        self.kv_type = KVStateType(
            dtype="f16",
            num_layers=1,
            num_kv_heads=2,
            head_dim=16,
        )

    def test_ssa_program_can_be_verified_and_printed(self) -> None:
        builder = IRBuilder()
        left = builder.argument(self.tensor_type, "left")
        right = builder.argument(self.tensor_type, "right")
        add = builder.emit(
            "aten.add.Tensor", [left, right], [self.tensor_type]
        )
        module = Module(
            [Function("add", builder.block, [add.results[0]])]
        )

        verify_module(module)
        text = format_module(module)
        self.assertIn('func @add', text)
        self.assertIn('"aten.add.Tensor"', text)
        self.assertIn("return %v0", text)

    def test_verifier_rejects_use_before_definition(self) -> None:
        builder = IRBuilder()
        argument = builder.argument(self.tensor_type, "input")
        future = Value("%future", self.tensor_type)
        operation = builder.emit(
            "aten.add.Tensor",
            [argument, future],
            [self.tensor_type],
        )
        module = Module(
            [Function("broken", builder.block, [operation.results[0]])]
        )

        with self.assertRaisesRegex(
            VerificationError, "尚未定义"
        ):
            verify_module(module)

    def test_kv_operations_have_explicit_effects(self) -> None:
        builder = IRBuilder()
        state = builder.argument(self.kv_type, "state")
        key = builder.argument(self.tensor_type, "key")
        value = builder.argument(self.tensor_type, "value")
        read = kv_read(
            builder,
            state,
            self.tensor_type,
            self.tensor_type,
            slot=0,
        )
        append = kv_append(
            builder, state, key, value, slot=0
        )
        module = Module(
            [
                Function(
                    "decode",
                    builder.block,
                    [read.results[0], append.results[0]],
                )
            ]
        )

        verify_module(module)
        text = format_module(module)
        self.assertIn("effects[read(kv)]", text)
        self.assertIn("effects[read(kv), write(kv)]", text)

    def test_kv_verifier_rejects_missing_write_effect(self) -> None:
        block = Block()
        builder = IRBuilder(block)
        state = builder.argument(self.kv_type, "state")
        key = builder.argument(self.tensor_type, "key")
        value = builder.argument(self.tensor_type, "value")
        operation = builder.emit(
            "serve.kv.append",
            [state, key, value],
            [self.kv_type],
        )
        module = Module(
            [Function("broken_kv", block, [operation.results[0]])]
        )

        with self.assertRaisesRegex(
            VerificationError, r"write\(kv\)"
        ):
            verify_module(module)


if __name__ == "__main__":
    unittest.main()

