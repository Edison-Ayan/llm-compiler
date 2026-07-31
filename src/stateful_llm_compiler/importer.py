"""把 torch.export 的 Functional ATen 图导入 ServeIR。"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path
from typing import Any

import torch
from torch.fx import Node

from .ir import (
    Function,
    IRBuilder,
    IRType,
    Module,
    ScalarType,
    StaticDim,
    SymbolicDim,
    TensorType,
    TupleType,
    UnknownType,
    Value,
    format_module,
    verify_module,
)


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int8: "i8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.bool: "i1",
    }
    return names.get(dtype, str(dtype).removeprefix("torch."))


def _infer_type(value: Any, ranges: dict[str, str]) -> IRType:
    if isinstance(value, torch.Tensor):
        dimensions = []
        for dimension in value.shape:
            if isinstance(dimension, int):
                dimensions.append(StaticDim(dimension))
            else:
                name = str(dimension)
                dimensions.append(SymbolicDim(name, ranges.get(name)))
        return TensorType(
            tuple(dimensions),
            _dtype_name(value.dtype),
            str(value.device),
        )
    if isinstance(value, (tuple, list)):
        return TupleType(tuple(_infer_type(item, ranges) for item in value))
    if isinstance(value, bool):
        return ScalarType("i1")
    if isinstance(value, int):
        return ScalarType("index")
    if isinstance(value, float):
        return ScalarType("f64")
    # SymInt 不能稳定地作为公开类型导入，使用字符串特征保留符号信息。
    if type(value).__name__ in {"SymInt", "SymFloat", "SymBool"}:
        return ScalarType(type(value).__name__.lower())
    return UnknownType(type(value).__name__)


def _node_type(node: Node, ranges: dict[str, str]) -> IRType:
    if "val" in node.meta:
        return _infer_type(node.meta["val"], ranges)
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is not None:
        return _infer_type(tensor_meta, ranges)
    return UnknownType(f"fx_{node.name}")


def _operation_name(target: Any) -> tuple[str, str | None]:
    target_text = str(target)
    builtin_targets = {
        operator.getitem: "builtin.getitem",
        operator.eq: "builtin.eq",
        operator.add: "builtin.add",
    }
    if target in builtin_targets:
        return builtin_targets[target], None
    if target_text.startswith("aten."):
        return target_text, None
    return "serve.external", target_text


def _encode_argument(
    argument: Any,
    values: dict[Node, Value],
    operands: list[Value],
) -> Any:
    """保留 FX 参数树，同时把 Node 引用转换成 SSA 操作数。"""

    if isinstance(argument, Node):
        value = values[argument]
        operands.append(value)
        return {"ssa": value.name}
    if isinstance(argument, tuple):
        return {
            "tuple": [
                _encode_argument(item, values, operands) for item in argument
            ]
        }
    if isinstance(argument, list):
        return [
            _encode_argument(item, values, operands) for item in argument
        ]
    if isinstance(argument, dict):
        return {
            str(key): _encode_argument(item, values, operands)
            for key, item in argument.items()
        }
    if isinstance(argument, torch.dtype):
        return _dtype_name(argument)
    if isinstance(argument, torch.device):
        return str(argument)
    if isinstance(argument, (str, int, float, bool)) or argument is None:
        return argument
    return str(argument)


def import_exported_program(
    program: torch.export.ExportedProgram,
    *,
    function_name: str = "decoder",
) -> Module:
    """导入完整 FX 图，并在返回前执行 ServeIR 校验。"""

    ranges = {
        str(symbol): str(constraint)
        for symbol, constraint in program.range_constraints.items()
    }
    builder = IRBuilder()
    values: dict[Node, Value] = {}
    returns: list[Value] = []

    for node in program.graph_module.graph.nodes:
        if node.op == "placeholder":
            values[node] = builder.argument(
                _node_type(node, ranges), hint=node.name
            )
            continue

        if node.op == "output":
            output_operands: list[Value] = []
            _encode_argument(node.args, values, output_operands)
            returns.extend(output_operands)
            continue

        operands: list[Value] = []
        argument_spec = _encode_argument(node.args, values, operands)
        keyword_spec = _encode_argument(node.kwargs, values, operands)
        operation_name, external_target = _operation_name(node.target)
        attributes: dict[str, Any] = {
            "fx_name": node.name,
            "args": argument_spec,
        }
        if keyword_spec:
            attributes["kwargs"] = keyword_spec
        if external_target is not None:
            attributes["target"] = external_target

        operation = builder.emit(
            operation_name,
            operands,
            [_node_type(node, ranges)],
            attributes=attributes,
        )
        values[node] = operation.results[0]

    module = Module(
        [Function(function_name, builder.block, returns)]
    )
    verify_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 torch.export 程序导入 ServeIR"
    )
    parser.add_argument("program", type=Path, help="输入的 .pt2 文件")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--function-name", default="decoder")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    program = torch.export.load(args.program)
    module = import_exported_program(
        program, function_name=args.function_name
    )
    text = format_module(module)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")

    function = module.functions[0]
    external_count = sum(
        operation.name == "serve.external"
        for operation in function.block.operations
    )
    print(
        f"导入完成：{len(function.block.arguments)} 个参数，"
        f"{len(function.block.operations)} 个操作，"
        f"{external_count} 个 external fallback"
    )
    print(f"ServeIR：{args.out}")


if __name__ == "__main__":
    main()
