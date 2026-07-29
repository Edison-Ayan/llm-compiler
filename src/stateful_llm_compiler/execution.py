"""ServeIR 的 CPU/PyTorch 参考执行器。

参考执行器追求语义清晰和可验证性，不追求性能。它让优化后的高层 IR 可以与原始
ExportedProgram 做差分测试，为后续 Triton Lowering 提供正确性基线。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .ir import (
    Function,
    IRType,
    Module,
    ScalarType,
    StaticDim,
    SymbolicDim,
    TensorType,
    TupleType,
    UnknownType,
)


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    outputs: tuple[Any, ...]
    executed_operations: tuple[str, ...]
    symbolic_dimensions: dict[str, int]


_DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "i1": torch.bool,
}


class ReferenceExecutor:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {
            "aten.sym_size.int": lambda tensor, dim: tensor.shape[dim],
            "aten.to.dtype": self._to_dtype,
            "aten.linear.default": F.linear,
            "aten.split_with_sizes.default": torch.split,
            "builtin.getitem": lambda value, index: value[index],
            "aten.view.default": lambda tensor, shape: tensor.view(shape),
            "aten.transpose.int": torch.transpose,
            "aten.repeat_interleave.self_int": torch.repeat_interleave,
            "aten.matmul.default": torch.matmul,
            "aten.div.Tensor": torch.div,
            "aten.add.Tensor": torch.add,
            "aten.softmax.int": torch.softmax,
            "aten.reshape.default": torch.reshape,
            "aten.chunk.default": torch.chunk,
            "aten.silu.default": F.silu,
            "aten.mul.Tensor": torch.mul,
        }

    def run(
        self,
        module: Module,
        arguments: Sequence[Any],
        *,
        function_name: str | None = None,
    ) -> ExecutionResult:
        function = _select_function(module, function_name)
        if len(arguments) != len(function.block.arguments):
            raise ExecutionError(
                f"函数 @{function.name} 需要 {len(function.block.arguments)} "
                f"个参数，实际收到 {len(arguments)} 个"
            )

        environment: dict[str, Any] = {}
        symbols: dict[str, int] = {}
        for argument, runtime_value in zip(
            function.block.arguments, arguments
        ):
            _validate_runtime_value(
                runtime_value,
                argument.type,
                symbols,
                context=f"参数 {argument.name}",
            )
            environment[argument.name] = runtime_value

        trace = []
        for operation in function.block.operations:
            runtime_results = self._execute_operation(
                operation.name,
                operation.attributes,
                operation.operands,
                environment,
            )
            values = _normalize_results(
                runtime_results, len(operation.results), operation.name
            )
            for result, runtime_value in zip(operation.results, values):
                _validate_runtime_value(
                    runtime_value,
                    result.type,
                    symbols,
                    context=f"{operation.name} 的结果 {result.name}",
                )
                environment[result.name] = runtime_value
            trace.append(operation.name)

        outputs = tuple(environment[value.name] for value in function.returns)
        return ExecutionResult(outputs, tuple(trace), dict(symbols))

    def _execute_operation(
        self,
        name: str,
        attributes: dict[str, Any],
        operands,
        environment: dict[str, Any],
    ) -> Any:
        if name == "serve.rms_norm":
            return self._serve_rms_norm(
                [environment[value.name] for value in operands],
                attributes,
            )
        handler = self._handlers.get(name)
        if handler is None:
            raise ExecutionError(f"参考执行器尚不支持操作 {name}")

        if "args" in attributes:
            decoded = _decode_attribute(attributes["args"], environment)
            args = decoded if isinstance(decoded, tuple) else (decoded,)
        else:
            args = tuple(environment[value.name] for value in operands)
        kwargs = {}
        if "kwargs" in attributes:
            decoded_kwargs = _decode_attribute(
                attributes["kwargs"], environment
            )
            if not isinstance(decoded_kwargs, dict):
                raise ExecutionError(f"{name} 的 kwargs 不是字典")
            kwargs = decoded_kwargs
        return handler(*args, **kwargs)

    @staticmethod
    def _to_dtype(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
        torch_dtype = _DTYPES.get(dtype)
        if torch_dtype is None:
            raise ExecutionError(f"不支持的 DType：{dtype}")
        return tensor.to(torch_dtype)

    @staticmethod
    def _serve_rms_norm(
        operands: list[Any], attributes: dict[str, Any]
    ) -> torch.Tensor:
        if len(operands) != 2:
            raise ExecutionError("serve.rms_norm 需要 input 和 weight")
        tensor, weight = operands
        epsilon = float(attributes["epsilon"])
        axis = int(attributes.get("axis", -1))
        if axis not in {-1, tensor.ndim - 1}:
            raise ExecutionError("参考 RMSNorm 当前只支持最后一个维度")

        variance = tensor.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = tensor.float() * torch.rsqrt(variance + epsilon)
        output_dtype = attributes.get("output_dtype")
        torch_dtype = _DTYPES.get(output_dtype, tensor.dtype)
        return (normalized * weight.float()).to(torch_dtype)


def bind_exported_program_arguments(
    program: torch.export.ExportedProgram,
    user_inputs: Sequence[Any] | Mapping[str, Any],
) -> list[Any]:
    """按 Graph Signature 顺序绑定参数、常量和用户输入。"""

    positional = list(user_inputs) if not isinstance(user_inputs, Mapping) else []
    positional_index = 0
    arguments = []
    for spec in program.graph_signature.input_specs:
        kind = spec.kind.name
        name = spec.arg.name
        if kind in {"PARAMETER", "BUFFER"}:
            arguments.append(program.state_dict[spec.target])
        elif kind == "CONSTANT_TENSOR":
            arguments.append(program.constants[spec.target])
        elif kind == "USER_INPUT":
            if isinstance(user_inputs, Mapping):
                if name not in user_inputs:
                    raise ExecutionError(f"缺少用户输入 {name}")
                arguments.append(user_inputs[name])
            else:
                if positional_index >= len(positional):
                    raise ExecutionError(f"缺少用户输入 {name}")
                arguments.append(positional[positional_index])
                positional_index += 1
        else:
            raise ExecutionError(f"暂不支持 Graph InputKind：{kind}")
    if not isinstance(user_inputs, Mapping) and positional_index != len(positional):
        raise ExecutionError(
            f"收到 {len(positional)} 个用户输入，但只消费了 "
            f"{positional_index} 个"
        )
    return arguments


def _select_function(
    module: Module, function_name: str | None
) -> Function:
    if function_name is None:
        if len(module.functions) != 1:
            raise ExecutionError("Module 包含多个函数，必须指定 function_name")
        return module.functions[0]
    for function in module.functions:
        if function.name == function_name:
            return function
    raise ExecutionError(f"找不到函数 @{function_name}")


def _decode_attribute(value: Any, environment: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"ssa"}:
            name = value["ssa"]
            if name not in environment:
                raise ExecutionError(f"Attribute 引用了未定义值 {name}")
            return environment[name]
        if set(value) == {"tuple"}:
            return tuple(
                _decode_attribute(item, environment)
                for item in value["tuple"]
            )
        return {
            key: _decode_attribute(item, environment)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_decode_attribute(item, environment) for item in value]
    return value


def _normalize_results(
    runtime_results: Any, expected: int, operation_name: str
) -> tuple[Any, ...]:
    if expected == 0:
        if runtime_results is not None:
            raise ExecutionError(f"{operation_name} 不应产生结果")
        return ()
    if expected == 1:
        return (runtime_results,)
    if not isinstance(runtime_results, (tuple, list)):
        raise ExecutionError(
            f"{operation_name} 应产生 {expected} 个结果"
        )
    if len(runtime_results) != expected:
        raise ExecutionError(
            f"{operation_name} 应产生 {expected} 个结果，"
            f"实际为 {len(runtime_results)} 个"
        )
    return tuple(runtime_results)


def _validate_runtime_value(
    value: Any,
    type_: IRType,
    symbols: dict[str, int],
    *,
    context: str,
) -> None:
    if isinstance(type_, UnknownType):
        return
    if isinstance(type_, TensorType):
        if not isinstance(value, torch.Tensor):
            raise ExecutionError(f"{context} 应为 Tensor")
        expected_dtype = _DTYPES.get(type_.dtype)
        if expected_dtype is not None and value.dtype != expected_dtype:
            raise ExecutionError(
                f"{context} DType 应为 {expected_dtype}，实际为 {value.dtype}"
            )
        if str(value.device) != type_.device:
            raise ExecutionError(
                f"{context} Device 应为 {type_.device}，实际为 {value.device}"
            )
        if value.ndim != len(type_.shape):
            raise ExecutionError(
                f"{context} Rank 应为 {len(type_.shape)}，实际为 {value.ndim}"
            )
        for index, (actual, dimension) in enumerate(
            zip(value.shape, type_.shape)
        ):
            if isinstance(dimension, StaticDim):
                if actual != dimension.value:
                    raise ExecutionError(
                        f"{context} 第 {index} 维应为 {dimension.value}，"
                        f"实际为 {actual}"
                    )
            elif isinstance(dimension, SymbolicDim):
                _bind_symbol(
                    dimension, int(actual), symbols, context, index
                )
        return
    if isinstance(type_, TupleType):
        if not isinstance(value, (tuple, list)):
            raise ExecutionError(f"{context} 应为 Tuple")
        if len(value) != len(type_.elements):
            raise ExecutionError(
                f"{context} Tuple 长度应为 {len(type_.elements)}"
            )
        for index, (item, item_type) in enumerate(
            zip(value, type_.elements)
        ):
            _validate_runtime_value(
                item,
                item_type,
                symbols,
                context=f"{context}[{index}]",
            )
        return
    if isinstance(type_, ScalarType):
        if type_.dtype in {"index", "symint"} and not isinstance(value, int):
            raise ExecutionError(f"{context} 应为整数标量")
        return
    raise ExecutionError(f"{context} 使用了无法执行的类型 {type_}")


def _bind_symbol(
    dimension: SymbolicDim,
    actual: int,
    symbols: dict[str, int],
    context: str,
    index: int,
) -> None:
    existing = symbols.get(dimension.name)
    if existing is not None and existing != actual:
        raise ExecutionError(
            f"{context} 第 {index} 维违反符号 {dimension.name} 的相等约束："
            f"已有 {existing}，实际为 {actual}"
        )
    lower, upper = _parse_bounds(dimension.bounds)
    if lower is not None and actual < lower:
        raise ExecutionError(
            f"{context} 第 {index} 维的 {dimension.name}={actual} "
            f"小于下界 {lower}"
        )
    if upper is not None and actual > upper:
        raise ExecutionError(
            f"{context} 第 {index} 维的 {dimension.name}={actual} "
            f"大于上界 {upper}"
        )
    symbols[dimension.name] = actual


def _parse_bounds(bounds: str | None) -> tuple[int | None, int | None]:
    if not bounds:
        return None, None
    match = re.fullmatch(r"VR\[(-?\d+),\s*(-?\d+|int_oo)\]", bounds)
    if match is None:
        return None, None
    lower = int(match.group(1))
    upper = None if match.group(2) == "int_oo" else int(match.group(2))
    return lower, upper
