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
    KVStateType,
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


@dataclass(frozen=True)
class KVCacheState:
    """按 Layer Slot 保存 K/V Tensor 的不可变运行时状态。"""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(self.values):
            raise ExecutionError("KV Cache 的 Key/Value Slot 数量必须相同且非空")
        for key, value in zip(self.keys, self.values):
            if key.shape != value.shape:
                raise ExecutionError("同一 Slot 的 Key/Value Shape 必须一致")
            if key.ndim != 4:
                raise ExecutionError("KV Cache Tensor 必须是四维 B×H×S×D")

    @classmethod
    def from_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> "KVCacheState":
        return cls((key,), (value,))

    @classmethod
    def from_layer_tensors(
        cls,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> "KVCacheState":
        """从 Hugging Face 风格的多层 ``(key, value)`` 序列创建状态。"""

        if not past_key_values:
            raise ExecutionError("多层 KV Cache 至少需要一个 Layer Slot")
        keys, values = zip(*past_key_values)
        return cls(tuple(keys), tuple(values))

    def read(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self.keys[slot], self.values[slot]
        except IndexError as error:
            raise ExecutionError(f"KV Cache 不存在 Slot {slot}") from error

    def append(
        self,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        axis: int = 2,
    ) -> "KVCacheState":
        past_key, past_value = self.read(slot)
        _validate_kv_append_inputs(past_key, past_value, key, value, axis)
        keys = list(self.keys)
        values = list(self.values)
        keys[slot] = torch.cat((past_key, key), dim=axis)
        values[slot] = torch.cat((past_value, value), dim=axis)
        return KVCacheState(
            tuple(keys),
            tuple(values),
            generation=self.generation + 1,
        )


@dataclass(frozen=True)
class PreallocatedKVCacheState:
    """共享物理 Buffer、用独立 Length 保持逻辑 SSA 版本的 KV 状态。"""

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    lengths: tuple[torch.Tensor, ...]
    capacity: int
    generation: int = 0

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ExecutionError("预分配 KV Capacity 必须为正数")
        if (
            not self.keys
            or len(self.keys) != len(self.values)
            or len(self.keys) != len(self.lengths)
        ):
            raise ExecutionError("预分配 KV 的 K/V/Length Slot 数量必须一致")
        for key, value, lengths in zip(
            self.keys,
            self.values,
            self.lengths,
        ):
            if key.shape != value.shape or key.ndim != 4:
                raise ExecutionError(
                    "预分配 KV Buffer 必须是相同 Shape 的 B×C×H×D"
                )
            if key.shape[1] != self.capacity:
                raise ExecutionError("KV Buffer 第二维必须等于 Capacity")
            if lengths.shape != (key.shape[0],):
                raise ExecutionError("KV Length 必须是一维 Batch Tensor")
            if lengths.dtype != torch.int64:
                raise ExecutionError("KV Length DType 必须是 i64")

    @classmethod
    def empty_from_key_template(
        cls,
        key: torch.Tensor,
        *,
        num_layers: int,
        capacity: int,
    ) -> "PreallocatedKVCacheState":
        """根据B×H×T×D Key模板创建所有Layer Slot的空物理Buffer。"""

        if key.ndim != 4 or num_layers <= 0 or capacity <= 0:
            raise ExecutionError("KV Init收到非法模板、Layer数量或Capacity")
        batch, heads, _, head_dim = key.shape
        keys = tuple(
            torch.zeros(
                batch,
                capacity,
                heads,
                head_dim,
                device=key.device,
                dtype=key.dtype,
            )
            for _ in range(num_layers)
        )
        values = tuple(torch.zeros_like(buffer) for buffer in keys)
        lengths = tuple(
            torch.zeros(batch, device=key.device, dtype=torch.int64)
            for _ in range(num_layers)
        )
        return cls(keys, values, lengths, capacity)

    @classmethod
    def from_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        capacity: int,
    ) -> "PreallocatedKVCacheState":
        if key.shape != value.shape or key.ndim != 4:
            raise ExecutionError("初始 KV Tensor 必须是相同 Shape 的 B×H×S×D")
        batch, heads, sequence, head_dim = key.shape
        if sequence > capacity:
            raise ExecutionError(
                f"初始 KV 长度 {sequence} 超过 Capacity {capacity}"
            )
        key_buffer = torch.zeros(
            batch,
            capacity,
            heads,
            head_dim,
            device=key.device,
            dtype=key.dtype,
        )
        value_buffer = torch.zeros_like(key_buffer)
        key_buffer[:, :sequence].copy_(key.transpose(1, 2))
        value_buffer[:, :sequence].copy_(value.transpose(1, 2))
        lengths = torch.full(
            (batch,),
            sequence,
            device=key.device,
            dtype=torch.int64,
        )
        return cls(
            (key_buffer,),
            (value_buffer,),
            (lengths,),
            capacity,
        )

    @classmethod
    def from_layer_tensors(
        cls,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        capacity: int,
    ) -> "PreallocatedKVCacheState":
        """把多层逻辑 B×H×S×D Cache 复制到共享契约的预分配 Slot。"""

        if not past_key_values:
            raise ExecutionError("多层 KV Cache 至少需要一个 Layer Slot")
        keys = []
        values = []
        lengths_by_layer = []
        expected_structure = None
        for slot, (key, value) in enumerate(past_key_values):
            if key.shape != value.shape or key.ndim != 4:
                raise ExecutionError(
                    f"第 {slot} 层初始 KV 必须是相同 Shape 的 B×H×S×D"
                )
            batch, heads, sequence, head_dim = key.shape
            structure = (batch, heads, sequence, head_dim, key.dtype, key.device)
            if expected_structure is None:
                expected_structure = structure
            elif structure != expected_structure:
                raise ExecutionError("所有 Layer Slot 的 KV Shape、DType 和设备必须一致")
            if sequence > capacity:
                raise ExecutionError(
                    f"第 {slot} 层初始 KV 长度 {sequence} 超过 Capacity {capacity}"
                )
            key_buffer = torch.zeros(
                batch,
                capacity,
                heads,
                head_dim,
                device=key.device,
                dtype=key.dtype,
            )
            value_buffer = torch.zeros_like(key_buffer)
            key_buffer[:, :sequence].copy_(key.transpose(1, 2))
            value_buffer[:, :sequence].copy_(value.transpose(1, 2))
            keys.append(key_buffer)
            values.append(value_buffer)
            lengths_by_layer.append(
                torch.full(
                    (batch,),
                    sequence,
                    device=key.device,
                    dtype=torch.int64,
                )
            )
        return cls(
            tuple(keys),
            tuple(values),
            tuple(lengths_by_layer),
            capacity,
        )

    def positions(self, slot: int) -> torch.Tensor:
        try:
            return self.lengths[slot]
        except IndexError as error:
            raise ExecutionError(f"KV Buffer 不存在 Slot {slot}") from error

    def read(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            key = self.keys[slot]
            value = self.values[slot]
            lengths = self.lengths[slot]
        except IndexError as error:
            raise ExecutionError(f"KV Buffer 不存在 Slot {slot}") from error
        used = int(lengths.max().item())
        # 物理布局是 B×Capacity×H×D，ServeIR 逻辑布局仍是 B×H×S×D。
        return (
            key[:, :used].transpose(1, 2),
            value[:, :used].transpose(1, 2),
        )

    def decode_attention(
        self,
        slot: int,
        query: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        groups: int,
        scale: float,
        runner=None,
    ) -> torch.Tensor:
        """直接读取 B×Capacity×H×D Buffer，不构造完整逻辑 KV Cache。"""

        key_buffer, value_buffer, lengths = self._attention_inputs(
            slot,
            query,
            attention_mask,
            groups,
        )
        if runner is not None:
            return runner(
                query,
                key_buffer,
                value_buffer,
                lengths,
                attention_mask,
                scale=scale,
            )

        sequence = attention_mask.shape[-1]
        key = key_buffer[:, :sequence].permute(0, 2, 1, 3)
        value = value_buffer[:, :sequence].permute(0, 2, 1, 3)
        head_indices = torch.arange(
            query.shape[1],
            device=query.device,
        ) // groups
        key = key.index_select(1, head_indices)
        value = value.index_select(1, head_indices)

        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        scores = scores.float() + attention_mask.float()
        valid = torch.arange(
            sequence,
            device=query.device,
        ).view(1, 1, 1, sequence) < lengths.view(-1, 1, 1, 1)
        scores = scores.masked_fill(~valid, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(query.dtype)
        return torch.matmul(probabilities, value)

    def _attention_inputs(
        self,
        slot: int,
        query: torch.Tensor,
        attention_mask: torch.Tensor,
        groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            key_buffer = self.keys[slot]
            value_buffer = self.values[slot]
            lengths = self.lengths[slot]
        except IndexError as error:
            raise ExecutionError(f"KV Buffer 不存在 Slot {slot}") from error
        if query.ndim != 4 or query.shape[2] != 1:
            raise ExecutionError(
                "Decode Attention 的 Query 必须是 B×H×1×D"
            )
        if attention_mask.ndim != 4 or attention_mask.shape[2] != 1:
            raise ExecutionError(
                "Decode Attention 的 Mask 必须是 B×1×1×S"
            )
        if (
            query.shape[0] != key_buffer.shape[0]
            or query.shape[0] != attention_mask.shape[0]
            or query.shape[3] != key_buffer.shape[3]
        ):
            raise ExecutionError(
                "Decode Attention 的 Batch 或 Head Dim 与 KV Buffer 不匹配"
            )
        if groups <= 0 or query.shape[1] != key_buffer.shape[2] * groups:
            raise ExecutionError("Decode Attention 的 GQA Groups 不匹配")
        if attention_mask.shape[-1] > self.capacity:
            raise ExecutionError(
                "Decode Attention 的 Mask 长度超过 KV Capacity"
            )
        _ensure_kv_capacity(
            lengths,
            attention_mask.shape[-1],
            "Decode Attention 的逻辑长度超过 Mask 长度",
        )
        return key_buffer, value_buffer, lengths

    def store(
        self,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        *,
        writer=None,
    ) -> "PreallocatedKVCacheState":
        buffer_key, buffer_value = self._validate_store(
            slot,
            key,
            value,
            positions,
        )
        if writer is None:
            native_kv_store(
                buffer_key,
                buffer_value,
                key,
                value,
                positions,
            )
        else:
            writer(
                buffer_key,
                buffer_value,
                key,
                value,
                positions,
            )
        # Store 只修改当前逻辑长度之外的物理位置，旧状态的可见前缀保持不变。
        return self

    def prefill_store(
        self,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        writer=None,
    ) -> "PreallocatedKVCacheState":
        """从位置0批量写入一个Layer的Prompt K/V并推进逻辑长度。"""

        positions = self.positions(slot)
        if torch.any(positions != 0):
            raise ExecutionError("Prefill KV Store要求目标Slot的初始长度为0")
        self.store(slot, key, value, positions, writer=writer)
        return self.advance(slot, delta=key.shape[2])

    def advance(
        self,
        slot: int,
        *,
        delta: int,
    ) -> "PreallocatedKVCacheState":
        if delta <= 0:
            raise ExecutionError("KV Advance Delta 必须为正数")
        positions = self.positions(slot)
        _ensure_kv_capacity(
            positions + delta,
            self.capacity,
            f"KV Advance 超过 Capacity {self.capacity}",
        )
        lengths = list(self.lengths)
        lengths[slot] = positions + delta
        return PreallocatedKVCacheState(
            self.keys,
            self.values,
            tuple(lengths),
            self.capacity,
            generation=self.generation + 1,
        )

    def _validate_store(
        self,
        slot: int,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            buffer_key = self.keys[slot]
            buffer_value = self.values[slot]
        except IndexError as error:
            raise ExecutionError(f"KV Buffer 不存在 Slot {slot}") from error
        if key.shape != value.shape or key.ndim != 4:
            raise ExecutionError("KV Store 的 K/V 必须是相同 Shape 的 B×H×T×D")
        batch, heads, tokens, head_dim = key.shape
        if (
            batch != buffer_key.shape[0]
            or heads != buffer_key.shape[2]
            or head_dim != buffer_key.shape[3]
        ):
            raise ExecutionError("KV Store 的 Batch、Head 或 Head Dim 不匹配")
        if positions.shape != (batch,) or positions.dtype != torch.int64:
            raise ExecutionError("KV Store Positions 必须是一维 Batch i64 Tensor")
        if positions.device != key.device:
            raise ExecutionError("KV Store Positions 和 K/V 必须位于同一 Device")
        _ensure_kv_capacity(
            positions + tokens,
            self.capacity,
            f"KV Store 超过 Capacity {self.capacity}",
        )
        return buffer_key, buffer_value


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
            "aten.embedding.default": torch.ops.aten.embedding.default,
            "aten.split_with_sizes.default": torch.split,
            "builtin.getitem": lambda value, index: value[index],
            "aten.view.default": lambda tensor, shape: tensor.view(shape),
            "aten.transpose.int": torch.transpose,
            "aten.contiguous.default": torch.ops.aten.contiguous.default,
            "aten.unsqueeze.default": torch.unsqueeze,
            "aten.slice.Tensor": torch.ops.aten.slice.Tensor,
            "aten.cat.default": lambda tensors, dim=0: torch.cat(
                tuple(tensors),
                dim=dim,
            ),
            "aten.neg.default": torch.neg,
            "aten.cos.default": torch.cos,
            "aten.sin.default": torch.sin,
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
        if name == "serve.linear":
            return self._serve_linear(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.prefill_attention":
            return self._serve_prefill_attention(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.rope":
            return self._serve_rope(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.append":
            return self._serve_kv_append(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.init":
            return self._serve_kv_init(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.prefill_store":
            return self._serve_kv_prefill_store(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.read":
            return self._serve_kv_read(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.length":
            return self._serve_kv_length(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.store":
            return self._serve_kv_store(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.kv.advance":
            return self._serve_kv_advance(
                [environment[value.name] for value in operands],
                attributes,
            )
        if name == "serve.decode_attention":
            return self._serve_decode_attention(
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

    @staticmethod
    def _serve_linear(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) not in {2, 3}:
            raise ExecutionError("serve.linear 需要 input、weight 和可选 bias")
        tensor, weight = operands[:2]
        bias = operands[2] if len(operands) == 3 else None
        if bool(attributes.get("has_bias")) != (bias is not None):
            raise ExecutionError("serve.linear 的 has_bias 与运行时参数不一致")
        return F.linear(tensor, weight, bias)

    @staticmethod
    def _serve_prefill_attention(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) != 4:
            raise ExecutionError(
                "serve.prefill_attention需要query、key、value和mask"
            )
        query, key, value, attention_mask = operands
        groups = int(attributes["groups"])
        if query.shape[1] != key.shape[1] * groups:
            raise ExecutionError("serve.prefill_attention的GQA groups不匹配")
        expanded_key = key.repeat_interleave(groups, dim=1)
        expanded_value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, expanded_key.transpose(-2, -1))
        # 保留原图“Score DType缩放后再Cast到FP32”的舍入顺序。
        scores = scores * float(attributes["scale"])
        scores = scores.float() + attention_mask.float()
        if attributes.get("causal") is True:
            tokens = query.shape[2]
            future = torch.triu(
                torch.ones(
                    tokens,
                    tokens,
                    dtype=torch.bool,
                    device=query.device,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(future, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(query.dtype)
        return torch.matmul(probabilities, expanded_value)

    @staticmethod
    def _serve_rope(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行Qwen2约定的前后半维旋转位置编码。"""

        if len(operands) != 4:
            raise ExecutionError("serve.rope需要query、key、cosine和sine")
        query, key, cosine, sine = operands
        head_dim = query.shape[-1]
        if head_dim != int(attributes["head_dim"]) or head_dim % 2:
            raise ExecutionError("serve.rope的Head Dim契约不匹配")

        def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
            half = tensor.shape[-1] // 2
            return torch.cat(
                (-tensor[..., half:], tensor[..., :half]),
                dim=-1,
            )

        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )

    @staticmethod
    def _serve_kv_append(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> KVCacheState:
        if len(operands) != 3:
            raise ExecutionError("serve.kv.append 需要 state、key 和 value")
        state, key, value = operands
        if not isinstance(state, KVCacheState):
            raise ExecutionError("serve.kv.append 的 state 类型非法")
        if not isinstance(key, torch.Tensor) or not isinstance(
            value,
            torch.Tensor,
        ):
            raise ExecutionError("serve.kv.append 的 key/value 必须是 Tensor")
        return state.append(
            int(attributes.get("slot", 0)),
            key,
            value,
            axis=int(attributes.get("axis", 2)),
        )

    @staticmethod
    def _serve_kv_init(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> PreallocatedKVCacheState:
        if len(operands) != 1 or not isinstance(operands[0], torch.Tensor):
            raise ExecutionError("serve.kv.init需要一个Key Tensor模板")
        return PreallocatedKVCacheState.empty_from_key_template(
            operands[0],
            num_layers=int(attributes["num_layers"]),
            capacity=int(attributes["capacity"]),
        )

    @staticmethod
    def _serve_kv_prefill_store(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> PreallocatedKVCacheState:
        if len(operands) != 3 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.kv.prefill_store需要state、key和value"
            )
        state, key, value = operands
        return state.prefill_store(
            int(attributes.get("slot", 0)),
            key,
            value,
        )

    @staticmethod
    def _serve_kv_read(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(operands) != 1 or not isinstance(
            operands[0],
            (KVCacheState, PreallocatedKVCacheState),
        ):
            raise ExecutionError("serve.kv.read 需要一个 KV Runtime State")
        return operands[0].read(int(attributes.get("slot", 0)))

    @staticmethod
    def _serve_kv_length(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) != 1 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.kv.length 需要 PreallocatedKVCacheState"
            )
        return operands[0].positions(int(attributes.get("slot", 0)))

    @staticmethod
    def _serve_kv_store(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> PreallocatedKVCacheState:
        if len(operands) != 4 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.kv.store 需要 state、key、value 和 positions"
            )
        state, key, value, positions = operands
        return state.store(
            int(attributes.get("slot", 0)),
            key,
            value,
            positions,
        )

    @staticmethod
    def _serve_kv_advance(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> PreallocatedKVCacheState:
        if len(operands) != 1 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.kv.advance 需要 PreallocatedKVCacheState"
            )
        return operands[0].advance(
            int(attributes.get("slot", 0)),
            delta=int(attributes.get("delta", 1)),
        )

    @staticmethod
    def _serve_decode_attention(
        operands: list[Any],
        attributes: dict[str, Any],
    ) -> torch.Tensor:
        if len(operands) != 3 or not isinstance(
            operands[0],
            PreallocatedKVCacheState,
        ):
            raise ExecutionError(
                "serve.decode_attention 需要 state、query 和 mask"
            )
        state, query, attention_mask = operands
        return state.decode_attention(
            int(attributes.get("slot", 0)),
            query,
            attention_mask,
            groups=int(attributes["groups"]),
            scale=float(attributes["scale"]),
        )


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
            arguments.append(_exported_state_value(program, spec.target))
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


def bind_stateful_decode_arguments(
    program: torch.export.ExportedProgram,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    state: KVCacheState | PreallocatedKVCacheState,
    *,
    extra_user_inputs: Mapping[str, Any] | None = None,
    primary_input_name: str = "hidden_states",
) -> list[Any]:
    """绑定改写后的 Decode 参数，用一个状态替代全部 Layer Cache 输入。"""

    user_values = {
        primary_input_name: hidden_states,
        "attention_mask": attention_mask,
    }
    if extra_user_inputs is not None:
        overlap = set(user_values) & set(extra_user_inputs)
        if overlap:
            raise ExecutionError(
                f"额外 Stateful 输入不能覆盖保留参数：{sorted(overlap)}"
            )
        user_values.update(extra_user_inputs)
    arguments = []
    state_inserted = False
    cache_components: dict[int, set[str]] = {}
    for spec in program.graph_signature.input_specs:
        kind = spec.kind.name
        name = spec.arg.name
        if kind in {"PARAMETER", "BUFFER"}:
            arguments.append(_exported_state_value(program, spec.target))
        elif kind == "CONSTANT_TENSOR":
            arguments.append(program.constants[spec.target])
        elif kind == "USER_INPUT":
            cache_input = _stateful_cache_input(name)
            if cache_input is not None:
                slot, component = cache_input
                cache_components.setdefault(slot, set()).add(component)
                if not state_inserted:
                    arguments.append(state)
                    state_inserted = True
                continue
            if name in user_values:
                arguments.append(user_values[name])
            else:
                raise ExecutionError(f"不支持的 Stateful Decode 输入 {name}")
        else:
            raise ExecutionError(f"暂不支持 Graph InputKind：{kind}")
    if not state_inserted:
        raise ExecutionError("导出程序没有 KV Cache 输入")
    expected_slots = list(range(len(cache_components)))
    if sorted(cache_components) != expected_slots or any(
        components != {"key", "value"}
        for components in cache_components.values()
    ):
        raise ExecutionError("导出程序的多层 KV Cache 输入不完整")
    if len(state.keys) != len(cache_components):
        raise ExecutionError(
            f"运行时状态有 {len(state.keys)} 个 Slot，导出程序需要 "
            f"{len(cache_components)} 个"
        )
    return arguments


def _exported_state_value(
    program: torch.export.ExportedProgram,
    target: str,
) -> Any:
    """读取持久参数/Buffer，兼容 persistent=False Buffer 的常量存储。"""

    if target in program.state_dict:
        return program.state_dict[target]
    if target in program.constants:
        return program.constants[target]
    raise ExecutionError(f"导出程序缺少参数或 Buffer：{target}")


def _stateful_cache_input(name: str) -> tuple[int, str] | None:
    """解析单层旧名称和 torch.export 展平后的多层 Cache 名称。"""

    if name == "past_key":
        return 0, "key"
    if name == "past_value":
        return 0, "value"
    nested = re.fullmatch(r"past_key_values_(\d+)_(0|1)", name)
    if nested is not None:
        return (
            int(nested.group(1)),
            "key" if nested.group(2) == "0" else "value",
        )
    indexed = re.fullmatch(r"past_(key|value)_(\d+)", name)
    if indexed is not None:
        return int(indexed.group(2)), indexed.group(1)
    return None


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
    if isinstance(type_, KVStateType):
        _validate_kv_state(value, type_, context)
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


def _validate_kv_state(
    value: Any,
    type_: KVStateType,
    context: str,
) -> None:
    if not isinstance(
        value,
        (KVCacheState, PreallocatedKVCacheState),
    ):
        raise ExecutionError(f"{context} 应为 KV Runtime State")
    if isinstance(value, PreallocatedKVCacheState):
        _validate_preallocated_kv_state(value, type_, context)
        return
    if len(value.keys) != type_.num_layers:
        raise ExecutionError(
            f"{context} 应包含 {type_.num_layers} 个 Layer Slot"
        )
    expected_dtype = _DTYPES.get(type_.dtype)
    for slot, (key, cached_value) in enumerate(
        zip(value.keys, value.values)
    ):
        if key.shape != cached_value.shape:
            raise ExecutionError(f"{context} Slot {slot} 的 K/V Shape 不一致")
        if key.ndim != 4:
            raise ExecutionError(f"{context} Slot {slot} 必须是四维 Tensor")
        if key.shape[1] != type_.num_kv_heads:
            raise ExecutionError(
                f"{context} Slot {slot} 的 KV Head 数量应为 "
                f"{type_.num_kv_heads}"
            )
        if key.shape[3] != type_.head_dim:
            raise ExecutionError(
                f"{context} Slot {slot} 的 Head Dim 应为 {type_.head_dim}"
            )
        if key.dtype != cached_value.dtype:
            raise ExecutionError(f"{context} Slot {slot} 的 K/V DType 不一致")
        if expected_dtype is not None and key.dtype != expected_dtype:
            raise ExecutionError(
                f"{context} Slot {slot} DType 应为 {expected_dtype}"
            )


def _validate_preallocated_kv_state(
    value: PreallocatedKVCacheState,
    type_: KVStateType,
    context: str,
) -> None:
    if type_.layout != "contiguous_bshd":
        raise ExecutionError(
            f"{context} 的预分配 Runtime 不支持 Layout {type_.layout}"
        )
    if type_.capacity is not None and value.capacity != type_.capacity:
        raise ExecutionError(
            f"{context} Capacity 应为 {type_.capacity}，实际为 {value.capacity}"
        )
    if len(value.keys) != type_.num_layers:
        raise ExecutionError(
            f"{context} 应包含 {type_.num_layers} 个 Layer Slot"
        )
    expected_dtype = _DTYPES.get(type_.dtype)
    for slot, (key, cached_value) in enumerate(
        zip(value.keys, value.values)
    ):
        if key.shape != cached_value.shape or key.ndim != 4:
            raise ExecutionError(
                f"{context} Slot {slot} 必须是 B×Capacity×H×D"
            )
        if key.shape[2] != type_.num_kv_heads:
            raise ExecutionError(
                f"{context} Slot {slot} 的 KV Head 数量应为 "
                f"{type_.num_kv_heads}"
            )
        if key.shape[3] != type_.head_dim:
            raise ExecutionError(
                f"{context} Slot {slot} 的 Head Dim 应为 {type_.head_dim}"
            )
        if expected_dtype is not None and key.dtype != expected_dtype:
            raise ExecutionError(
                f"{context} Slot {slot} DType 应为 {expected_dtype}"
            )


def native_kv_store(
    buffer_key: torch.Tensor,
    buffer_value: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    batch, _, tokens, _ = key.shape
    rows = torch.arange(
        batch,
        device=positions.device,
        dtype=torch.int64,
    ).view(batch, 1)
    offsets = torch.arange(
        tokens,
        device=positions.device,
        dtype=torch.int64,
    ).view(1, tokens)
    indices = (
        rows * buffer_key.shape[1]
        + positions.view(batch, 1)
        + offsets
    ).reshape(-1)
    source_key = key.transpose(1, 2).reshape(
        batch * tokens,
        key.shape[1],
        key.shape[3],
    )
    source_value = value.transpose(1, 2).reshape_as(source_key)
    buffer_key.view(
        -1,
        buffer_key.shape[2],
        buffer_key.shape[3],
    ).index_copy_(0, indices, source_key)
    buffer_value.view(
        -1,
        buffer_value.shape[2],
        buffer_value.shape[3],
    ).index_copy_(0, indices, source_value)


def _ensure_kv_capacity(
    end_positions: torch.Tensor,
    capacity: int,
    message: str,
) -> None:
    condition = (end_positions <= capacity).all()
    if end_positions.is_cuda:
        # 异步设备断言不会为正常路径引入 `.item()` 的主机同步。
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ExecutionError(message)


def _validate_kv_append_inputs(
    past_key: torch.Tensor,
    past_value: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    axis: int,
) -> None:
    if axis not in {2, -2}:
        raise ExecutionError("KV Cache 当前只支持沿 Sequence 维追加")
    if key.shape != value.shape:
        raise ExecutionError("待追加的 Key/Value Shape 必须一致")
    if key.ndim != 4 or past_key.ndim != 4:
        raise ExecutionError("KV Cache Tensor 必须是四维 B×H×S×D")
    if past_key.shape != past_value.shape:
        raise ExecutionError("历史 Key/Value Shape 必须一致")
    for dimension in (0, 1, 3):
        if past_key.shape[dimension] != key.shape[dimension]:
            raise ExecutionError(
                f"KV Append 第 {dimension} 维不匹配："
                f"{past_key.shape[dimension]} != {key.shape[dimension]}"
            )
    if past_key.dtype != key.dtype or past_value.dtype != value.dtype:
        raise ExecutionError("KV Append 的历史和当前 DType 必须一致")
    if past_key.device != key.device or past_value.device != value.device:
        raise ExecutionError("KV Append 的历史和当前 Device 必须一致")


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
