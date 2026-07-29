"""用于编译器前端测试的小型 Qwen 风格 Decoder。

模型只使用普通 PyTorch 算子，既能暴露 RMSNorm、QKV 投影、Attention、SwiGLU
和残差等编译模式，又足够小，可以在 CPU 上快速导出和测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DecoderConfig:
    hidden_size: int = 64
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads 必须能被 num_kv_heads 整除")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class TinyDecoderBlock(nn.Module):
    """包含 GQA 和显式因果 Mask 输入的单层 Decoder。"""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        cfg = self.config
        q_size = cfg.num_heads * cfg.head_dim
        kv_size = cfg.num_kv_heads * cfg.head_dim

        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.qkv_proj = nn.Linear(
            cfg.hidden_size, q_size + 2 * kv_size, bias=True
        )
        self.o_proj = nn.Linear(q_size, cfg.hidden_size, bias=False)
        self.post_attention_norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps
        )
        self.gate_up_proj = nn.Linear(
            cfg.hidden_size, 2 * cfg.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            cfg.intermediate_size, cfg.hidden_size, bias=False
        )

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        cfg = self.config
        batch, sequence, _ = hidden_states.shape
        residual = hidden_states
        x = self.input_norm(hidden_states)

        q_size = cfg.num_heads * cfg.head_dim
        kv_size = cfg.num_kv_heads * cfg.head_dim
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(
            qkv, (q_size, kv_size, kv_size), dim=-1
        )

        query = query.view(
            batch, sequence, cfg.num_heads, cfg.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch, sequence, cfg.num_kv_heads, cfg.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch, sequence, cfg.num_kv_heads, cfg.head_dim
        ).transpose(1, 2)

        groups = cfg.num_heads // cfg.num_kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(cfg.head_dim)
        probabilities = torch.softmax(
            scores.float() + attention_mask.float(), dim=-1
        ).to(query.dtype)
        context = torch.matmul(probabilities, value)
        context = context.transpose(1, 2).reshape(
            batch, sequence, cfg.hidden_size
        )
        hidden_states = residual + self.o_proj(context)

        residual = hidden_states
        x = self.post_attention_norm(hidden_states)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        mlp = torch.nn.functional.silu(gate) * up
        return residual + self.down_proj(mlp)


class StatefulTinyDecoderBlock(TinyDecoderBlock):
    """接收历史 KV Cache，并返回追加后 Cache 的单层 Decode 模型。

    输入的当前序列长度可以大于一，但前端里程碑只导出单 Token Decode。
    Cache 保留 GQA 原始 KV Head 数量，只有执行 Attention 前才扩展到 Query Head。
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        past_key: Tensor,
        past_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        batch, sequence, _ = hidden_states.shape
        residual = hidden_states
        x = self.input_norm(hidden_states)

        q_size = cfg.num_heads * cfg.head_dim
        kv_size = cfg.num_kv_heads * cfg.head_dim
        qkv = self.qkv_proj(x)
        query, current_key, current_value = torch.split(
            qkv,
            (q_size, kv_size, kv_size),
            dim=-1,
        )

        query = query.view(
            batch,
            sequence,
            cfg.num_heads,
            cfg.head_dim,
        ).transpose(1, 2)
        current_key = current_key.view(
            batch,
            sequence,
            cfg.num_kv_heads,
            cfg.head_dim,
        ).transpose(1, 2)
        current_value = current_value.view(
            batch,
            sequence,
            cfg.num_kv_heads,
            cfg.head_dim,
        ).transpose(1, 2)

        key_cache = torch.cat((past_key, current_key), dim=2)
        value_cache = torch.cat((past_value, current_value), dim=2)

        groups = cfg.num_heads // cfg.num_kv_heads
        attention_key = key_cache.repeat_interleave(groups, dim=1)
        attention_value = value_cache.repeat_interleave(groups, dim=1)

        scores = torch.matmul(
            query,
            attention_key.transpose(-2, -1),
        )
        scores = scores / math.sqrt(cfg.head_dim)
        probabilities = torch.softmax(
            scores.float() + attention_mask.float(),
            dim=-1,
        ).to(query.dtype)
        context = torch.matmul(probabilities, attention_value)
        context = context.transpose(1, 2).reshape(
            batch,
            sequence,
            cfg.hidden_size,
        )
        hidden_states = residual + self.o_proj(context)

        residual = hidden_states
        x = self.post_attention_norm(hidden_states)
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        mlp = torch.nn.functional.silu(gate) * up
        output = residual + self.down_proj(mlp)
        return output, key_cache, value_cache


def make_inputs(
    config: DecoderConfig,
    batch: int,
    sequence: int,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """创建确定性的 Hidden State 和加法因果 Mask。"""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        batch,
        sequence,
        config.hidden_size,
        generator=generator,
        dtype=torch.float32,
    )
    causal = torch.triu(
        torch.full((sequence, sequence), float("-inf")), diagonal=1
    )
    attention_mask = causal.view(1, 1, sequence, sequence).expand(
        batch, 1, sequence, sequence
    ).clone()
    return hidden_states, attention_mask


def make_decode_inputs(
    config: DecoderConfig,
    batch: int,
    past_length: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """创建单 Token Decode 输入，历史 Cache 长度必须至少为 1。"""

    if past_length < 1:
        raise ValueError("Decode 的 past_length 必须至少为 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        batch,
        1,
        config.hidden_size,
        generator=generator,
        dtype=dtype,
    )
    past_key = torch.randn(
        batch,
        config.num_kv_heads,
        past_length,
        config.head_dim,
        generator=generator,
        dtype=dtype,
    )
    past_value = torch.randn(
        batch,
        config.num_kv_heads,
        past_length,
        config.head_dim,
        generator=generator,
        dtype=dtype,
    )
    # 单 Token Decode 只能看到历史 Token 和当前 Token，因此 Mask 全为零。
    attention_mask = torch.zeros(
        batch,
        1,
        1,
        past_length + 1,
        dtype=dtype,
    )
    return hidden_states, attention_mask, past_key, past_value
