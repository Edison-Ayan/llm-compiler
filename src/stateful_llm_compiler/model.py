"""A compact Qwen-style decoder block used as the compiler frontend fixture.

The module is intentionally made from ordinary PyTorch operators. It is large
enough to expose useful compiler patterns (RMSNorm, QKV projection, attention,
SwiGLU and residual connections), but small enough to export and test on CPU.
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
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")

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
    """One decoder layer with GQA and an explicit causal-mask input."""

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


def make_inputs(
    config: DecoderConfig,
    batch: int,
    sequence: int,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Create deterministic hidden states and an additive causal mask."""

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

