"""可与 Hugging Face Qwen2 随机权重逐项对照的 Stateful Decode 模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import RMSNorm


@dataclass(frozen=True)
class Qwen2CompatConfig:
    """项目实际支持的 Qwen2 Decoder 配置子集。"""

    hidden_size: int = 64
    intermediate_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 32768
    vocab_size: int = 32000
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size 必须能被 num_heads 整除")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads 必须能被 num_kv_heads 整除")
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须为正数")
        if self.head_dim % 2:
            raise ValueError("Qwen2 RoPE 要求 head_dim 为偶数")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta 必须为正数")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须为正数")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_huggingface(cls, config: Any) -> "Qwen2CompatConfig":
        """从 Hugging Face Qwen2Config 提取已验证的配置子集。"""

        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError("当前 Qwen2 兼容层只支持 SiLU MLP")
        if bool(getattr(config, "use_sliding_window", False)):
            raise ValueError("当前 Qwen2 兼容层尚未支持 Sliding Window Attention")
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"当前只支持默认 RoPE，收到 {rope_type}")
        return cls(
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_layers=int(config.num_hidden_layers),
            num_heads=int(config.num_attention_heads),
            num_kv_heads=int(config.num_key_value_heads),
            rms_norm_eps=float(config.rms_norm_eps),
            rope_theta=float(rope_parameters.get("rope_theta", 10000.0)),
            max_position_embeddings=int(config.max_position_embeddings),
            vocab_size=int(config.vocab_size),
            tie_word_embeddings=bool(config.tie_word_embeddings),
        )


def rotate_half(tensor: Tensor) -> Tensor:
    """按照 Qwen2 约定交换 Head Dim 的前后两半并旋转符号。"""

    half = tensor.shape[-1] // 2
    first = tensor[..., :half]
    second = tensor[..., half:]
    return torch.cat((-second, first), dim=-1)


def apply_rotary_position_embedding(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    """把 B×T×D 的旋转系数广播到 B×H×T×D Query/Key。"""

    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


class Qwen2RotaryEmbedding(nn.Module):
    """默认 RoPE；公式与 Hugging Face Qwen2RotaryEmbedding 对齐。"""

    def __init__(self, config: Qwen2CompatConfig) -> None:
        super().__init__()
        self._head_dim = config.head_dim
        self._rope_theta = config.rope_theta
        inverse_frequency = self._make_inverse_frequency(device=None)
        self.register_buffer(
            "inv_freq",
            inverse_frequency,
            persistent=False,
        )

    def _make_inverse_frequency(
        self,
        device: torch.device | None,
    ) -> Tensor:
        """始终以FP32重建频率，避免权重DType转换损失RoPE精度。"""

        dimensions = torch.arange(
            0,
            self._head_dim,
            2,
            dtype=torch.float32,
            device=device,
        )
        return 1.0 / (
            self._rope_theta ** (dimensions / self._head_dim)
        )

    def _apply(self, fn, recurse: bool = True):
        """迁移Device后重建FP32 Buffer，不跟随模型权重转成BF16。"""

        result = super()._apply(fn, recurse=recurse)
        self.inv_freq = self._make_inverse_frequency(self.inv_freq.device)
        return result

    def forward(
        self,
        tensor: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        frequencies = (
            position_ids.float().unsqueeze(-1)
            * self.inv_freq.float().view(1, 1, -1)
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embedding.cos().to(tensor.dtype),
            embedding.sin().to(tensor.dtype),
        )


class StatefulQwen2Attention(nn.Module):
    """独立 Q/K/V 投影、RoPE、GQA 和显式 Tensor Cache 的 Qwen2 Attention。"""

    def __init__(self, config: Qwen2CompatConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * config.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_kv_heads * config.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            config.num_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key: Tensor | None,
        past_value: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        batch, tokens, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch,
            tokens,
            cfg.num_heads,
            cfg.head_dim,
        ).transpose(1, 2)
        current_key = self.k_proj(hidden_states).view(
            batch,
            tokens,
            cfg.num_kv_heads,
            cfg.head_dim,
        ).transpose(1, 2)
        current_value = self.v_proj(hidden_states).view(
            batch,
            tokens,
            cfg.num_kv_heads,
            cfg.head_dim,
        ).transpose(1, 2)

        cosine, sine = position_embeddings
        query, current_key = apply_rotary_position_embedding(
            query,
            current_key,
            cosine,
            sine,
        )
        if past_key is None or past_value is None:
            # Prefill没有历史状态，当前K/V本身就是首个Cache版本。
            key_cache = current_key
            value_cache = current_value
        else:
            key_cache = torch.cat((past_key, current_key), dim=2)
            value_cache = torch.cat((past_value, current_value), dim=2)

        groups = cfg.num_heads // cfg.num_kv_heads
        attention_key = key_cache.repeat_interleave(groups, dim=1)
        attention_value = value_cache.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, attention_key.transpose(-2, -1))
        scores = scores * (cfg.head_dim ** -0.5)
        probabilities = torch.softmax(
            scores.float() + attention_mask.float(),
            dim=-1,
        ).to(query.dtype)
        context = torch.matmul(probabilities, attention_value)
        context = context.transpose(1, 2).contiguous().reshape(
            batch,
            tokens,
            cfg.num_heads * cfg.head_dim,
        )
        return self.o_proj(context), key_cache, value_cache


class Qwen2MLP(nn.Module):
    """使用独立 Gate/Up Projection 的 Qwen2 SwiGLU。"""

    def __init__(self, config: Qwen2CompatConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


class StatefulQwen2DecoderLayer(nn.Module):
    """与 Hugging Face Qwen2DecoderLayer 参数命名一致的 Stateful Layer。"""

    def __init__(self, config: Qwen2CompatConfig) -> None:
        super().__init__()
        self.self_attn = StatefulQwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key: Tensor,
        past_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        attention_output, present_key, present_value = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_embeddings,
            past_key,
            past_value,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key, present_value


class StatefulQwen2Model(nn.Module):
    """不含 Embedding/LM Head、直接接收 Input Embedding 的 Qwen2 Decoder。"""

    def __init__(self, config: Qwen2CompatConfig | None = None) -> None:
        super().__init__()
        self.config = config or Qwen2CompatConfig()
        self.layers = nn.ModuleList(
            StatefulQwen2DecoderLayer(self.config)
            for _ in range(self.config.num_layers)
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(self.config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: tuple[tuple[Tensor, Tensor], ...],
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        if len(past_key_values) != len(self.layers):
            raise ValueError("KV Cache Slot 数量必须等于 Qwen2 Decoder Layer 数量")
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        present_key_values = []
        for layer, (past_key, past_value) in zip(
            self.layers,
            past_key_values,
        ):
            hidden_states, present_key, present_value = layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                past_key,
                past_value,
            )
            present_key_values.append((present_key, present_value))
        return self.norm(hidden_states), tuple(present_key_values)

    def prefill(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """从空状态执行多Token Prefill，并返回每层首个K/V版本。"""

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        present_key_values = []
        for layer in self.layers:
            hidden_states, present_key, present_value = layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                None,
                None,
            )
            present_key_values.append((present_key, present_value))
        return self.norm(hidden_states), tuple(present_key_values)

    @classmethod
    def from_huggingface(cls, model: nn.Module) -> "StatefulQwen2Model":
        """构造同配置模型，并复制 Hugging Face Qwen2Model 的 Decoder 权重。"""

        config = Qwen2CompatConfig.from_huggingface(model.config)
        converted = cls(config)
        reference_parameter = next(model.parameters())
        converted.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        expected_names = set(converted.state_dict())
        source_state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if name in expected_names
        }
        missing = sorted(expected_names - set(source_state))
        if missing:
            raise ValueError(f"Hugging Face Qwen2 缺少 Decoder 权重：{missing}")
        converted.load_state_dict(source_state, strict=True)
        return converted


class StatefulQwen2ForCausalLM(nn.Module):
    """覆盖Embedding、Decoder、KV输出和LM Head的完整Qwen2边界。"""

    def __init__(self, config: Qwen2CompatConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoder = StatefulQwen2Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def prefill(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """把Prompt Token编译边界转换为Logits和可复用的多层KV Cache。"""

        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_key_values = self.decoder.prefill(
            hidden_states,
            attention_mask,
            position_ids,
        )
        return self.lm_head(hidden_states), present_key_values

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """默认前向入口表示从空状态开始的Prefill。"""

        return self.prefill(input_ids, attention_mask, position_ids)

    def decode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_key_values: tuple[tuple[Tensor, Tensor], ...],
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """消费Prefill状态执行单Token Decode并输出词表Logits。"""

        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_key_values = self.decoder(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
        )
        return self.lm_head(hidden_states), present_key_values

    @classmethod
    def from_huggingface(cls, model: nn.Module) -> "StatefulQwen2ForCausalLM":
        """复制官方Qwen2ForCausalLM的Embedding、Decoder和LM Head权重。"""

        config = Qwen2CompatConfig.from_huggingface(model.config)
        converted = cls(config)
        reference_parameter = next(model.parameters())
        converted.to(
            device=reference_parameter.device,
            dtype=reference_parameter.dtype,
        )
        converted.embed_tokens.load_state_dict(
            {"weight": model.model.embed_tokens.weight.detach()},
            strict=True,
        )
        converted.decoder = StatefulQwen2Model.from_huggingface(model.model)
        if config.tie_word_embeddings:
            converted.lm_head.weight = converted.embed_tokens.weight
        else:
            converted.lm_head.load_state_dict(
                {"weight": model.lm_head.weight.detach()},
                strict=True,
            )
        return converted


def make_qwen2_decode_inputs(
    config: Qwen2CompatConfig,
    batch: int,
    past_length: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    tuple[tuple[Tensor, Tensor], ...],
]:
    """创建单 Token Qwen2 Decode 的 Input Embedding、Position 和多层 Cache。"""

    if past_length < 1:
        raise ValueError("Qwen2 Decode 的 past_length 必须至少为 1")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        batch,
        1,
        config.hidden_size,
        generator=generator,
        dtype=dtype,
    )
    attention_mask = torch.zeros(
        batch,
        1,
        1,
        past_length + 1,
        dtype=dtype,
    )
    position_ids = torch.full(
        (batch, 1),
        past_length,
        dtype=torch.int64,
    )
    past_key_values = tuple(
        (
            torch.randn(
                batch,
                config.num_kv_heads,
                past_length,
                config.head_dim,
                generator=generator,
                dtype=dtype,
            ),
            torch.randn(
                batch,
                config.num_kv_heads,
                past_length,
                config.head_dim,
                generator=generator,
                dtype=dtype,
            ),
        )
        for _ in range(config.num_layers)
    )
    return hidden_states, attention_mask, position_ids, past_key_values


def make_qwen2_prefill_inputs(
    config: Qwen2CompatConfig,
    batch: int,
    tokens: int,
    *,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """创建动态Prompt、四维加性Causal Mask和Position ID。"""

    if batch <= 0 or tokens < 2:
        raise ValueError("Qwen2 Prefill要求正Batch且Token数量至少为2")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch, tokens),
        generator=generator,
        dtype=torch.int64,
    )
    causal = torch.triu(
        torch.ones(tokens, tokens, dtype=torch.bool),
        diagonal=1,
    )
    attention_mask = torch.zeros(
        batch,
        1,
        tokens,
        tokens,
        dtype=torch.float32,
    ).masked_fill(causal, float("-inf"))
    position_ids = torch.arange(tokens, dtype=torch.int64).view(1, -1)
    position_ids = position_ids.expand(batch, -1).clone()
    return input_ids, attention_mask, position_ids
