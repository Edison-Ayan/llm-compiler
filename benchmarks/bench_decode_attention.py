"""比较物化 GQA、PyTorch SDPA 和直接读取 KV Buffer 的 Triton Attention。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateful_llm_compiler.backends import triton_decode_attention


@dataclass
class DecodeAttentionBenchmarkRow:
    batch: int
    cache_length: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    expanded_eager_us: float
    sdpa_us: float
    triton_us: float
    triton_vs_expanded: float
    triton_vs_sdpa: float
    max_abs_error: float


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def benchmark_one(
    batch: int,
    cache_length: int,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    warmup_ms: int,
    rep_ms: int,
) -> DecodeAttentionBenchmarkRow:
    if num_query_heads % num_kv_heads:
        raise ValueError("Query Head 数量必须能被 KV Head 数量整除")
    dtype = torch.float16
    generator = torch.Generator(device="cuda").manual_seed(
        batch * 10000 + cache_length
    )
    query = torch.randn(
        batch,
        num_query_heads,
        1,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key_buffer = torch.randn(
        batch,
        cache_length,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value_buffer = torch.randn_like(key_buffer)
    lengths = torch.full(
        (batch,),
        cache_length,
        device="cuda",
        dtype=torch.int64,
    )
    attention_mask = torch.zeros(
        batch,
        1,
        1,
        cache_length,
        device="cuda",
        dtype=dtype,
    )
    groups = num_query_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    def materialized_kv() -> tuple[torch.Tensor, torch.Tensor]:
        key = key_buffer.permute(0, 2, 1, 3)
        value = value_buffer.permute(0, 2, 1, 3)
        return (
            key.repeat_interleave(groups, dim=1),
            value.repeat_interleave(groups, dim=1),
        )

    def expanded_call() -> torch.Tensor:
        key, value = materialized_kv()
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        probabilities = torch.softmax(
            scores.float() + attention_mask.float(),
            dim=-1,
        ).to(dtype)
        return torch.matmul(probabilities, value)

    def sdpa_call() -> torch.Tensor:
        key, value = materialized_kv()
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=scale,
        )

    triton_call = lambda: triton_decode_attention(
        query,
        key_buffer,
        value_buffer,
        lengths,
        attention_mask,
        scale=scale,
    )

    expected = expanded_call()
    actual = triton_call()
    max_abs_error = float((actual.float() - expected.float()).abs().max())
    expanded_ms = triton.testing.do_bench(
        expanded_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    sdpa_ms = triton.testing.do_bench(
        sdpa_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    triton_ms = triton.testing.do_bench(
        triton_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    return DecodeAttentionBenchmarkRow(
        batch=batch,
        cache_length=cache_length,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype="fp16",
        expanded_eager_us=expanded_ms * 1000,
        sdpa_us=sdpa_ms * 1000,
        triton_us=triton_ms * 1000,
        triton_vs_expanded=expanded_ms / triton_ms,
        triton_vs_sdpa=sdpa_ms / triton_ms,
        max_abs_error=max_abs_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--lengths", default="64,256,1024")
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/decode_attention_benchmark.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA GPU")
    rows = []
    for batch in parse_ints(args.batches):
        for length in parse_ints(args.lengths):
            row = benchmark_one(
                batch,
                length,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            rows.append(row)
            print(
                f"B={batch:>2} L={length:>4} | "
                f"expanded={row.expanded_eager_us:8.2f}us "
                f"sdpa={row.sdpa_us:8.2f}us "
                f"triton={row.triton_us:8.2f}us "
                f"vs-expanded={row.triton_vs_expanded:6.2f}x "
                f"vs-sdpa={row.triton_vs_sdpa:6.2f}x"
            )

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    payload = {
        "schema_version": 1,
        "target": {
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": f"{major}.{minor}",
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
        },
        "benchmark": {
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "compile_time_included": False,
        },
        "results": [asdict(row) for row in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"结果：{args.out}")


if __name__ == "__main__":
    main()
