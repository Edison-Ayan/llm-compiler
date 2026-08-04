"""比较展开GQA、PyTorch SDPA和Triton Prefill Attention。"""

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

from stateful_llm_compiler.backends import triton_prefill_attention


@dataclass
class PrefillAttentionBenchmarkRow:
    batch: int
    tokens: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    expanded_us: float
    sdpa_us: float
    triton_us: float
    triton_vs_expanded: float
    triton_vs_sdpa: float
    max_abs_error: float


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def benchmark_one(
    batch: int,
    tokens: int,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    warmup_ms: int,
    rep_ms: int,
) -> PrefillAttentionBenchmarkRow:
    if query_heads % kv_heads:
        raise ValueError("Query Head数量必须能被KV Head数量整除")
    dtype = torch.float16
    generator = torch.Generator(device="cuda").manual_seed(
        batch * 10000 + tokens
    )
    query = torch.randn(
        batch,
        query_heads,
        tokens,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        batch,
        kv_heads,
        tokens,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn_like(key)
    future = torch.triu(
        torch.ones(tokens, tokens, device="cuda", dtype=torch.bool),
        diagonal=1,
    )
    mask = torch.zeros(
        batch,
        1,
        tokens,
        tokens,
        device="cuda",
    ).masked_fill(future, float("-inf"))
    groups = query_heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    def expanded_call() -> torch.Tensor:
        expanded_key = key.repeat_interleave(groups, dim=1)
        expanded_value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query, expanded_key.transpose(-2, -1)) * scale
        probabilities = torch.softmax(scores.float() + mask, dim=-1)
        return torch.matmul(probabilities.to(dtype), expanded_value)

    def sdpa_call() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
            enable_gqa=True,
        )

    triton_call = lambda: triton_prefill_attention(
        query,
        key,
        value,
        mask,
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
    return PrefillAttentionBenchmarkRow(
        batch=batch,
        tokens=tokens,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        expanded_us=expanded_ms * 1000,
        sdpa_us=sdpa_ms * 1000,
        triton_us=triton_ms * 1000,
        triton_vs_expanded=expanded_ms / triton_ms,
        triton_vs_sdpa=sdpa_ms / triton_ms,
        max_abs_error=max_abs_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2")
    parser.add_argument("--tokens", default="16,64,128")
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/prefill_attention_benchmark.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的CUDA GPU")

    results = []
    for batch in _parse_ints(args.batches):
        for tokens in _parse_ints(args.tokens):
            result = benchmark_one(
                batch,
                tokens,
                query_heads=args.query_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            results.append(result)
            print(
                f"B={batch:>2} T={tokens:>4} | "
                f"expanded={result.expanded_us:8.2f}us "
                f"sdpa={result.sdpa_us:8.2f}us "
                f"triton={result.triton_us:8.2f}us "
                f"vs-expanded={result.triton_vs_expanded:5.2f}x "
                f"vs-sdpa={result.triton_vs_sdpa:5.2f}x"
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
        "results": [asdict(result) for result in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"结果：{args.out}")


if __name__ == "__main__":
    main()
