"""比较PyTorch Eager、TorchInductor和Triton Qwen2 RoPE。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateful_llm_compiler.backends import triton_rope


@dataclass
class RoPEBenchmarkRow:
    batch: int
    tokens: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    eager_us: float
    inductor_us: float
    triton_us: float
    triton_vs_eager: float
    triton_vs_inductor: float
    max_abs_error: float


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _native_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """与Qwen2前后半维布局一致的PyTorch参考图。"""

    query_half = query.shape[-1] // 2
    key_half = key.shape[-1] // 2
    query_rotated = torch.cat(
        (-query[..., query_half:], query[..., :query_half]),
        dim=-1,
    )
    key_rotated = torch.cat(
        (-key[..., key_half:], key[..., :key_half]),
        dim=-1,
    )
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + query_rotated * sine,
        key * cosine + key_rotated * sine,
    )


def benchmark_one(
    batch: int,
    tokens: int,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    warmup_ms: int,
    rep_ms: int,
) -> RoPEBenchmarkRow:
    dtype = torch.float16
    generator = torch.Generator(device="cuda").manual_seed(
        batch * 100000 + tokens * 100 + head_dim
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
    angles = torch.randn(
        batch,
        tokens,
        head_dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cosine = angles.cos().to(dtype)
    sine = angles.sin().to(dtype)

    eager_call = lambda: _native_rope(query, key, cosine, sine)
    compiled = torch.compile(
        _native_rope,
        fullgraph=True,
        dynamic=False,
        mode="default",
    )
    inductor_call = lambda: compiled(query, key, cosine, sine)
    triton_call = lambda: triton_rope(query, key, cosine, sine)

    expected_query, expected_key = eager_call()
    actual_query, actual_key = triton_call()
    max_abs_error = max(
        float((actual_query.float() - expected_query.float()).abs().max()),
        float((actual_key.float() - expected_key.float()).abs().max()),
    )
    # 首次调用只触发TorchInductor编译，不计入稳态延迟。
    inductor_call()
    eager_ms = triton.testing.do_bench(
        eager_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    inductor_ms = triton.testing.do_bench(
        inductor_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    triton_ms = triton.testing.do_bench(
        triton_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    return RoPEBenchmarkRow(
        batch=batch,
        tokens=tokens,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        eager_us=eager_ms * 1000,
        inductor_us=inductor_ms * 1000,
        triton_us=triton_ms * 1000,
        triton_vs_eager=eager_ms / triton_ms,
        triton_vs_inductor=inductor_ms / triton_ms,
        max_abs_error=max_abs_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--tokens", default="1,128,512")
    parser.add_argument("--query-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/rope_benchmark.json"),
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
                f"eager={result.eager_us:8.2f}us "
                f"inductor={result.inductor_us:8.2f}us "
                f"triton={result.triton_us:8.2f}us "
                f"vs-eager={result.triton_vs_eager:5.2f}x "
                f"vs-inductor={result.triton_vs_inductor:5.2f}x"
            )

    geometric_vs_eager = math.prod(
        result.triton_vs_eager for result in results
    ) ** (1.0 / len(results))
    geometric_vs_inductor = math.prod(
        result.triton_vs_inductor for result in results
    ) ** (1.0 / len(results))
    print(
        "几何平均："
        f"vs-eager={geometric_vs_eager:.3f}x，"
        f"vs-inductor={geometric_vs_inductor:.3f}x"
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
        "summary": {
            "geomean_triton_vs_eager": geometric_vs_eager,
            "geomean_triton_vs_inductor": geometric_vs_inductor,
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
