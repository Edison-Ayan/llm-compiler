"""比较增长式 KV Cat、预分配 Index Copy 和 Triton 位置写入。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateful_llm_compiler.backends import triton_kv_store
from stateful_llm_compiler.execution import native_kv_store


@dataclass
class KVStoreBenchmarkRow:
    batch: int
    cache_length: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    cat_us: float
    index_copy_us: float
    triton_us: float
    triton_vs_cat: float
    triton_vs_index_copy: float
    theoretical_traffic_ratio: float
    max_abs_error: float


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def benchmark_one(
    batch: int,
    cache_length: int,
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> KVStoreBenchmarkRow:
    generator = torch.Generator(device="cuda").manual_seed(
        batch * 10000 + cache_length
    )
    past_key = torch.randn(
        batch,
        num_kv_heads,
        cache_length,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    past_value = torch.randn_like(past_key)
    current_key = torch.randn(
        batch,
        num_kv_heads,
        1,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    current_value = torch.randn_like(current_key)
    capacity = cache_length + 1
    key_buffer = torch.zeros(
        batch,
        capacity,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    value_buffer = torch.zeros_like(key_buffer)
    key_buffer[:, :cache_length].copy_(past_key.transpose(1, 2))
    value_buffer[:, :cache_length].copy_(past_value.transpose(1, 2))
    positions = torch.full(
        (batch,),
        cache_length,
        device="cuda",
        dtype=torch.int64,
    )

    cat_call = lambda: (
        torch.cat((past_key, current_key), dim=2),
        torch.cat((past_value, current_value), dim=2),
    )
    index_call = lambda: native_kv_store(
        key_buffer,
        value_buffer,
        current_key,
        current_value,
        positions,
    )
    triton_call = lambda: triton_kv_store(
        key_buffer,
        value_buffer,
        current_key,
        current_value,
        positions,
    )

    expected_key, expected_value = cat_call()
    triton_call()
    actual_key = key_buffer.transpose(1, 2)
    actual_value = value_buffer.transpose(1, 2)
    max_abs_error = max(
        float((actual_key - expected_key).abs().max()),
        float((actual_value - expected_value).abs().max()),
    )

    cat_ms = triton.testing.do_bench(
        cat_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    index_ms = triton.testing.do_bench(
        index_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    triton_ms = triton.testing.do_bench(
        triton_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    # Cat 需要读历史并写出新 Tensor；位置写入只读写当前 Token 的 K/V。
    traffic_ratio = cache_length + 1
    return KVStoreBenchmarkRow(
        batch=batch,
        cache_length=cache_length,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=str(dtype).removeprefix("torch."),
        cat_us=cat_ms * 1000,
        index_copy_us=index_ms * 1000,
        triton_us=triton_ms * 1000,
        triton_vs_cat=cat_ms / triton_ms,
        triton_vs_index_copy=index_ms / triton_ms,
        theoretical_traffic_ratio=traffic_ratio,
        max_abs_error=max_abs_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--lengths", default="64,256,1024")
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/kv_store_benchmark.json"),
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
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                dtype=torch.float16,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            rows.append(row)
            print(
                f"B={batch:>2} L={length:>4} | "
                f"cat={row.cat_us:8.2f}us "
                f"index={row.index_copy_us:8.2f}us "
                f"triton={row.triton_us:8.2f}us "
                f"speedup(cat)={row.triton_vs_cat:6.2f}x "
                f"speedup(index)={row.triton_vs_index_copy:5.2f}x"
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
