"""比较 PyTorch/cuBLAS 与当前 Triton Linear Lowering。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateful_llm_compiler.backends import triton_linear


@dataclass
class LinearBenchmarkRow:
    rows: int
    input_features: int
    output_features: int
    dtype: str
    bias: bool
    eager_us: float
    triton_us: float
    triton_speedup: float
    max_abs_error: float


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        input_features, output_features = item.lower().split("x")
        shapes.append((int(input_features), int(output_features)))
    return shapes


def benchmark_one(
    rows: int,
    input_features: int,
    output_features: int,
    *,
    bias: bool,
    warmup_ms: int,
    rep_ms: int,
) -> LinearBenchmarkRow:
    dtype = torch.float16
    generator = torch.Generator(device="cuda").manual_seed(
        rows * 100000 + input_features * 100 + output_features
    )
    tensor = torch.randn(
        rows,
        input_features,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        output_features,
        input_features,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    bias_tensor = (
        torch.randn(
            output_features,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        if bias
        else None
    )
    eager_call = lambda: F.linear(tensor, weight, bias_tensor)
    triton_call = lambda: triton_linear(tensor, weight, bias_tensor)

    expected = eager_call()
    actual = triton_call()
    max_abs_error = float((actual.float() - expected.float()).abs().max())
    eager_ms = triton.testing.do_bench(
        eager_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    triton_ms = triton.testing.do_bench(
        triton_call,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    return LinearBenchmarkRow(
        rows=rows,
        input_features=input_features,
        output_features=output_features,
        dtype="fp16",
        bias=bias,
        eager_us=eager_ms * 1000,
        triton_us=triton_ms * 1000,
        triton_speedup=eager_ms / triton_ms,
        max_abs_error=max_abs_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,32")
    parser.add_argument("--shapes", default="512x512,1536x1536")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/linear_benchmark.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA GPU")

    results = []
    for input_features, output_features in _parse_shapes(args.shapes):
        for rows in _parse_ints(args.rows):
            result = benchmark_one(
                rows,
                input_features,
                output_features,
                bias=args.bias,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            results.append(result)
            print(
                f"M={rows:>4} K={input_features:>5} N={output_features:>5} | "
                f"eager={result.eager_us:8.2f}us "
                f"triton={result.triton_us:8.2f}us "
                f"speedup={result.triton_speedup:5.2f}x "
                f"max_abs={result.max_abs_error:.2e}"
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
