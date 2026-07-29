"""在 GPU 上比较 RMSNorm 的不同执行后端。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateful_llm_compiler.backends import triton_rms_norm


@dataclass
class BenchmarkRow:
    rows: int
    hidden_size: int
    dtype: str
    expanded_eager_us: float
    native_eager_us: float
    inductor_us: float | None
    triton_us: float
    triton_vs_expanded: float
    triton_vs_native: float
    triton_vs_inductor: float | None
    max_abs_error: float
    relative_l2_error: float


def expanded_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """与前端模型一致的展开形式，包含多个独立 PyTorch Operation。"""

    input_dtype = tensor.dtype
    variance = tensor.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = tensor.float() * torch.rsqrt(variance + epsilon)
    return (normalized * weight.float()).to(input_dtype)


def native_rms_norm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return F.rms_norm(
        tensor,
        (tensor.shape[-1],),
        weight,
        epsilon,
    )


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_dtypes(value: str) -> list[torch.dtype]:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    result = []
    for item in value.split(","):
        name = item.strip().lower()
        if name not in mapping:
            raise ValueError(f"不支持的 DType：{name}")
        result.append(mapping[name])
    return result


def dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }[dtype]


def benchmark_one(
    rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    *,
    epsilon: float,
    use_inductor: bool,
    warmup_ms: int,
    rep_ms: int,
) -> BenchmarkRow:
    generator = torch.Generator(device="cuda").manual_seed(
        rows * 10000 + hidden_size
    )
    tensor = torch.randn(
        rows,
        hidden_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        hidden_size,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )

    expanded = lambda: expanded_rms_norm(tensor, weight, epsilon)
    native = lambda: native_rms_norm(tensor, weight, epsilon)
    triton_call = lambda: triton_rms_norm(
        tensor,
        weight,
        epsilon=epsilon,
        output_dtype=dtype,
    )

    # 在计时前完成 JIT 编译和正确性检查，避免把首次编译时间混入 Kernel 延迟。
    expected = expanded()
    triton_output = triton_call()
    difference = triton_output.float() - expected.float()
    max_abs_error = float(difference.abs().max())
    relative_l2_error = float(
        difference.norm() / expected.float().norm().clamp(min=1e-12)
    )

    compiled_call = None
    if use_inductor:
        # 每个 Shape 都是独立实验，清空 Dynamo 缓存以免不同 Shape/DType 的
        # 特化版本共同触发全局 recompile_limit。
        torch._dynamo.reset()
        compiled = torch.compile(
            expanded_rms_norm,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
        compiled(tensor, weight, epsilon)
        compiled_call = lambda: compiled(tensor, weight, epsilon)

    torch.cuda.synchronize()
    expanded_ms = triton.testing.do_bench(
        expanded, warmup=warmup_ms, rep=rep_ms
    )
    native_ms = triton.testing.do_bench(
        native, warmup=warmup_ms, rep=rep_ms
    )
    triton_ms = triton.testing.do_bench(
        triton_call, warmup=warmup_ms, rep=rep_ms
    )
    inductor_ms = None
    if compiled_call is not None:
        inductor_ms = triton.testing.do_bench(
            compiled_call, warmup=warmup_ms, rep=rep_ms
        )

    return BenchmarkRow(
        rows=rows,
        hidden_size=hidden_size,
        dtype=dtype_name(dtype),
        expanded_eager_us=expanded_ms * 1000,
        native_eager_us=native_ms * 1000,
        inductor_us=inductor_ms * 1000 if inductor_ms is not None else None,
        triton_us=triton_ms * 1000,
        triton_vs_expanded=expanded_ms / triton_ms,
        triton_vs_native=native_ms / triton_ms,
        triton_vs_inductor=(
            inductor_ms / triton_ms if inductor_ms is not None else None
        ),
        max_abs_error=max_abs_error,
        relative_l2_error=relative_l2_error,
    )


def build_profile(
    rows: list[BenchmarkRow],
    *,
    epsilon: float,
    warmup_ms: int,
    rep_ms: int,
) -> dict:
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return {
        "schema_version": 1,
        "target": {
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": f"{major}.{minor}",
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
        },
        "benchmark": {
            "epsilon": epsilon,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            # 所有候选后端均已在计时前完成首次编译和预热。
            "compile_time_included": False,
        },
        "results": [asdict(row) for row in rows],
    }


def write_results(
    rows: list[BenchmarkRow],
    output: Path,
    *,
    epsilon: float,
    warmup_ms: int,
    rep_ms: int,
) -> None:
    if not rows:
        raise ValueError("至少需要一条 Benchmark 结果")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_profile(
                rows,
                epsilon=epsilon,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1,8,32,128")
    parser.add_argument("--hidden-sizes", default="64,1536")
    parser.add_argument("--dtypes", default="fp16")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--inductor", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/rmsnorm_benchmark.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要可用的 CUDA GPU")

    results = []
    for dtype in parse_dtypes(args.dtypes):
        for hidden_size in parse_ints(args.hidden_sizes):
            for rows in parse_ints(args.rows):
                row = benchmark_one(
                    rows,
                    hidden_size,
                    dtype,
                    epsilon=args.epsilon,
                    use_inductor=args.inductor,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                results.append(row)
                inductor = (
                    f"{row.inductor_us:8.2f}"
                    if row.inductor_us is not None
                    else "       -"
                )
                print(
                    f"M={row.rows:>4} N={row.hidden_size:>5} "
                    f"{row.dtype} | expanded={row.expanded_eager_us:8.2f}us "
                    f"native={row.native_eager_us:8.2f}us "
                    f"inductor={inductor}us "
                    f"triton={row.triton_us:8.2f}us "
                    f"speedup(exp)={row.triton_vs_expanded:5.2f}x "
                    f"max_abs={row.max_abs_error:.2e}"
                )
    write_results(
        results,
        args.out,
        epsilon=args.epsilon,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
    )
    print(f"结果：{args.out} 和 {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
