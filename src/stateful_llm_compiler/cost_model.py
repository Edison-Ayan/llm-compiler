"""基于实测 Profile 的 RMSNorm Lowering Cost Model。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class ProfileError(ValueError):
    """Profile 格式或数值不满足成本模型约束。"""

    pass


@dataclass(frozen=True)
class TargetProfile:
    device_name: str = "unknown"
    compute_capability: str = "unknown"
    torch_version: str = "unknown"
    triton_version: str = "unknown"


@dataclass(frozen=True)
class RMSNormProfileEntry:
    rows: int
    hidden_size: int
    dtype: str
    native_us: float
    triton_us: float
    inductor_us: float | None = None

    def backend_costs(self) -> dict[str, float]:
        costs = {
            "native": self.native_us,
            "triton": self.triton_us,
        }
        if self.inductor_us is not None:
            costs["inductor"] = self.inductor_us
        return costs


@dataclass(frozen=True)
class LoweringVariant:
    min_rows: int
    max_rows: int | None
    profile_rows: int
    backend: str
    estimated_us: float
    num_warps: int | None = None

    def contains(self, rows: int) -> bool:
        return rows >= self.min_rows and (
            self.max_rows is None or rows <= self.max_rows
        )


@dataclass(frozen=True)
class LoweringDecision:
    backend: str
    source: str
    profile_rows: int | None = None
    estimated_us: float | None = None
    num_warps: int | None = None


class RMSNormCostModel:
    """从离散测量点构建动态 Rows 的分桶 Lowering 方案。"""

    def __init__(
        self,
        target: TargetProfile,
        entries: Iterable[RMSNormProfileEntry],
        *,
        source: str = "memory",
    ) -> None:
        self.target = target
        self.entries = tuple(entries)
        self.source = source
        if not self.entries:
            raise ProfileError("Profile 至少需要一个测量点")
        _validate_entries(self.entries)

    @classmethod
    def load(cls, path: str | Path) -> "RMSNormCostModel":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload, source=str(path))

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        source: str = "memory",
    ) -> "RMSNormCostModel":
        # 兼容第五阶段产生的旧版纯数组结果。
        if isinstance(payload, list):
            target_data = {}
            rows_data = payload
        elif isinstance(payload, dict):
            if payload.get("schema_version") != 1:
                raise ProfileError("不支持的 Profile schema_version")
            target_data = payload.get("target", {})
            rows_data = payload.get("results", [])
        else:
            raise ProfileError("Profile 顶层必须是对象或数组")

        target = TargetProfile(
            device_name=str(target_data.get("device_name", "unknown")),
            compute_capability=str(
                target_data.get("compute_capability", "unknown")
            ),
            torch_version=str(target_data.get("torch_version", "unknown")),
            triton_version=str(
                target_data.get("triton_version", "unknown")
            ),
        )
        entries = [_parse_entry(row) for row in rows_data]
        return cls(target, entries, source=source)

    def variants(
        self,
        hidden_size: int,
        dtype: str,
        *,
        allowed_backends: Iterable[str] = (
            "triton",
            "inductor",
            "native",
        ),
    ) -> list[LoweringVariant]:
        dtype = normalize_dtype(dtype)
        allowed = set(allowed_backends)
        candidates = sorted(
            (
                entry
                for entry in self.entries
                if entry.hidden_size == hidden_size
                and entry.dtype == dtype
            ),
            key=lambda entry: entry.rows,
        )
        if not candidates:
            return []

        thresholds = [
            math.floor(math.sqrt(left.rows * right.rows))
            for left, right in zip(candidates, candidates[1:])
        ]
        variants = []
        minimum = 1
        for index, entry in enumerate(candidates):
            maximum = thresholds[index] if index < len(thresholds) else None
            costs = {
                backend: cost
                for backend, cost in entry.backend_costs().items()
                if backend in allowed
            }
            if not costs:
                raise ProfileError(
                    f"M={entry.rows},N={hidden_size},{dtype} "
                    "没有允许的 Backend"
                )
            backend = min(costs, key=costs.get)
            variants.append(
                LoweringVariant(
                    min_rows=minimum,
                    max_rows=maximum,
                    profile_rows=entry.rows,
                    backend=backend,
                    estimated_us=costs[backend],
                    num_warps=(
                        select_num_warps(entry.rows, hidden_size)
                        if backend == "triton"
                        else None
                    ),
                )
            )
            minimum = (maximum + 1) if maximum is not None else minimum
        return variants

    def plan_attribute(
        self,
        hidden_size: int,
        dtype: str,
        *,
        fallback: str = "inductor",
    ) -> dict[str, Any]:
        variants = self.variants(hidden_size, dtype)
        return {
            "schema_version": 1,
            "profile_source": self.source,
            "target": asdict(self.target),
            "hidden_size": hidden_size,
            "dtype": normalize_dtype(dtype),
            "fallback": fallback,
            "variants": [asdict(variant) for variant in variants],
        }


def select_variant(
    variants: Iterable[LoweringVariant] | Iterable[dict[str, Any]],
    rows: int,
) -> LoweringVariant | None:
    for item in variants:
        variant = (
            item
            if isinstance(item, LoweringVariant)
            else LoweringVariant(**item)
        )
        if variant.contains(rows):
            return variant
    return None


def resolve_lowering_plan(
    plan: dict[str, Any] | None,
    rows: int,
    *,
    default_backend: str = "triton",
) -> LoweringDecision:
    """按运行时 Rows 解析编译期生成的多版本计划。"""

    if rows <= 0:
        raise ProfileError("运行时 Rows 必须为正数")
    if not plan:
        return LoweringDecision(default_backend, "default")

    variant = select_variant(plan.get("variants", []), rows)
    if variant is None:
        return LoweringDecision(
            str(plan.get("fallback", default_backend)),
            "fallback",
        )
    return LoweringDecision(
        backend=variant.backend,
        source="profile",
        profile_rows=variant.profile_rows,
        estimated_us=variant.estimated_us,
        num_warps=variant.num_warps,
    )


def select_num_warps(rows: int, hidden_size: int) -> int:
    block_size = 1 << max(0, hidden_size - 1).bit_length()
    if block_size < 2048:
        return 4
    if 4 <= rows <= 16:
        return 4
    return 8


def normalize_dtype(dtype: str) -> str:
    aliases = {
        "f16": "fp16",
        "float16": "fp16",
        "fp16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f32": "fp32",
        "float32": "fp32",
        "fp32": "fp32",
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as error:
        raise ProfileError(f"不支持的 DType：{dtype}") from error


def _parse_entry(row: dict[str, Any]) -> RMSNormProfileEntry:
    try:
        return RMSNormProfileEntry(
            rows=int(row["rows"]),
            hidden_size=int(row["hidden_size"]),
            dtype=normalize_dtype(str(row["dtype"])),
            native_us=float(row["native_eager_us"]),
            triton_us=float(row["triton_us"]),
            inductor_us=(
                float(row["inductor_us"])
                if row.get("inductor_us") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProfileError(f"非法 Profile Row：{row}") from error


def _validate_entries(entries: tuple[RMSNormProfileEntry, ...]) -> None:
    keys = set()
    for entry in entries:
        if entry.rows <= 0 or entry.hidden_size <= 0:
            raise ProfileError("Rows 和 Hidden Size 必须为正数")
        if any(
            not math.isfinite(cost) or cost <= 0
            for cost in entry.backend_costs().values()
        ):
            raise ProfileError("Backend 延迟必须是有限正数")
        key = (entry.rows, entry.hidden_size, entry.dtype)
        if key in keys:
            raise ProfileError(f"Profile 测量点重复：{key}")
        keys.add(key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="查看 RMSNorm Lowering 方案")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = RMSNormCostModel.load(args.profile)
    plan = model.plan_attribute(args.hidden_size, args.dtype)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
