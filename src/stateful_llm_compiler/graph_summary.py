"""Stable, machine-readable summaries of exported ATen graphs."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch


def _target_name(target: Any) -> str:
    if hasattr(target, "name") and callable(target.name):
        return str(target.name())
    return str(target)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def summarize_exported_program(
    program: torch.export.ExportedProgram,
) -> dict[str, Any]:
    """Return an intentionally version-tolerant graph summary.

    The future ServeIR importer consumes operator names and tensor metadata from
    this representation instead of depending on pretty-printed FX source.
    """

    nodes = []
    op_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for node in program.graph_module.graph.nodes:
        target = _target_name(node.target)
        op_counts[node.op] += 1
        target_counts[target] += 1
        tensor_meta = node.meta.get("tensor_meta")
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": target,
                "args": _json_value(node.args),
                "kwargs": _json_value(node.kwargs),
                "tensor_meta": _json_value(tensor_meta),
            }
        )

    return {
        "schema_version": 1,
        "dialect": "functional_aten",
        "op_counts": dict(sorted(op_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "range_constraints": {
            str(symbol): str(constraint)
            for symbol, constraint in program.range_constraints.items()
        },
        "graph_signature": str(program.graph_signature),
        "nodes": nodes,
    }

