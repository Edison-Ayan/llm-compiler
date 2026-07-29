"""基于 Profile 为 `serve.rms_norm` 生成动态 Lowering 方案。"""

from __future__ import annotations

from collections import Counter

from ..cost_model import RMSNormCostModel
from ..ir import Module, StaticDim, TensorType
from ..pass_manager import CompilerPass, PassResult


class SelectRMSNormLoweringPass(CompilerPass):
    name = "select-rmsnorm-lowering"

    def __init__(
        self,
        cost_model: RMSNormCostModel,
        *,
        fallback: str = "inductor",
    ) -> None:
        self.cost_model = cost_model
        self.fallback = fallback

    def run(self, module: Module) -> PassResult:
        planned = 0
        missing_profile = 0
        variant_backends: Counter[str] = Counter()
        changed = False

        for function in module.functions:
            for operation in function.block.operations:
                if operation.name != "serve.rms_norm":
                    continue
                input_type = operation.operands[0].type
                if (
                    not isinstance(input_type, TensorType)
                    or not input_type.shape
                    or not isinstance(input_type.shape[-1], StaticDim)
                ):
                    missing_profile += 1
                    continue
                hidden_size = input_type.shape[-1].value
                dtype = operation.attributes.get(
                    "output_dtype", input_type.dtype
                )
                plan = self.cost_model.plan_attribute(
                    hidden_size,
                    dtype,
                    fallback=self.fallback,
                )
                if not plan["variants"]:
                    missing_profile += 1
                else:
                    planned += 1
                    variant_backends.update(
                        variant["backend"]
                        for variant in plan["variants"]
                    )
                if operation.attributes.get("lowering_plan") != plan:
                    operation.attributes["lowering_plan"] = plan
                    changed = True

        return PassResult(
            self.name,
            changed=changed,
            statistics={
                "planned_operations": planned,
                "missing_profile": missing_profile,
                "variant_backends": dict(sorted(variant_backends.items())),
            },
        )

