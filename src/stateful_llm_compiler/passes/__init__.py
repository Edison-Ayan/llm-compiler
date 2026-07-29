"""ServeIR 优化 Pass。"""

from .cleanup import RemoveExportAssertionsPass
from .rmsnorm_fusion import FuseRMSNormPass

__all__ = ["RemoveExportAssertionsPass", "FuseRMSNormPass"]

