"""ServeIR 优化 Pass。"""

from .cleanup import RemoveExportAssertionsPass
from .decode_attention import FuseDecodeAttentionPass
from .kv_bufferize import BufferizeKVCachePass
from .kv_state import MaterializeKVStatePass
from .linear import NormalizeLinearPass
from .prefill_attention import FusePrefillAttentionPass
from .prefill_kv import MaterializePrefillKVStatePass
from .rope import FuseRoPEPass
from .lowering_selection import SelectRMSNormLoweringPass
from .rmsnorm_fusion import FuseRMSNormPass
from ..lowering import LowerToKernelIRPass

__all__ = [
    "RemoveExportAssertionsPass",
    "FuseRMSNormPass",
    "MaterializeKVStatePass",
    "BufferizeKVCachePass",
    "FuseDecodeAttentionPass",
    "NormalizeLinearPass",
    "FusePrefillAttentionPass",
    "MaterializePrefillKVStatePass",
    "FuseRoPEPass",
    "SelectRMSNormLoweringPass",
    "LowerToKernelIRPass",
]
