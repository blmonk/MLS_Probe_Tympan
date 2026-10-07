"""Import every concrete algorithm module so its @register_algorithm
decorators actually run -- importing just `algorithms` (this package) is
enough to populate the registry."""

from .base import AFCAlgorithm
from .registry import create_algorithm, register_algorithm, registered_names
from . import pemafc  # noqa: F401 (registers pemafc_f, pemafc_a, nlms)
from . import pemafc_pca  # noqa: F401 (registers pemafc_pca_a, pemafc_pca_f, nlms_pca)
from . import pemafc_soft_pca  # noqa: F401 (registers soft_pca)
from . import blend_pca  # noqa: F401 (registers blend_pca)
from . import null_afc  # noqa: F401 (registers null_afc)

__all__ = ["AFCAlgorithm", "create_algorithm", "register_algorithm", "registered_names"]
