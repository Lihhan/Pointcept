from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier
from .modules import PointModule, PointModel

try:
    from .sparse_unet import *
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import sparse_unet: {e}", ImportWarning)

try:
    from .point_transformer import *
except ImportError:
    pass

try:
    from .point_transformer_v2 import *
except ImportError:
    pass

try:
    from .point_transformer_v3 import *
except ImportError:
    pass

try:
    from .stratified_transformer import *
except ImportError:
    pass

try:
    from .spvcnn import *
except ImportError:
    pass

try:
    from .octformer import *
except ImportError:
    pass

try:
    from .oacnns import *
except ImportError:
    pass

try:
    from .context_aware_classifier import *
except ImportError:
    pass

try:
    from .point_group import *
except ImportError:
    pass

try:
    from .masked_scene_contrast import *
except ImportError:
    pass

try:
    from .point_prompt_training import *
except ImportError:
    pass

try:
    from .sonata import *
except ImportError:
    pass

try:
    from .MSC_pointcnnpp import *
except ImportError:
    pass