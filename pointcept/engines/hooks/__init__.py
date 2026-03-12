# 使用条件导入，避免依赖不可用的模块
try:
    from .default import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .misc import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .evaluator import *
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import evaluator: {e}", ImportWarning)

try:
    from .ema import *
except (ImportError, ModuleNotFoundError):
    pass

try:
    from .builder import build_hooks
except (ImportError, ModuleNotFoundError) as e:
    import warnings
    warnings.warn(f"Failed to import build_hooks: {e}", ImportWarning)
    build_hooks = None
