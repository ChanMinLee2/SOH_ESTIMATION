"""Compatibility shim: enables numpy < 2.0 to unpickle data saved with numpy >= 2.0.

numpy 2.0 restructured internal modules from `numpy.core` to `numpy._core`.
PKL files created in a numpy 2.x environment cannot be loaded in numpy 1.x
without this shim, which registers all numpy.core.* modules under numpy._core.*.

Import this module before any pickle.load() call that may involve numpy arrays.
"""

from __future__ import annotations
import sys
import types
import numpy as np


def install_numpy2_shim() -> None:
    """Register numpy.core.* aliases under numpy._core.* for pickle compatibility."""
    if hasattr(np, "_core"):
        return  # numpy >= 2.0: no shim needed

    import numpy.core as _nc
    import importlib
    import pkgutil

    # Create a fake numpy._core module mirroring numpy.core
    fake = types.ModuleType("numpy._core")
    fake.__dict__.update(_nc.__dict__)
    sys.modules.setdefault("numpy._core", fake)

    # Register all numpy.core.* submodules as numpy._core.* as well
    for info in pkgutil.iter_modules(_nc.__path__, prefix="numpy.core."):
        new_name = info.name.replace("numpy.core.", "numpy._core.")
        if new_name not in sys.modules:
            try:
                mod = importlib.import_module(info.name)
                sys.modules[new_name] = mod
            except Exception:
                pass


# Apply shim on import
install_numpy2_shim()
