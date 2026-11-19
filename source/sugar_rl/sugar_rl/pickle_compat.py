"""Cross-version NumPy pickle compatibility helpers (NumPy 1.x <-> NumPy 2.x)."""

from __future__ import annotations

import io
import pickle
from typing import Any, BinaryIO


class NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that maps `numpy._core` <-> `numpy.core` when loading across NumPy 1.x and 2.x."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == "numpy._core" or module.startswith("numpy._core."):
                alt = "numpy.core" + module[len("numpy._core") :]
                return super().find_class(alt, name)
            if module == "numpy.core" or module.startswith("numpy.core."):
                alt = "numpy._core" + module[len("numpy.core") :]
                return super().find_class(alt, name)
            raise


def load_pickle_compat(file_obj: BinaryIO) -> Any:
    """Load an open binary pickle file saved under either NumPy 1.x or NumPy 2.x."""
    return NumpyCompatUnpickler(file_obj).load()


def loads_pickle_compat(data: bytes) -> Any:
    """Load raw pickle bytes saved under either NumPy 1.x or NumPy 2.x."""
    return NumpyCompatUnpickler(io.BytesIO(data)).load()
