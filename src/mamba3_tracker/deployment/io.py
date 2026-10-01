"""Version-independent numeric NPZ writer (never accepts object/pickle arrays)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if not arrays or any(not name.isidentifier() for name in arrays):
        raise ValueError("NPZ members must have nonempty identifier names")
    if any(not isinstance(array, np.ndarray) or array.dtype.hasobject for array in arrays.values()):
        raise TypeError("Only non-object NumPy arrays may be written")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, array in arrays.items():
                buffer = io.BytesIO()
                np.save(buffer, array, allow_pickle=False)
                archive.writestr(f"{name}.npy", buffer.getvalue())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
