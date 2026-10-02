"""Pinned minival download using the existing CRC-checked streaming downloader."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "subset_download", ROOT / "scripts/download_tapvid3d_subset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HF = (
    "https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D/resolve/"
    "1575ec135a22e924d1702cd15e76527bcb89cc6e"
)
if __name__ == "__main__":
    module.main()
