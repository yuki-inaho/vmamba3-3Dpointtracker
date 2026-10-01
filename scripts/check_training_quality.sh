#!/bin/sh
set -eu
uv run ruff check src/mamba3_tracker/train/profiling.py \
    src/mamba3_tracker/data/frozen_cache.py src/mamba3_tracker/data/synthetic.py \
    src/mamba3_tracker/data/fixed_da.py \
    src/mamba3_tracker/data/dataset.py tests/unit/test_fixed_da.py \
    src/mamba3_tracker/model/dino_encoder.py scripts/smoke_train_synthetic.py \
    scripts/train_v64_staged.py scripts/train_depth_refined_tracker.py \
    tests/unit/test_profile_cache.py
uv run ty check
uv run pytest -q
uv lock --check
