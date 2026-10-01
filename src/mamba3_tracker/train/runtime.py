"""Optimizer construction, averaged evaluation and bounded resumable checkpoints."""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .vendor.amuse import AMUSE


def manifest_train_paths(manifest_path, data_root, subsets):
    """Require every selected official training clip; never silently shrink a run."""
    from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES

    manifest = json.loads(Path(manifest_path).expanduser().read_text())
    root = Path(data_root).expanduser() / "tapvid3d"
    paths = []
    for subset in subsets:
        names = manifest["files_by_subset"][subset]
        if len(names) != len(set(names)) or not names:
            raise ValueError(f"empty/duplicate training selection: {subset}")
        if not set(names) <= set(FULL_EVAL_FILES[subset]):
            raise ValueError(f"non-official training clips in manifest: {subset}")
        if set(names) & set(MINIVAL_FILES[subset]):
            raise ValueError("minival leakage in training manifest")
        paths.extend(root / subset / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected training clips missing; first: {missing[0]}"
        )
    return sorted(paths)


def build_optimizer(model, train_cfg):
    name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name != "amuse":
        raise ValueError(f"Unknown optimizer: {name}")
    ac = train_cfg.get("amuse", {})
    muon, aux = [], []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Follow upstream: embeddings, output heads, scalars and vectors use aux updates.
        is_aux = param.ndim < 2 or any(
            "head" in part or "embed" in part for part in pname.split(".")
        )
        (aux if is_aux else muon).append(param)
    aux_type = ac.get("aux_update_type", "adamw")
    opt = AMUSE(
        [
            {
                "params": muon,
                "use_muon": True,
                "lr": float(ac.get("muon_lr", 0.02)),
                "weight_decay": wd,
                "momentum": float(ac.get("momentum", 0.95)),
                "aux_update_type": aux_type,
            },
            {
                "params": aux,
                "use_muon": False,
                "lr": float(ac.get("aux_lr", lr)),
                "weight_decay": wd,
                "update_type": aux_type,
            },
        ],
        beta1=float(ac.get("beta1", 0.8)),
        rho=float(ac.get("rho", 0.3)),
        warmup_steps=int(train_cfg["warmup"]),
        weight_decay_at_y=float(ac.get("weight_decay_at_y", 0.0)),
    )
    opt.train()
    return opt


@contextmanager
def evaluation_weights(optimizer):
    """Temporarily expose schedule-free X weights, restoring Y even after errors."""
    was_train = isinstance(optimizer, AMUSE) and optimizer.train_mode
    if isinstance(optimizer, AMUSE):
        optimizer.eval()
    try:
        yield
    finally:
        if was_train:
            optimizer.train()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class ClipBudget:
    """Stop a run by examples consumed rather than by optimizer-update count.

    A batch-size change must not silently multiply the training budget.  ``count``
    is persisted in the checkpoint so a resumed run retains the same limit.
    A non-positive ``limit`` disables the budget for legacy configurations.
    """

    def __init__(self, limit=0, count=0):
        self.limit = int(limit)
        self.count = int(count)
        if self.limit < 0 or self.count < 0:
            raise ValueError("clip budget and count must be nonnegative")

    @property
    def enabled(self):
        return self.limit > 0

    @property
    def exhausted(self):
        return self.enabled and self.count >= self.limit

    @property
    def remaining(self):
        return max(0, self.limit - self.count) if self.enabled else None

    def consume(self, clips):
        clips = int(clips)
        if clips < 1:
            raise ValueError("an optimizer update must consume at least one clip")
        self.count += clips

    def state_dict(self):
        return {"limit": self.limit, "count": self.count}


def _atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


class CheckpointManager:
    """Keep latest plus K ranked validation checkpoints. All model weights are X."""

    def __init__(self, out_dir, k=3, mode="min"):
        if k < 1 or mode not in ("min", "max"):
            raise ValueError("checkpoint_k must be positive; mode must be min/max")
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.k, self.mode = k, mode
        self.manifest = self.root / "checkpoints.json"
        self.best = []
        if self.manifest.exists():
            stored = json.loads(self.manifest.read_text())
            if stored["mode"] != mode or stored["k"] != k:
                raise ValueError(
                    "checkpoint ranking configuration differs from existing run"
                )
            self.best = stored["best"]
            for row in self.best:
                if not (self.root / row["path"]).is_file():
                    raise FileNotFoundError(self.root / row["path"])

    def save(
        self,
        step,
        model,
        optimizer,
        scheduler=None,
        history=None,
        cfg=None,
        score=None,
        extra=None,
    ):
        old_best = list(self.best)
        ranked = [row for row in self.best if row["step"] != step]
        candidate = {"path": f"best_{step}.pt", "step": step, "score": score}
        if score is not None and math.isfinite(score):
            ranked.append(candidate)
        else:
            ranked = list(self.best)
        ranked.sort(
            key=lambda row: (
                row["score"] if self.mode == "min" else -row["score"],
                row["step"],
            )
        )
        ranked = ranked[: self.k]
        with evaluation_weights(optimizer):
            payload = {
                "step": step,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "optimizer_name": type(optimizer).__name__,
                "weights_mode": "eval",
                "sched": scheduler.state_dict() if scheduler else None,
                "history": history or [],
                "cfg": cfg,
                "rng": rng_state(),
                "extra": extra or {},
            }
            latest = self.root / "latest.pt"
            tmp = self.root / "latest.pt.tmp"
            torch.save(payload, tmp)
            tmp.replace(latest)
            if candidate in ranked and score is not None:
                tmp = self.root / (candidate["path"] + ".tmp")
                torch.save(payload, tmp)
                tmp.replace(self.root / candidate["path"])
        _atomic_json(
            self.manifest,
            {"k": self.k, "mode": self.mode, "latest_step": step, "best": ranked},
        )
        self.best = ranked
        retained = {row["path"] for row in ranked}
        for row in old_best:
            if row["path"] not in retained:
                (self.root / row["path"]).unlink(missing_ok=True)
        return latest


def restore_checkpoint(path, model, optimizer, scheduler=None, device="cpu"):
    st = torch.load(path, map_location=device, weights_only=False)
    expected = st.get("optimizer_name", "AdamW")
    if expected != type(optimizer).__name__:
        raise ValueError(
            f"checkpoint optimizer {expected} != {type(optimizer).__name__}"
        )
    if isinstance(optimizer, AMUSE) and st.get("weights_mode") != "eval":
        raise ValueError("AMUSE checkpoint must contain averaged evaluation weights")
    model.load_state_dict(st["model"], strict=True)
    optimizer.load_state_dict(st["optim"])
    if isinstance(optimizer, AMUSE):
        # train_mode is an upstream Python attribute, absent from state_dict().
        optimizer.train_mode = False
        optimizer.train()
    if scheduler is not None and st.get("sched") is not None:
        scheduler.load_state_dict(st["sched"])
    restore_rng(st.get("rng"))
    return st
