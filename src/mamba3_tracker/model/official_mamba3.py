"""Use the official pretrained Mamba-3 SISO mixer on tracker feature streams."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

# Register the pinned upstream namespace without importing optional LM backends.
import visionmamba3  # noqa: F401
from mamba_ssm.modules.mamba3 import Mamba3


MAMBA3_REPO = "state-spaces/mamba3-siso-187m"
MAMBA3_REVISION = "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"


class OfficialMamba3Adapter(nn.Module):
    """128-D tracker features -> intact 768-D pretrained mixer -> 128-D.

    Forward and reverse scans share the same weights. Their average supplies
    both temporal directions, while retaining the official causal operator in
    each direction. This is an experimental temporal mixer, not VSSD-2pool.
    Tracks are independent, so splitting their batch is mathematically exact.
    """

    def __init__(self, dim=128, bidirectional=True, track_chunk_size=128):
        super().__init__()
        if track_chunk_size < 1:
            raise ValueError("official_mamba3_track_chunk must be positive")
        self.bidirectional = bool(bidirectional)
        self.track_chunk_size = int(track_chunk_size)
        self.up = nn.Linear(dim, 768, bias=False)
        self.norm = nn.RMSNorm(768, eps=1e-5)
        self.mixer = Mamba3(d_model=768, d_state=128, expand=2, headdim=64,
                            ngroups=1, rope_fraction=0.5, chunk_size=64,
                            is_mimo=False, is_outproj_norm=False)
        self.down = nn.Linear(768, dim, bias=False)

    def forward(self, q_tokens, kv_tokens):
        if q_tokens is not kv_tokens:
            raise ValueError("official Mamba-3 adapter requires the same temporal stream")
        if not q_tokens.is_cuda:
            raise ValueError("official Mamba-3 Triton execution requires CUDA")
        outputs = []
        for part in q_tokens.split(self.track_chunk_size, dim=0):
            tokens = self.norm(self.up(part))
            # The official Triton kernels require FP16/BF16 inputs. Training
            # uses autocast; validation/evaluation also call without autocast.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward = self.mixer(tokens.to(torch.bfloat16))
                if self.bidirectional:
                    reverse = self.mixer(tokens.flip(1).to(torch.bfloat16)).flip(1)
                    forward = (forward + reverse) * 0.5
                outputs.append(self.down(forward))
        return torch.cat(outputs, dim=0).to(q_tokens.dtype)


def load_pretrained_mixers(layers, checkpoint, source_layers=(0, 1), expected_sha256=None):
    """Load exact mixer tensors with strict=False; require 100% mixer coverage.

    The LM embedding, MLP and LM head are deliberately excluded. No tensor is
    cropped, padded or reshaped to force compatibility. Validate all layers
    before mutating any target so a bad checkpoint cannot load half a model.
    """
    path = Path(checkpoint).expanduser()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if len(layers) != len(source_layers):
        raise ValueError("pretrained source_layers must match tracker num_layers")
    prepared = []
    mappings = []
    for target_index, (layer, source_index) in enumerate(zip(layers, source_layers)):
        if not isinstance(layer, OfficialMamba3Adapter):
            raise ValueError("pretrained Mamba-3 needs temporal_mixer: official_mamba3")
        prefix = f"backbone.layers.{int(source_index)}."
        own = {**{f"mixer.{k}": v for k, v in layer.mixer.state_dict().items()},
               "norm.weight": layer.norm.weight}
        selected = {}
        for name, value in own.items():
            source_key = prefix + name
            if source_key not in state or state[source_key].shape != value.shape:
                raise ValueError(f"pretrained tensor missing or incompatible: {source_key}")
            selected[name] = state[source_key]
            mappings.append({"source": source_key, "target": f"layers.{target_index}.{name}",
                             "shape": list(value.shape), "numel": value.numel()})
        prepared.append((layer, selected))
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError(f"pretrained checkpoint sha256 differs from pinned source: {path}")
    for layer, selected in prepared:
        layer.load_state_dict(selected, strict=False)  # Adapter up/down are new.
    return {"repo": MAMBA3_REPO, "revision": MAMBA3_REVISION, "path": str(path),
            "sha256": digest.hexdigest(), "checkpoint_bytes": path.stat().st_size,
            "source_layers": list(source_layers), "loaded_tensors": len(mappings),
            "loaded_parameters": sum(x["numel"] for x in mappings),
            "mixer_coverage": 1.0, "mappings": mappings}
