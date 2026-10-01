"""Portable FP32 refiner with standard ONNX operators and unchanged weights.

The SISO formula follows the upstream Mamba-3 forward reference, including
rotary state, trapezoidal discretization, diagonal correction, D, and Z gating.
It materializes causal attention (quadratic in frames); training keeps Triton.
Frozen DINO features, metric depth and optical-flow tracks are external inputs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn

from .heads import _mlp


class PortableRMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight


def _rotate(x: Tensor, angles: Tensor) -> Tensor:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    # Exactly half of the state dimensions rotate in this checkpoint.
    cosine = F.pad(torch.cos(angles), (0, angles.shape[-1]), value=1.0)
    sine = F.pad(torch.sin(angles), (0, angles.shape[-1]), value=0.0)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).reshape_as(x)


def siso_attention(q: Tensor, k: Tensor, v: Tensor, adt: Tensor, dt: Tensor,
                   trap: Tensor, angles: Tensor, d: Tensor, z: Tensor) -> Tensor:
    """Inputs Q/K/V: (streams, frames, heads, channels), scalars: (..., heads)."""
    angle_state = torch.cumsum(torch.tanh(angles) * math.pi * dt.unsqueeze(-1), dim=1)
    angle_state = angle_state - 2 * math.pi * torch.floor(angle_state / (2 * math.pi))
    qr, kr = _rotate(q, angle_state), _rotate(k, angle_state)
    gate = torch.sigmoid(trap)
    shifted_gamma = F.pad(dt[:, 1:] * (1 - gate[:, 1:]), (0, 0, 0, 1))
    scale = dt * gate + shifted_gamma
    scores = torch.matmul(qr.transpose(1, 2),
                          (kr * scale.unsqueeze(-1)).permute(0, 2, 3, 1))
    length = v.shape[1]
    times = torch.arange(length, device=v.device)
    causal = times[:, None] >= times[None, :]
    strict = times[:, None] > times[None, :]
    # Stable segment sums: avoid subtracting large, nearly equal prefix sums.
    segments = torch.cumsum(torch.where(strict, adt.transpose(1, 2).unsqueeze(-1), 0.0), -2)
    decay = torch.where(causal, torch.exp(segments), 0.0)
    result = torch.matmul(scores * decay, v.transpose(1, 2)).transpose(1, 2)
    diagonal = (q * k).sum(-1) * shifted_gamma
    result = result + d[None, None, :, None] * v - diagonal.unsqueeze(-1) * v
    return result * F.silu(z)


class PortableMamba3SISO(nn.Module):
    """The checkpoint's 768/1536/128/24-head SISO mixer in FP32."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(768, 3432, bias=False)
        self.out_proj = nn.Linear(1536, 768, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(24))
        self.B_bias = nn.Parameter(torch.ones(24, 1, 128))
        self.C_bias = nn.Parameter(torch.ones(24, 1, 128))
        self.B_norm = PortableRMSNorm(128)
        self.C_norm = PortableRMSNorm(128)
        self.D = nn.Parameter(torch.ones(24))

    def forward(self, u: Tensor) -> Tensor:
        n, t, _ = u.shape
        z, x, b, c, raw_dt, raw_a, trap, angles = torch.split(
            self.in_proj(u), [1536, 1536, 128, 128, 24, 24, 24, 32], dim=-1)
        z, x = z.reshape(n, t, 24, 64), x.reshape(n, t, 24, 64)
        k = self.B_norm(b).unsqueeze(2) + self.B_bias.squeeze(1)[None, None]
        q = self.C_norm(c).unsqueeze(2) + self.C_bias.squeeze(1)[None, None]
        dt = F.softplus(raw_dt + self.dt_bias)
        a = -(raw_a.clamp_min(0) + torch.reciprocal(1 - raw_a.clamp_max(0)))
        adt = a.clamp_max(-1e-4) * dt
        angles = angles.unsqueeze(2).expand(-1, -1, 24, -1)
        y = siso_attention(q, k, x, adt, dt, trap, angles, self.D, z)
        return self.out_proj(y.reshape(n, t, 1536))


class PortableMamba3Adapter(nn.Module):
    def __init__(self, dim: int, bidirectional: bool) -> None:
        super().__init__()
        self.up = nn.Linear(dim, 768, bias=False)
        self.norm = PortableRMSNorm(768)
        self.mixer = PortableMamba3SISO()
        self.down = nn.Linear(768, dim, bias=False)
        self.bidirectional = bidirectional

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.norm(self.up(x))
        y = self.mixer(tokens)
        if self.bidirectional:
            y = (y + self.mixer(tokens.flip(1)).flip(1)) * 0.5
        return self.down(y)


class OnnxV35Refiner(nn.Module):
    """Uses every trainable tracker tensor, with frozen DINO features as input."""

    def __init__(self, config: dict[str, Any], dino_dim: int = 384) -> None:
        super().__init__()
        if config.get("temporal_mixer") != "official_mamba3":
            raise ValueError("This exporter requires the official_mamba3 SISO tracker")
        if config.get("feat_encoder", "dinov3") != "dinov3":
            raise ValueError("This exporter expects frozen DINO features")
        if any(config.get(flag, False) for flag in ("two_pool", "per_frame_scale", "within_frame", "pose_head")):
            raise ValueError("This exporter supports the selected v35 best200 configuration")
        if dino_dim != 384 or int(config.get("dino_image_size", 448)) != 448:
            raise ValueError("The deployment contract requires 384-channel, 28x28 DINO features")
        if int(config.get("num_layers", 2)) < 1 or int(config.get("dim", 128)) < 1:
            raise ValueError("num_layers and dim must be positive")
        patch = int(config.get("patch_size", 5))
        if patch < 1 or patch % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if float(config.get("image_size", 896)) != 896:
            raise ValueError("The deployment contract uses 896-pixel coordinates")
        self.dim = int(config.get("dim", 128))
        self.image_size = float(config.get("image_size", 896))
        self.patch_size = int(config.get("patch_size", 5))
        self.max_log_correction = float(config.get("max_log_correction", 2.0))
        self.max_delta_uv = float(config.get("max_delta_uv", 2.0))
        self.gate_by_vis = bool(config.get("gate_by_vis", True))
        d_proj, layers = int(config.get("d_proj", 64)), int(config.get("num_layers", 2))
        self.feat_proj = nn.Linear(dino_dim, d_proj)
        self.embed = _mlp(4 + self.patch_size**2 + d_proj, self.dim, self.dim)
        self.layers = nn.ModuleList([PortableMamba3Adapter(
            self.dim, bool(config.get("official_mamba3_bidirectional", True))) for _ in range(layers)])
        self.pre_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(self.dim)
        self.dz_head, self.duv_head, self.vis_head = (
            _mlp(self.dim, 64, width) for width in (1, 2, 1))

    def forward(self, ray: Tensor, z_raw: Tensor, vis: Tensor, uv: Tensor,
                depth_map: Tensor, dino_features: Tensor, K: Tensor,
                z_ref: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, t, n, _ = ray.shape
        k = self.patch_size
        uv_norm = 2 * uv / self.image_size - 1
        offsets = (torch.arange(k, device=uv.device, dtype=uv.dtype) - k // 2) * (2.0 / 14)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        d_offsets = torch.stack((dx.reshape(-1), dy.reshape(-1)), -1)
        grid = (uv_norm.unsqueeze(-2) + d_offsets).reshape(b * t, 1, n * k * k, 2)
        depth = depth_map.reshape(b * t, 1, depth_map.shape[-2], depth_map.shape[-1])
        patch = F.grid_sample(depth, grid, mode="bilinear", padding_mode="border",
                              align_corners=False).reshape(b, t, n, k * k)
        patch = patch / patch[..., k * k // 2].unsqueeze(-1).clamp_min(1e-6)
        features = F.grid_sample(
            dino_features.reshape(b * t, dino_features.shape[2], dino_features.shape[3], dino_features.shape[4]),
            uv_norm.reshape(b * t, 1, n, 2), mode="bilinear", padding_mode="border", align_corners=False)
        features = features.reshape(b, t, dino_features.shape[2], n).permute(0, 1, 3, 2)
        gate = vis.unsqueeze(-1) if self.gate_by_vis else torch.ones_like(vis).unsqueeze(-1)
        feat = torch.cat((ray, (z_raw / z_ref).unsqueeze(-1), vis.unsqueeze(-1),
                          patch * gate, self.feat_proj(features) * gate), -1)
        x = self.embed(feat).permute(0, 2, 1, 3).reshape(b * n, t, self.dim)
        for pre, layer, post in zip(self.pre_norms, self.layers, self.post_norms):
            x = post(x + layer(pre(x)))
        x = self.out_norm(x).reshape(b, n, t, self.dim).permute(0, 2, 1, 3)
        dlog = self.dz_head(x).squeeze(-1).clamp(-self.max_log_correction, self.max_log_correction)
        delta_uv = self.max_delta_uv * torch.tanh(self.duv_head(x))
        new_uv = uv + delta_uv
        z = F.grid_sample(depth, (2 * new_uv / self.image_size - 1).reshape(b * t, 1, n, 2),
                          mode="bilinear", padding_mode="border", align_corners=False).reshape(b, t, n)
        z = z * torch.exp(dlog)
        fx, fy, cx, cy = (K[:, i, j].reshape(b, 1, 1) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
        xyz = torch.stack(((new_uv[..., 0] - cx) / fx * z, (new_uv[..., 1] - cy) / fy * z, z), -1)
        return xyz, new_uv, self.vis_head(x).squeeze(-1), delta_uv


@jaxtyped(typechecker=beartype)
def reference_depth(z_raw: Float[Tensor, "batch frames tracks"]) -> Float[Tensor, ""]:
    """Host-side normalization; torch.median selects the lower middle value."""
    if z_raw.numel() == 0 or not torch.isfinite(z_raw).all():
        raise ValueError("z_raw must be nonempty and finite")
    return z_raw.detach().flatten().median() + 1e-6
