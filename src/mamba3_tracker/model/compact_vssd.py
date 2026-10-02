"""Minimal two-pool projections, with an explicit legacy/normalized distinction.

Legacy mode reproduces the repository's terminal-state two-pool mixer; it is
not the scan-free NC-SSD operator of the VSSD paper. Normalized mode is a new
function whose first pool has token-local weights and whose second pool has
unit total mass. Both implementations store O(F * state_dim) activations and
never form a token-by-token matrix.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from visionmamba3.cross_attention import COLLAPSE, Mamba3CrossAttention
from visionmamba3.mask import build_cross_scale
from visionmamba3.projections import BCNorm


class _QueryProjection(nn.Module):
    def __init__(self, dim: int, heads: int, state_dim: int) -> None:
        super().__init__()
        self.heads, self.state_dim = heads, state_dim
        self.proj = nn.Linear(dim, heads * state_dim, bias=False)
        self.bc_norm_c = BCNorm(heads, state_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, frames, _ = x.shape
        c = self.proj(x).reshape(batch, frames, self.heads, self.state_dim)
        return self.bc_norm_c(c.transpose(1, 2))


class _KeyValueProjection(nn.Module):
    def __init__(
        self, dim: int, heads: int, state_dim: int, *, alignment_rows: int = 0
    ) -> None:
        super().__init__()
        self.heads, self.state_dim, self.head_dim = heads, state_dim, dim // heads
        self.alignment_rows = alignment_rows
        self.proj = nn.Linear(dim, heads * state_dim + dim + 2 * heads, bias=False)
        self.bc_norm_b = BCNorm(heads, state_dim)
        self.delta_bias = nn.Parameter(torch.zeros(heads))
        self.A_bias = nn.Parameter(torch.zeros(heads))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, frames, _ = x.shape
        heads, state_dim, head_dim = self.heads, self.state_dim, self.head_dim
        if self.alignment_rows:
            # Legacy packed Linear ends with delta/A plus unused lambda rows.
            # Fixed zero rows preserve its CPU parallel GEMM remainder rounding,
            # without storing/training lambda parameters. The normalized v2
            # operator has alignment_rows=0 and retains its original arithmetic.
            packed = F.linear(
                x, F.pad(self.proj.weight, (0, 0, 0, self.alignment_rows))
            )[..., : self.proj.out_features]
        else:
            packed = self.proj(x)
        end_b = heads * state_dim
        end_v = end_b + heads * head_dim
        b = packed[..., :end_b].reshape(batch, frames, heads, state_dim)
        v = packed[..., end_b:end_v].reshape(batch, frames, heads, head_dim)
        delta = packed[..., end_v : end_v + heads].transpose(1, 2)
        a = packed[..., end_v + heads : end_v + 2 * heads].transpose(1, 2)
        return (
            self.bc_norm_b(b.transpose(1, 2)),
            F.silu(v.transpose(1, 2)),
            F.softplus(delta + self.delta_bias[None, :, None]),
            -F.softplus(a + self.A_bias[None, :, None]),
        )


class CompactCrossAttention(nn.Module):
    """C-only queries and B/V/delta/A-only keys, with independently read pools."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        state_dim: int = 64,
        *,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        if dim < 1 or num_heads < 1 or state_dim < 1 or dim % num_heads:
            raise ValueError(
                "Positive dimensions and dim divisible by heads are required"
            )
        self.dim, self.num_heads, self.state_dim = dim, num_heads, state_dim
        self.normalized = normalized
        self.q_proj = _QueryProjection(dim, num_heads, state_dim)
        self.q_proj2 = _QueryProjection(dim, num_heads, state_dim)
        self.kv_proj = _KeyValueProjection(
            dim,
            num_heads,
            state_dim,
            alignment_rows=0 if normalized else num_heads,
        )
        self.m2 = nn.Linear(dim, num_heads)
        nn.init.zeros_(self.m2.weight)
        nn.init.zeros_(self.m2.bias)
        self.pool2_gate = nn.Parameter(torch.zeros(1))
        self.out = nn.Linear(dim, dim)

    @classmethod
    def from_legacy(
        cls, source: Mamba3CrossAttention, *, normalized: bool = False
    ) -> CompactCrossAttention:
        """Copy only used tensors; source tensors/grad flags are never modified."""
        if not source.two_pool:
            raise ValueError("Legacy migration requires two_pool=True")
        if source.bidirectional_mask:
            raise ValueError("Legacy migration does not support bidirectional masks")
        if source.variant != COLLAPSE or source.dim_q != source.dim_kv:
            raise ValueError("Legacy migration requires equal-width collapsed mixing")
        if not isinstance(source.out, nn.Linear) or source.out.bias is None:
            raise ValueError("Legacy migration requires a biased output Linear")
        if not all(
            torch.isfinite(value).all() for value in source.state_dict().values()
        ):
            raise ValueError("Legacy migration requires finite parameters")
        layer = cls(
            source.dim_q, source.num_heads, source.state_dim, normalized=normalized
        ).to(device=source.out.weight.device, dtype=source.out.weight.dtype)
        heads, states, dim = source.num_heads, source.state_dim, source.dim_q
        end_b, end_c, end_v = (
            heads * states,
            2 * heads * states,
            2 * heads * states + dim,
        )
        with torch.no_grad():
            for target, original in (
                (layer.q_proj, source.q_proj),
                (layer.q_proj2, source.q_proj2),
            ):
                target.proj.weight.copy_(original.proj.weight[end_b:end_c])
                target.bc_norm_c.load_state_dict(
                    copy.deepcopy(original.bc_norm_c.state_dict())
                )
            used_kv = torch.cat(
                (
                    source.kv_proj.proj.weight[:end_b],
                    source.kv_proj.proj.weight[end_c : end_v + 2 * heads],
                )
            )
            layer.kv_proj.proj.weight.copy_(used_kv)
            layer.kv_proj.bc_norm_b.load_state_dict(
                copy.deepcopy(source.kv_proj.bc_norm_b.state_dict())
            )
            layer.kv_proj.delta_bias.copy_(source.kv_proj.delta_bias)
            layer.kv_proj.A_bias.copy_(source.kv_proj.A_bias)
            layer.m2.load_state_dict(copy.deepcopy(source.m2.state_dict()))
            layer.pool2_gate.copy_(source.pool2_gate)
            layer.out.load_state_dict(copy.deepcopy(source.out.state_dict()))
        layer.train(source.training)
        return layer

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        c, c2 = self.q_proj(q), self.q_proj2(q)
        b, v, delta, a = self.kv_proj(kv)
        if self.normalized:
            weights1 = torch.softmax(torch.log(delta + 1e-6) + delta * a, dim=-1)
        else:
            weights1 = build_cross_scale(delta, a, bidirectional=False)
        pool_logits = self.m2(kv).transpose(1, 2)
        if self.normalized:
            # log(softplus(z)) = z + O(exp(z)); below -20 the correction is
            # smaller than FP32 rounding. Log-space normalization preserves
            # relative token mass even when exp(-200) underflows in FP32.
            # Promote AMP dtypes so the unselected log branch is finite too.
            stable_logits = (
                pool_logits.float()
                if pool_logits.dtype in (torch.float16, torch.bfloat16)
                else pool_logits
            )
            log_weights = torch.where(
                stable_logits < -20,
                stable_logits,
                torch.log(
                    F.softplus(stable_logits).clamp_min(
                        torch.finfo(stable_logits.dtype).tiny
                    )
                ),
            )
            weights2 = torch.softmax(log_weights, dim=-1).to(pool_logits.dtype)
        else:
            weights2 = F.softplus(pool_logits)
        h1 = torch.einsum("bhts,bhtd->bhsd", b * weights1.unsqueeze(-1), v)
        h2 = torch.einsum("bhts,bhtd->bhsd", b * weights2.unsqueeze(-1), v)
        y = torch.einsum("bhqs,bhsd->bhqd", c, h1)
        y = y + self.pool2_gate * torch.einsum("bhqs,bhsd->bhqd", c2, h2)
        batch, _, frames, _ = y.shape
        return self.out(y.transpose(1, 2).contiguous().view(batch, frames, self.dim))
