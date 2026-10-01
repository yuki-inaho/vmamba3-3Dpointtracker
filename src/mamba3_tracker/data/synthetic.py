"""Analytic tracks and images for smoke training, unrelated to TAPVid scores."""

import torch


def synthetic_batch(seed=0, frames=4, points=12, image_size=64, device="cpu", batch_size=1):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.rand(1, 1, points, 2, generator=generator) * (image_size * 0.4)
    starts += image_size * 0.3
    times = torch.arange(frames).view(1, frames, 1, 1)
    true_uv = starts + times * torch.tensor([0.3, -0.1]).view(1, 1, 1, 2)
    uv = true_uv + torch.tensor([0.8, 0.2])
    ray = (uv - image_size / 2) / image_size
    true_z = torch.full((1, frames, points), 2.0)
    target = torch.cat(
        [
            (true_uv - image_size / 2) / image_size * true_z.unsqueeze(-1),
            true_z.unsqueeze(-1),
        ],
        dim=-1,
    )
    z_raw = true_z * 1.2
    depth = torch.full((1, frames, image_size, image_size), 2.4)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, image_size),
        torch.linspace(0, 1, image_size),
        indexing="ij",
    )
    images = torch.stack([xs, ys, (xs * 12).sin() * 0.25 + 0.5])
    images = images.view(1, 1, 3, image_size, image_size).repeat(1, frames, 1, 1, 1)
    K = torch.tensor(
        [[[image_size, 0, image_size / 2], [0, image_size, image_size / 2], [0, 0, 1]]],
        dtype=torch.float32,
    )
    vis = torch.ones(1, frames, points)
    fwd = (
        torch.tensor([0.3, -0.1]).view(1, 1, 1, 2).expand(1, frames, points, 2).clone()
    )
    bwd = -fwd.clone()
    fwd[:, -1] = 0
    bwd[:, 0] = 0
    vis[:, frames // 2, ::3] = 0
    fwd[:, frames // 2, ::3] += 5.0
    result = {
        name: value.repeat(batch_size, *([1] * (value.ndim - 1))).to(device)
        for name, value in {
            "ray": ray,
            "z_raw": z_raw,
            "vis": vis,
            "uv": uv,
            "depth_map": depth,
            "images": images,
            "K": K,
            "target": target,
            "flow_fwd": fwd,
            "flow_bwd": bwd,
        }.items()
    }
    if batch_size > 1:
        # Distinct frames prevent unrealistically perfect cache deduplication in profiles.
        offset = torch.arange(batch_size * frames, device=device).view(batch_size, frames, 1, 1, 1)
        result["images"] = (result["images"] + offset * 0.0001).clamp(0, 1)
    return result
