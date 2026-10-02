"""Geometry reuse, independent tracks, and train-to-deploy fusion gates."""

import pytest
import torch

from mamba3_tracker.model.student_refiner import StudentRefiner
from mamba3_tracker.model.local_global_refiner import LocalGlobalMixer


def _inputs(batch=1, frames=8, tracks=3):
    uv = torch.rand(batch, frames, tracks, 2) * 800 + 48
    z = torch.full((batch, frames, tracks), 2.5)
    depth = torch.rand(batch, frames, 17, 21) + 2
    features = torch.randn(batch, frames, 384, 7, 9)
    k = torch.tensor([[500.0, 0, 448], [0, 500, 448], [0, 0, 1]]).repeat(batch, 1, 1)
    return (
        (uv - 448) / 500,
        z,
        torch.ones_like(z),
        uv,
        depth,
        features,
        k,
        z.median() + 1e-6,
    )


def _track_subset(args, indices):
    return tuple(
        value[:, :, indices] if i < 4 else value for i, value in enumerate(args)
    )


def _nontrivial_readout(model):
    with torch.no_grad():
        model.dz_head[-1].weight.normal_(std=0.03)
        model.duv_head[-1].weight.normal_(std=0.03)
        for layer in model.layers:
            layer.global_mixer.pool2_gate.fill_(0.2)
            layer.global_mixer.m2.weight.normal_(std=0.03)
            for scale in (layer.local_scale, layer.global_scale, layer.ffn_scale):
                scale.fill_(0.05)


def test_architecture_and_parameter_budget():
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    train = LocalGlobalStudentRefiner(deploy=False)
    deploy = LocalGlobalStudentRefiner(deploy=True)
    assert isinstance(train, StudentRefiner)
    assert train.architecture == "vssd_local_global_128x2_v2_train"
    assert deploy.architecture == "vssd_local_global_128x2_v2"
    assert sum(p.numel() for p in train.parameters()) == 617374
    assert sum(p.numel() for p in deploy.parameters()) == 615838
    assert not any("aux" in key or "teacher" in key for key in deploy.state_dict())
    for layer in train.layers:
        assert isinstance(layer, LocalGlobalMixer)
        assert layer.global_mixer.normalized
        assert layer.global_mixer.dim == 128
        assert layer.global_mixer.state_dim == 64
        assert layer.global_mixer.num_heads == 4
        expand, project = layer.ffn[0], layer.ffn[2]
        assert isinstance(expand, torch.nn.Linear)
        assert isinstance(project, torch.nn.Linear)
        assert expand.out_features == 512
        assert project.in_features == 512
        assert torch.equal(layer.local_scale, torch.full((128,), 0.01))


@pytest.mark.parametrize(
    "batch,frames,tracks", [(1, 1, 1), (2, 8, 7), (1, 31, 3), (1, 600, 1)]
)
def test_all_four_outputs_train_to_deploy_parity(batch, frames, tracks):
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    torch.manual_seed(56)
    train = LocalGlobalStudentRefiner().eval()
    _nontrivial_readout(train)
    before = {name: value.clone() for name, value in train.state_dict().items()}
    deployed = train.to_deploy()
    args = _inputs(batch, frames, tracks)
    with torch.no_grad():
        expected, actual = train(*args), deployed(*args)
    for got, want in zip(actual, expected):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert all(
        torch.equal(value, train.state_dict()[name]) for name, value in before.items()
    )
    assert not train.deploy and deployed.deploy
    assert deployed.architecture == "vssd_local_global_128x2_v2"
    assert not any("branch" in name for name in deployed.state_dict())
    for layer in train.layers:
        assert isinstance(layer, LocalGlobalMixer)
        assert not layer.local.deploy
    for layer in deployed.layers:
        assert isinstance(layer, LocalGlobalMixer)
        assert layer.local.deploy
    assert not deployed.training
    assert train.feat_proj.weight.data_ptr() != deployed.feat_proj.weight.data_ptr()


def test_shared_geometry_initialization_and_finite_backward():
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    torch.manual_seed(59)
    source = StudentRefiner()
    target = LocalGlobalStudentRefiner()
    copied = target.initialize_common(source.state_dict())
    assert copied
    assert not any(key.startswith("layers.") for key in copied)
    for key in copied:
        torch.testing.assert_close(
            target.state_dict()[key], source.state_dict()[key], rtol=0, atol=0
        )
    _nontrivial_readout(target)
    result = target.forward_train(*_inputs(frames=31))
    assert len(result["hidden"]) == 2
    assert result["final_hidden"].shape == (1, 31, 3, 128)
    assert all(value.shape == (1, 31, 3, 128) for value in result["hidden"])
    loss = result["xyz"].square().mean() + result["vis_logits"].square().mean()
    loss.backward()
    for parameter in target.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_track_permutation_and_chunking_do_not_mix_tracks():
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    torch.manual_seed(61)
    model = LocalGlobalStudentRefiner().eval()
    _nontrivial_readout(model)
    args = _inputs(batch=2, frames=31, tracks=7)
    permutation = torch.tensor([5, 1, 6, 0, 3, 2, 4])
    with torch.no_grad():
        baseline = model(*args)
        permuted = model(*_track_subset(args, permutation))
        chunked = model(*_track_subset(args, torch.tensor([1, 5])))
    for base, perm, chunk in zip(baseline, permuted, chunked):
        torch.testing.assert_close(perm, base[:, :, permutation], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunk, base[:, :, [1, 5]], rtol=1e-5, atol=1e-6)


def test_deployment_is_an_independent_copy_and_preserves_nonlinear_ffn():
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    model = LocalGlobalStudentRefiner()
    deployed = model.to_deploy()
    copied = deployed.to_deploy()
    assert deployed.training == model.training
    assert copied is not deployed
    for layer in deployed.layers:
        assert isinstance(layer, LocalGlobalMixer)
        assert isinstance(layer.ffn[1], torch.nn.GELU)
        assert isinstance(layer.local_norm, torch.nn.LayerNorm)
    with torch.no_grad():
        copied.feat_proj.weight.zero_()
    assert not torch.equal(copied.feat_proj.weight, deployed.feat_proj.weight)


def test_nonfinite_geometry_is_rejected_before_deployment():
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    model = LocalGlobalStudentRefiner()
    with torch.no_grad():
        model.feat_proj.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.to_deploy()
