"""Exact projection pruning and separately specified normalized global mixing."""

import pytest
import torch
import copy

from visionmamba3.cross_attention import Mamba3CrossAttention


@pytest.fixture(params=[32, 48])
def parallel_cpu_arithmetic(request):
    """Exercise the packed GEMM remainder path without leaking global settings."""
    original_threads = torch.get_num_threads()
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_num_threads(request.param)
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_num_threads(original_threads)
        torch.set_float32_matmul_precision(original_precision)


@pytest.mark.parametrize("frames,batch", [(31, 2), (128, 1), (600, 1)])
@pytest.mark.parametrize("gate", [0.0, 0.37])
def test_parallel_legacy_fp32_forward_parity(
    parallel_cpu_arithmetic, frames, batch, gate
):
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    torch.manual_seed(17)
    old = Mamba3CrossAttention(128, 128, num_heads=4, state_dim=64, two_pool=True)
    with torch.no_grad():
        old.pool2_gate.fill_(gate)
        old.m2.weight.normal_(std=0.02)
    compact = CompactCrossAttention.from_legacy(old)
    q = torch.randn(batch, frames, 128, requires_grad=True)
    kv = torch.randn(batch, frames, 128, requires_grad=True)
    expected, actual = old(q, kv), compact(q, kv)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_alignment_rows_are_fixed_legacy_only_and_do_not_add_parameters(monkeypatch):
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    legacy = CompactCrossAttention(normalized=False)
    normalized = CompactCrossAttention(normalized=True)
    assert legacy.kv_proj.alignment_rows == 4
    assert normalized.kv_proj.alignment_rows == 0
    assert legacy.kv_proj.proj.weight.shape == (392, 128)
    assert normalized.kv_proj.proj.weight.shape == (392, 128)
    assert sum(p.numel() for p in legacy.parameters()) == 134285
    assert sum(p.numel() for p in normalized.parameters()) == 134285
    before = {key: value.clone() for key, value in normalized.state_dict().items()}

    def reject_padding(*args, **kwargs):
        raise AssertionError("The normalized v2 arithmetic must not be padded")

    monkeypatch.setattr(torch.nn.functional, "pad", reject_padding)
    assert torch.isfinite(
        normalized(torch.randn(1, 31, 128), torch.randn(1, 31, 128))
    ).all()
    assert all(
        torch.equal(value, normalized.state_dict()[key])
        for key, value in before.items()
    )


def test_registered_factory_guards_alignment_rows():
    from mamba3_tracker.deployment.student_checkpoint import (
        create_student,
        validate_student,
    )
    from mamba3_tracker.model.local_global_refiner import LocalGlobalMixer

    student = create_student("vssd_local_global_128x2_v2_train")
    for layer in student.layers:
        assert isinstance(layer, LocalGlobalMixer)
        assert layer.global_mixer.kv_proj.alignment_rows == 0
    validate_student(student)
    first_layer = student.layers[0]
    assert isinstance(first_layer, LocalGlobalMixer)
    first_layer.global_mixer.kv_proj.alignment_rows = 4
    with pytest.raises(ValueError, match="semantic contract"):
        validate_student(student)


@pytest.mark.parametrize("frames,batch", [(1, 1), (8, 6), (31, 2), (128, 1), (600, 1)])
@pytest.mark.parametrize("gate", [0.0, 0.37])
def test_legacy_function_and_input_gradient_parity(frames, batch, gate):
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    torch.manual_seed(17)
    old = Mamba3CrossAttention(128, 128, num_heads=4, state_dim=64, two_pool=True)
    with torch.no_grad():
        old.pool2_gate.fill_(gate)
        old.m2.weight.normal_(std=0.02)
    before = {name: value.clone() for name, value in old.state_dict().items()}
    compact = CompactCrossAttention.from_legacy(old)
    q = torch.randn(batch, frames, 128, requires_grad=True)
    kv = torch.randn(batch, frames, 128, requires_grad=True)
    expected = old(q, kv)
    actual = compact(q, kv)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Pruning packed GEMM output rows changes FP32 backward summation order.
    # Verify the derivative of the same function in double precision, while
    # retaining the strict FP32 forward gate above and finite FP32 gradients below.
    old.double()
    compact.double()
    q64, kv64 = (
        q.detach().double().requires_grad_(),
        kv.detach().double().requires_grad_(),
    )
    expected64, actual64 = old(q64, kv64), compact(q64, kv64)
    weight = torch.randn_like(expected64)
    old_grad = torch.autograd.grad((expected64 * weight).sum(), (q64, kv64))
    new_grad = torch.autograd.grad((actual64 * weight).sum(), (q64, kv64))
    for got, want in zip(new_grad, old_grad):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert all(
        torch.equal(value.double(), old.state_dict()[name])
        for name, value in before.items()
    )


def test_projection_budget_and_used_rows():
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    old = Mamba3CrossAttention(128, 128, num_heads=4, state_dim=64, two_pool=True)
    compact = CompactCrossAttention.from_legacy(old)
    assert sum(p.numel() for p in old.parameters()) == 270505
    assert sum(p.numel() for p in compact.parameters()) == 134285
    assert 2 * (270505 - 134285) == 272440
    assert compact.q_proj.proj.weight.shape == (256, 128)
    assert compact.kv_proj.proj.weight.shape == (392, 128)
    assert not any("lam" in name for name, _ in compact.named_parameters())
    with torch.no_grad():
        compact.pool2_gate.fill_(0.2)
    compact(torch.randn(2, 11, 128), torch.randn(2, 11, 128)).square().mean().backward()
    for parameter in compact.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_normalized_formula_and_duplicate_length_invariance():
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    torch.manual_seed(12)
    layer = CompactCrossAttention(normalized=True)
    with torch.no_grad():
        layer.pool2_gate.fill_(0.3)
        layer.m2.weight.normal_(std=0.02)
    q, kv = torch.randn(2, 7, 128), torch.randn(2, 11, 128)
    b, v, delta, a = layer.kv_proj(kv)
    c, c2 = layer.q_proj(q), layer.q_proj2(q)
    weights1 = torch.softmax(torch.log(delta + 1e-6) + delta * a, dim=-1)
    weights2 = torch.nn.functional.softplus(layer.m2(kv)).transpose(1, 2)
    weights2 = weights2 / weights2.sum(dim=-1, keepdim=True)
    h1 = torch.einsum("bhts,bhtd->bhsd", b * weights1.unsqueeze(-1), v)
    h2 = torch.einsum("bhts,bhtd->bhsd", b * weights2.unsqueeze(-1), v)
    y = torch.einsum("bhqs,bhsd->bhqd", c, h1)
    y = y + layer.pool2_gate * torch.einsum("bhqs,bhsd->bhqd", c2, h2)
    expected = layer.out(y.transpose(1, 2).reshape(2, 7, 128))
    torch.testing.assert_close(layer(q, kv), expected, rtol=1e-5, atol=1e-6)
    # Repeating identical observations must not multiply a normalized pooled state.
    torch.testing.assert_close(
        layer(q, kv.repeat(1, 3, 1)), expected, rtol=1e-5, atol=1e-6
    )
    assert torch.isfinite(layer(q[:, :1], kv[:, :1])).all()


@pytest.mark.parametrize("bias,variable", [(-50.0, False), (-200.0, True)])
def test_normalized_pool2_underflow_matches_unclamped_double_reference(bias, variable):
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    torch.manual_seed(71)
    layer = CompactCrossAttention(normalized=True)
    with torch.no_grad():
        layer.pool2_gate.fill_(0.3)
        layer.m2.bias.fill_(bias)
        if variable:
            layer.m2.weight[:, 0] = torch.arange(1.0, 5.0)
    q, kv = torch.randn(2, 7, 128), torch.randn(2, 11, 128)
    kv[..., 0] = torch.arange(11).remainder(5).float() - 2
    reference = copy.deepcopy(layer).double()
    with torch.no_grad():
        c, c2 = reference.q_proj(q.double()), reference.q_proj2(q.double())
        b, v, delta, a = reference.kv_proj(kv.double())
        weights1 = torch.softmax(torch.log(delta + 1e-6) + delta * a, dim=-1)
        # In FP64 even -200 remains representable: no clamp or fallback.
        weights2 = torch.nn.functional.softplus(reference.m2(kv.double())).transpose(
            1, 2
        )
        weights2 = weights2 / weights2.sum(dim=-1, keepdim=True)
        h1 = torch.einsum("bhts,bhtd->bhsd", b * weights1.unsqueeze(-1), v)
        h2 = torch.einsum("bhts,bhtd->bhsd", b * weights2.unsqueeze(-1), v)
        y = torch.einsum("bhqs,bhsd->bhqd", c, h1)
        y = y + reference.pool2_gate * torch.einsum("bhqs,bhsd->bhqd", c2, h2)
        expected = reference.out(y.transpose(1, 2).reshape(2, 7, 128)).float()
    torch.testing.assert_close(layer(q, kv), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        layer(q, kv.repeat(1, 3, 1)), expected, rtol=1e-5, atol=1e-6
    )
    single = layer(q[:, :1], kv[:, :1])
    assert torch.isfinite(single).all()
    layer(q.requires_grad_(), kv.requires_grad_()).square().mean().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert q.grad is not None and kv.grad is not None
    assert torch.isfinite(q.grad).all() and torch.isfinite(kv.grad).all()


def test_unsupported_legacy_variants_are_rejected():
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention

    with pytest.raises(ValueError, match="two_pool"):
        CompactCrossAttention.from_legacy(Mamba3CrossAttention(128, 128))
    with pytest.raises(ValueError, match="bidirectional"):
        CompactCrossAttention.from_legacy(
            Mamba3CrossAttention(
                128,
                128,
                num_heads=4,
                state_dim=64,
                two_pool=True,
                bidirectional_mask=True,
            )
        )


@pytest.mark.parametrize(
    "batch,frames,tracks", [(1, 1, 1), (2, 8, 7), (1, 31, 3), (1, 600, 1)]
)
def test_all_four_refiner_outputs_after_projection_pruning(batch, frames, tracks):
    from mamba3_tracker.model.compact_vssd import CompactCrossAttention
    from mamba3_tracker.model.student_refiner import StudentRefiner

    torch.manual_seed(48)
    source = StudentRefiner().eval()
    with torch.no_grad():
        for head in (source.dz_head, source.duv_head):
            linear = head[-1]
            assert isinstance(linear, torch.nn.Linear)
            linear.weight.normal_(std=0.01)
        for layer in source.layers:
            assert isinstance(layer, Mamba3CrossAttention)
            layer.pool2_gate.fill_(0.025)
    before = {key: value.clone() for key, value in source.state_dict().items()}
    compact = copy.deepcopy(source)
    layers = []
    for layer in source.layers:
        assert isinstance(layer, Mamba3CrossAttention)
        layers.append(CompactCrossAttention.from_legacy(layer))
    compact.layers = torch.nn.ModuleList(layers)
    assert sum(p.numel() for p in compact.parameters()) == 348062
    uv = torch.rand(batch, frames, tracks, 2) * 800 + 48
    z = torch.full((batch, frames, tracks), 2.5)
    depth = torch.rand(batch, frames, 17, 21) + 2
    features = torch.randn(batch, frames, 384, 7, 9)
    k = torch.tensor([[500.0, 0, 448], [0, 500, 448], [0, 0, 1]]).repeat(batch, 1, 1)
    args = (
        (uv - 448) / 500,
        z,
        torch.ones_like(z),
        uv,
        depth,
        features,
        k,
        z.median() + 1e-6,
    )
    with torch.no_grad():
        expected, actual = source(*args), compact(*args)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    assert all(
        torch.equal(value, source.state_dict()[key]) for key, value in before.items()
    )
