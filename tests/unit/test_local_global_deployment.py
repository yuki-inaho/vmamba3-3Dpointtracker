"""Train/fused public-format separation and explicit deployment export."""

import pytest
import torch


@pytest.mark.parametrize("deploy", [False, True])
def test_local_global_strict_roundtrip(tmp_path, deploy):
    from mamba3_tracker.deployment.student_checkpoint import save_student, load_student
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    model = LocalGlobalStudentRefiner(deploy=deploy)
    path = tmp_path / "student.pt"
    save_student(model, path)
    loaded = load_student(path)
    assert isinstance(loaded, LocalGlobalStudentRefiner)
    assert loaded.architecture == model.architecture
    assert loaded.deploy == deploy
    assert all(
        torch.equal(value, loaded.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    payload = torch.load(path, weights_only=True)
    payload["architecture"] = (
        "vssd_local_global_128x2_v2_train" if deploy else "vssd_local_global_128x2_v2"
    )
    torch.save(payload, path)
    with pytest.raises(ValueError, match="tensors"):
        load_student(path)


def test_wrapper_public_save_is_rejected(tmp_path):
    from mamba3_tracker.deployment.student_checkpoint import save_student
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner
    from mamba3_tracker.train.distillation import DistillationWrapper

    wrapper = DistillationWrapper(
        LocalGlobalStudentRefiner(), LocalGlobalStudentRefiner()
    )
    with pytest.raises(ValueError, match="wrapper"):
        save_student(wrapper, tmp_path / "bad.pt")


@pytest.mark.parametrize(
    "mutation", ["tag", "aux", "normalized", "padding", "activation", "geometry"]
)
@pytest.mark.parametrize("operation", ["save", "export"])
def test_modified_public_semantics_are_rejected(tmp_path, mutation, operation):
    from mamba3_tracker.deployment.student_checkpoint import save_student
    from mamba3_tracker.model.local_global_refiner import (
        LocalGlobalMixer,
        LocalGlobalStudentRefiner,
    )
    from scripts.export_student_refiner_onnx import export_graph

    model = LocalGlobalStudentRefiner(deploy=mutation != "tag")
    layer = model.layers[0]
    assert isinstance(layer, LocalGlobalMixer)
    if mutation == "tag":
        model.architecture = "vssd_local_global_128x2_v2"
    elif mutation == "aux":
        model.aux = torch.nn.Linear(128, 2)
    elif mutation == "normalized":
        layer.global_mixer.normalized = False
    elif mutation == "padding":
        layer.local.reparam.padding = (0,)
    elif mutation == "activation":
        layer.ffn[1] = torch.nn.ReLU()
    else:
        model.max_delta_uv = 17.0
    path = tmp_path / ("bad.pt" if operation == "save" else "bad.onnx")
    with pytest.raises(ValueError, match="contract|structure|semantic|tensors"):
        (save_student if operation == "save" else export_graph)(model, path)
    assert not path.exists()


def test_public_save_preserves_cpu_rng(tmp_path):
    from mamba3_tracker.deployment.student_checkpoint import save_student
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner

    model = LocalGlobalStudentRefiner()
    before = torch.get_rng_state().clone()
    save_student(model, tmp_path / "student.pt")
    assert torch.equal(before, torch.get_rng_state())


def test_unfused_export_requires_explicit_conversion(tmp_path):
    from mamba3_tracker.model.local_global_refiner import LocalGlobalStudentRefiner
    from scripts.export_student_refiner_onnx import export_graph

    with pytest.raises(ValueError, match="explicitly fused"):
        export_graph(LocalGlobalStudentRefiner(), tmp_path / "bad.onnx")


def test_fused_dynamic_onnx_preserves_chunk_and_permutation(tmp_path):
    from mamba3_tracker.model.local_global_refiner import (
        LocalGlobalMixer,
        LocalGlobalStudentRefiner,
    )
    from mamba3_tracker.deployment.runtime import session_for
    from mamba3_tracker.deployment.student_runtime import run_student_chunked
    from scripts.export_refiner_onnx import synthetic_inputs, as_feed, compare
    from scripts.export_student_refiner_onnx import export_graph

    torch.manual_seed(19)
    model = LocalGlobalStudentRefiner().eval()
    with torch.no_grad():
        for head in (model.dz_head[-1], model.duv_head[-1]):
            assert isinstance(head, torch.nn.Linear)
            head.weight.normal_(std=0.01)
        for layer in model.layers:
            assert isinstance(layer, LocalGlobalMixer)
            layer.global_mixer.pool2_gate.fill_(0.2)
    deployed = model.to_deploy().eval()
    path = tmp_path / "fused.onnx"
    export_graph(deployed, path)
    session = session_for(path)
    for b, f, n in ((1, 1, 1), (2, 31, 7), (1, 128, 3)):
        values = synthetic_inputs(b, f, n, 384, 896, depth_shape=(17, 21))
        with torch.no_grad():
            expected = model(*values)
        feed = as_feed(values)
        compare(expected, run_student_chunked(session, feed, 2))
        order = torch.arange(n - 1, -1, -1).numpy()
        permuted = dict(feed)
        for name in tuple(feed)[:4]:
            permuted[name] = feed[name][:, :, order].copy()
        compare(
            [value[:, :, order] for value in expected],
            run_student_chunked(session, permuted, 3),
        )
