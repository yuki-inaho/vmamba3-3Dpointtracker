import torch

from mamba3_tracker.train.early_stop import EarlyStopping
from mamba3_tracker.train.runtime import CheckpointManager


def test_small_gains_do_not_reset_patience_but_cumulative_gain_does():
    stop = EarlyStopping(patience=3, min_delta=.001)
    assert not stop.observe(.1, 0)
    assert not stop.observe(.0995, 500)
    assert not stop.observe(.0994, 1000)
    assert stop.since == 2
    assert not stop.observe(.098, 1500)
    assert stop.since == 0 and stop.best_step == 1500
    stop.observe(.098, 2000)
    stop.observe(.0981, 2500)
    assert stop.observe(.0979, 3000)


def test_resume_new_policy_replays_disabled_legacy_history():
    stop = EarlyStopping(5, .001)
    stop.restore({"best": float("inf"), "since": 0},
                 [(0, .342), (500, .182), (1000, .131), (1500, .1305), (2000, .146),
                  (2500, .123), (3000, .114), (3500, .111)])
    assert stop.best == .111 and stop.best_step == 3500 and stop.since == 0
    stop.observe(.1111, 4000)
    resumed = EarlyStopping(5, .001)
    resumed.restore(stop.state_dict())
    assert resumed.since == 1 and resumed.last_step == 4000
    resumed.observe(.111, 4500)
    assert resumed.since == 2


def test_ranked_checkpoint_keeps_raw_best_and_updated_stop_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path)
    stop = EarlyStopping(1, .001)
    stop.observe(.1, 0)
    assert stop.observe(.0995, 500)
    manager.save(500, model, optimizer, score=.0995, extra={"early_stop": stop.state_dict()})
    state = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert state["extra"]["early_stop"]["since"] == 1
    assert manager.best[0]["score"] == .0995


def test_accuracy_policy_max_and_resume_after_stop():
    stopper = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    assert not stopper.observe(0.8, 250)
    assert not stopper.observe(0.805, 500)
    assert stopper.observe(0.79, 750)
    restored = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    restored.restore(stopper.state_dict())
    assert restored.stopped
    assert restored.best == 0.8
    assert restored.best_step == 250
