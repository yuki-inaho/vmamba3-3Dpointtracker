"""Resumable early stopping, separate from raw K-best ranking."""

import math


class EarlyStopping:
    def __init__(self, patience=0, min_delta=0.001, mode="min"):
        if patience < 0 or min_delta < 0 or mode not in ("min", "max"):
            raise ValueError("Invalid early stopping policy")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = None
        self.best_step = -1
        self.since = 0
        self.last_step = -1

    @property
    def stopped(self):
        return self.patience > 0 and self.since >= self.patience

    def observe(self, score, step):
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite early stopping metric")
        if step <= self.last_step:
            raise ValueError("Early stopping validation steps must increase")
        improvement = (self.best - score if self.mode == "min" else score - self.best) if self.best is not None else math.inf
        if improvement > self.min_delta:
            self.best, self.best_step, self.since = score, step, 0
        else:
            self.since += 1
        self.last_step = step
        return self.stopped

    def state_dict(self):
        return {"patience": self.patience, "min_delta": self.min_delta, "mode": self.mode,
                "best": self.best, "best_step": self.best_step, "since": self.since,
                "last_step": self.last_step}

    def restore(self, state, observations=()):
        policy = ("patience", "min_delta", "mode")
        if state and all(state.get(k) == getattr(self, k) for k in policy):
            for name in ("best", "best_step", "since", "last_step"):
                setattr(self, name, state[name])
        else:
            # An older disabled implementation stored best=inf/since=0. A new
            # policy must replay validation history, not trust that sentinel.
            for step, score in sorted(observations):
                self.observe(float(score), int(step))
