from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "max"
    min_delta: float = 0.0
    best: float | None = None
    num_bad_epochs: int = 0

    def step(self, value: float | None) -> bool:
        if value is None:
            self.num_bad_epochs += 1
            return self.num_bad_epochs > self.patience
        if self.best is None:
            self.best = value
            self.num_bad_epochs = 0
            return False
        improved = value > self.best + self.min_delta if self.mode == "max" else value < self.best - self.min_delta
        if improved:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience
