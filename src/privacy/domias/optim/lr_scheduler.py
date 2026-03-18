# stdlib
from typing import Any, Callable, Optional

# third party
import torch


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
        self, *args: Any, early_stopping: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.early_stopping = early_stopping
        self.early_stopping_counter = 0
        self.my_best = None  # 🔥 não usar self.best!

    def step(
        self,
        metrics: Any,
        epoch: Optional[int] = None,
        callback_best: Optional[Callable] = None,
        callback_reduce: Optional[Callable] = None,
    ) -> bool:

        super().step(metrics, epoch)

        current = metrics

        if self.my_best is None or current < self.my_best:
            self.my_best = current
            self.early_stopping_counter = 0
            if callback_best is not None:
                callback_best()
        else:
            self.early_stopping_counter += 1

        if self.early_stopping is not None:
            return self.early_stopping_counter >= self.early_stopping

        return False