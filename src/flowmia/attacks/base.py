"""Common interface every membership inference attack implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttackResult:
    """Standard output shape for every membership inference attack.

    Args:
        scores: 1-D array of attack scores, members followed by non-members.
        auc: ROC-AUC of the attack.
        extra: Attack-specific extra data (e.g. per-class metrics, thresholds,
            score distributions used for plotting). Always a dict, may be empty.
    """

    scores: np.ndarray
    auc: float
    extra: dict = field(default_factory=dict)


class BaseAttack(ABC):
    """Common interface every membership inference attack must implement.

    Instances are constructed with attack hyperparameters only (no data).
    ``fit()`` trains/prepares the attack against the member/non-member/synthetic
    data; ``attack()`` scores a member/non-member sample and returns a standard
    :class:`AttackResult`.
    """

    name: str = "base"

    def __init__(self, device=None, **hyperparams):
        self.device = device
        self.hyperparams = hyperparams

    @abstractmethod
    def fit(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        X_synth: np.ndarray,
        **fit_kwargs,
    ) -> "BaseAttack":
        """Train/prepare the attack. Must return self for chaining."""
        raise NotImplementedError

    @abstractmethod
    def attack(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        test_size: int = 1000,
        **attack_kwargs,
    ) -> AttackResult:
        """Score a member/non-member sample and return an AttackResult."""
        raise NotImplementedError

    def run(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        X_synth: np.ndarray,
        fit_kwargs: dict | None = None,
        attack_kwargs: dict | None = None,
    ) -> AttackResult:
        """Convenience: fit() then attack() in one call."""
        self.fit(X_member, X_non_member, X_synth, **(fit_kwargs or {}))
        return self.attack(X_member, X_non_member, **(attack_kwargs or {}))
