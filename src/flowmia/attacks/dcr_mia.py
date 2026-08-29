"""
DCR (Distance to Closest Record) membership inference attack.

The intuition is that training members tend to be closer to the synthetic
dataset than held-out non-members, because the generative model was exposed
to member samples during training.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def dcr_mia(
    train_members: np.ndarray,
    non_members: np.ndarray,
    synthetic_data: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute a DCR-based membership inference attack score.

    For each query point, the attack score is the *negative* Euclidean distance
    to the closest record in the synthetic dataset. A higher score (smaller
    distance) indicates a higher likelihood of membership.

    Args:
        train_members: Pre-transformed member (training) samples, shape (N, D).
        non_members: Pre-transformed non-member samples, shape (M, D).
        synthetic_data: Pre-transformed synthetic samples used as the reference
            set for nearest-neighbor search, shape (S, D).

    Returns:
        Tuple of:
            - scores (np.ndarray): Concatenated negative-distance scores for
              members followed by non-members, shape (N + M,).
            - auc (float): ROC-AUC of the attack.
    """
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(synthetic_data)

    dist_members, _ = nn.kneighbors(train_members)
    dist_non_members, _ = nn.kneighbors(non_members)

    # Negate distances: smaller distance → higher score → predicted as member
    scores = np.concatenate([-dist_members.flatten(), -dist_non_members.flatten()])
    y_true = np.concatenate([np.ones(len(train_members)), np.zeros(len(non_members))])

    auc = roc_auc_score(y_true, scores)
    return scores, auc


# ------------------------------------------------------------------
# BaseAttack adapter
# ------------------------------------------------------------------

from flowmia.attacks.base import AttackResult, BaseAttack
from flowmia.attacks.registry import register_attack


@register_attack("dcr")
class DcrAttack(BaseAttack):
    """BaseAttack adapter around the dcr_mia() function."""

    def __init__(self, device=None, **hyperparams):
        super().__init__(device=device, **hyperparams)
        self._X_synth: np.ndarray = None

    def fit(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        X_synth: np.ndarray,
        **_,
    ) -> "DcrAttack":
        self._X_synth = X_synth
        return self

    def attack(
        self,
        X_member: np.ndarray,
        X_non_member: np.ndarray,
        test_size: int = 1000,
        seed: int = 42,
        **kwargs,
    ) -> AttackResult:
        rng = np.random.default_rng(seed)
        idx_m = rng.choice(len(X_member), size=test_size, replace=False)
        idx_nm = rng.choice(len(X_non_member), size=test_size, replace=False)
        scores, auc = dcr_mia(
            train_members=X_member[idx_m],
            non_members=X_non_member[idx_nm],
            synthetic_data=self._X_synth,
        )
        return AttackResult(scores=scores, auc=auc)