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