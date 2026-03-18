"""
DOMIAS (Density-based Overfitting Membership Inference Attack on Synthetic data).

Uses a Block Neural Autoregressive Flow (BNAF) to estimate the log-density of
the synthetic distribution. Points with higher log-density are predicted to be
training members. An optional reference model (fitted on non-member data) can
be used to compute a density ratio, which corrects for distributional shift.

Reference:
    van Breugel et al., "Membership Inference Attacks Against Synthetic Data",
    NeurIPS 2023.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.privacy.domias.domias import density_estimator_trainer, compute_log_p_x


def domias_bnaf(
    train_members: np.ndarray,
    non_members: np.ndarray,
    synthetic_data: np.ndarray,
    device: torch.device,
    reference_data: np.ndarray = None,
) -> tuple[np.ndarray, float]:
    """
    Compute DOMIAS membership inference scores using a normalizing flow density model.

    A BNAF model is trained on ``synthetic_data`` to approximate its density
    p_S. The log-density log p_S(x) is then evaluated at every query point.
    If ``reference_data`` is provided, a second model is trained on it to
    estimate p_R, and the final score becomes log p_S(x) − log p_R(x) (the
    DOMIAS formulation). Without reference data, log p_S alone is used.

    Args:
        train_members: Pre-transformed member samples, shape (N, D).
        non_members: Pre-transformed non-member samples, shape (M, D).
        synthetic_data: Pre-transformed synthetic samples used to fit the
            density model, shape (S, D).
        device: Torch device for model inference.
        reference_data: Optional pre-transformed reference (non-member)
            samples used to fit a calibration density model p_R. When
            provided, the score is the log density ratio log p_S / p_R.

    Returns:
        Tuple of:
            - scores (np.ndarray): Attack scores for members followed by
              non-members, shape (N + M,).
            - auc (float): ROC-AUC of the attack.
    """
    X_test = np.vstack([train_members, non_members])
    y_test = np.concatenate([np.ones(len(train_members)), np.zeros(len(non_members))])
    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Fit density model on synthetic data (train/val split at midpoint)
    mid = len(synthetic_data) // 2
    _, p_S_model = density_estimator_trainer(
        synthetic_data,
        synthetic_data[:mid],
        synthetic_data[mid:],
        epochs=5,
        load=True,
    )
    log_p_S = compute_log_p_x(p_S_model, X_test_torch).detach().cpu().numpy()

    if reference_data is not None:
        # Density ratio variant: corrects for distributional shift
        mid_r = len(reference_data) // 2
        _, p_R_model = density_estimator_trainer(
            reference_data,
            reference_data[:mid_r],
            reference_data[mid_r:],
        )
        log_p_R = compute_log_p_x(p_R_model, X_test_torch).detach().cpu().numpy()
        scores = log_p_S - log_p_R
    else:
        scores = log_p_S

    auc = roc_auc_score(y_test, scores)
    return scores, auc