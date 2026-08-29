import numpy as np
import torch

from flowmia.attacks.flowmia_gan import FlowMIAGANAttack


def _random_arrays(seed):
    rng = np.random.default_rng(seed)
    X_member = rng.random((30, 5)).astype(np.float32)
    X_non_member = rng.random((30, 5)).astype(np.float32)
    X_synth = rng.random((30, 5)).astype(np.float32)
    return X_member, X_non_member, X_synth


def test_flowmia_gan_attack_wrapper(tmp_path):
    X_member, X_non_member, X_synth = _random_arrays(0)
    attack = FlowMIAGANAttack(device=torch.device("cpu"))
    attack.fit(
        X_member, X_non_member, X_synth,
        epochs=2, batch_size=8, fcheckpoint=2, save_path=str(tmp_path),
    )
    result = attack.attack(X_member, X_non_member, test_size=10)

    assert result.scores.shape == (20,)
    assert 0.0 <= result.auc <= 1.0
    assert "threshold" in result.extra
