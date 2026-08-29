import numpy as np

from flowmia.attacks.dcr_mia import DcrAttack, dcr_mia


def _random_arrays(seed):
    rng = np.random.default_rng(seed)
    X_member = rng.random((30, 5)).astype(np.float32)
    X_non_member = rng.random((30, 5)).astype(np.float32)
    X_synth = rng.random((30, 5)).astype(np.float32)
    return X_member, X_non_member, X_synth


def test_dcr_attack_wrapper():
    X_member, X_non_member, X_synth = _random_arrays(0)
    attack = DcrAttack()
    attack.fit(X_member, X_non_member, X_synth)
    result = attack.attack(X_member, X_non_member, test_size=10)

    assert result.scores.shape == (20,)
    assert 0.0 <= result.auc <= 1.0


def test_dcr_mia_function_still_usable_standalone():
    X_member, X_non_member, X_synth = _random_arrays(1)
    scores, auc = dcr_mia(train_members=X_member, non_members=X_non_member, synthetic_data=X_synth)

    assert scores.shape == (60,)
    assert 0.0 <= auc <= 1.0
