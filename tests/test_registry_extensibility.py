import numpy as np

from flowmia import FlowMIA
from flowmia.attacks import AttackResult, BaseAttack, get_attack, list_attacks, register_attack


@register_attack("dummy")
class DummyAttack(BaseAttack):
    """A minimal 3rd-party-style attack used only to prove extensibility."""

    def fit(self, X_member, X_non_member, X_synth, **_):
        return self

    def attack(self, X_member, X_non_member, test_size=1000, **_):
        return AttackResult(scores=np.zeros(2 * test_size), auc=0.5)


def test_dummy_attack_is_registered():
    assert "dummy" in list_attacks()
    assert get_attack("dummy") is DummyAttack


def test_flowmia_can_run_a_custom_attack(tiny_flow_config):
    flowmia_instance = FlowMIA(config=tiny_flow_config)
    results = flowmia_instance.evaluate_privacy(attacks=["dummy"])

    assert results["dummy"].auc == 0.5
