import flowmia
from flowmia import (
    AttackResult,
    BaseAttack,
    FlowMIA,
    FlowMIAConfig,
    get_attack,
    list_attacks,
    load_config,
    register_attack,
)


def test_builtin_attacks_registered():
    assert set(list_attacks()) >= {"flowmia_gan", "domias", "dcr"}


def test_get_attack_returns_class():
    cls = get_attack("dcr")
    assert issubclass(cls, BaseAttack)


def test_get_attack_unknown_raises():
    try:
        get_attack("does_not_exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown attack name")
