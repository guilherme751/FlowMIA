"""Attack plugin registry. Importing this subpackage registers all built-in attacks."""

from flowmia.attacks.base import AttackResult, BaseAttack
from flowmia.attacks.registry import get_attack, list_attacks, register_attack

# Imported for registration side-effects only (each module self-registers via decorator).
from flowmia.attacks import dcr_mia as _dcr_mia  # noqa: F401
from flowmia.attacks import domias_bnaf as _domias_bnaf  # noqa: F401
from flowmia.attacks import flowmia_gan as _flowmia_gan  # noqa: F401

__all__ = [
    "AttackResult",
    "BaseAttack",
    "get_attack",
    "list_attacks",
    "register_attack",
]
