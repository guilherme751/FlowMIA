"""FlowMIA: membership-inference privacy evaluation for synthetic network flow data."""

from flowmia.core import FlowMIA
from flowmia.config import FlowMIAConfig, load_config
from flowmia.attacks import (
    AttackResult,
    BaseAttack,
    get_attack,
    list_attacks,
    register_attack,
)

__version__ = "0.1.0"

__all__ = [
    "FlowMIA",
    "FlowMIAConfig",
    "load_config",
    "AttackResult",
    "BaseAttack",
    "get_attack",
    "list_attacks",
    "register_attack",
]
