"""Internal registry of attack implementations, keyed by name.

New attacks register themselves with the ``@register_attack("name")`` class
decorator; ``FlowMIA.evaluate_privacy`` looks them up by name via
``get_attack``. No plugin discovery beyond this in-process dict is provided.
"""

from flowmia.attacks.base import BaseAttack

_REGISTRY: dict[str, type[BaseAttack]] = {}


def register_attack(name: str):
    """Class decorator: register a BaseAttack subclass under ``name``."""

    def _decorator(cls: type[BaseAttack]) -> type[BaseAttack]:
        if not issubclass(cls, BaseAttack):
            raise TypeError(f"{cls!r} must subclass BaseAttack")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_attack(name: str) -> type[BaseAttack]:
    """Look up a registered attack class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Registered: {list_attacks()}")
    return _REGISTRY[name]


def list_attacks() -> list[str]:
    """List the names of all currently registered attacks."""
    return sorted(_REGISTRY)
