"""Attack registry for the benchmark package."""

from typing import Dict, List, Optional, Type

# Registry dict - populated by @register_attack decorators
_ATTACK_REGISTRY: Dict[str, Type] = {}


def register_attack(name: Optional[str] = None, aliases: Optional[List[str]] = None):
    """Decorator to register an attack class."""

    def decorator(cls):
        attack_name = name or cls.__name__
        _ATTACK_REGISTRY[attack_name.lower()] = cls
        for alias in aliases or []:
            _ATTACK_REGISTRY[alias.lower()] = cls
        return cls

    return decorator


def get_attack(name: str, **kwargs):
    """Get an attack instance by name."""

    name_lower = name.lower()
    if name_lower not in _ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: '{name}'. Available: {list_attacks()}")
    return _ATTACK_REGISTRY[name_lower](**kwargs)


def list_attacks() -> List[str]:
    """List all registered attack names (without duplicates from aliases)."""
    seen_classes = set()
    unique_names = []
    for name, cls in _ATTACK_REGISTRY.items():
        if id(cls) not in seen_classes:
            unique_names.append(name)
            seen_classes.add(id(cls))
    return sorted(unique_names)


def get_all_attacks():
    """Get instances of all registered attacks with default parameters."""

    attacks = []
    seen_classes = set()
    for name, cls in _ATTACK_REGISTRY.items():
        if id(cls) not in seen_classes:
            seen_classes.add(id(cls))
            try:
                attacks.append(cls())
            except Exception:
                print(f"Skipping '{name}': requires mandatory parameters")
    return attacks
