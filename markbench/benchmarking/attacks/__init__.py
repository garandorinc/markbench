"""Attack registry and parser utilities.

This module provides a registry pattern for image attacks, allowing
attacks to be registered and instantiated by name.

Usage:
    from markbench.benchmarking.attacks import get_attack, list_attacks, AbstractAttack

    # Get an attack instance
    attack = get_attack("jpeg", quality=75)

    # List available attacks
    print(list_attacks())
"""

import torch

from .abstract_attack import AbstractAttack
from .registry import (
    _ATTACK_REGISTRY,
    register_attack,
    get_attack,
    list_attacks,
    get_all_attacks,
)

# Import all attack modules to trigger self-registration via @register_attack decorators
# These imports are for side effects only (decorator registration)
from .no_attack import NoAttack  # noqa: F401
from . import composite as _composite  # noqa: F401
from . import compression as _compression  # noqa: F401
from .geometric import transformation as _transformation  # noqa: F401
from .geometric import distortion as _distortion  # noqa: F401
from . import noise as _noise  # noqa: F401
from .signal_processing import blurring as _blurring  # noqa: F401
from .signal_processing import color_adjustment as _color_adjustment  # noqa: F401
from .signal_processing import filtering as _filtering  # noqa: F401
from .signal_processing import image_enhancement as _image_enhancement  # noqa: F401
from . import image_modification as _image_modification  # noqa: F401

# Optional imports may fail when users install only the core package.
import warnings

try:
    from . import special_attacks as _special_attacks  # noqa: F401
except ImportError as e:
    warnings.warn(f"Optional special attacks are unavailable: {e}")

try:
    from . import splicing as _splicing  # noqa: F401
except ImportError as e:
    warnings.warn(f"Optional splicing attacks are unavailable: {e}")

try:
    from .generative_ai import stable_diffusion_inpainting as _sd_inpaint  # noqa: F401
except ImportError as e:
    warnings.warn(f"Optional diffusion edit attacks are unavailable: {e}")


def parse_attack_string(attack_str: str) -> AbstractAttack:
    """Parse attack string like 'jpeg:50' and return attack instance."""
    # Named kernels for convolutional_filter
    NAMED_KERNELS = {
        "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        "edge": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        "emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    }

    attack_str = attack_str.strip()

    # Handle composite attacks with parentheses
    if "(" in attack_str and attack_str.endswith(")"):
        paren_idx = attack_str.index("(")
        name = attack_str[:paren_idx].lower()
        content = attack_str[paren_idx + 1:-1]

        if name == "chain":
            attacks = [parse_attack_string(a.strip()) for a in _split_preserving_parens(content)]
            return get_attack("chain", attacks=attacks)

        if name == "random":
            parts = _split_preserving_parens(content)
            if not parts:
                raise ValueError("random() requires count and pool")
            try:
                count = int(parts[0])
            except ValueError:
                raise ValueError(f"random() count must be integer, got: {parts[0]}")
            pool = [parse_attack_string(a.strip()) for a in parts[1:]]
            if not pool:
                raise ValueError("random() requires at least one attack in pool")
            return get_attack("random", count=count, pool=pool)

    parts = attack_str.split(":")
    name = parts[0]

    if len(parts) == 1:
        return get_attack(name)

    params = []
    for p in parts[1:]:
        try:
            params.append(int(p))
        except ValueError:
            try:
                params.append(float(p))
            except ValueError:
                params.append(p)

    attack_cls = _ATTACK_REGISTRY.get(name.lower())
    if not attack_cls:
        raise ValueError(f"Unknown attack: {name}")

    # Special handling for convolutional_filter with named kernels
    if name.lower() in ("convolutional_filter", "conv_filter") and len(params) == 1:
        if isinstance(params[0], str) and params[0] in NAMED_KERNELS:
            kernel = torch.tensor(NAMED_KERNELS[params[0]], dtype=torch.float32)
            return attack_cls(kernel=kernel)

    # Special handling for interpolation (fx, fy from single scale)
    if name.lower() == "interpolation" and len(params) == 1:
        scale = params[0]
        return attack_cls(fx=scale, fy=scale)

    if len(params) == 1:
        return attack_cls(params[0])
    return attack_cls(*params)


def _split_preserving_parens(s: str) -> list[str]:
    """Split on commas, but preserve commas inside parentheses."""
    parts = []
    current = []
    depth = 0

    for char in s:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current))

    return parts


def parse_attack_config(config: object, device=None) -> list[AbstractAttack]:
    """Parse attack config from a list or dict loaded from YAML.

    Supported shapes:
      - list of strings (uses parse_attack_string)
      - list of dicts: {name: "...", ...params}
      - compact list entries with variants: {name: "...", variants: [...]}
      - dict with "attacks" key containing either of the above

    Args:
        config: Attack configuration (list or dict)
        device: Optional device to pass to attacks that support it
    """
    if config is None:
        return []

    if isinstance(config, dict):
        if "attacks" in config:
            config = config["attacks"]
        else:
            raise ValueError("Attack config dict must include an 'attacks' key")

    if not isinstance(config, list):
        raise ValueError("Attack config must be a list of attack entries")

    attacks: list[AbstractAttack] = []
    for entry in config:
        if isinstance(entry, str):
            attacks.append(parse_attack_string(entry))
            continue
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("attack")
            if not name:
                raise ValueError("Attack config entry is missing 'name'")
            params = {k: v for k, v in entry.items() if k not in ("name", "attack")}

            if "variants" in params:
                from .runtime_specs import build_runtime_attack_spec

                attacks.append(build_runtime_attack_spec(name, params, device=device))
                continue

            # Add device parameter if not already specified, device is provided, and attack accepts it
            if device is not None and "device" not in params:
                # Check if the attack class accepts a 'device' parameter
                from .registry import _ATTACK_REGISTRY
                import inspect

                name_lower = name.lower()
                if name_lower in _ATTACK_REGISTRY:
                    attack_class = _ATTACK_REGISTRY[name_lower]
                    sig = inspect.signature(attack_class.__init__)
                    if 'device' in sig.parameters:
                        params["device"] = device

            # Handle composite attacks
            if name.lower() == "chain" and "attacks" in params:
                attacks_list = params["attacks"]
                if isinstance(attacks_list, list):
                    parsed_attacks = [parse_attack_string(a) if isinstance(a, str) else get_attack(a["name"], **{k: v for k, v in a.items() if k != "name"}) for a in attacks_list]
                    attacks.append(get_attack("chain", attacks=parsed_attacks))
                else:
                    raise ValueError("chain 'attacks' must be a list")
                continue

            if name.lower() == "random" and "pool" in params:
                pool_list = params["pool"]
                count = params.get("count", 1)
                replacement = params.get("replacement", False)
                if isinstance(pool_list, list):
                    parsed_pool = [parse_attack_string(a) if isinstance(a, str) else get_attack(a["name"], **{k: v for k, v in a.items() if k != "name"}) for a in pool_list]
                    attacks.append(get_attack("random", count=count, pool=parsed_pool, replacement=replacement))
                else:
                    raise ValueError("random 'pool' must be a list")
                continue

            attacks.append(get_attack(name, **params))
            continue
        raise ValueError(f"Unsupported attack entry type: {type(entry)}")

    return attacks


# Public API
__all__ = [
    "AbstractAttack",
    "NoAttack",
    "register_attack",
    "get_attack",
    "list_attacks",
    "get_all_attacks",
    "parse_attack_string",
    "parse_attack_config",
    "_ATTACK_REGISTRY",
]
