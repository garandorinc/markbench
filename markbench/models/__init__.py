"""Model registry for user-provided watermark adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from markbench.interface import WatermarkModel

ModelBuilder = Callable[..., WatermarkModel]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str, builder: ModelBuilder) -> None:
    """Register a model builder for later use through ``load_model``."""
    normalized = name.lower().strip()
    if not normalized:
        raise ValueError("Model name must be non-empty")
    _MODEL_REGISTRY[normalized] = builder


def list_models() -> list[str]:
    """Return registered model names."""
    return sorted(_MODEL_REGISTRY)


def load_model(name: str, **kwargs: Any) -> WatermarkModel:
    """Load a registered model or a ``module:function`` builder.

    The ``module:function`` form keeps this package independent of third-party
    model repositories. The function must return a ``WatermarkModel``.
    """
    if ":" in name:
        module_name, function_name = name.split(":", 1)
        module = importlib.import_module(module_name)
        builder = getattr(module, function_name)
        model = builder(**kwargs)
    else:
        normalized = name.lower().strip()
        if normalized not in _MODEL_REGISTRY:
            available = ", ".join(list_models()) or "none"
            raise ValueError(
                f"Unknown model '{name}'. Register it with markbench.models.register_model, "
                f"or pass a module:function builder. Registered models: {available}."
            )
        model = _MODEL_REGISTRY[normalized](**kwargs)

    if not isinstance(model, WatermarkModel):
        raise TypeError(
            f"Model builder for '{name}' returned {type(model)!r}, expected WatermarkModel."
        )
    return model
