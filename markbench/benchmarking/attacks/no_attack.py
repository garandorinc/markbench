from typing import Optional

import torch
from .abstract_attack import AbstractAttack
from .registry import register_attack
from markbench.interface import AttackResult


@register_attack("none", aliases=["no_attack", "identity"])
class NoAttack(AbstractAttack):
    """Returns a clone of the input image and passes masks through."""

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Return cloned image and masks unchanged."""
        return AttackResult(
            attacked=image.clone(),
            localization_map=localization_map,
            metadata={}
        )
