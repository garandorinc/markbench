from typing import Optional

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("invert_colors", aliases=["invert"])
class InvertColorsAttack(AbstractAttack):
    """Invert colors attack (negative image) using pure PyTorch."""

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Invert image colors (masks unchanged)."""
        attacked_image = 255 - image.float()

        # Color inversion is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
