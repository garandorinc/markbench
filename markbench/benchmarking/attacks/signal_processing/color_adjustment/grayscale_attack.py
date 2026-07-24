from typing import Optional

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("grayscale", aliases=["gray", "greyscale"])
class GrayscaleAttack(AbstractAttack):
    """Convert RGB image to grayscale."""

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Convert image to grayscale (masks unchanged)."""
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        gray = (image * weights.view(3, 1, 1)).sum(dim=0, keepdim=True)
        attacked_image = gray.repeat(3, 1, 1)

        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
