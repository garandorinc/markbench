from typing import Optional

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("diagonal_flip")
class DiagonalFlipAttack(AbstractAttack):
    """Diagonal flip (transpose) attack using pure PyTorch."""

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply diagonal flip to image and optionally masks."""
        # For tensor format (C, H, W), transpose H and W (dims 1 and 2)
        attacked_image = image.transpose(1, 2)

        # Transpose masks if provided (N, H, W) -> (N, W, H)
        transformed_masks = None
        if localization_map is not None:
            transformed_masks = localization_map.transpose(1, 2)

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={}
        )
