from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("saturation")
class AdjustSaturationAttack(AbstractAttack):
    """
    Wraps kornia's RandomSaturation.
    """

    def __init__(self, saturation: Union[float, Tuple[float, float]] = 1.0):
        """
        Args:
            saturation: Saturation factor.
                       Can be float or tuple (min, max) for random range.
                       1.0 = no change
                       < 1.0 = less saturated (0.0 = grayscale)
                       > 1.0 = more saturated
        """
        # Handle tuple for randomization
        if isinstance(saturation, (list, tuple)):
            saturation_range = tuple(saturation)
        else:
            saturation_range = (saturation, saturation)

        self.saturation = saturation
        self.saturation_range = saturation_range
        self.transform = aug.RandomSaturation(saturation=saturation_range, p=1.0)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply saturation adjustment to image (masks unchanged)."""
        # kornia expects (B, C, H, W) with values in [0, 1]
        img = image.float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)

        result = self.transform(img)

        if result.dim() == 4:
            result = result.squeeze(0)

        attacked_image = (result * 255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
