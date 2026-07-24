from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("contrast")
class AdjustContrastAttack(AbstractAttack):
    """
    Wraps kornia.augmentation.RandomContrast.
    """

    def __init__(self, factor: Union[float, Tuple[float, float]] = 1.5):
        """
        Args:
            factor: Contrast factor (0=gray, 1=original, >1=more contrast).
                    Can be float or tuple (min, max) for random range.
        """
        # Convert fixed value to tuple for RandomContrast
        if isinstance(factor, (list, tuple)):
            contrast_range = tuple(factor)
        else:
            contrast_range = (factor, factor)

        self.factor = factor
        self.contrast_range = contrast_range
        self.transform = aug.RandomContrast(contrast=contrast_range, p=1.0)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply contrast adjustment to image (masks unchanged)."""
        # kornia expects (B, C, H, W) with values in [0, 1]
        img = image.float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)

        result = self.transform(img)

        # Remove batch dimension and convert back to [0, 255]
        if result.dim() == 4:
            result = result.squeeze(0)

        attacked_image = (result * 255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
