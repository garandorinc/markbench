from typing import Optional, Tuple, Union

import torch
import kornia.enhance

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("sharpening", aliases=["sharpen"])
class SharpeningAttack(AbstractAttack):
    """
    Sharpening attack using kornia.enhance.sharpness.

    Normalizes [0, 255] input to [0, 1], applies the transformation, then undoes normalization.
    Supports randomization by passing tuples (min, max) for sharpness parameter.
    """

    def __init__(self, sharpness: Union[float, Tuple[float, float]] = 0.5):
        """
        Args:
            sharpness: Sharpening intensity. 0.0 means no change, higher values
                       increase sharpening effect.
                       Can be float or tuple (min, max) for random range.
        """
        self.sharpness = sharpness

    def _resolve_sharpness(self) -> float:
        """Resolve sharpness to a concrete value, randomizing if given a range."""
        import random

        if isinstance(self.sharpness, (list, tuple)):
            return random.uniform(self.sharpness[0], self.sharpness[1])
        return self.sharpness

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply sharpening to image (masks unchanged)."""
        # kornia.enhance.sharpness requires [0, 1] normalized input
        # Normalize, apply sharpness, denormalize in one expression to minimize allocations
        sharpness = self._resolve_sharpness()

        # Add batch dimension, apply sharpness, remove batch dimension
        img_normed = image.float().div(255.0).unsqueeze(0)
        sharpened = kornia.enhance.sharpness(img_normed, sharpness)

        attacked_image = sharpened.squeeze(0).mul(255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"sharpness": sharpness}
        )
