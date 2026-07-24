import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("dither_noise", aliases=["dither"])
class DitherNoiseAttack(AbstractAttack):
    """
    Dither noise attack.
    """

    def __init__(self, intensity: Union[float, Tuple[float, float]] = 0.1):
        """
        Args:
            intensity: Intensity of dither noise (0 to 1).
                       Can be float or tuple (min, max) for random range.
        """
        if isinstance(intensity, (list, tuple)):
            self.intensity_min, self.intensity_max = intensity
            self.randomize = True
        else:
            self.intensity = intensity
            self.randomize = False

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply dither noise to image (masks unchanged)."""
        img = image.float()

        # Get intensity (possibly random)
        intensity = (
            random.uniform(self.intensity_min, self.intensity_max)
            if self.randomize
            else self.intensity
        )
        intensity_scale = int(255 * intensity)

        # Generate binary mask and compute noise
        binary_mask = torch.rand(img.shape, device=img.device) > 0.5
        noise = torch.where(
            binary_mask,
            torch.tensor(intensity_scale, dtype=img.dtype, device=img.device),
            torch.tensor(-intensity_scale, dtype=img.dtype, device=img.device),
        )

        # Add noise and clamp
        attacked_image = torch.clamp(img + noise, 0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"intensity": intensity, "intensity_scale": intensity_scale}
        )
