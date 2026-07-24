import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("poisson_noise")
class PoissonNoiseAttack(AbstractAttack):
    """
    Poisson noise attack using torch.poisson.
    """

    def __init__(self, scale: Union[float, Tuple[float, float]] = 1.0):
        """
        Args:
            scale: Multiplier applied to pixel values before Poisson sampling.
                   Higher values reduce relative noise (Poisson variance = mean).
                   scale=1.0 uses original pixel values as Poisson lambda.
                   scale<1.0 increases noise intensity; scale>1.0 decreases it.
                   Can be float or tuple (min, max) for random range.
        """
        if isinstance(scale, (list, tuple)):
            self.scale_min, self.scale_max = scale
            self.randomize = True
        else:
            self.scale = scale
            self.randomize = False

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply Poisson noise to image (masks unchanged)."""
        img = image.float()

        # Get scale
        scale = (
            random.uniform(self.scale_min, self.scale_max)
            if self.randomize
            else self.scale
        )

        # Normalize to range for Poisson distribution
        img_normalized = img / 255.0 * scale * 255.0

        # Ensure positive values for Poisson
        img_normalized = torch.clamp(img_normalized, min=0.01)

        # Generate Poisson noise
        noisy = torch.poisson(img_normalized)

        attacked_image = torch.clamp(noisy, 0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"scale": scale}
        )
