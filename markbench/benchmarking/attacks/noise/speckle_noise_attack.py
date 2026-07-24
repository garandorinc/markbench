import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("speckle_noise")
class SpeckleNoiseAttack(AbstractAttack):
    """
    Speckle (multiplicative) noise attack.

    Supports randomization by passing tuples (min, max) for std parameter.
    """

    def __init__(self, std: Union[float, Tuple[float, float]] = 0.2):
        """
        Args:
            std: Standard deviation of the speckle noise.
                 Can be float or tuple (min, max) for random range.
        """
        self.std = std

    def _resolve_std(self) -> float:
        """Resolve std to a concrete value, randomizing if given a range."""
        if isinstance(self.std, (list, tuple)):
            return random.uniform(self.std[0], self.std[1])
        return self.std

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply speckle noise to image (masks unchanged)."""
        img = image.float()

        # Get std (possibly random)
        std = self._resolve_std()

        # Generate speckle noise (multiplicative)
        noise = torch.normal(0, std, img.shape, dtype=img.dtype, device=img.device)
        noisy = img + img * noise

        attacked_image = torch.clamp(noisy, 0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"std": std}
        )
