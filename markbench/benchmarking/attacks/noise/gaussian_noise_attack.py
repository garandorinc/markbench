import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("gaussian_noise", aliases=["noise"])
class GaussianNoiseAttack(AbstractAttack):
    """
    Additive Gaussian noise attack.

    Supports randomization by passing tuples (min, max) for std parameter.
    """

    def __init__(
        self, mean: float = 0.0, std: Union[float, Tuple[float, float]] = 10.0
    ):
        """
        Args:
            mean: Mean of the Gaussian noise (in [0, 255] scale).
            std: Standard deviation of the noise (in [0, 255] scale).
                 Can be float or tuple (min, max) for random range.
        """
        self.mean = mean
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
        """Apply Gaussian noise to image (masks unchanged)."""
        # Get std (possibly random)
        std = self._resolve_std()

        # Generate noise directly in [0, 255] scale - no normalization needed
        noise = torch.normal(
            self.mean, std, image.shape, dtype=image.dtype, device=image.device
        )

        attacked_image = (image + noise).clamp(0, 255)

        # Noise is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"mean": self.mean, "std": std}
        )
