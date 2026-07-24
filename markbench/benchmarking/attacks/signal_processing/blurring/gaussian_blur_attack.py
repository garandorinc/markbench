import random
from typing import Optional, Union, Tuple

import torch
from torchvision.transforms import GaussianBlur

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("gaussian_blur", aliases=["blur"])
class GaussianBlurAttack(AbstractAttack):
    """Gaussian blur attack using torchvision for better performance."""

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: Union[float, Tuple[float, float]] = (0.1, 2.0),
    ):
        """
        Args:
            kernel_size: Size of the Gaussian kernel (odd number)
            sigma: Standard deviation. Can be float or tuple (min, max) for random range.
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _resolve_sigma(self) -> float:
        """Resolve sigma to a concrete value, randomizing if given a range."""
        if isinstance(self.sigma, (tuple, list)):
            return random.uniform(self.sigma[0], self.sigma[1])
        return float(self.sigma)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply Gaussian blur to image (masks unchanged)."""
        # Get sigma (possibly random)
        sigma = self._resolve_sigma()

        # Create and apply GaussianBlur transform
        # torchvision.transforms.GaussianBlur works on CHW format directly
        blur = GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)

        attacked_image = blur(image)

        # Blur is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"kernel_size": self.kernel_size, "sigma": sigma}
        )
