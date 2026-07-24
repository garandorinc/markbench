import random
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("convolutional_filter", aliases=["conv_filter"])
class ConvolutionalFilterAttack(AbstractAttack):
    """Apply custom convolution kernel to image."""

    def __init__(self, kernel, strength: Union[float, Tuple[float, float]] = 1.0):
        """
        Args:
            kernel: 2D array/tensor representing the convolution filter
            strength: Scaling factor for the filter effect.
                     Can be float or tuple (min, max) for random range.
                     1.0 = full effect, 0.0 = no effect (original image)
        """
        self.kernel = (
            kernel
            if isinstance(kernel, torch.Tensor)
            else torch.tensor(kernel, dtype=torch.float32)
        )
        self.strength = strength

    def _resolve_strength(self) -> float:
        """Resolve strength to a concrete value, randomizing if given a range."""
        if isinstance(self.strength, (tuple, list)):
            return random.uniform(self.strength[0], self.strength[1])
        return self.strength

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply convolution filter to image (masks unchanged)."""
        # Prepare kernel for convolution
        kernel = self.kernel.clone()
        if kernel.dim() == 2:
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k_h, k_w)
            kernel = kernel.repeat(3, 1, 1, 1)  # (3, 1, k_h, k_w) for RGB

        kernel = kernel.to(image.device)

        # Add batch dimension
        img_batch = image.unsqueeze(0)

        # Calculate padding
        padding_h = kernel.shape[2] // 2
        padding_w = kernel.shape[3] // 2

        # Apply convolution
        filtered = F.conv2d(img_batch, kernel, padding=(padding_h, padding_w), groups=3)

        # Get strength (possibly random)
        strength = self._resolve_strength()

        # Blend original and filtered based on strength
        if strength < 1.0:
            result = strength * filtered + (1.0 - strength) * img_batch
        else:
            result = filtered

        attacked_image = result.squeeze(0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"strength": strength}
        )
