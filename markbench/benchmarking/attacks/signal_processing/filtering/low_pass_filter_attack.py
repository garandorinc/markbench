import random
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("low_pass_filter", aliases=["low_pass"])
class LowPassFilterAttack(AbstractAttack):
    """Low-pass (averaging) filter attack using PyTorch convolution."""

    def __init__(self, kernel_size: Union[int, Tuple[int, int]] = 3):
        """
        Args:
            kernel_size: Size of the averaging kernel.
                        Can be int or tuple (min, max) for random range.
        """
        self.kernel_size = kernel_size

    def _resolve_kernel_size(self) -> int:
        """Resolve kernel_size to a concrete value, randomizing if given a range."""
        if isinstance(self.kernel_size, (tuple, list)):
            k = random.randint(self.kernel_size[0], self.kernel_size[1])
        else:
            k = self.kernel_size
        # Ensure odd
        return k if k % 2 == 1 else k + 1

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply low-pass filter to image (masks unchanged)."""
        img = image.float()

        # Get kernel size (possibly random)
        kernel_size = self._resolve_kernel_size()

        # Define low-pass kernel (averaging filter)
        kernel = torch.ones(
            (kernel_size, kernel_size), dtype=img.dtype, device=img.device
        ) / (kernel_size**2)

        # Prepare kernel for convolution: (out_channels, in_channels/groups, H, W)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)  # (3, 1, 3, 3) for RGB channels

        # Add batch dimension for conv2d: (1, C, H, W)
        img_batch = img.unsqueeze(0)

        # Apply convolution with padding
        padding = kernel_size // 2
        filtered = F.conv2d(img_batch, kernel, padding=padding, groups=3)

        attacked_image = filtered.squeeze(0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"kernel_size": kernel_size}
        )
