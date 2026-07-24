import torch
import torch.nn.functional as F
from typing import Optional

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("high_pass_filter", aliases=["high_pass"])
class HighPassFilterAttack(AbstractAttack):
    """High-pass (edge detection) filter attack using PyTorch convolution."""

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply high-pass filter to image (masks unchanged)."""
        img = image.float()

        # Define high-pass kernel (Laplacian)
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=img.dtype,
            device=img.device,
        )

        # Prepare kernel for convolution: (out_channels, in_channels/groups, H, W)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)  # (3, 1, 3, 3) for RGB channels

        # Add batch dimension for conv2d: (1, C, H, W)
        img_batch = img.unsqueeze(0)

        # Apply convolution
        filtered = F.conv2d(img_batch, kernel, padding=1, groups=3)

        attacked_image = filtered.squeeze(0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
