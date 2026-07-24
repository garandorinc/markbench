import torch
from typing import Optional

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("histogram_equalization", aliases=["hist_eq"])
class HistogramEqualizationAttack(AbstractAttack):
    """Apply histogram equalization to image."""

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply histogram equalization to image (masks unchanged)."""
        img = image.float()
        r, g, b = img[0], img[1], img[2]

        # Calculate Y (luminance)
        y = 0.299 * r + 0.587 * g + 0.114 * b

        # Perform histogram equalization on Y channel
        y_flat = y.flatten()
        y_hist = torch.histc(y_flat, bins=256, min=0, max=255)

        # Calculate CDF
        cdf = torch.cumsum(y_hist, dim=0)
        cdf_normalized = cdf / cdf[-1] * 255.0

        # Apply equalization
        y_flat_clamped = torch.clamp(y_flat, 0, 255)
        indices = torch.searchsorted(
            torch.arange(256, dtype=torch.float32, device=image.device), y_flat_clamped
        )
        indices = torch.clamp(indices, 0, 255)
        y_equalized = cdf_normalized[indices].view(y.shape)

        # Reconstruct RGB with equalized luminance
        r_eq = y_equalized + (r - y)
        g_eq = y_equalized + (g - y)
        b_eq = y_equalized + (b - y)

        attacked_image = torch.clamp(torch.stack([r_eq, g_eq, b_eq], dim=0), 0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
