import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from markbench.interface import AttackResult
from ...abstract_attack import AbstractAttack
from ...registry import register_attack


@register_attack("small_distortions", aliases=["distortions"])
class SmallDistortionsAttack(AbstractAttack):
    """Small random geometric distortions: corner shifts, bending, and wave patterns."""

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        seed: int = 0,
        strength: Union[float, Tuple[float, float]] = 1.0,
        bending_factor: Union[float, Tuple[float, float]] = 0.01,
        random_factor: Union[float, Tuple[float, float]] = 0.008,
    ):
        """
        Args:
            seed: Random seed for reproducible distortions
            strength: Overall distortion strength multiplier.
                      Can be float or tuple (min, max) for random range.
            bending_factor: Sinusoidal bending strength.
                           Can be float or tuple (min, max) for random range.
            random_factor: Random wave pattern strength.
                          Can be float or tuple (min, max) for random range.
        """
        self.strength = strength
        self.bending_factor = bending_factor
        self.random_factor = random_factor

        # Use PyTorch generator for reproducible random numbers
        generator = torch.Generator().manual_seed(seed)

        # Pre-generate random parameters using torch
        self.corner_devs = (torch.rand((4, 2), generator=generator) - 0.5) * 0.04
        self.phase = torch.rand(2, generator=generator) * math.pi

    def _resolve_strength(self) -> float:
        """Resolve strength to a concrete value, randomizing if given a range."""
        if isinstance(self.strength, (tuple, list)):
            return random.uniform(self.strength[0], self.strength[1])
        return self.strength

    def _resolve_bending_factor(self) -> float:
        """Resolve bending_factor to a concrete value, randomizing if given a range."""
        if isinstance(self.bending_factor, (tuple, list)):
            return random.uniform(self.bending_factor[0], self.bending_factor[1])
        return self.bending_factor

    def _resolve_random_factor(self) -> float:
        """Resolve random_factor to a concrete value, randomizing if given a range."""
        if isinstance(self.random_factor, (tuple, list)):
            return random.uniform(self.random_factor[0], self.random_factor[1])
        return self.random_factor

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        img = image.float()
        if img.dim() == 3:
            img = img.unsqueeze(0)

        _, _, h, w = img.shape
        device = img.device

        # Get parameters (possibly random)
        strength = self._resolve_strength()
        bending_factor = self._resolve_bending_factor()
        random_factor = self._resolve_random_factor()

        # Create normalized coordinate grid [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=device)
        x_coords = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Convert to [0, 1] for distortion calculations
        nx = (grid_x + 1) / 2
        ny = (grid_y + 1) / 2

        # Corner deviations (move pre-computed tensor to device)
        corner_devs = self.corner_devs.to(device=device, dtype=torch.float32)
        ul, ur, ll, lr = corner_devs[0], corner_devs[1], corner_devs[2], corner_devs[3]

        # Bilinear interpolation of corner shifts
        shift_x = ((1 - nx) * ul[0] + nx * ur[0]) * (1 - ny) + (
            (1 - nx) * ll[0] + nx * lr[0]
        ) * ny
        shift_y = ((1 - nx) * ul[1] + nx * ur[1]) * (1 - ny) + (
            (1 - nx) * ll[1] + nx * lr[1]
        ) * ny

        # Apply corner shifts
        nx = nx + shift_x * strength
        ny = ny + shift_y * strength

        # Add sinusoidal bending (move pre-computed tensor to device)
        phase = self.phase.to(device=device, dtype=torch.float32)
        nx = nx + bending_factor * torch.sin(2 * math.pi * ny + phase[0])
        ny = ny + bending_factor * torch.sin(2 * math.pi * nx + phase[1])

        # Add random wave pattern
        nx = nx + random_factor * torch.sin(10 * math.pi * nx) * torch.sin(
            10 * math.pi * ny
        )
        ny = ny + random_factor * torch.sin(10 * math.pi * nx) * torch.sin(
            10 * math.pi * ny
        )

        # Convert back to [-1, 1] for grid_sample
        grid_x = nx * 2 - 1
        grid_y = ny * 2 - 1

        # Create sampling grid (B, H, W, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        # Apply distortion to image
        distorted = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        if distorted.dim() == 4:
            distorted = distorted.squeeze(0)

        attacked_image = distorted.clamp(0, 255)

        # Apply SAME grid to masks if provided
        transformed_masks = None
        if localization_map is not None:
            masks = localization_map.float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (N, 1, H, W)
            distorted_masks = F.grid_sample(
                masks, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            if distorted_masks.dim() == 4 and distorted_masks.shape[1] == 1:
                distorted_masks = distorted_masks.squeeze(1)
            transformed_masks = distorted_masks

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={
                "strength": strength,
                "bending_factor": bending_factor,
                "random_factor": random_factor
            }
        )
