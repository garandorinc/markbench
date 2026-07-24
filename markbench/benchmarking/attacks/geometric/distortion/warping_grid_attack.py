import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from markbench.interface import AttackResult
from ...abstract_attack import AbstractAttack
from ...registry import register_attack


@register_attack("warping_grid", aliases=["warp"])
class WarpingGridAttack(AbstractAttack):
    """Sinusoidal grid-based warping distortion using PyTorch."""

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        distortion_strength: Union[float, Tuple[float, float]] = 1.0,
        grid_size: Union[float, Tuple[float, float]] = 20.0,
    ):
        """
        Args:
            distortion_strength: Amplitude of distortion in pixels.
                                 Can be float or tuple (min, max) for random range.
            grid_size: Period of the sinusoidal pattern.
                       Can be float or tuple (min, max) for random range.
        """
        self.distortion_strength = distortion_strength
        self.grid_size = grid_size

    def _resolve_distortion_strength(self) -> float:
        """Resolve distortion_strength to a concrete value, randomizing if given a range."""
        if isinstance(self.distortion_strength, (tuple, list)):
            return random.uniform(
                self.distortion_strength[0], self.distortion_strength[1]
            )
        return self.distortion_strength

    def _resolve_grid_size(self) -> float:
        """Resolve grid_size to a concrete value, randomizing if given a range."""
        if isinstance(self.grid_size, (tuple, list)):
            return random.uniform(self.grid_size[0], self.grid_size[1])
        return self.grid_size

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
        distortion_strength = self._resolve_distortion_strength()
        grid_size = self._resolve_grid_size()

        # Create coordinate grids
        y_coords = torch.arange(h, dtype=torch.float32, device=device)
        x_coords = torch.arange(w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Apply sinusoidal distortion
        dx = distortion_strength * torch.sin(2 * math.pi * grid_y / grid_size)
        dy = distortion_strength * torch.cos(2 * math.pi * grid_x / grid_size)

        # Calculate new positions
        new_x = grid_x + dx
        new_y = grid_y + dy

        # Normalize to [-1, 1] for grid_sample
        norm_x = 2 * new_x / (w - 1) - 1
        norm_y = 2 * new_y / (h - 1) - 1

        # Create sampling grid (B, H, W, 2)
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)

        # Apply warping to image
        warped = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        if warped.dim() == 4:
            warped = warped.squeeze(0)

        attacked_image = warped.clamp(0, 255)

        # Apply SAME grid to masks if provided
        transformed_masks = None
        if localization_map is not None:
            masks = localization_map.float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (N, 1, H, W)
            warped_masks = F.grid_sample(
                masks, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            if warped_masks.dim() == 4 and warped_masks.shape[1] == 1:
                warped_masks = warped_masks.squeeze(1)
            transformed_masks = warped_masks

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={
                "distortion_strength": distortion_strength,
                "grid_size": grid_size
            }
        )
