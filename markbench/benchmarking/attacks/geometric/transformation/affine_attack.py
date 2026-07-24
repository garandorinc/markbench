import random
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("affine")
class AffineAttack(AbstractAttack):
    """
    Affine transformation attack using PyTorch operations.

    Supports randomization by passing tuples (min, max) for any parameter.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        rotation: Union[float, Tuple[float, float]] = 0.0,
        scale_x: Union[float, Tuple[float, float]] = 1.0,
        scale_y: Union[float, Tuple[float, float]] = 1.0,
        shear_x: Union[float, Tuple[float, float]] = 0.0,
        shear_y: Union[float, Tuple[float, float]] = 0.0,
        translate_x: Union[float, Tuple[float, float]] = 0.0,
        translate_y: Union[float, Tuple[float, float]] = 0.0,
    ):
        """
        Initialize with affine transformation parameters.

        Args:
            rotation: Rotation angle in degrees. Can be float or (min, max) tuple.
            scale_x: Scale factor in X direction. Can be float or (min, max) tuple.
            scale_y: Scale factor in Y direction. Can be float or (min, max) tuple.
            shear_x: Shear factor in X direction. Can be float or (min, max) tuple.
            shear_y: Shear factor in Y direction. Can be float or (min, max) tuple.
            translate_x: Translation in X in normalized grid coordinates.
                         Values in [-1, 1] range (e.g., 0.5 shifts by half image width).
                         Can be float or (min, max) tuple.
            translate_y: Translation in Y in normalized grid coordinates.
                         Values in [-1, 1] range (e.g., 0.5 shifts by half image height).
                         Can be float or (min, max) tuple.
        """
        self.rotation = rotation
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.translate_x = translate_x
        self.translate_y = translate_y

    def _resolve_param(self, param: Union[float, Tuple[float, float]]) -> float:
        """Resolve a parameter to a concrete value, randomizing if given a range."""
        if isinstance(param, (list, tuple)):
            return random.uniform(param[0], param[1])
        return param

    def _get_affine_params(
        self,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Get all affine parameters, resolving any randomization."""
        return (
            self._resolve_param(self.rotation),
            self._resolve_param(self.scale_x),
            self._resolve_param(self.scale_y),
            self._resolve_param(self.shear_x),
            self._resolve_param(self.shear_y),
            self._resolve_param(self.translate_x),
            self._resolve_param(self.translate_y),
        )

    def _build_affine_matrix(
        self,
        rotation: float,
        scale_x: float,
        scale_y: float,
        shear_x: float,
        shear_y: float,
        translate_x: float,
        translate_y: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 2x3 affine transformation matrix."""
        # Convert rotation to radians
        theta = math.radians(rotation)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Rotation matrix
        R = torch.tensor(
            [[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float32, device=device
        )

        # Scale matrix
        S = torch.tensor(
            [[scale_x, 0], [0, scale_y]], dtype=torch.float32, device=device
        )

        # Shear matrix
        Sh = torch.tensor(
            [[1, shear_x], [shear_y, 1]], dtype=torch.float32, device=device
        )

        # Combined transformation: T = R @ S @ Sh
        transform = R @ S @ Sh

        # Build 2x3 matrix with translation
        affine_matrix = torch.zeros((2, 3), dtype=torch.float32, device=device)
        affine_matrix[:2, :2] = transform
        affine_matrix[0, 2] = translate_x
        affine_matrix[1, 2] = translate_y

        return affine_matrix

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply affine transformation to image and optionally masks."""
        # Get the input image tensor and ensure it's in the right format
        img = image.float()

        # Add batch dimension if not present (C, H, W) -> (1, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Get affine parameters (once for both image and masks)
        rotation, scale_x, scale_y, shear_x, shear_y, translate_x, translate_y = (
            self._get_affine_params()
        )

        # Build affine matrix
        affine_matrix = self._build_affine_matrix(
            rotation,
            scale_x,
            scale_y,
            shear_x,
            shear_y,
            translate_x,
            translate_y,
            img.device,
        )

        # Transform image
        grid = F.affine_grid(affine_matrix.unsqueeze(0), img.shape, align_corners=False)
        transformed_img = F.grid_sample(
            img, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

        # Remove batch dimension if it was added
        if transformed_img.dim() == 4:
            transformed_img = transformed_img.squeeze(0)

        attacked_image = transformed_img.clamp(0, 255)

        # Transform masks if provided
        transformed_masks = None
        if localization_map is not None:
            masks = localization_map.float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)

            # Expand affine_matrix to match batch size of masks
            n_masks = masks.shape[0]
            affine_matrix_batch = affine_matrix.expand(n_masks, -1, -1)  # (N, 2, 3)

            grid = F.affine_grid(
                affine_matrix_batch, masks.shape, align_corners=False
            )

            # Apply transformation to masks
            transformed = F.grid_sample(
                masks, grid, align_corners=False,
                mode="bilinear", padding_mode="zeros"
            )

            # Remove channel dimension if needed (N, 1, H', W') -> (N, H', W')
            if transformed.dim() == 4 and transformed.shape[1] == 1:
                transformed = transformed.squeeze(1)

            transformed_masks = (transformed > 0.5).float()

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={
                "rotation": rotation,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "shear_x": shear_x,
                "shear_y": shear_y,
                "translate_x": translate_x,
                "translate_y": translate_y,
            }
        )
