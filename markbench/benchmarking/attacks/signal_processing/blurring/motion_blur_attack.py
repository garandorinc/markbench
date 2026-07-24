from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("motion_blur")
class MotionBlurAttack(AbstractAttack):
    """
    Motion blur attack using kornia.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]] = 7,
        angle: Union[float, Tuple[float, float]] = (0.0, 360.0),
        direction: Union[float, Tuple[float, float]] = (-1.0, 1.0),
    ):
        """
        Args:
            kernel_size: Size of the motion blur kernel. Can be int or tuple (min, max) for random odd sizes.
            angle: Angle of the motion blur in degrees. Can be float or tuple (min, max).
            direction: Direction of the motion blur. Can be float or tuple (min, max).
        """
        # Normalize to tuples for kornia
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(angle, (int, float)):
            angle = (float(angle), float(angle))
        if isinstance(direction, (int, float)):
            direction = (float(direction), float(direction))

        self.kernel_size = kernel_size
        self.angle = angle
        self.direction = direction

        # Use kornia's built-in RandomMotionBlur with randomization support
        self.transform = aug.RandomMotionBlur(
            kernel_size=kernel_size, angle=angle, direction=direction, p=1.0
        )

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply motion blur to image (masks unchanged)."""
        # kornia expects (B, C, H, W) with values in [0, 1]
        img = image.float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Use kornia augmentation (handles randomization internally)
        blurred = self.transform(img)

        # Remove batch dimension and convert back to [0, 255]
        if blurred.dim() == 4:
            blurred = blurred.squeeze(0)

        attacked_image = (blurred * 255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
