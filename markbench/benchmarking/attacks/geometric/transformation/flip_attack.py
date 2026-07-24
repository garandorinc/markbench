from typing import Optional

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


class FlipAttack(AbstractAttack):
    """Flip attack using pure PyTorch (horizontal or vertical flip)."""

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(self, axis: str = "x"):
        """
        Args:
            axis: 'x' for vertical flip (top-bottom), 'y' for horizontal flip (left-right)
        """
        self.axis = axis.lower()
        if self.axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply flip to image and optionally masks."""
        # For tensor format (C, H, W) or (N, H, W):
        # - dim=1 is H (vertical flip, top-bottom)
        # - dim=2 is W (horizontal flip, left-right)
        flip_dim = 1 if self.axis == "x" else 2

        # Flip image
        attacked_image = image.flip(flip_dim)

        # Flip masks if provided
        transformed_masks = None
        if localization_map is not None:
            transformed_masks = localization_map.flip(flip_dim)

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={"axis": self.axis}
        )


@register_attack("flip")
class AxisFlipAttack(FlipAttack):
    """Generic axis flip. Prefer horizontal_flip or vertical_flip in benchmark reports."""


@register_attack("horizontal_flip", aliases=["hflip"])
class HorizontalFlipAttack(FlipAttack):
    """Left-right image flip."""

    def __init__(self):
        super().__init__(axis="y")


@register_attack("vertical_flip", aliases=["vflip"])
class VerticalFlipAttack(FlipAttack):
    """Top-bottom image flip."""

    def __init__(self):
        super().__init__(axis="x")
