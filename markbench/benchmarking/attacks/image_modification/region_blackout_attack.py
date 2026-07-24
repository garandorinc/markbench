import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("region_blackout", aliases=["blackout"])
class RegionBlackoutAttack(AbstractAttack):
    """
    Black out a rectangular region of the image.

    Supports randomization:
    - Pass tuples (min, max) for x, y, width, height for random values within range
    - Set random_location=True to randomly position the blackout region
    """

    def __init__(
        self,
        x: Optional[Union[int, Tuple[int, int]]] = None,
        y: Optional[Union[int, Tuple[int, int]]] = None,
        width: Union[int, Tuple[int, int]] = 50,
        height: Union[int, Tuple[int, int]] = 50,
        random_location: bool = False,
        fill_value: float = 0.0,
        mask_aware: bool = False,
        min_retention: float = 0.25,
    ):
        """
        Args:
            x: Left edge of region. Tuple for range, None for random_location mode.
            y: Top edge of region. Tuple for range, None for random_location mode.
            width: Width of region. Can be int or tuple (min, max).
            height: Height of region. Can be int or tuple (min, max).
            random_location: If True, position region randomly within image bounds.
            fill_value: Value to fill the blacked out region (default 0 for black).
        """
        self.random_location = random_location
        self.fill_value = fill_value

        # Handle x position
        if x is None:
            self.x_randomize = True
            self.x = None
        elif isinstance(x, (list, tuple)):
            self.x_min, self.x_max = x
            self.x_randomize = True
            self.x = None
        else:
            self.x = x
            self.x_randomize = False

        # Handle y position
        if y is None:
            self.y_randomize = True
            self.y = None
        elif isinstance(y, (list, tuple)):
            self.y_min, self.y_max = y
            self.y_randomize = True
            self.y = None
        else:
            self.y = y
            self.y_randomize = False

        # Handle width
        if isinstance(width, (list, tuple)):
            self.w_min, self.w_max = width
            self.w_randomize = True
        else:
            self.width = width
            self.w_randomize = False

        # Handle height
        if isinstance(height, (list, tuple)):
            self.h_min, self.h_max = height
            self.h_randomize = True
        else:
            self.height = height
            self.h_randomize = False

        self.mask_aware = mask_aware
        self.min_retention = min_retention

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Black out a rectangular region in image and zero out same region in masks.

        Args:
            image: Input tensor (C, H, W) in [0, 255] range
            localization_map: Optional ground truth localization maps (N_masks, H, W)

        Returns:
            AttackResult with blacked out image and zeroed mask regions
        """
        _, img_h, img_w = image.shape

        # Get dimensions (once for both image and masks)
        w = random.randint(self.w_min, self.w_max) if self.w_randomize else self.width
        h = random.randint(self.h_min, self.h_max) if self.h_randomize else self.height

        # Ensure dimensions fit within image
        w = min(w, img_w)
        h = min(h, img_h)

        # Get position (once for both image and masks)
        if self.mask_aware and localization_map is not None:
            from ..mask_aware_utils import find_mask_aware_region
            x, y, _, _ = find_mask_aware_region(
                localization_map, w, h, img_h, img_w,
                min_retention=self.min_retention, operation="delete"
            )
        elif self.random_location or self.x is None:
            max_x = img_w - w
            x = random.randint(0, max(0, max_x))
        elif self.x_randomize:
            x = random.randint(self.x_min, min(self.x_max, img_w - w))
        else:
            x = self.x

        if self.mask_aware and localization_map is not None:
            pass
        elif self.random_location or self.y is None:
            max_y = img_h - h
            y = random.randint(0, max(0, max_y))
        elif self.y_randomize:
            y = random.randint(self.y_min, min(self.y_max, img_h - h))
        else:
            y = self.y

        # Black out image region
        attacked_image = image.clone()
        attacked_image[:, y : y + h, x : x + w] = self.fill_value

        # Zero out masks in same region
        transformed_masks = None
        if localization_map is not None:
            transformed_masks = localization_map.clone()
            transformed_masks[:, y : y + h, x : x + w] = 0

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={"region_x": x, "region_y": y, "region_w": w, "region_h": h}
        )
