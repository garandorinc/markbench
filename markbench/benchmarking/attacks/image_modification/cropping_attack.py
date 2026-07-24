import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("crop", aliases=["cropping"])
class CroppingAttack(AbstractAttack):
    """
    Crop a rectangular region from the image.

    Supports randomization:
    - Pass tuples (min, max) for x, y, width, height for random values within range
    - Set random_location=True to randomly position a fixed-size crop

    Note: This attack changes the image dimensions (extracts a sub-region).
    For blacking out a region, use RegionBlackoutAttack.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        x: Optional[Union[int, Tuple[int, int]]] = None,
        y: Optional[Union[int, Tuple[int, int]]] = None,
        width: Union[int, Tuple[int, int]] = 100,
        height: Union[int, Tuple[int, int]] = 100,
        width_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        height_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        random_location: bool = False,
        mask_aware: bool = False,
        min_retention: float = 0.25,
    ):
        """
        Args:
            x: Left edge of crop. Tuple for range, None for random_location mode.
            y: Top edge of crop. Tuple for range, None for random_location mode.
            width: Width of crop. Can be int or tuple (min, max).
            height: Height of crop. Can be int or tuple (min, max).
            width_ratio: Width as a ratio of image width (0-1+). Can be float or tuple (min, max).
            height_ratio: Height as a ratio of image height (0-1+). Can be float or tuple (min, max).
            random_location: If True, position crop randomly within image bounds.
        """
        self.random_location = random_location

        # Handle x position
        if x is None:
            self.x_randomize = True
            self.x_min = 0
            self.x_max = None  # Will be constrained by image bounds at runtime
        elif isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("x must be a tuple of (min, max)")
            self.x_min, self.x_max = x
            if self.x_min < 0:
                raise ValueError(f"x_min must be non-negative, got {self.x_min}")
            if self.x_max < self.x_min:
                raise ValueError(f"x_max ({self.x_max}) must be >= x_min ({self.x_min})")
            self.x_randomize = True
        else:
            if x < 0:
                raise ValueError(f"x must be non-negative, got {x}")
            self.x = x
            self.x_randomize = False

        # Handle y position
        if y is None:
            self.y_randomize = True
            self.y_min = 0
            self.y_max = None  # Will be constrained by image bounds at runtime
        elif isinstance(y, (list, tuple)):
            if len(y) != 2:
                raise ValueError("y must be a tuple of (min, max)")
            self.y_min, self.y_max = y
            if self.y_min < 0:
                raise ValueError(f"y_min must be non-negative, got {self.y_min}")
            if self.y_max < self.y_min:
                raise ValueError(f"y_max ({self.y_max}) must be >= y_min ({self.y_min})")
            self.y_randomize = True
        else:
            if y < 0:
                raise ValueError(f"y must be non-negative, got {y}")
            self.y = y
            self.y_randomize = False

        # Handle width
        if isinstance(width, (list, tuple)):
            if len(width) != 2:
                raise ValueError("width must be a tuple of (min, max)")
            self.w_min, self.w_max = width
            if self.w_min <= 0:
                raise ValueError(f"w_min must be positive, got {self.w_min}")
            if self.w_max < self.w_min:
                raise ValueError(f"w_max ({self.w_max}) must be >= w_min ({self.w_min})")
            self.w_randomize = True
        else:
            if width <= 0:
                raise ValueError(f"width must be positive, got {width}")
            self.width = width
            self.w_randomize = False

        # Handle height
        if isinstance(height, (list, tuple)):
            if len(height) != 2:
                raise ValueError("height must be a tuple of (min, max)")
            self.h_min, self.h_max = height
            if self.h_min <= 0:
                raise ValueError(f"h_min must be positive, got {self.h_min}")
            if self.h_max < self.h_min:
                raise ValueError(f"h_max ({self.h_max}) must be >= h_min ({self.h_min})")
            self.h_randomize = True
        else:
            if height <= 0:
                raise ValueError(f"height must be positive, got {height}")
            self.height = height
            self.h_randomize = False

        # Optional ratio-based crop dimensions (takes precedence over absolute width/height)
        if width_ratio is None:
            self.width_ratio = None
            self.wr_randomize = False
        elif isinstance(width_ratio, (list, tuple)):
            if len(width_ratio) != 2:
                raise ValueError("width_ratio must be a tuple of (min, max)")
            self.wr_min, self.wr_max = float(width_ratio[0]), float(width_ratio[1])
            if self.wr_min <= 0:
                raise ValueError(f"width_ratio min must be positive, got {self.wr_min}")
            if self.wr_max < self.wr_min:
                raise ValueError(f"width_ratio max ({self.wr_max}) must be >= min ({self.wr_min})")
            self.wr_randomize = True
            self.width_ratio = None
        else:
            if width_ratio <= 0:
                raise ValueError(f"width_ratio must be positive, got {width_ratio}")
            self.width_ratio = float(width_ratio)
            self.wr_randomize = False

        if height_ratio is None:
            self.height_ratio = None
            self.hr_randomize = False
        elif isinstance(height_ratio, (list, tuple)):
            if len(height_ratio) != 2:
                raise ValueError("height_ratio must be a tuple of (min, max)")
            self.hr_min, self.hr_max = float(height_ratio[0]), float(height_ratio[1])
            if self.hr_min <= 0:
                raise ValueError(f"height_ratio min must be positive, got {self.hr_min}")
            if self.hr_max < self.hr_min:
                raise ValueError(f"height_ratio max ({self.hr_max}) must be >= min ({self.hr_min})")
            self.hr_randomize = True
            self.height_ratio = None
        else:
            if height_ratio <= 0:
                raise ValueError(f"height_ratio must be positive, got {height_ratio}")
            self.height_ratio = float(height_ratio)
            self.hr_randomize = False

        self.mask_aware = mask_aware
        self.min_retention = min_retention

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Crop a rectangular region from image and optionally masks.

        Args:
            image: Input tensor (C, H, W) in [0, 255] range
            localization_map: Optional ground truth masks (N_masks, H, W) in [0, 1] range

        Returns:
            AttackResult with cropped image and optionally cropped masks
        """
        _, img_h, img_w = image.shape

        # Resolve dimensions (once for both image and masks)
        if self.wr_randomize:
            ratio_w = random.uniform(self.wr_min, self.wr_max)
            w = max(1, int(round(img_w * ratio_w)))
        elif self.width_ratio is not None:
            w = max(1, int(round(img_w * self.width_ratio)))
        else:
            w = random.randint(self.w_min, self.w_max) if self.w_randomize else self.width

        if self.hr_randomize:
            ratio_h = random.uniform(self.hr_min, self.hr_max)
            h = max(1, int(round(img_h * ratio_h)))
        elif self.height_ratio is not None:
            h = max(1, int(round(img_h * self.height_ratio)))
        else:
            h = random.randint(self.h_min, self.h_max) if self.h_randomize else self.height

        # Clamp dimensions to image bounds to ensure consistent output size
        # If crop is larger than image, this will result in smaller output
        if w > img_w or h > img_h:
            raise ValueError(
                f"Crop dimensions ({w}x{h}) exceed image dimensions ({img_w}x{img_h}). "
                f"This would result in inconsistent crop sizes across batches."
            )
        w = min(w, img_w)
        h = min(h, img_h)

        # Resolve position (once for both image and masks)
        if self.mask_aware and localization_map is not None:
            from ..mask_aware_utils import find_mask_aware_region
            x, y, _, _ = find_mask_aware_region(
                localization_map, w, h, img_h, img_w,
                min_retention=self.min_retention, operation="keep"
            )
        elif self.random_location:
            max_x = img_w - w
            x = random.randint(0, max(0, max_x))
        elif self.x_randomize:
            # Use x_min and x_max, clamping to ensure valid crop
            x_max = self.x_max if self.x_max is not None else img_w - w
            x_min = self.x_min
            # Ensure bounds are valid
            x_max = min(x_max, img_w - w)
            x_min = min(x_min, x_max)
            if x_min > img_w - w:
                raise ValueError(
                    f"x_min ({x_min}) is too large for image width ({img_w}) "
                    f"with crop width ({w}). Maximum valid x is {img_w - w}."
                )
            x = random.randint(x_min, max(x_min, x_max))
        else:
            # Clamp fixed position to ensure crop fits
            if self.x > img_w - w:
                raise ValueError(
                    f"x ({self.x}) is too large for image width ({img_w}) "
                    f"with crop width ({w}). Maximum valid x is {img_w - w}."
                )
            x = self.x

        if self.mask_aware and localization_map is not None:
            pass
        elif self.random_location:
            max_y = img_h - h
            y = random.randint(0, max(0, max_y))
        elif self.y_randomize:
            # Use y_min and y_max, clamping to ensure valid crop
            y_max = self.y_max if self.y_max is not None else img_h - h
            y_min = self.y_min
            # Ensure bounds are valid
            y_max = min(y_max, img_h - h)
            y_min = min(y_min, y_max)
            if y_min > img_h - h:
                raise ValueError(
                    f"y_min ({y_min}) is too large for image height ({img_h}) "
                    f"with crop height ({h}). Maximum valid y is {img_h - h}."
                )
            y = random.randint(y_min, max(y_min, y_max))
        else:
            # Clamp fixed position to ensure crop fits
            if self.y > img_h - h:
                raise ValueError(
                    f"y ({self.y}) is too large for image height ({img_h}) "
                    f"with crop height ({h}). Maximum valid y is {img_h - h}."
                )
            y = self.y

        # Crop image
        attacked_image = image[:, y : y + h, x : x + w]

        # Crop masks if provided
        transformed_masks = None
        if localization_map is not None:
            transformed_masks = (localization_map[:, y : y + h, x : x + w] > 0.5).float()

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={"crop_x": x, "crop_y": y, "crop_w": w, "crop_h": h}
        )
