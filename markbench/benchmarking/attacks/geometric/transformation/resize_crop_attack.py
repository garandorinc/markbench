import random
from typing import Optional, Tuple

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("resize_crop", aliases=["rc"])
class ResizeCropAttack(AbstractAttack):
    """
    Random resized crop attack.

    Randomly crops a region and resizes it back to the original size.
    This simulates zoom-in/zoom-out with different aspect ratios.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        scale: Tuple[float, float] = (0.75, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        mask_aware: bool = False,
        min_retention: float = 0.25,
    ):
        """
        Args:
            scale: Range of size of the crop relative to original image (min, max)
            ratio: Range of aspect ratio of the crop (min, max)
        """
        self.scale = scale
        self.ratio = ratio
        self.mask_aware = mask_aware
        self.min_retention = min_retention

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply random resized crop to image and optionally masks.

        Args:
            image: Input tensor (C, H, W) in [0, 255] range
            localization_map: Optional ground truth localization maps (N_masks, H, W)

        Returns:
            AttackResult with cropped and resized image (same size as input)
        """
        # Normalize image to [0, 1] for kornia
        img_norm = image.float() / 255.0  # (C, H, W)
        h, w = img_norm.shape[-2:]

        if self.mask_aware and localization_map is not None:
            # Generate crop dimensions
            area = h * w
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            crop_h = int(round((target_area / aspect_ratio) ** 0.5))
            crop_w = int(round(crop_h * aspect_ratio))
            crop_h = min(crop_h, h)
            crop_w = min(crop_w, w)

            from ...mask_aware_utils import find_mask_aware_region
            x, y, _, _ = find_mask_aware_region(
                localization_map, crop_w, crop_h, h, w,
                min_retention=self.min_retention, operation="keep"
            )

            # Crop and resize
            cropped = img_norm[:, y:y+crop_h, x:x+crop_w]
            attacked_image = torch.nn.functional.interpolate(
                cropped.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0) * 255.0

            if localization_map is not None:
                cropped_mask = localization_map[:, y:y+crop_h, x:x+crop_w]
                transformed_masks = torch.nn.functional.interpolate(
                    cropped_mask.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
                ).squeeze(0)
                transformed_masks = (transformed_masks > 0.5).float()
            else:
                transformed_masks = None
        else:
            transform = aug.RandomResizedCrop(
                size=(h, w), scale=self.scale, ratio=self.ratio, p=1.0
            )

            if localization_map is not None:
                # Concatenate image and masks along channel dimension
                masks_norm = localization_map.float()  # (N, H, W)
                combined = torch.cat([img_norm, masks_norm], dim=0)  # (C+N, H, W)
                combined = combined.unsqueeze(0)  # (1, C+N, H, W)

                # Apply transformation once to both
                transformed = transform(combined).squeeze(0)  # (C+N, H', W')

                # Split back
                attacked_image = (transformed[:3] * 255.0).clamp(0, 255)  # (3, H', W')
                transformed_masks = (transformed[3:] > 0.5).float()  # (N, H', W')
            else:
                # Transform image only
                img_batch = img_norm.unsqueeze(0)  # (1, C, H, W)
                result = transform(img_batch).squeeze(0)  # (C, H', W')
                attacked_image = (result * 255.0).clamp(0, 255)
                transformed_masks = None

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={"scale": self.scale, "ratio": self.ratio}
        )
