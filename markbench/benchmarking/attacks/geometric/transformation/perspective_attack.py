import random
from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("perspective")
class PerspectiveAttack(AbstractAttack):
    """
    Random perspective transformation attack.

    Uses kornia's RandomPerspective.
    Supports randomization by passing a tuple (min, max) for distortion_scale.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(self, distortion_scale: Union[float, Tuple[float, float]] = 0.1):
        """
        Args:
            distortion_scale: Amount of distortion (0.0-1.0). Higher = more distortion.
                             Can be a single value or tuple (min, max) for random range.
        """
        if isinstance(distortion_scale, (list, tuple)):
            self.ds_min, self.ds_max = distortion_scale
            self.randomize = True
        else:
            self.distortion_scale = distortion_scale
            self.randomize = False
            self.transform = aug.RandomPerspective(
                distortion_scale=distortion_scale, p=1.0
            )

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply perspective transformation to image and optionally masks.

        Args:
            image: Input tensor (C, H, W) in [0, 255] range
            localization_map: Optional ground truth localization maps (N_masks, H, W)

        Returns:
            AttackResult with perspective-transformed image and optionally masks
        """
        # Get distortion scale
        if self.randomize:
            ds = random.uniform(self.ds_min, self.ds_max)
            transform = aug.RandomPerspective(distortion_scale=ds, p=1.0)
        else:
            ds = self.distortion_scale
            transform = self.transform

        # Normalize image to [0, 1] for kornia
        img_norm = image.float() / 255.0  # (C, H, W)

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
            metadata={"distortion_scale": ds}
        )
