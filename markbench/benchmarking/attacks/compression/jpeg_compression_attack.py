from typing import Optional, Tuple, Union

import torch
from torchvision.transforms import v2

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("jpeg", aliases=["jpeg_compression"])
class JPEGCompressionAttack(AbstractAttack):
    """
    JPEG compression attack using torchvision.

    Uses v2.JPEG's built-in randomization when quality is a tuple (min, max).
    """

    def __init__(self, quality: Union[int, Tuple[int, int]] = 50):
        """
        Args:
            quality: JPEG quality (1-100). Lower = more compression.
                     Can be a single value or tuple (min, max) for random range.
                     v2.JPEG handles the randomization internally.
        """
        if isinstance(quality, (list, tuple)):
            # v2.JPEG accepts [min, max] for built-in randomization
            self.quality = list(quality)
        else:
            # Fixed quality
            self.quality = [quality, quality]

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply JPEG compression to image (masks unchanged)."""
        # torchvision JPEG expects uint8 on CPU with shape [..., C, H, W]
        img = image.byte().cpu()

        # v2.JPEG handles randomization internally when given [min, max]
        jpeg = v2.JPEG(quality=self.quality)
        compressed = jpeg(img)

        attacked_image = compressed.float().to(image.device)

        # JPEG compression is shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,  # Pass through unchanged
            metadata={"quality_range": self.quality}
        )
