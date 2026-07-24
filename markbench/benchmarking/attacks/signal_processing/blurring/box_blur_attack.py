import random
from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("box_blur")
class BoxBlurAttack(AbstractAttack):
    """
    Box blur (averaging filter) attack.

    Uses kornia's RandomBoxBlur.
    """

    def __init__(self, kernel_size: Union[int, Tuple[int, int]] = 5):
        """
        Args:
            kernel_size: Size of the blur kernel. Can be:
                - int: fixed kernel size
                - tuple (h, w): fixed height/width kernel (if h == w, treated as fixed)
                - tuple (min, max) for random range (if min != max)
        """
        self.kernel_size = kernel_size
        self._is_random = (
            isinstance(kernel_size, (tuple, list)) and kernel_size[0] != kernel_size[1]
        )

        # Pre-create transform if not random
        if not self._is_random:
            resolved = self._resolve_kernel_size()
            self._transform = aug.RandomBoxBlur(kernel_size=resolved, p=1.0)
        else:
            self._transform = None

    def _resolve_kernel_size(self) -> Tuple[int, int]:
        """Resolve kernel_size to a concrete value, randomizing if given a range."""
        if isinstance(self.kernel_size, (tuple, list)):
            # If both values are the same, treat as fixed (h, w) kernel
            if self.kernel_size[0] == self.kernel_size[1]:
                k = self.kernel_size[0]
            else:
                # Treat as (min, max) range
                k = random.randint(self.kernel_size[0], self.kernel_size[1])
            # Ensure odd
            k = k if k % 2 == 1 else k + 1
            return (k, k)
        else:
            k = self.kernel_size
            k = k if k % 2 == 1 else k + 1
            return (k, k)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply box blur to image (masks unchanged)."""
        # kornia expects (B, C, H, W) with values in [0, 1]
        img = image.float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Use cached transform or create new one if random
        if self._is_random:
            kernel_size = self._resolve_kernel_size()
            transform = aug.RandomBoxBlur(kernel_size=kernel_size, p=1.0)
        else:
            kernel_size = self._resolve_kernel_size()
            transform = self._transform

        result = transform(img)

        if result.dim() == 4:
            result = result.squeeze(0)

        attacked_image = (result * 255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"kernel_size": kernel_size}
        )
