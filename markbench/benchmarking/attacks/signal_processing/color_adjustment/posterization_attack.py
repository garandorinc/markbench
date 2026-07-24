import random
from typing import Optional, Tuple, Union

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("posterization", aliases=["posterize"])
class PosterizationAttack(AbstractAttack):
    """Reduce number of bits per color channel (posterization effect).

    Implements similar algorithm as kornia.enhance.posterize, using bit-shift
    operations, but optimized for [0, 255] input range.
    """

    def __init__(self, bits: Union[int, Tuple[int, int]] = 4):
        """
        Args:
            bits: Number of bits to keep per channel (1-8).
                  Can be int or tuple (min, max) for random range.
                  Lower values = more posterized (fewer colors).
                  bits=1 gives 2 colors per channel (8 total colors)
                  bits=4 gives 16 colors per channel
                  bits=8 gives full 256 colors (no effect)
        """
        if isinstance(bits, (list, tuple)):
            self.bits_min, self.bits_max = bits
            if not 1 <= self.bits_min <= 8 or not 1 <= self.bits_max <= 8:
                raise ValueError(
                    f"bits must be between 1 and 8, got ({self.bits_min}, {self.bits_max})"
                )
            self.randomize = True
        else:
            if not 1 <= bits <= 8:
                raise ValueError(f"bits must be between 1 and 8, got {bits}")
            self.bits = bits
            self.randomize = False

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply posterization to image (masks unchanged)."""
        # Get bits value (possibly random)
        bits = (
            random.randint(self.bits_min, self.bits_max)
            if self.randomize
            else self.bits
        )

        # Handle edge cases
        if bits == 0:
            attacked_image = torch.zeros_like(image)
        elif bits == 8:
            attacked_image = image.clone()
        else:
            # Exact kornia algorithm: bit-shift operations
            # shift = 8 - bits (number of bits to remove)
            shift = 8 - bits

            # Convert to uint8, right-shift to remove lower bits, left-shift back
            # For input in [0, 255] range:
            img_uint8 = image.to(torch.uint8)

            # Right shift: divide by 2^shift (integer division truncates)
            shifted_right = img_uint8 >> shift

            # Left shift: multiply by 2^shift
            shifted_back = shifted_right << shift

            # Convert back to original dtype
            attacked_image = shifted_back.to(image.dtype)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"bits": bits}
        )
