import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("quantization", aliases=["quant"])
class QuantizationAttack(AbstractAttack):
    """
    Reduces image precision by quantizing pixel values.

    For color/posterization effects, use PosterizationAttack instead.
    """

    def __init__(self, quantization_level: Union[int, Tuple[int, int]] = 4):
        """
        Args:
            quantization_level: Quantization step size (1-255).
                               Can be int or tuple (min, max) for random range.
                               Higher values = more aggressive quantization.
                               Example: level=4 → ~64 quantization levels per channel,
                                        level=32 → ~8 levels per channel.
                               Note: Total distinct RGB colors = levels³ (e.g., 64³ ≈ 262K colors).
        """
        if isinstance(quantization_level, (list, tuple)):
            self.level_min, self.level_max = quantization_level
            self.level_min = max(1, min(255, self.level_min))
            self.level_max = max(1, min(255, self.level_max))
            self.randomize = True
        else:
            self.quantization_level = max(1, min(255, quantization_level))
            self.randomize = False

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply quantization to image (masks unchanged)."""
        # Get quantization level (possibly random)
        level = (
            random.randint(self.level_min, self.level_max)
            if self.randomize
            else self.quantization_level
        )

        quantized = torch.round(image.float() / level) * level
        attacked_image = torch.clamp(quantized, 0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"quantization_level": level}
        )
