import random
from typing import Optional, Tuple, Union

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("brightness")
class AdjustBrightnessAttack(AbstractAttack):
    """
    Brightness adjustment attack using direct multiplication.

    Supports randomization by passing tuples (min, max) for factor parameter.
    """

    def __init__(self, factor: Union[float, Tuple[float, float]] = 1.2):
        """
        Args:
            factor: Brightness factor (0=black, 1=original, >1=brighter).
                    Can be float or tuple (min, max) for random range.
        """
        self.factor = factor

    def _resolve_factor(self) -> float:
        """Resolve factor to a concrete value, randomizing if given a range."""
        if isinstance(self.factor, (list, tuple)):
            return random.uniform(self.factor[0], self.factor[1])
        return self.factor

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply brightness adjustment to image (masks unchanged)."""
        factor = self._resolve_factor()
        # Direct multiplication - no normalization needed
        attacked_image = (image * factor).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"factor": factor}
        )
