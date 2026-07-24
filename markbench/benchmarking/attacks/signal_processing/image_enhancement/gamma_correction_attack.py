import random
from typing import Optional, Tuple, Union

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("gamma_correction", aliases=["gamma"])
class GammaCorrectionAttack(AbstractAttack):
    """
    Apply gamma correction to image.

    Gamma correction: output = input^gamma
    - gamma > 1: darkens the image
    - gamma < 1: brightens the image
    - gamma = 1: no change

    Supports randomization by passing tuples (min, max) for gamma parameter.
    """

    def __init__(
        self, gamma: Union[float, Tuple[float, float]] = 1.0, gain: float = 1.0
    ):
        """
        Args:
            gamma: Gamma correction factor.
                   Can be float or tuple (min, max) for random range.
            gain: Constant multiplier for the result.
        """
        self.gamma = gamma
        self.gain = gain

    def _resolve_gamma(self) -> float:
        """Resolve gamma to a concrete value, randomizing if given a range."""
        if isinstance(self.gamma, (list, tuple)):
            return random.uniform(self.gamma[0], self.gamma[1])
        return self.gamma

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply gamma correction to image (masks unchanged)."""
        gamma = self._resolve_gamma()

        # Normalize to [0, 1], apply gamma, scale back to [0, 255]
        # output = gain * (input/255)^gamma * 255
        normalized = image.float() / 255.0
        corrected = self.gain * torch.pow(normalized, gamma)

        attacked_image = (corrected * 255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"gamma": gamma, "gain": self.gain}
        )
