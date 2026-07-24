import random
from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("rgb_shift_trustmark")
class RGBShiftTrustMarkAttack(AbstractAttack):
    """
    TrustMark-compatible RGB channel shift attack.

    Uses kornia's RandomRGBShift wrapper which internally converts single positive values
    to symmetric ranges. For example, r_shift=0.05 becomes [-0.05, 0.05] internally.

    This matches the behavior expected by the TrustMark paper configurations.
    For precise control over exact shift amounts, use the 'rgb_shift' attack instead.
    """

    def __init__(
        self,
        r_shift: Union[float, Tuple[float, float]] = 0.05,
        g_shift: Union[float, Tuple[float, float]] = 0.05,
        b_shift: Union[float, Tuple[float, float]] = 0.05,
    ):
        """
        Args:
            r_shift: Red channel shift limit. Single positive value becomes [-value, value].
            g_shift: Green channel shift limit. Single positive value becomes [-value, value].
            b_shift: Blue channel shift limit. Single positive value becomes [-value, value].
        """
        self.r_shift = r_shift
        self.g_shift = g_shift
        self.b_shift = b_shift

        # Check if any param is random
        self._is_random = any(
            isinstance(p, (tuple, list)) for p in [r_shift, g_shift, b_shift]
        )

        # Pre-create transform if not random
        if not self._is_random:
            self._transform = aug.RandomRGBShift(
                r_shift_limit=r_shift,
                g_shift_limit=g_shift,
                b_shift_limit=b_shift,
                p=1.0,
            )
        else:
            self._transform = None

    def _resolve_param(self, param: Union[float, Tuple[float, float]]) -> float:
        """Resolve a parameter to a concrete value, randomizing if given a range."""
        if isinstance(param, (tuple, list)):
            return random.uniform(param[0], param[1])
        return param

    def _get_params(self) -> Tuple[float, float, float]:
        """Get all resolved parameters."""
        return (
            self._resolve_param(self.r_shift),
            self._resolve_param(self.g_shift),
            self._resolve_param(self.b_shift),
        )

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply RGB shift to image (masks unchanged)."""
        # kornia expects (B, C, H, W) with values in [0, 1]
        img = image.float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Use cached transform or create new one if random
        if self._is_random:
            r_shift, g_shift, b_shift = self._get_params()
            transform = aug.RandomRGBShift(
                r_shift_limit=r_shift,
                g_shift_limit=g_shift,
                b_shift_limit=b_shift,
                p=1.0,
            )
        else:
            r_shift, g_shift, b_shift = self._get_params()
            transform = self._transform

        result = transform(img)

        if result.dim() == 4:
            result = result.squeeze(0)

        attacked_image = (result * 255.0).clamp(0, 255)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={
                "r_shift": r_shift,
                "g_shift": g_shift,
                "b_shift": b_shift
            }
        )
