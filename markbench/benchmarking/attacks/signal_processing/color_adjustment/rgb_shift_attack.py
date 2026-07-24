import random
from typing import Optional, Tuple, Union

import torch
from kornia import enhance

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("rgb_shift")
class RGBShiftAttack(AbstractAttack):
    """
    RGB channel shift attack.

    Independently shifts R, G, B channel values by specified amounts.
    """

    def __init__(
        self,
        r_shift: Union[float, Tuple[float, float]] = 0.0,
        g_shift: Union[float, Tuple[float, float]] = 0.0,
        b_shift: Union[float, Tuple[float, float]] = 0.0,
    ):
        """
        Args:
            r_shift: Shift for red channel in [0, 1] range. Can be float or tuple (min, max) for random range.
                     Positive values increase red, negative values decrease red.
            g_shift: Shift for green channel in [0, 1] range. Can be float or tuple (min, max) for random range.
                     Positive values increase green, negative values decrease green.
            b_shift: Shift for blue channel in [0, 1] range. Can be float or tuple (min, max) for random range.
                     Positive values increase blue, negative values decrease blue.
        """
        self.r_shift = r_shift
        self.g_shift = g_shift
        self.b_shift = b_shift

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
        # Get shift values
        r_shift, g_shift, b_shift = self._get_params()

        # Work in [0, 1] range for shifting
        img = image.float() / 255.0  # (C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, C, H, W)

        # Convert shift values to tensors
        r_shift_t = torch.tensor([r_shift], dtype=img.dtype, device=img.device)
        g_shift_t = torch.tensor([g_shift], dtype=img.dtype, device=img.device)
        b_shift_t = torch.tensor([b_shift], dtype=img.dtype, device=img.device)

        result = enhance.shift_rgb(img, r_shift_t, g_shift_t, b_shift_t)

        # Remove batch dimension
        if result.dim() == 4:
            result = result.squeeze(0)

        # Convert back to [0, 255]
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
