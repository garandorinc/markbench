import random
from typing import Optional, Tuple, Union

import torch
import kornia.augmentation as aug

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("color_jiggle_trustmark")
class ColorJiggleTrustMarkAttack(AbstractAttack):
    """
    TrustMark-compatible color jitter attack.

    Uses kornia's ColorJiggle wrapper which internally converts single positive values
    to symmetric ranges. For example, hue=0.05 becomes [-0.05, 0.05] internally.

    This matches the behavior expected by the TrustMark paper configurations.
    For precise control over exact shift amounts, use the 'color_jiggle' attack instead.
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.1,
        contrast: Union[float, Tuple[float, float]] = 0.1,
        saturation: Union[float, Tuple[float, float]] = 0.1,
        hue: Union[float, Tuple[float, float]] = 0.02,
    ):
        """
        Args:
            brightness: Brightness jitter amount. Single positive value becomes [-value, value].
            contrast: Contrast jitter amount. Single positive value becomes [-value, value].
            saturation: Saturation jitter amount. Single positive value becomes [-value, value].
            hue: Hue jitter amount (0.0-0.5). Single positive value becomes [-value, value].
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # Check if any param is random
        self._is_random = any(
            isinstance(p, (tuple, list))
            for p in [brightness, contrast, saturation, hue]
        )

        # Pre-create transform if not random
        if not self._is_random:
            self._transform = aug.ColorJiggle(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=1.0,
            )
        else:
            self._transform = None

    def _resolve_param(self, param: Union[float, Tuple[float, float]]) -> float:
        """Resolve a parameter to a concrete value, randomizing if given a range."""
        if isinstance(param, (tuple, list)):
            return random.uniform(param[0], param[1])
        return param

    def _get_params(self) -> Tuple[float, float, float, float]:
        """Get all resolved parameters."""
        return (
            self._resolve_param(self.brightness),
            self._resolve_param(self.contrast),
            self._resolve_param(self.saturation),
            self._resolve_param(self.hue),
        )

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply color jiggle to image (masks unchanged)."""
        # kornia expects (B, C, H, W) with values in [0, 1]
        img = image.float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Use cached transform or create new one if random
        if self._is_random:
            brightness, contrast, saturation, hue = self._get_params()
            transform = aug.ColorJiggle(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=1.0,
            )
        else:
            brightness, contrast, saturation, hue = self._get_params()
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
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue
            }
        )
