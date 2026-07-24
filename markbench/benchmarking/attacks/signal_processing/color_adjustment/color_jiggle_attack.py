import random
from typing import Optional, Tuple, Union

import torch
from kornia import enhance

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("color_jiggle", aliases=["jiggle"])
class ColorJiggleAttack(AbstractAttack):
    """
    Combined color jitter attack.

    Randomly adjusts brightness, contrast, saturation, and hue simultaneously.
    Uses kornia's enhance module for GPU-accelerated processing.
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0.0,
        contrast: Union[float, Tuple[float, float]] = 0.0,
        saturation: Union[float, Tuple[float, float]] = 0.0,
        hue: Union[float, Tuple[float, float]] = 0.0,
    ):
        """
        Args:
            brightness: Brightness adjustment factor. Can be float or tuple (min, max) for random range.
                        Factor of 0 means no change. Negative values darken, positive brighten.
            contrast: Contrast adjustment factor. Can be float or tuple (min, max) for random range.
                      Factor of 1 means no change. <1 reduces contrast, >1 increases contrast.
            saturation: Saturation adjustment factor. Can be float or tuple (min, max) for random range.
                        Factor of 1 means no change. 0 gives grayscale, >1 increases saturation.
            hue: Hue shift in radians [-π, π]. Can be float or tuple (min, max) for random range.
                 0 means no shift. Negative and positive shift hue in opposite directions.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

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

        # Resolve parameters to concrete values
        brightness, contrast, saturation, hue = self._get_params()

        # Apply adjustments using kornia.enhance functions directly
        result = img

        # Brightness: 0 = no change, negative = darken, positive = brighten
        if brightness != 0.0:
            result = enhance.adjust_brightness(result, brightness)

        # Contrast: 1 = no change, <1 = reduce, >1 = increase
        if contrast != 0.0:
            result = enhance.adjust_contrast(result, 1.0 + contrast)

        # Saturation: 1 = no change, 0 = grayscale, >1 = more saturated
        if saturation != 0.0:
            result = enhance.adjust_saturation(result, 1.0 + saturation)

        # Hue: 0 = no shift, [-π, π] for shift range
        if hue != 0.0:
            result = enhance.adjust_hue(result, hue)

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
