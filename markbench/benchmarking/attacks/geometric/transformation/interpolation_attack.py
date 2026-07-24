import random
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("interpolation")
class InterpolationAttack(AbstractAttack):
    """
    Unified interpolation/scaling attack using PyTorch F.interpolate.

    Supports two modes:
    1. Scaling mode: Use fx, fy scale factors
    2. Absolute mode: Use new_width, new_height for exact dimensions
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        fx: Optional[Union[float, Tuple[float, float]]] = None,
        fy: Optional[Union[float, Tuple[float, float]]] = None,
        new_width: Optional[int] = None,
        new_height: Optional[int] = None,
        interpolation_mode: str = "linear",
    ):
        """
        Args:
            fx: Scale factor for width (1.0 = no change).
                Can be float or tuple (min, max) for random range.
            fy: Scale factor for height (1.0 = no change).
                Can be float or tuple (min, max) for random range.
            new_width: Fixed width for absolute resizing
            new_height: Fixed height for absolute resizing
            interpolation_mode: Interpolation method ('nearest', 'linear', 'bicubic')

        Note: Either use (fx, fy) for scaling OR (new_width, new_height) for absolute resizing
        """
        if (fx is not None or fy is not None) and (
            new_width is not None or new_height is not None
        ):
            raise ValueError(
                "Cannot specify both scale factors and absolute dimensions"
            )

        if (fx is None and fy is None) and (new_width is None and new_height is None):
            raise ValueError(
                "Must specify either scale factors (fx, fy) or absolute dimensions"
            )

        self.use_scaling = fx is not None or fy is not None
        self.fx = fx if fx is not None else 1.0
        self.fy = fy if fy is not None else 1.0
        self.new_width = new_width
        self.new_height = new_height
        self.interpolation_mode = interpolation_mode.lower()

        valid_modes = {"nearest", "linear", "bicubic"}
        if self.interpolation_mode not in valid_modes:
            raise ValueError(f"interpolation_mode must be one of {valid_modes}")

        self.mode_map = {
            "nearest": "nearest",
            "linear": "bilinear",
            "bicubic": "bicubic",
        }

    def _resolve_fx(self) -> float:
        """Resolve fx to a concrete value, randomizing if given a range."""
        if isinstance(self.fx, (tuple, list)):
            return random.uniform(self.fx[0], self.fx[1])
        return float(self.fx)

    def _resolve_fy(self) -> float:
        """Resolve fy to a concrete value, randomizing if given a range."""
        if isinstance(self.fy, (tuple, list)):
            return random.uniform(self.fy[0], self.fy[1])
        return float(self.fy)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Resize image and optionally masks to new dimensions."""
        img = image.float()

        # Add batch dimension if needed (C, H, W) -> (1, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        _, _, h, w = img.shape

        # Resolve new dimensions (once for both image and masks)
        if self.use_scaling:
            fx = self._resolve_fx()
            fy = self._resolve_fy()
            new_height = max(1, int(h * fy))
            new_width = max(1, int(w * fx))
        else:
            fx = None
            fy = None
            new_height = self.new_height if self.new_height is not None else h
            new_width = self.new_width if self.new_width is not None else w

        pytorch_mode = self.mode_map[self.interpolation_mode]

        resize_kwargs = {"size": (new_height, new_width), "mode": pytorch_mode}

        # Only add align_corners for modes that support it
        if pytorch_mode != "nearest":
            resize_kwargs["align_corners"] = False

        # Resize image
        resized = F.interpolate(img, **resize_kwargs)

        # Remove batch dimension
        if resized.dim() == 4:
            resized = resized.squeeze(0)

        attacked_image = resized

        # Resize masks if provided
        transformed_masks = None
        if localization_map is not None:
            masks = localization_map.float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (N, 1, H, W)

            resized_masks = F.interpolate(masks, **resize_kwargs)

            if resized_masks.dim() == 4:
                resized_masks = resized_masks.squeeze(1)

            transformed_masks = (resized_masks > 0.5).float()

        # Build metadata
        metadata = {
            "new_height": new_height,
            "new_width": new_width,
            "interpolation_mode": self.interpolation_mode
        }
        if fx is not None:
            metadata["fx"] = fx
            metadata["fy"] = fy

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata=metadata
        )

    @classmethod
    def scale_by_factor(cls, fx: float, fy: float, interpolation_mode: str = "linear"):
        """Convenience method for scaling by factors"""
        return cls(fx=fx, fy=fy, interpolation_mode=interpolation_mode)

    @classmethod
    def resize_absolute(
        cls, new_width: int, new_height: int, interpolation_mode: str = "linear"
    ):
        """Convenience method for absolute resizing"""
        return cls(
            new_width=new_width,
            new_height=new_height,
            interpolation_mode=interpolation_mode,
        )


# Alias for backwards compatibility
ScalingAttack = InterpolationAttack
