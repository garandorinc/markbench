import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import kornia.geometry.transform as K

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult
from markbench.utils.device import get_device


@register_attack("translation", aliases=["translate"])
class TranslationAttack(AbstractAttack):
    """
    Translation attack with automatic device selection.

    Uses kornia for GPU execution or OpenCV for CPU execution (faster on CPU).
    Supports randomization by passing tuples (min, max) for tx and ty.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        tx: Union[int, Tuple[int, int]] = 10,
        ty: Union[int, Tuple[int, int]] = 10,
        device: str = "",
    ):
        """
        Args:
            tx: X translation in pixels.
                Can be a single value or tuple (min, max) for random range.
            ty: Y translation in pixels.
                Can be a single value or tuple (min, max) for random range.
            device: 'cuda', 'cpu', or '' for auto-detect (default).
        """
        self.tx = tx
        self.ty = ty
        self.device = get_device(device)

    def _resolve_translation(self, translation: Union[int, Tuple[int, int]]) -> int:
        """Resolve a translation to a concrete value, randomizing if given a range."""
        if isinstance(translation, (list, tuple)):
            return random.randint(translation[0], translation[1])
        return translation

    def _get_translations(self) -> Tuple[int, int]:
        """Get the translation values for x and y, resolving any randomization."""
        tx_val = self._resolve_translation(self.tx)
        ty_val = self._resolve_translation(self.ty)
        return tx_val, ty_val

    def _apply_cpu(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """Fast CV2-based translation for CPU."""
        # Convert to numpy (H, W, C)
        img_np = image.byte().permute(1, 2, 0).cpu().numpy()
        h, w = img_np.shape[:2]

        # Get translations (once for both image and masks)
        tx_val, ty_val = self._get_translations()

        # Create translation matrix
        matrix = np.float32([[1, 0, tx_val], [0, 1, ty_val]])

        # Apply translation to image
        translated = cv2.warpAffine(img_np, matrix, (w, h))

        # Convert back to tensor (C, H, W)
        attacked_image = torch.from_numpy(translated.transpose(2, 0, 1)).float().to(image.device)

        # Apply translation to masks if provided
        transformed_masks = None
        if localization_map is not None:
            masks_list = []
            localization_map_np = localization_map.cpu().numpy()
            for mask_np in localization_map_np:
                mask_np = mask_np
                translated_mask = cv2.warpAffine(mask_np, matrix, (w, h))
                masks_list.append(torch.from_numpy(translated_mask))
            transformed_masks = torch.stack(masks_list)
            transformed_masks = (transformed_masks > 0.5).float().to(image.device)

        return attacked_image, transformed_masks, tx_val, ty_val

    def _apply_gpu(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
        """Kornia-based translation for GPU."""
        # kornia expects (B, C, H, W)
        img = image.float()
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Move to GPU if needed
        if self.device == "cuda" and not img.is_cuda:
            img = img.cuda()

        # Get translations (once for both image and masks)
        tx_val, ty_val = self._get_translations()

        # Create translation tensor (B, 2) for x and y translation
        translation = torch.tensor(
            [[tx_val, ty_val]], dtype=img.dtype, device=img.device
        )

        # Translate image using kornia
        translated = K.translate(img, translation)

        # Remove batch dimension and return
        if translated.dim() == 4:
            translated = translated.squeeze(0)

        attacked_image = translated.clamp(0, 255).to(image.device)

        # Translate masks if provided
        transformed_masks = None
        if localization_map is not None:
            masks = localization_map.float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (N, 1, H, W)
            if self.device == "cuda" and not masks.is_cuda:
                masks = masks.cuda()

            # Use same translation
            translated_masks = K.translate(masks, translation)

            if translated_masks.dim() == 4:
                translated_masks = translated_masks.squeeze(1)

            transformed_masks = (translated_masks > 0.5).float().to(image.device)

        return attacked_image, transformed_masks, tx_val, ty_val

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply translation attack to image and optionally masks.

        Args:
            image: Input tensor (C, H, W) in [0, 255] range.
            localization_map: Optional ground truth localization maps (N_masks, H, W)

        Returns:
            AttackResult containing translated image and optionally translated masks.
        """
        if self.device == "cpu":
            attacked_image, transformed_masks, tx_val, ty_val = self._apply_cpu(image, localization_map)
        else:
            attacked_image, transformed_masks, tx_val, ty_val = self._apply_gpu(image, localization_map)

        return AttackResult(
            attacked=attacked_image,
            localization_map=(transformed_masks > 0.5).float() if transformed_masks is not None else None,
            metadata={"tx": tx_val, "ty": ty_val}
        )
