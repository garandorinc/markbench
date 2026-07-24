import random
from typing import Optional, Tuple, Union

import cv2
import torch
import kornia.filters

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult
from markbench.utils.device import get_device


@register_attack("median_filter")
class MedianFilterAttack(AbstractAttack):
    """
    Median filter attack with automatic device selection.

    Uses kornia for GPU execution or OpenCV for CPU execution (42x faster on CPU).
    """

    def __init__(self, ksize: Union[int, Tuple[int, int]] = 3, device: str = ""):
        """
        Args:
            ksize: Kernel size for median filter. Must be odd.
                   Can be int or tuple (min, max) for random range.
            device: 'cuda', 'cpu', or '' for auto-detect (default).
        """
        self.ksize = ksize
        self.device = get_device(device)

    def _resolve_ksize(self) -> int:
        """Resolve ksize to a concrete value, randomizing if given a range."""
        if isinstance(self.ksize, (tuple, list)):
            k = random.randint(self.ksize[0], self.ksize[1])
        else:
            k = self.ksize
        # Ensure odd
        return k if k % 2 == 1 else k + 1

    def _apply_cpu(self, image: torch.Tensor) -> torch.Tensor:
        """Fast CV2-based median filter for CPU."""
        # Convert to numpy (H, W, C) uint8
        img_np = image.byte().permute(1, 2, 0).cpu().numpy()

        # Get kernel size
        ksize = self._resolve_ksize()

        # Apply median blur
        filtered = cv2.medianBlur(img_np, ksize)

        # Convert back to tensor (C, H, W)
        return torch.from_numpy(filtered.transpose(2, 0, 1)).float().to(image.device)

    def _apply_gpu(self, image: torch.Tensor) -> torch.Tensor:
        """Kornia-based median filter for GPU."""
        img = image.float() / 255.0

        # Add batch dimension if needed
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Move to GPU if needed
        if self.device == "cuda" and not img.is_cuda:
            img = img.cuda()

        # Get ksize (possibly random)
        ksize = self._resolve_ksize()

        # Apply median blur using kornia
        filtered = kornia.filters.median_blur(img, (ksize, ksize))

        # Remove batch dimension
        if filtered.dim() == 4:
            filtered = filtered.squeeze(0)

        return (filtered * 255.0).clamp(0, 255).to(image.device)

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply median filter to image (masks unchanged)."""
        # Get ksize before applying filter
        ksize = self._resolve_ksize()

        if self.device == "cpu":
            attacked_image = self._apply_cpu(image)
        else:
            attacked_image = self._apply_gpu(image)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"ksize": ksize}
        )
