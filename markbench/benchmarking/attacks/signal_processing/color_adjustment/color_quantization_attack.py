"""Color quantization attack using k-means clustering."""

import random
from typing import Optional, Tuple, Union

import torch

from ...abstract_attack import AbstractAttack
from ...registry import register_attack
from markbench.interface import AttackResult


@register_attack("color_quantization")
class ColorQuantizationAttack(AbstractAttack):
    """Reduce colors using k-means clustering."""

    def __init__(self, k: Union[int, Tuple[int, int]] = 8):
        """
        Args:
            k: Number of colors to quantize to.
               Can be int or tuple (min, max) for random range.
        """
        self.k = k

    def _resolve_k(self) -> int:
        """Resolve k to a concrete value, randomizing if given a range."""
        if isinstance(self.k, (tuple, list)):
            return random.randint(self.k[0], self.k[1])
        return self.k

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply color quantization to image (masks unchanged)."""
        img = image.float()
        h, w = img.shape[1], img.shape[2]

        # Get k (possibly random)
        k = self._resolve_k()

        # Reshape to (H*W, C) for clustering
        pixels = img.view(3, -1).t()  # (H*W, 3)

        # Initialize random centroids
        indices = torch.randperm(pixels.shape[0])[:k]
        centroids = pixels[indices].clone()

        # Run k-means iterations
        for _ in range(10):
            distances = torch.cdist(pixels, centroids)
            labels = torch.argmin(distances, dim=1)

            for i in range(k):
                mask = labels == i
                if torch.any(mask):
                    centroids[i] = pixels[mask].mean(dim=0)

        # Assign each pixel to its quantized color
        quantized_pixels = centroids[labels]

        attacked_image = quantized_pixels.t().view(3, h, w)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"k": k}
        )
