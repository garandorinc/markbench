import io
from typing import Optional

import torch
import numpy as np
from PIL import Image as PILImage

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("gif", aliases=["gif_compression"])
class GIFCompressionAttack(AbstractAttack):
    """GIF compression attack using PIL GIF encode/decode."""

    def __init__(self):
        pass

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply GIF compression to image (masks unchanged)."""
        # Convert to numpy RGB (H, W, C)
        img_np = image.byte().permute(1, 2, 0).cpu().numpy()

        # Convert to PIL, apply GIF compression via encode/decode
        pil_img = PILImage.fromarray(img_np)

        # Encode to GIF bytes
        buffer = io.BytesIO()
        pil_img.save(buffer, format="GIF", optimize=True)
        buffer.seek(0)

        # Decode from GIF bytes
        gif_img = PILImage.open(buffer).convert("RGB")
        result = np.array(gif_img, dtype=np.uint8)

        # Convert back to torch tensor (C, H, W)
        attacked_image = torch.from_numpy(result.transpose(2, 0, 1)).float().to(image.device)

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={}
        )
