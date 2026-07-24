import cv2
import torch
from typing import Optional

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("jpeg2000", aliases=["jp2", "jpeg_2000", "jpeg_2000_compression"])
class JPEG2000CompressionAttack(AbstractAttack):
    """JPEG2000 compression attack with configurable compression ratio."""

    def __init__(self, compression_ratio: int = 200):
        """Initialize the attack.

        Args:
            compression_ratio: JPEG2000 compression ratio (higher = more compression)
        """
        self.compression_ratio = compression_ratio

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None,
    ) -> AttackResult:
        """Apply JPEG2000 compression to image (masks unchanged)."""
        attacked_image = image.float()

        # Convert to numpy HWC format (BGR for cv2)
        img_np = image.byte().permute(1, 2, 0).cpu().numpy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Apply JPEG2000 compression
        encode_param = [
            cv2.IMWRITE_JPEG2000_COMPRESSION_X1000,
            int(self.compression_ratio),
        ]
        success, encoded_img = cv2.imencode(".jp2", img_bgr, encode_param)

        if success:
            result = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            if result is not None:
                # Convert BGR back to RGB
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                attacked_image = (
                    torch.from_numpy(result_rgb.transpose(2, 0, 1))
                    .float()
                    .to(image.device)
                )

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=attacked_image,
            localization_map=localization_map,
            metadata={"compression_ratio": self.compression_ratio, "success": bool(success)},
        )
