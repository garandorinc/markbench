import cv2
import numpy as np
import torch
import tempfile
import os
from typing import Optional

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("mpeg", aliases=["mpeg_compression"])
class MPEGCompressionAttack(AbstractAttack):
    """
    MPEG Compression Attack using OpenCV's video encoder.

    Encodes image as single-frame video and decodes back, providing
    authentic MPEG compression artifacts.
    """

    def __init__(self, codec: str = "MJPEG", quality: int = 50, fps: int = 1):
        """
        Args:
            codec: Video codec ('MJPEG', 'MPEG', 'MP4V', 'XVID', 'DIVX', 'H264')
            quality: Compression quality (1-100, lower = more compression)
            fps: Frames per second for temporary video
        """
        self.codec = codec
        self.quality = quality
        self.fps = fps
        self._validate_codec()

    def _validate_codec(self):
        """Validate codec availability."""
        codec_map = {
            "MJPEG": cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            "MPEG": cv2.VideoWriter_fourcc("M", "P", "E", "G"),
            "MP4V": cv2.VideoWriter_fourcc("M", "P", "4", "V"),
            "XVID": cv2.VideoWriter_fourcc("X", "V", "I", "D"),
            "DIVX": cv2.VideoWriter_fourcc("D", "I", "V", "X"),
            "H264": cv2.VideoWriter_fourcc("H", "2", "6", "4"),
        }

        if self.codec not in codec_map:
            raise ValueError(
                f"Unsupported codec: {self.codec}. Available: {list(codec_map.keys())}"
            )

        self.fourcc = codec_map[self.codec]

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply MPEG compression to image.

        Note: Video codecs may change dimensions (e.g., rounding to multiples of 2/4/8/16).
        If dimensions change, localization_map is cropped to match.
        """
        # Convert to numpy HWC format
        img_np = image.byte().permute(1, 2, 0).cpu().numpy()

        # Apply compression
        result = self._compress(img_np)

        # Convert back to tensor CHW format
        attacked_image = torch.from_numpy(result.transpose(2, 0, 1)).float().to(image.device)

        # Handle dimension changes - crop masks if needed
        transformed_masks = None
        if localization_map is not None:
            h_out, w_out = attacked_image.shape[1], attacked_image.shape[2]
            h_in, w_in = localization_map.shape[1], localization_map.shape[2]

            if (h_out, w_out) != (h_in, w_in):
                # Codec changed dimensions - crop masks to match
                transformed_masks = localization_map[:, :h_out, :w_out]
            else:
                # Dimensions unchanged
                transformed_masks = localization_map

        return AttackResult(
            attacked=attacked_image,
            localization_map=transformed_masks,
            metadata={
                "codec": self.codec,
                "quality": self.quality,
                "fps": self.fps,
                "dimension_changed": (attacked_image.shape[1], attacked_image.shape[2]) != (image.shape[1], image.shape[2])
            }
        )

    def _compress(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply MPEG compression using video encoder."""
        height, width = img_rgb.shape[:2]

        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temp_file:
            temp_video_path = temp_file.name

        try:
            # Create video writer
            writer = cv2.VideoWriter(
                temp_video_path, self.fourcc, self.fps, (width, height), isColor=True
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Failed to create video writer with codec {self.codec}"
                )

            try:
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, self.quality)
            except (AttributeError, cv2.error):
                # Some codecs may not support quality setting
                pass

            # Write frame (convert RGB to BGR for OpenCV)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)
            writer.release()

            # Read back compressed frame
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                raise RuntimeError("Failed to open temporary video file")

            ret, compressed_bgr = cap.read()
            cap.release()

            if not ret or compressed_bgr is None:
                raise RuntimeError("Failed to read compressed frame")

            # Convert back to RGB
            return cv2.cvtColor(compressed_bgr, cv2.COLOR_BGR2RGB)

        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
