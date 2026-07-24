"""Minimal example adapter for the ``WatermarkModel`` interface.

This is a toy method. It shows the tensor shapes that the runner sends and
expects, and it lets you run the benchmark end to end without model weights.
It is not robust: any attack that changes pixel values or geometry destroys it.

Embedding writes two parity signals inside the mask:

- red channel: a fixed pseudo-random sync pattern, used to find the region;
- blue channel: the message bits, tiled over the region.

Run it with::

    markbench run --path /path/to/images \\
        --model examples.parity_adapter:build_model \\
        --attacks "none,jpeg:75" --jsonl_only True
"""

import time

import torch
import torch.nn.functional as F

from markbench.interface import DecodeResult, EmbedResult, WatermarkData, WatermarkModel


def _set_parity(values: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Round ``values`` to integers in [0, 255] whose parity equals ``bits``."""
    values = values.round().clamp(0, 255)
    wrong = (values % 2) != bits
    step = torch.where(values >= 255, -1.0, 1.0)
    return torch.where(wrong, values + step, values)


class ParityWatermark(WatermarkModel):
    def __init__(self, device: str = "cpu", nbits: int = 32, window: int = 17, threshold: float = 0.75):
        self.device = device
        self._nbits = nbits
        self.window = window
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "parity_example"

    @property
    def nbits(self) -> int:
        return self._nbits

    def _sync_pattern(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator().manual_seed(1234)
        pattern = torch.randint(0, 2, (height, width), generator=generator)
        return pattern.to(device=device, dtype=torch.float32)

    def _bit_index(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        return (torch.arange(height * width, device=device) % self._nbits).view(height, width)

    def embed(self, image: torch.Tensor, watermark_data: WatermarkData) -> EmbedResult:
        # image: (B, 3, H, W) in [0, 255]
        # watermark_data.message: (B, n_masks, nbits), values in {0, 1}
        # watermark_data.localization_map: (B, n_masks, H, W), values in {0, 1}
        t0 = time.time()
        _, _, height, width = image.shape
        mask = watermark_data.localization_map[:, 0] > 0.5  # (B, H, W)
        message = watermark_data.message[:, 0].to(image.device)  # (B, nbits)

        sync = self._sync_pattern(height, width, image.device).expand_as(mask)
        bit_index = self._bit_index(height, width, image.device)
        message_plane = message[:, bit_index].to(image.dtype)  # (B, H, W)

        watermarked = image.clone()
        watermarked[:, 0] = torch.where(mask, _set_parity(image[:, 0], sync), image[:, 0])
        watermarked[:, 2] = torch.where(mask, _set_parity(image[:, 2], message_plane), image[:, 2])

        return EmbedResult(
            watermarked=watermarked,
            watermark_data=watermark_data,
            embed_time_ms=(time.time() - t0) * 1000,
        )

    def decode(self, image: torch.Tensor) -> DecodeResult:
        # image: (B, 3, H, W) in [0, 255]. H and W can differ from the embed size.
        t0 = time.time()
        batch_size, _, height, width = image.shape
        sync = self._sync_pattern(height, width, image.device)
        bit_index = self._bit_index(height, width, image.device).flatten()

        # Localization: fraction of pixels in a window whose red parity agrees with the sync pattern.
        agreement = ((image[:, 0].round() % 2) == sync).float().unsqueeze(1)  # (B, 1, H, W)
        local_agreement = F.avg_pool2d(agreement, self.window, stride=1, padding=self.window // 2)
        localization_map = (local_agreement > self.threshold).float()  # (B, 1, H, W)

        # Message: majority vote of the blue parity over the detected region.
        blue_parity = (image[:, 2].round() % 2).flatten(start_dim=1)  # (B, H*W)
        detected = localization_map.flatten(start_dim=1)  # (B, H*W)
        message = torch.zeros((batch_size, 1, self._nbits), dtype=torch.int64, device=image.device)
        for b in range(batch_size):
            ones = torch.zeros(self._nbits, device=image.device).index_add_(0, bit_index, blue_parity[b] * detected[b])
            total = torch.zeros(self._nbits, device=image.device).index_add_(0, bit_index, detected[b])
            message[b, 0] = (2 * ones > total).long()

        return DecodeResult(
            watermark_data=WatermarkData(message=message, localization_map=localization_map),
            decode_time_ms=(time.time() - t0) * 1000,
        )


def build_model(device: str = "cpu", **kwargs) -> ParityWatermark:
    """Builder for ``--model examples.parity_adapter:build_model``."""
    return ParityWatermark(device=device, **kwargs)
