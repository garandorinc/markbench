"""Centralized image utilities for the markbench framework.

This module provides consistent image loading, saving, and conversion functions
used throughout the codebase.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from markbench.utils.device import get_device


def load_image(
    path: Union[Path, str],
    device: str = "",
    size: Optional[Tuple[int, int]] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Load image from file as tensor (C, H, W) in [0, 255].

    Args:
        path: Path to image file
        device: Device to load tensor onto (default: auto-detect)
        size: Optional (width, height) tuple to resize image
        transform: Optional transform to apply to the loaded tensor

    Returns:
        Image tensor (C, H, W) with float values in [0, 255]

    Example:
        >>> img = load_image("test.png", device="cuda")
        >>> img.shape
        torch.Size([3, 512, 512])
    """
    # Check if we're in a DataLoader worker process
    # Workers must use CPU only (CUDA not allowed in forked processes)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Force CPU when in DataLoader worker
        device = "cpu"
    else:
        device = get_device(device)

    pil_img = Image.open(path).convert("RGB")

    if size is not None:
        pil_img = pil_img.resize(size)

    img_np = np.array(pil_img, dtype=np.float32)
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
    tensor = tensor.to(device)

    if transform is not None:
        tensor = transform(tensor)

    return tensor


def save_image(image: torch.Tensor, path: str, quality: int = 95, round_before_quantize: bool = True):
    """Save a tensor image to a file.

    Args:
        image: Image tensor in (C, H, W) format with values in [0, 255]
        path: Path to save to
        quality: JPEG quality (if applicable)
        round_before_quantize: If True, round values before converting to uint8.
                              If False, truncate (floor) values. Default: True.

    Example:
        >>> img = torch.rand(3, 256, 256) * 255
        >>> save_image(img, "output.png")
    """
    pil_img = tensor_to_pil(image, round_before_quantize=round_before_quantize)
    pil_img.save(path, quality=quality)


def tensor_to_pil(tensor: torch.Tensor, round_before_quantize: bool = True) -> Image.Image:
    """Convert tensor (C, H, W) to PIL Image.

    Args:
        tensor: Image tensor (C, H, W) with values in [0, 255]
        round_before_quantize: If True, round values before converting to uint8.
                              If False, truncate (floor) values. Default: True.

    Returns:
        PIL Image in RGB format

    Example:
        >>> tensor = torch.rand(3, 256, 256) * 255
        >>> pil_img = tensor_to_pil(tensor)
        >>> isinstance(pil_img, Image.Image)
        True
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    tensor = tensor.clamp(0, 255)

    if round_before_quantize:
        # Round to nearest integer before converting to uint8
        img_np = tensor.round().byte().numpy()
    else:
        # Truncate (floor) by directly converting to uint8
        img_np = tensor.byte().numpy()

    return Image.fromarray(img_np)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor (C, H, W) in [0, 255].

    Args:
        image: PIL Image in RGB format

    Returns:
        Image tensor (C, H, W) with float values in [0, 255]

    Example:
        >>> pil_img = Image.new('RGB', (256, 256))
        >>> tensor = pil_to_tensor(pil_img)
        >>> tensor.shape
        torch.Size([3, 256, 256])
    """
    img_np = np.array(image, dtype=np.float32)
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
    return tensor


def tensor_to_numpy_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) numpy array.

    Args:
        tensor: Image tensor in (C, H, W) format

    Returns:
        Numpy array in (H, W, C) format as uint8

    Example:
        >>> tensor = torch.rand(3, 256, 256) * 255
        >>> arr = tensor_to_numpy_hwc(tensor)
        >>> arr.shape
        (256, 256, 3)
    """
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor.byte().permute(1, 2, 0).numpy()


def numpy_hwc_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert (H, W, C) numpy array to (C, H, W) tensor.

    Args:
        array: Numpy array in (H, W, C) format

    Returns:
        Image tensor in (C, H, W) format with float values

    Example:
        >>> arr = np.random.rand(256, 256, 3).astype(np.float32)
        >>> tensor = numpy_hwc_to_tensor(arr)
        >>> tensor.shape
        torch.Size([3, 256, 256])
    """
    return torch.from_numpy(array.transpose(2, 0, 1)).float()
