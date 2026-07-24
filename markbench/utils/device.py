"""Device utilities for markbench."""

from typing import Union

import torch

VALID_DEVICES = {"", "cpu", "cuda"}


def get_device(device: Union[str, torch.device] = "") -> str:
    """Get and resolve device with validation.

    Args:
        device: 'cpu', 'cuda', 'cuda:<idx>', torch.device, or '' for auto-detect

    Returns:
        Resolved device string ('cpu' or 'cuda')

    Raises:
        ValueError: If device is invalid
    """
    if isinstance(device, torch.device):
        device = device.type

    if device == "auto":
        raise ValueError("device='auto' deprecated. Use device='' for auto-detect.")
    if isinstance(device, str) and device.startswith("cuda:"):
        device = "cuda"
    if device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid device '{device}'. Must be 'cpu', 'cuda', or '' for auto."
        )
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
