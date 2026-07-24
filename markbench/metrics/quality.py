"""Metrics for watermark evaluation.

Measuring image quality and watermark recovery performance.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker

from markbench.utils.device import get_device

_LPIPS_MODELS = {}

def get_lpips_model(net: str, device: str):
    if net not in _LPIPS_MODELS:
        import lpips
        _LPIPS_MODELS[net] = lpips.LPIPS(net=net, verbose=False).to(device)
    return _LPIPS_MODELS[net]

@jaxtyped(typechecker=typechecker)
def compute_psnr(
    original: Float[Tensor, "3 height width"],
    compared: Float[Tensor, "3 height width"],
    max_val: float = 255.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Higher values indicate better quality (less distortion).

    Args:
        original: Reference image tensor (C, H, W) in [0, 255]
        compared: Compared image tensor (C, H, W) in [0, 255]
        max_val: Maximum pixel value (255 for 8-bit images)

    Returns:
        PSNR in decibels (dB). Returns inf if images are identical.
    """

    if original.device != compared.device:
        compared = compared.to(original.device)

    mse = torch.mean((original.float() - compared.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val**2) / mse)


@jaxtyped(typechecker=typechecker)
def compute_ssim(
    original: Float[Tensor, "3 height width"],
    compared: Float[Tensor, "3 height width"],
) -> float:
    """Compute Structural Similarity Index between two images.

    Values range from -1 to 1, where 1 indicates perfect similarity (so higher is better).

    Args:
        original: Reference image tensor (C, H, W) in [0, 255]
        compared: Compared image tensor (C, H, W) in [0, 255]

    Returns:
        SSIM value in [-1, 1].
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    # Keep on same device as input
    device = original.device

    # torchmetrics expects (N, C, H, W) in [0, 1] range, all on same device
    img1 = original.float().to(device).unsqueeze(0) / 255.0
    img2 = compared.float().to(device).unsqueeze(0) / 255.0

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        return ssim(img1, img2).item()


@jaxtyped(typechecker=typechecker)
def compute_lpips(
    original: Float[Tensor, "3 height width"],
    compared: Float[Tensor, "3 height width"],
    net: str = "alex",
    device: str = "",
) -> Optional[float]:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity) distance.

    LPIPS uses deep network features to measure perceptual similarity.
    Lower values indicate more similar images (unlike PSNR/SSIM).

    Args:
        original: Reference image tensor (C, H, W) in [0, 255]
        compared: Compared image tensor (C, H, W) in [0, 255]
        net: Network backbone ("alex", "vgg", or "squeeze")
        device: Device for computation (default: auto-detect)

    Returns:
        LPIPS distance, or None if lpips not installed.
    """
    device = get_device(device)
    try:
        import lpips
    except ImportError:
        return None

    # LPIPS expects batch dimension and [-1, 1] range
    img1 = original.float().unsqueeze(0) / 255.0  # [0, 1]
    img2 = compared.float().unsqueeze(0) / 255.0

    # Normalize to [-1, 1]
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    # Move to device
    img1 = img1.to(device)
    img2 = img2.to(device)

    loss_fn = get_lpips_model(net, device)
    with torch.no_grad():
        return loss_fn(img1, img2).item()


@jaxtyped(typechecker=typechecker)
def compute_mse(
    original: Float[Tensor, "3 height width"],
    compared: Float[Tensor, "3 height width"],
) -> float:
    """Compute Mean Squared Error between two images.

    Args:
        original: Reference image tensor (C, H, W)
        compared: Compared image tensor (C, H, W)

    Returns:
        MSE value
    """

    if original.device != compared.device:
        compared = compared.to(original.device)

    return torch.mean((original.float() - compared.float()) ** 2).item()


@jaxtyped(typechecker=typechecker)
def compute_mae(
    original: Float[Tensor, "3 height width"],
    compared: Float[Tensor, "3 height width"],
) -> float:
    """Compute Mean Absolute Error between two images.

    Args:
        original: Reference image tensor (C, H, W)
        compared: Compared image tensor (C, H, W)

    Returns:
        MAE value (lower = more similar)
    """

    if original.device != compared.device:
        compared = compared.to(original.device)

    return torch.mean(torch.abs(original.float() - compared.float())).item()


# Batched versions for parallel computation

def compute_psnr_batch(
    original: Float[Tensor, "batch 3 height width"],
    compared: Float[Tensor, "batch 3 height width"],
    max_val: float = 255.0,
) -> Float[Tensor, "batch"]:
    """Batched PSNR computation.

    Args:
        original: Batch of reference images (N, C, H, W)
        compared: Batch of compared images (N, C, H, W)
        max_val: Maximum pixel value

    Returns:
        Tensor of PSNR values for each image pair
    """
    if original.device != compared.device:
        compared = compared.to(original.device)

    mse = torch.mean((original.float() - compared.float()) ** 2, dim=(1, 2, 3))
    psnr = torch.where(
        mse == 0,
        torch.tensor(float("inf"), device=mse.device),
        10 * torch.log10((max_val**2) / mse)
    )
    return psnr


def compute_ssim_batch(
    original: Float[Tensor, "batch 3 height width"],
    compared: Float[Tensor, "batch 3 height width"],
) -> Float[Tensor, "batch"]:
    """Batched SSIM computation using torchmetrics.

    Args:
        original: Batch of reference images (N, C, H, W)
        compared: Batch of compared images (N, C, H, W)

    Returns:
        Tensor of SSIM values for each image pair
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    # Keep on same device as input
    device = original.device

    # torchmetrics expects [0, 1] range and tensors on same device
    img1 = original.float().to(device) / 255.0
    img2 = compared.float().to(device) / 255.0

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        # Compute per-image SSIM by processing batch
        batch_size = original.shape[0]
        ssim_values = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            ssim_values[i] = ssim(img1[i:i+1], img2[i:i+1])
            ssim.reset()  # Reset metric state to avoid accumulation issues

        return ssim_values


def compute_lpips_batch(
    original: Float[Tensor, "batch 3 height width"],
    compared: Float[Tensor, "batch 3 height width"],
    net: str = "alex",
    device: str = "",
) -> Float[Tensor, "batch"]:
    """Batched LPIPS computation.

    Args:
        original: Batch of reference images (N, C, H, W)
        compared: Batch of compared images (N, C, H, W)
        net: Network backbone
        device: Device for computation

    Returns:
        Tensor of LPIPS values for each image pair
    """
    device = get_device(device)
    try:
        import lpips
    except ImportError:
        return None

    # LPIPS expects [-1, 1] range
    img1 = original.float() / 255.0
    img2 = compared.float() / 255.0
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    img1 = img1.to(device)
    img2 = img2.to(device)

    loss_fn = get_lpips_model(net, device)
    with torch.no_grad():
        result = loss_fn(img1, img2)
        # Flatten to (batch,) - squeeze all trailing dimensions but keep batch
        return result.view(-1)


def compute_mse_batch(
    original: Float[Tensor, "batch 3 height width"],
    compared: Float[Tensor, "batch 3 height width"],
) -> Float[Tensor, "batch"]:
    """Batched MSE computation.

    Args:
        original: Batch of reference images (N, C, H, W)
        compared: Batch of compared images (N, C, H, W)

    Returns:
        Tensor of MSE values for each image pair
    """
    if original.device != compared.device:
        compared = compared.to(original.device)

    return torch.mean((original.float() - compared.float()) ** 2, dim=(1, 2, 3))


def compute_mae_batch(
    original: Float[Tensor, "batch 3 height width"],
    compared: Float[Tensor, "batch 3 height width"],
) -> Float[Tensor, "batch"]:
    """Batched MAE computation.

    Args:
        original: Batch of reference images (N, C, H, W)
        compared: Batch of compared images (N, C, H, W)

    Returns:
        Tensor of MAE values for each image pair
    """
    if original.device != compared.device:
        compared = compared.to(original.device)

    return torch.mean(torch.abs(original.float() - compared.float()), dim=(1, 2, 3))
