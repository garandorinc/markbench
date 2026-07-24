"""Visualization utilities for debugging mask transformations."""

import torch
from typing import Optional


def blend_mask_with_image(
    image: torch.Tensor,  # (3, H, W) in [0, 255]
    mask: torch.Tensor,   # (H, W) in [0, 1]
) -> torch.Tensor:
    """
    Blend a single mask with image.

    Applies the formula: 0.25 * image + 0.75 * mask * image
    This creates a visualization where:
    - mask=1 regions (watermarked): full intensity
    - mask=0 regions (non-watermarked): 25% intensity

    Args:
        image: Input image tensor (3, H, W) in [0, 255] range
        mask: Binary mask tensor (H, W) in [0, 1] range

    Returns:
        Blended image tensor (3, H, W) in [0, 255] range
    """
    # Expand mask to 3 channels and ensure same device as image
    mask_3d = mask.to(image.device).unsqueeze(0).expand(3, -1, -1)  # (3, H, W)

    blended = 0.25 * image + 0.75 * mask_3d * image
    return blended.clamp(0, 255)


def combine_masks(
    masks: torch.Tensor,  # (N, H, W) in [0, 1]
    method: str = "max"   # "max", "mean", or "sum"
) -> torch.Tensor:
    """
    Combine multiple masks into a single mask.

    Args:
        masks: N masks of shape (N, H, W) in [0, 1] range
        method: How to combine masks:
                - "max": Take maximum value across all masks (union)
                - "mean": Take average value across all masks
                - "sum": Sum all masks and clamp to [0, 1]

    Returns:
        Single combined mask (H, W) in [0, 1] range
    """
    if method == "max":
        combined = masks.max(dim=0)[0]
    elif method == "mean":
        combined = masks.mean(dim=0)
    elif method == "sum":
        combined = masks.sum(dim=0).clamp(0, 1)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'max', 'mean', or 'sum'")

    return combined


def visualize_mask_transformation(
    original_image: torch.Tensor,      # (3, H, W) in [0, 255]
    original_masks: Optional[torch.Tensor],   # (N, H, W) in [0, 1]
    attacked_image: torch.Tensor,      # (3, H', W') in [0, 255]
    transformed_masks: Optional[torch.Tensor],  # (N, H', W') in [0, 1]
    combine_method: str = "max"
) -> torch.Tensor:
    """
    Create side-by-side visualization of mask transformations.

    This function creates a visualization showing the original image with masks
    overlaid on the left, and the attacked image with transformed masks on the right.
    This helps verify that masks were correctly transformed by the attack.

    Args:
        original_image: Original image tensor (3, H, W) in [0, 255] range
        original_masks: Original localization masks (N, H, W) in [0, 1] range
        attacked_image: Attacked image tensor (3, H', W') in [0, 255] range
        transformed_masks: Transformed localization masks (N, H', W') in [0, 1] range
        combine_method: Method to combine multiple masks ("max", "mean", or "sum")

    Returns:
        Visualization image (3, H_max, W_orig + W_attacked) showing:
        - Left: original image with masks overlaid
        - Right: attacked image with transformed masks overlaid
    """
    # If no masks, just return side-by-side images
    if original_masks is None and transformed_masks is None:
        return torch.cat([original_image, attacked_image], dim=2)

    # Combine multiple masks into single mask
    if original_masks is not None:
        orig_mask_combined = combine_masks(original_masks, method=combine_method)
        orig_blended = blend_mask_with_image(original_image, orig_mask_combined)
    else:
        orig_blended = original_image

    if transformed_masks is not None:
        trans_mask_combined = combine_masks(transformed_masks, method=combine_method)
        trans_blended = blend_mask_with_image(attacked_image, trans_mask_combined)
    else:
        trans_blended = attacked_image

    # Handle different sizes - pad to match height
    h_orig, w_orig = orig_blended.shape[1:]
    h_trans, w_trans = trans_blended.shape[1:]
    h_max = max(h_orig, h_trans)

    # Pad if needed
    if h_orig < h_max:
        pad_h = h_max - h_orig
        orig_blended = torch.nn.functional.pad(
            orig_blended, (0, 0, 0, pad_h), value=0
        )
    if h_trans < h_max:
        pad_h = h_max - h_trans
        trans_blended = torch.nn.functional.pad(
            trans_blended, (0, 0, 0, pad_h), value=0
        )

    # Concatenate side-by-side (ensure same device)
    trans_blended = trans_blended.to(orig_blended.device)
    visualization = torch.cat([orig_blended, trans_blended], dim=2)
    return visualization


def visualize_residual(
    original: torch.Tensor,
    watermarked: torch.Tensor,
    amplify_factor: float = 10.0
) -> torch.Tensor:
    residual = watermarked - original
    residual_vis = (residual * amplify_factor + 127.5).clamp(0, 255)
    return torch.cat([original, residual_vis, watermarked], dim=2)
