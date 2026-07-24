from typing import Any

import torch
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker

from markbench.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_mse,
    compute_mae,
)
from markbench.metrics.quality import (
    compute_psnr_batch,
    compute_ssim_batch,
    compute_lpips_batch,
    compute_mse_batch,
    compute_mae_batch,
)
from markbench.interface import DecodeResult, EmbedResult, WatermarkData, AttackResult


@jaxtyped(typechecker=typechecker)
def compute_bit_accuracy(
    original_message: Int[Tensor, "..."],
    extracted_message: Int[Tensor, "..."],
) -> float:
    """Compute bit accuracy between original and extracted messages.

    Args:
        original_message: Original binary message tensor
        extracted_message: Extracted binary message tensor

    Returns:
        Bit accuracy in [0, 1]
    """
    if original_message.numel() != extracted_message.numel():
        raise ValueError(
            "Original and extracted messages must have the same number of bits"
        )

    # Convert to same type and device for comparison
    original_message = original_message.to(extracted_message.device).float().view(-1)
    extracted_message = extracted_message.float().view(-1)

    return (original_message == extracted_message).float().mean().item()


@jaxtyped(typechecker=typechecker)
def compute_iou(
    pred_mask: Float[Tensor, "..."],
    gt_mask: Float[Tensor, "..."],
    threshold: float = 0.5,
) -> float:
    """Compute Intersection over Union (IoU) between two binary masks.

    Args:
        pred_mask: Predicted mask tensor (H, W) or (1, H, W)
        gt_mask: Ground truth mask tensor (H, W) or (1, H, W)
        threshold: Threshold to binarize masks (default 0.5)

    Returns:
        IoU value in [0, 1]
    """
    # Ensure masks are boolean/binary
    pred = (pred_mask > threshold).bool()
    gt = (gt_mask > threshold).bool()
    # Allow comparing masks across devices (e.g. CPU GT vs GPU decoded).
    if gt.device != pred.device:
        gt = gt.to(pred.device)

    # Flatten (reshape works on both contiguous and non-contiguous tensors)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    intersection = (pred & gt).sum().float().item()
    union = (pred | gt).sum().float().item()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


@jaxtyped(typechecker=typechecker)
def compute_bbox_iou(
    pred_mask: Float[Tensor, "..."],
    gt_mask: Float[Tensor, "..."],
    threshold: float = 0.5,
) -> float:
    """Convert masks to bboxes and compute bbox IoU.

    Args:
        pred_mask: Predicted mask tensor (H, W) or (1, H, W)
        gt_mask: Ground truth mask tensor (H, W) or (1, H, W)
        threshold: Threshold to binarize masks (default 0.5)

    Returns:
        Bbox IoU value in [0, 1]
    """
    from markbench.utils.masks import mask_to_bbox, bbox_iou

    pred = (pred_mask > threshold).float()
    gt = (gt_mask > threshold).float()

    if pred.ndim == 3:
        pred = pred.squeeze(0)
    if gt.ndim == 3:
        gt = gt.squeeze(0)

    pred_empty = (pred.sum() == 0)
    gt_empty = (gt.sum() == 0)

    if pred_empty and gt_empty:
        return 1.0  # Correct rejection (Empty matches Empty)
    
    if pred_empty or gt_empty:
        return 0.0  # Mismatch (One is empty, one is not)

    pred_bbox = mask_to_bbox(pred)
    gt_bbox = mask_to_bbox(gt)

    return bbox_iou(pred_bbox, gt_bbox)


@jaxtyped(typechecker=typechecker)
def compute_dice(
    pred_mask: Float[Tensor, "..."],
    gt_mask: Float[Tensor, "..."],
    threshold: float = 0.5,
) -> float:
    """Compute Dice coefficient. (Equivalent to F1 score.)

    Args:
        pred_mask: Predicted mask tensor (H, W) or (1, H, W)
        gt_mask: Ground truth mask tensor (H, W) or (1, H, W)
        threshold: Threshold to binarize masks (default 0.5)

    Returns:
        Dice coefficient in [0, 1]
    """
    pred = (pred_mask > threshold).bool()
    gt = (gt_mask > threshold).bool()

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    intersection = (pred & gt).sum().float().item()
    pred_area = pred.sum().float().item()
    gt_area = gt.sum().float().item()

    if pred_area + gt_area == 0:
        return 1.0

    return 2 * intersection / (pred_area + gt_area)


@jaxtyped(typechecker=typechecker)
def compute_coverage_metrics(
    pred_mask: Float[Tensor, "..."],
    gt_mask: Float[Tensor, "..."],
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute GT coverage and prediction containment.

    Args:
        pred_mask: Predicted mask tensor (H, W) or (1, H, W)
        gt_mask: Ground truth mask tensor (H, W) or (1, H, W)
        threshold: Threshold to binarize masks (default 0.5)

    Returns:
        Tuple of (gt_coverage, pred_containment) values in [0, 1]
    """
    pred = (pred_mask > threshold).bool()
    gt = (gt_mask > threshold).bool()

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    intersection = (pred & gt).sum().float().item()
    pred_area = pred.sum().float().item()
    gt_area = gt.sum().float().item()

    if gt_area == 0:
        gt_coverage = 1.0
    else:
        gt_coverage = intersection / gt_area

    if pred_area == 0:
        pred_containment = 1.0
    else:
        pred_containment = intersection / pred_area

    return gt_coverage, pred_containment


def _score_iou(i, j, dec_map, orig_map, dec_msg, orig_msg):
    """Score based on IoU between masks."""
    return compute_iou(dec_map[i], orig_map[j])


def _score_joint(i, j, dec_map, orig_map, dec_msg, orig_msg, weight_iou):
    """Score based on weighted combination of IoU and message similarity."""
    iou = compute_iou(dec_map[i], orig_map[j])
    bit_acc = compute_bit_accuracy(orig_msg[j], dec_msg[i])
    return weight_iou * iou + (1.0 - weight_iou) * bit_acc


def _score_message(i, j, dec_map, orig_map, dec_msg, orig_msg):
    """Score based on message accuracy only."""
    return compute_bit_accuracy(orig_msg[j], dec_msg[i])


def _score_bbox_iou(i, j, dec_map, orig_map, dec_msg, orig_msg):
    """Score based on IoU between bounding boxes."""
    return compute_bbox_iou(dec_map[i], orig_map[j])


MATCHING_STRATEGIES = {
    "iou": {
        "score_fn": _score_iou,
        "requires_maps": True,
    },
    "bbox_iou": {
        "score_fn": _score_bbox_iou,
        "requires_maps": True,
    },
    "joint": {
        "score_fn": _score_joint,
        "requires_maps": True,
        "default_params": {"weight_iou": 0.5},
    },
    "best_message": {
        "score_fn": _score_message,
        "requires_maps": False,
    },
}


def _match_regions(
    decoded_map: Float[Tensor, "n_decoded height width"],
    original_map: Float[Tensor, "n_gt height width"],
    decoded_msg: Int[Tensor, "n_decoded n_bits"],
    original_msg: Int[Tensor, "n_gt n_bits"],
    strategy_config: dict,
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    """Match decoded regions to GT using configured scoring function."""
    score_fn = strategy_config["score_fn"]
    default_params = strategy_config.get("default_params", {})

    # Merge user config with defaults
    params = {}
    for key, default_value in default_params.items():
        config_key = f"joint_{key}" if "weight" in key else key
        params[key] = config.get(config_key, default_value)

    matches = []
    for i in range(decoded_msg.shape[0]):
        best_score = -1
        best_gt_idx = -1
        for j in range(original_msg.shape[0]):
            if params:
                score = score_fn(i, j, decoded_map, original_map, decoded_msg, original_msg, **params)
            else:
                score = score_fn(i, j, decoded_map, original_map, decoded_msg, original_msg)

            if score > best_score:
                best_score = score
                best_gt_idx = j

        if best_gt_idx >= 0:
            matches.append((i, best_gt_idx))

    return matches


def _compute_metrics_for_matches(
    matches: list[tuple[int, int]],
    decoded_map: Float[Tensor, "n_decoded height width"],
    original_map: Float[Tensor, "n_gt height width"],
    decoded_msg: Int[Tensor, "n_decoded n_bits"],
    original_msg: Int[Tensor, "n_gt n_bits"],
) -> dict[str, float]:
    """Compute all metrics for matched pairs."""
    if not matches:
        return {
            "bit_accuracy": 0.0,
            "localization_iou": 0.0,
            "localization_bbox_iou": 0.0,
            "localization_dice": 0.0,
            "localization_gt_coverage": 0.0,
            "localization_pred_containment": 0.0,
        }

    bit_accuracies = []
    iou_scores = []
    bbox_ious = []
    dice_scores = []
    gt_coverages = []
    pred_containments = []

    for decoded_idx, gt_idx in matches:
        bit_accuracies.append(
            compute_bit_accuracy(original_msg[gt_idx], decoded_msg[decoded_idx])
        )

        if original_map.numel() > 0 and decoded_map.numel() > 0:
            iou_scores.append(compute_iou(decoded_map[decoded_idx], original_map[gt_idx]))
            bbox_ious.append(compute_bbox_iou(decoded_map[decoded_idx], original_map[gt_idx]))
            dice_scores.append(compute_dice(decoded_map[decoded_idx], original_map[gt_idx]))

            gt_cov, pred_cont = compute_coverage_metrics(
                decoded_map[decoded_idx], original_map[gt_idx]
            )
            gt_coverages.append(gt_cov)
            pred_containments.append(pred_cont)

    metrics = {
        "bit_accuracy": sum(bit_accuracies) / len(bit_accuracies) if bit_accuracies else 0.0,
    }

    if iou_scores:
        metrics["localization_iou"] = sum(iou_scores) / len(iou_scores)
        metrics["localization_bbox_iou"] = sum(bbox_ious) / len(bbox_ious)
        metrics["localization_dice"] = sum(dice_scores) / len(dice_scores)
        metrics["localization_gt_coverage"] = sum(gt_coverages) / len(gt_coverages)
        metrics["localization_pred_containment"] = sum(pred_containments) / len(pred_containments)

    return metrics


@jaxtyped(typechecker=typechecker)
def compute_decode_metrics(
    original_watermark_data: WatermarkData,
    decode_result: DecodeResult,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute decoding quality metrics.

    Matches decoded masks to ground truth masks, then compares corresponding
    messages. Provides multiple matching strategies for robustness.

    Args:
        original_watermark_data: Ground truth watermark data from embedding
        decode_result: DecodeResult containing extracted watermark data
        config: Configuration dictionary

    Returns:
        Dictionary with nested metrics for each matching strategy:
        - iou: Metrics from IoU-based mask matching
        - joint: Metrics from joint IoU+message similarity matching
        - best_message: Metrics from best message accuracy matching
        - original_messages: Original messages as list of lists (optional)
        - decoded_messages: Decoded messages as list of lists (optional)
    """
    original_msg = original_watermark_data.message
    decoded_msg = decode_result.watermark_data.message
    original_map = original_watermark_data.localization_map
    decoded_map = decode_result.watermark_data.localization_map

    if original_msg.numel() == 0 or decoded_msg.numel() == 0:
        zero_metrics = {
            "bit_accuracy": 0.0,
            "localization_iou": 0.0,
            "localization_bbox_iou": 0.0,
            "localization_dice": 0.0,
            "localization_gt_coverage": 0.0,
            "localization_pred_containment": 0.0,
        }
        metrics = {
            "iou": zero_metrics.copy(),
            "joint": zero_metrics.copy(),
            "best_message": zero_metrics.copy(),
        }
        if config.get("log_messages", True):
            metrics["original_messages"] = []
            metrics["decoded_messages"] = []
        return metrics

    if original_msg.ndim == 1:
        original_msg = original_msg.unsqueeze(0)
    if decoded_msg.ndim == 1:
        decoded_msg = decoded_msg.unsqueeze(0)

    if original_map.numel() > 0 and decoded_map.numel() > 0:
        if original_map.ndim == 2:
            original_map = original_map.unsqueeze(0)
        if decoded_map.ndim == 2:
            decoded_map = decoded_map.unsqueeze(0)

    zero_metrics = {
        "bit_accuracy": 0.0,
        "localization_iou": 0.0,
        "localization_bbox_iou": 0.0,
        "localization_dice": 0.0,
        "localization_gt_coverage": 0.0,
        "localization_pred_containment": 0.0,
    }

    metrics = {}
    for strategy_name, strategy_config in MATCHING_STRATEGIES.items():
        if strategy_config["requires_maps"] and (original_map.numel() == 0 or decoded_map.numel() == 0):
            metrics[strategy_name] = zero_metrics.copy()
        else:
            matches = _match_regions(decoded_map, original_map, decoded_msg, original_msg, strategy_config, config)
            metrics[strategy_name] = _compute_metrics_for_matches(matches, decoded_map, original_map, decoded_msg, original_msg)

    if config.get("log_messages", True):
        metrics["original_messages"] = original_msg.cpu().tolist()
        metrics["decoded_messages"] = decoded_msg.cpu().tolist()

    return metrics


@jaxtyped(typechecker=typechecker)
def compute_embed_metrics(
    original_image: Float[Tensor, "3 height width"],
    embed_result: EmbedResult,
    config: dict[str, Any],
) -> dict[str, float]:
    """Compute embedding quality metrics between original and watermarked images.

    Args:
        original_image: Original image tensor (C, H, W) in [0, 255]
        embed_result: EmbedResult containing watermarked image
        config: Configuration dictionary (not used currently)

    Returns:
        Dictionary of embedding metrics (psnr, ssim, lpips, mse, mae)
    """
    watermarked_image = embed_result.watermarked

    metrics = {}

    # PSNR
    metrics["psnr"] = compute_psnr(original_image, watermarked_image)

    # SSIM
    metrics["ssim"] = compute_ssim(original_image, watermarked_image)

    # LPIPS
    lpips_value = compute_lpips(original_image, watermarked_image)
    if lpips_value is not None:
        metrics["lpips"] = lpips_value

    # MSE
    metrics["mse"] = compute_mse(original_image, watermarked_image)

    # MAE
    metrics["mae"] = compute_mae(original_image, watermarked_image)

    return metrics

@jaxtyped(typechecker=typechecker)
def compute_attack_metrics(
    watermarked_image: Float[Tensor, "3 height1 width1"],
    attack_result: AttackResult,
    config: dict[str, Any],
) -> dict[str, float]:
    """Compute attack quality metrics between watermarked and attacked images.

    Args:
        watermarked_image: Watermarked image tensor (C, H, W) in [0, 255]
        attacked_image: Attacked image tensor (C, H, W) in [0, 255]
        config: Configuration dictionary (not used currently)

    Returns:
        Dictionary of attack metrics (psnr, ssim)
    """
    metrics = {}
    attacked_image = attack_result.attacked

    if attack_result.metadata.get("is_geometry_modifying", False):
        return metrics
    
    # PSNR
    metrics["psnr"] = compute_psnr(watermarked_image, attacked_image)

    # SSIM
    metrics["ssim"] = compute_ssim(watermarked_image, attacked_image)

    # LPIPS
    lpips_value = compute_lpips(watermarked_image, attacked_image)

    if lpips_value is not None:
        metrics["lpips"] = lpips_value
    
    # MSE
    metrics["mse"] = compute_mse(watermarked_image, attacked_image)

    # MAE
    metrics["mae"] = compute_mae(watermarked_image, attacked_image)

    return metrics


def compute_embed_metrics_batch(
    original_images: Float[Tensor, "batch 3 height width"],
    watermarked_images: Float[Tensor, "batch 3 height width"],
    config: dict[str, Any],
) -> list[dict[str, float]]:
    """Batched embedding quality metrics computation.

    Args:
        original_images: Batch of original images (N, C, H, W)
        watermarked_images: Batch of watermarked images (N, C, H, W)
        config: Configuration dictionary

    Returns:
        List of metric dictionaries, one per image
    """
    batch_size = original_images.shape[0]

    # Compute batched metrics
    psnr_values = compute_psnr_batch(original_images, watermarked_images)
    ssim_values = compute_ssim_batch(original_images, watermarked_images)
    lpips_values = compute_lpips_batch(original_images, watermarked_images)
    mse_values = compute_mse_batch(original_images, watermarked_images)
    mae_values = compute_mae_batch(original_images, watermarked_images)

    # Convert to list of dicts
    results = []
    for i in range(batch_size):
        metrics = {
            "psnr": psnr_values[i].item(),
            "ssim": ssim_values[i].item(),
            "mse": mse_values[i].item(),
            "mae": mae_values[i].item(),
        }
        if lpips_values is not None:
            metrics["lpips"] = lpips_values[i].item()
        results.append(metrics)

    return results


def compute_attack_metrics_batch(
    watermarked_images: Float[Tensor, "batch 3 height width"],
    attacked_images: Float[Tensor, "batch 3 height width"],
    is_geometry_modifying: list[bool],
    config: dict[str, Any],
) -> list[dict[str, float]]:
    """Batched attack quality metrics computation.

    Args:
        watermarked_images: Batch of watermarked images (N, C, H, W)
        attacked_images: Batch of attacked images (N, C, H, W)
        is_geometry_modifying: List of bools indicating geometry-modifying attacks
        config: Configuration dictionary

    Returns:
        List of metric dictionaries, one per image
    """
    batch_size = watermarked_images.shape[0]

    # If all attacks are geometry-modifying, return empty dicts
    if all(is_geometry_modifying):
        return [{}] * batch_size

    # Skip geometry-modifying attacks before computing metrics
    non_geometry_indices = [i for i in range(batch_size) if not is_geometry_modifying[i]]

    # Filter to only non-geometry-modifying images
    watermarked_filtered = torch.stack([watermarked_images[i] for i in non_geometry_indices])
    attacked_filtered = torch.stack([attacked_images[i] for i in non_geometry_indices])

    # Compute batched metrics
    psnr_values = compute_psnr_batch(watermarked_filtered, attacked_filtered)
    ssim_values = compute_ssim_batch(watermarked_filtered, attacked_filtered)
    lpips_values = compute_lpips_batch(watermarked_filtered, attacked_filtered)
    mse_values = compute_mse_batch(watermarked_filtered, attacked_filtered)
    mae_values = compute_mae_batch(watermarked_filtered, attacked_filtered)

    # Convert to list of dicts
    results = []
    non_geometry_idx = 0
    for i in range(batch_size):
        if is_geometry_modifying[i]:
            results.append({})
        else:
            metrics = {
                "psnr": psnr_values[non_geometry_idx].item(),
                "ssim": ssim_values[non_geometry_idx].item(),
                "mse": mse_values[non_geometry_idx].item(),
                "mae": mae_values[non_geometry_idx].item(),
            }
            if lpips_values is not None:
                metrics["lpips"] = lpips_values[non_geometry_idx].item()
            results.append(metrics)
            non_geometry_idx += 1

    return results
