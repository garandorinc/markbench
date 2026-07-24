"""Benchmark runner.

Orchestrates the full benchmark pipeline:
  Image → Embed → Attack → Decode → Log

All metric computation happens here. Logger just writes.
"""

import json
import time
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from markbench.benchmarking.attacks import AbstractAttack
from markbench.benchmarking.attacks.runtime_specs import RuntimeAttackSpec
from markbench.data.random_mask_generator import RandomMaskGenerator, get_benchmark_random_mask_kwargs
from markbench.data import ImageDataset
from markbench.interface import WatermarkModel, WatermarkData, EmbedResult, DecodeResult, DataPoint
from markbench.logger import BenchmarkLogger, BenchmarkRow
from markbench.utils.masks import bbox_to_mask, mask_to_bbox
from markbench.utils import get_device
from markbench.metrics import (
    compute_decode_metrics,
    compute_embed_metrics_batch,
    compute_attack_metrics,
    compute_attack_metrics_batch,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum")

@contextmanager
def _temporary_seed(seed: int):
    """Temporarily seed RNGs for deterministic, isolated randomness."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _compute_attack_seed(
    attack_seed: Optional[int],
    batch_idx: int,
    global_attack_idx: int,
    img_idx: Optional[int] = None,
    batched: bool = False,
) -> Optional[int]:
    """Compute a deterministic attack seed.

    Batched attacks keep a single seed for the whole batch because they execute
    via one joint call. Single-image attacks get a distinct per-image seed so
    range-based/randomized params vary across images instead of repeating within
    each batch.
    """
    if attack_seed is None:
        return None

    batch_seed = attack_seed + (batch_idx * 1000) + global_attack_idx
    if batched or img_idx is None:
        return batch_seed
    return (batch_seed * 1000) + img_idx


def _compute_random_mask_seed(
    mask_seed: Optional[int],
    batch_idx: int,
    img_idx: Optional[int] = None,
) -> Optional[int]:
    """Compute deterministic random-mask seeds.

    Random masks should be independent across images, unlike the historical
    flattened batch generation that coupled all images in a batch to one mask
    stream. This helper keeps reproducibility while avoiding cross-image
    interactions.
    """
    if mask_seed is None:
        return None

    batch_seed = mask_seed + batch_idx
    if img_idx is None:
        return batch_seed
    return (batch_seed * 1_000_003) + img_idx


def _load_and_prepare_batch(
    samples: Any,
    model: WatermarkModel,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int, bool]:
    """Prepare batch data from dataset samples.

    Args:
        samples: Batch samples (DataPoint or raw tensor)
        model: Watermark model (checked for supports_multiple_messages)
        device: Device to move tensors to

    Returns:
        Tuple of:
        - original_batch: (B, C, H, W) images on device
        - original: Same as original_batch (legacy compatibility)
        - dataset_masks: (B, n_masks, H, W) masks if available, else None
        - n_masks: Number of masks per image (adjusted for model support)
        - supports_multiple: Whether model supports multiple messages
    """
    original_batch = samples.image.to(device)  # (B, C, H, W)

    if isinstance(samples, DataPoint):
        original = samples.image
        dataset_masks = samples.masks
        if dataset_masks is not None:
            # For annotation-backed datasets, the dataset returns all masks for
            # each image. The runner is responsible for filtering and sampling
            # them deterministically before they reach the model.
            n_masks = dataset_masks.shape[1]
        else:
            n_masks = 1
    else:
        original = samples
        dataset_masks = None
        n_masks = 1

    # Check if model supports multiple messages per image
    supports_multiple = getattr(model, "supports_multiple_messages", True)
    if not supports_multiple and n_masks > 1:
        print(f"Warning: Model '{model.name}' only supports one message per image, "
              f"but dataset has {n_masks} masks. The runner will sample one valid "
              f"annotation mask per image.")
        n_masks = 1

    original = original.to(device)

    return original_batch, original, dataset_masks, n_masks, supports_multiple


def _compute_mask_area_ratios(masks: torch.Tensor) -> torch.Tensor:
    """Return per-mask area ratio for a batched mask tensor.

    Args:
        masks: Tensor shaped (B, N, H, W)

    Returns:
        Tensor shaped (B, N) with area ratios in [0, 1].
    """
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape (B, N, H, W), got {tuple(masks.shape)}")

    return masks.float().flatten(start_dim=2).mean(dim=2)


def _prefers_bbox_localization(model: WatermarkModel) -> bool:
    """Whether a model should be benchmarked with bbox-rasterized localization."""
    return getattr(model, "name", "").lower() in {"trustmark", "tm", "trust_mark"}


def _rasterize_masks_to_bboxes(masks: torch.Tensor) -> torch.Tensor:
    """Convert a batched mask tensor to tight rectangular bbox masks."""
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with shape (B, N, H, W), got {tuple(masks.shape)}")

    batch_size, num_masks, height, width = masks.shape
    bbox_masks = torch.zeros_like(masks, dtype=torch.float32, device=masks.device)
    for img_idx in range(batch_size):
        for mask_idx in range(num_masks):
            bbox_masks[img_idx, mask_idx] = bbox_to_mask(
                mask_to_bbox(masks[img_idx, mask_idx]),
                height,
                width,
                device=masks.device,
                dtype=torch.float32,
            )
    return bbox_masks


def _select_annotation_masks_per_image(
    dataset_masks: torch.Tensor,
    supports_multiple: bool,
    min_area_ratio: float,
    batch_idx: int,
    mask_seed: Optional[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Filter annotation masks by area and sample them per image.

    This fixes the previous batch-shared-index behavior. Each image now samples
    from its own valid annotation masks, and padded zero channels are never
    selected.

    Args:
        dataset_masks: Annotation masks shaped (B, N, H, W)
        supports_multiple: Whether the model supports multiple messages/masks
        min_area_ratio: Minimum accepted mask area ratio in [0, 1]
        batch_idx: Current batch index
        mask_seed: Base seed for deterministic sampling
        device: Target device

    Returns:
        Tuple of:
        - filtered/sampled localization maps
        - valid-image flags shaped (B,)
    """
    localization_map_batch = dataset_masks.to(device)
    batch_size, _, height, width = localization_map_batch.shape
    area_ratios = _compute_mask_area_ratios(localization_map_batch)

    # Even when min_area_ratio is zero, never allow padded all-zero masks.
    effective_min_area = max(float(min_area_ratio), 0.0)
    if effective_min_area > 0.0:
        valid_mask_flags = area_ratios >= effective_min_area
    else:
        valid_mask_flags = area_ratios > 0.0

    valid_image_flags = valid_mask_flags.any(dim=1)

    if supports_multiple:
        max_valid_masks = int(valid_mask_flags.sum(dim=1).max().item()) if valid_image_flags.any() else 0
        if max_valid_masks == 0:
            empty = torch.zeros((batch_size, 1, height, width), dtype=localization_map_batch.dtype, device=device)
            return empty, valid_image_flags

        filtered_masks = torch.zeros(
            (batch_size, max_valid_masks, height, width),
            dtype=localization_map_batch.dtype,
            device=device,
        )

        for img_idx in range(batch_size):
            valid_indices = torch.nonzero(valid_mask_flags[img_idx], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            selected_masks = localization_map_batch[img_idx, valid_indices]
            filtered_masks[img_idx, :selected_masks.shape[0]] = selected_masks

        return filtered_masks, valid_image_flags

    sampled_masks = torch.zeros(
        (batch_size, 1, height, width),
        dtype=localization_map_batch.dtype,
        device=device,
    )

    base_seed = mask_seed + batch_idx if mask_seed is not None else None

    for img_idx in range(batch_size):
        candidate_indices = torch.nonzero(valid_mask_flags[img_idx], as_tuple=False).flatten().tolist()
        if not candidate_indices:
            continue

        if base_seed is not None:
            img_rng = random.Random((base_seed * 1_000_003) + img_idx)
            selected_idx = img_rng.choice(candidate_indices)
        else:
            selected_idx = random.choice(candidate_indices)
        sampled_masks[img_idx, 0] = localization_map_batch[img_idx, selected_idx]

    return sampled_masks, valid_image_flags


def _filter_batch_to_valid_images(
    image_names: List[str],
    dataset_indices: List[int],
    original_batch: torch.Tensor,
    original: torch.Tensor,
    localization_map_batch: torch.Tensor,
    valid_image_flags: torch.Tensor,
) -> Tuple[List[str], List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop images whose annotation masks do not meet the validity threshold."""
    keep_indices = torch.nonzero(valid_image_flags, as_tuple=False).flatten()

    filtered_names = [image_names[idx] for idx in keep_indices.tolist()]
    filtered_dataset_indices = [dataset_indices[idx] for idx in keep_indices.tolist()]
    return (
        filtered_names,
        filtered_dataset_indices,
        original_batch[keep_indices],
        original[keep_indices],
        localization_map_batch[keep_indices],
    )


def _prepare_localization_masks(
    dataset_masks: Optional[torch.Tensor],
    supports_multiple: bool,
    model: WatermarkModel,
    original: torch.Tensor,
    batch_size: int,
    n_masks: int,
    batch_idx: int,
    mask_seed: Optional[int],
    annotation_min_area: float,
    device: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Handle mask generation or selection from dataset.

    Args:
        dataset_masks: Masks from dataset (B, n_masks, H, W) or None
        supports_multiple: Whether model supports multiple messages
        original: Original images (B, C, H, W)
        batch_size: Batch size
        n_masks: Number of masks per image
        batch_idx: Current batch index (for debugging)
        mask_seed: Base seed for mask generation/selection
        annotation_min_area: Minimum accepted annotation mask area ratio
        device: Device for tensors

    Returns:
        Tuple of:
        - localization_map_batch: (B, n_masks, H, W) masks on device
        - valid-image flags for annotation-backed datasets, else None
    """
    # Generate random mask for embedding if dataset masks are unavailable
    force_bbox_geometry = _prefers_bbox_localization(model)
    if dataset_masks is not None:
        localization_map_batch, valid_image_flags = _select_annotation_masks_per_image(
            dataset_masks=dataset_masks,
            supports_multiple=supports_multiple,
            min_area_ratio=annotation_min_area,
            batch_idx=batch_idx,
            mask_seed=mask_seed,
            device=device,
        )
        if force_bbox_geometry:
            localization_map_batch = _rasterize_masks_to_bboxes(localization_map_batch)
    else:
        _, _, H, W = original.shape
        mask_generator = RandomMaskGenerator(height=H, width=W, device=device)
        random_mask_kwargs = get_benchmark_random_mask_kwargs()
        if force_bbox_geometry:
            random_mask_kwargs.update(
                {
                    "allow_box": True,
                    "allow_irregular": False,
                    "allow_segmentation": False,
                    "allow_invert": False,
                }
            )
        image_masks = []
        for img_idx in range(batch_size):
            actual_mask_seed = _compute_random_mask_seed(mask_seed, batch_idx, img_idx)
            image_tensor = original[img_idx]
            if actual_mask_seed is not None:
                with _temporary_seed(actual_mask_seed):
                    image_mask = mask_generator.generate(
                        num_masks=n_masks,
                        image_tensor=image_tensor,
                        **random_mask_kwargs,
                    )
            else:
                image_mask = mask_generator.generate(
                    num_masks=n_masks,
                    image_tensor=image_tensor,
                    **random_mask_kwargs,
                )
            image_masks.append(image_mask)
        localization_map_batch = torch.stack(image_masks, dim=0)
        if force_bbox_geometry:
            localization_map_batch = _rasterize_masks_to_bboxes(localization_map_batch)
        valid_image_flags = None

    return localization_map_batch, valid_image_flags


def _generate_messages(
    model: WatermarkModel,
    batch_size: int,
    n_masks: int,
    device: torch.device,
    model_seed: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Generate random messages for watermark embedding.

    Args:
        model: Watermark model (must have message_length or nbits attribute)
        batch_size: Number of images in batch
        n_masks: Number of masks per image
        device: Device to create tensors on
        model_seed: Optional seed for reproducible message generation

    Returns:
        Message batch tensor (B, n_masks, message_length) or None if model
        doesn't specify message length
    """
    message_length = getattr(model, "message_length", getattr(model, "nbits", None))
    if not isinstance(message_length, int) or message_length <= 0:
        return None

    if model_seed is not None:
        with _temporary_seed(model_seed):
            message_batch = torch.randint(
                0,
                2,
                (batch_size, n_masks, message_length),
                dtype=torch.int64,
                device=device,
            )
    else:
        message_batch = torch.randint(
            0,
            2,
            (batch_size, n_masks, message_length),
            dtype=torch.int64,
            device=device,
        )

    return message_batch


def _embed_watermark(
    model: WatermarkModel,
    original_batch: torch.Tensor,
    message_batch: Optional[torch.Tensor],
    localization_map_batch: torch.Tensor,
) -> Tuple[EmbedResult, float]:
    """Embed watermark into images.

    Args:
        model: Watermark model
        original_batch: Original images (B, C, H, W)
        message_batch: Messages (B, n_masks, message_length) or None
        localization_map_batch: Localization masks (B, n_masks, H, W)

    Returns:
        Tuple of:
        - embed_result: Embedding result with watermarked images
        - embed_time_ms: Embedding time in milliseconds
    """
    t0 = time.time()
    embed_result = model.embed(
        original_batch,
        watermark_data=WatermarkData(
            message=message_batch,
            localization_map=localization_map_batch,
        ),
    )
    embed_time_ms = (time.time() - t0) * 1000
    return embed_result, embed_time_ms


def _ensure_4d_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure mask is 4D (B, C, H, W) by adding batch dim if needed.

    Args:
        mask: Mask tensor, either 3D (C, H, W) or 4D (B, C, H, W)

    Returns:
        4D mask tensor with batch dimension
    """
    if mask.dim() == 3:
        return mask.unsqueeze(0)
    return mask


def _compute_mask_stats(mask: Optional[torch.Tensor]) -> dict[str, Optional[float | str]]:
    """Compute union-mask area and bbox stats for an (N, H, W) mask tensor."""
    if mask is None:
        return {
            "mask_area": None,
            "mask_bbox_area": None,
            "mask_bbox": None,
        }

    if mask.dim() == 2:
        mask_2d = (mask > 0.5).float()
    elif mask.dim() == 3:
        mask_2d = (mask.sum(dim=0) > 0.5).float()
    else:
        raise ValueError(f"Expected 2D/3D mask tensor, got {tuple(mask.shape)}")

    height, width = mask_2d.shape
    area = float(mask_2d.mean().item())
    ymin, xmin, ymax, xmax = mask_to_bbox(mask_2d)
    if ymin == ymax or xmin == xmax:
        bbox_area = 0.0
        bbox = "0,0,0,0"
    else:
        bbox_area = float(((ymax - ymin) * (xmax - xmin)) / float(height * width))
        bbox = f"{ymin},{xmin},{ymax},{xmax}"

    return {
        "mask_area": round(area, 6),
        "mask_bbox_area": round(bbox_area, 6),
        "mask_bbox": bbox,
    }


def _compute_embed_metrics_and_save_images(
    image_names: List[str],
    dataset_indices: List[int],
    original_batch: torch.Tensor,
    embed_result: EmbedResult,
    embed_time_ms: float,
    logger: BenchmarkLogger,
    device: str,
) -> Tuple[List[dict], List[dict]]:
    """Prepare batch items by saving images and caching data.

    Args:
        image_names: List of image names in batch
        original_batch: Original images (B, C, H, W)
        embed_result: Embedding result
        embed_time_ms: Total embedding time in ms
        logger: Logger for saving images/tensors
        device: Device for tensors

    Returns:
        Tuple of:
        - batch_items: List of dicts with cached data per image
        - embed_metrics_list: List of embed metrics per image
    """
    watermarked_batch = embed_result.watermarked
    batch_size = original_batch.shape[0]

    # Compute embed metrics for entire batch
    embed_metrics_list = compute_embed_metrics_batch(original_batch, watermarked_batch, {})

    # Pre-process batch: Save static images and cache data
    batch_items = []
    for img_idx in range(batch_size):
        img_name = image_names[img_idx]
        dataset_idx = dataset_indices[img_idx]
        base_name = Path(img_name).stem
        original_single = original_batch[img_idx]
        watermarked_single = watermarked_batch[img_idx]

        wm_data_single = WatermarkData(
            message=embed_result.watermark_data.message[img_idx:img_idx+1] if embed_result.watermark_data.message is not None else None,
            localization_map=embed_result.watermark_data.localization_map[img_idx:img_idx+1] if embed_result.watermark_data.localization_map is not None else None
        )

        embed_result_single = EmbedResult(
            watermarked=watermarked_single,
            watermark_data=wm_data_single,
            embed_time_ms=embed_time_ms / batch_size
        )

        embed_metrics = embed_metrics_list[img_idx]

        orig_path = logger.save_image(original_single, "original", base_name)
        wm_path = logger.save_image(watermarked_single, "watermarked", base_name)

        residual_viz_path = None
        if logger.save_visualizations:
            from markbench.utils.visualization import visualize_residual
            residual_viz = visualize_residual(original_single, watermarked_single)
            residual_viz_path = logger.save_image(
                residual_viz, "visualizations", f"{base_name}_residual"
            )

        original_mask_path = None
        if embed_result.watermark_data.localization_map is not None:
            original_mask = _ensure_4d_mask(embed_result.watermark_data.localization_map[img_idx])
            original_mask_path = logger.save_tensor(original_mask, f"{base_name}_original_mask")

        # Ensure attack gets masks on the same device as the image.
        loc_map = None
        if embed_result.watermark_data.localization_map is not None:
            loc_map = embed_result.watermark_data.localization_map[img_idx]
            if isinstance(loc_map, torch.Tensor) and loc_map.device != watermarked_single.device:
                embed_result.watermark_data.localization_map[img_idx] = loc_map.to(watermarked_single.device)

        batch_items.append({
            "base_name": base_name,
            "img_name": img_name,
            "dataset_idx": dataset_idx,
            "watermarked_single": watermarked_single.to(device=device),
            "embed_metrics": embed_metrics,
            "orig_path": orig_path,
            "wm_path": wm_path,
            "residual_viz_path": residual_viz_path,
            "original_mask_path": original_mask_path,
            "original_message": embed_result.watermark_data.message[img_idx].to(device=device) if embed_result.watermark_data.message is not None else None,
            "original_mask_tensor": embed_result.watermark_data.localization_map[img_idx].to(device=device) if embed_result.watermark_data.localization_map is not None else None
        })

    return batch_items, embed_metrics_list


def _get_normalized_attack_config(attack: AbstractAttack) -> Tuple[dict, str]:
    """Get normalized attack config and JSON representation.

    Args:
        attack: Attack instance

    Returns:
        Tuple of:
        - attack_config_normalized: Normalized config dict
        - attack_config_json: JSON string representation
    """
    from markbench.resume import _normalize_config
    attack_config = _extract_attack_params(attack)
    attack_config_normalized = _normalize_config(attack_config)
    attack_config_json = json.dumps(attack_config_normalized, sort_keys=True)
    return attack_config_normalized, attack_config_json


def _normalize_attack_config_dict(config: dict) -> Tuple[dict, str]:
    """Normalize an already-resolved attack config."""
    from markbench.resume import _normalize_config

    attack_config_normalized = _normalize_config(config)
    attack_config_json = json.dumps(attack_config_normalized, sort_keys=True)
    return attack_config_normalized, attack_config_json


def _is_runtime_attack_spec(attack: AbstractAttack) -> bool:
    return isinstance(attack, RuntimeAttackSpec)


def _attack_uses_batched_call(attack: AbstractAttack) -> bool:
    return hasattr(attack, "__call__batch__") and not _is_runtime_attack_spec(attack)


def _resolve_attack_identity(
    attack: AbstractAttack,
    item: dict,
    attack_seed: Optional[int],
    batch_idx: int,
    global_attack_idx: int,
    img_idx: int,
    device: str,
) -> Tuple[str, dict, str, Optional[int]]:
    """Resolve the concrete attack config and seed for one image."""
    uses_batched_attack = _attack_uses_batched_call(attack)
    actual_attack_seed = _compute_attack_seed(
        attack_seed=attack_seed,
        batch_idx=batch_idx,
        global_attack_idx=global_attack_idx,
        img_idx=None if uses_batched_attack else img_idx,
        batched=uses_batched_attack,
    )

    if _is_runtime_attack_spec(attack):
        resolved_config = attack.resolve_config(
            item=item,
            seed=actual_attack_seed,
            device=device,
        )
        attack_name = attack.name
        config_normalized, config_json = _normalize_attack_config_dict(resolved_config)
        return attack_name, config_normalized, config_json, actual_attack_seed

    config_normalized, config_json = _get_normalized_attack_config(attack)
    return attack.name, config_normalized, config_json, actual_attack_seed


def _should_skip_batch_attack(
    batch_items: List[dict],
    attack: AbstractAttack,
    batch_idx: int,
    global_attack_idx: int,
    attack_seed: Optional[int],
    skip_set: Optional[set],
) -> bool:
    """Check if entire batch should skip this attack (for resume).

    Args:
        batch_items: List of batch item dicts
        attack: Attack to check
        batch_idx: Current batch index
        global_attack_idx: Global attack index
        attack_seed: Base attack seed
        skip_set: Set of completed (image, attack, config, seed) tuples

    Returns:
        True if all images in batch are already complete for this attack
    """
    if skip_set is None or attack_seed is None:
        return False

    # Check if all images in batch are complete
    for img_idx, item in enumerate(batch_items):
        attack_name, _, attack_config_json, actual_attack_seed = _resolve_attack_identity(
            attack=attack,
            item=item,
            attack_seed=attack_seed,
            batch_idx=batch_idx,
            global_attack_idx=global_attack_idx,
            img_idx=img_idx,
            device="",
        )
        skip_key = (
            item["img_name"],
            attack_name,
            attack_config_json,
            actual_attack_seed,
        )
        if skip_key not in skip_set:
            return False  # At least one image not complete

    return True  # All images complete


def _apply_attacks_to_batch(
    attack: AbstractAttack,
    batch_items: List[dict],
    batch_idx: int,
    global_attack_idx: int,
    attack_seed: Optional[int],
    device: str,
) -> Tuple[List[torch.Tensor], List[Tuple[Any, float]], List[Optional[int]], List[Tuple[str, dict, str]]]:
    """Apply attack to all images in batch.

    Args:
        attack: Attack to apply
        batch_items: List of dicts with watermarked images and masks
        batch_idx: Current batch index
        global_attack_idx: Global attack index
        attack_seed: Base seed for attacks
        device: Device for tensors

    Returns:
        Tuple of:
        - attacked_images_list: List of attacked images
        - attack_results_list: List of (attack_result, time_ms) tuples
        - actual_attack_seeds: Computed attack seeds (one per image, for logging)
        - resolved_attack_infos: (attack_name, normalized_config, config_json) per image
    """
    batch_size = len(batch_items)
    uses_runtime_spec = _is_runtime_attack_spec(attack)
    uses_batched_attack = _attack_uses_batched_call(attack)
    batch_attack_seed = _compute_attack_seed(
        attack_seed=attack_seed,
        batch_idx=batch_idx,
        global_attack_idx=global_attack_idx,
        batched=uses_batched_attack,
    )

    attacked_images_list = []
    attack_results_list = []
    actual_attack_seeds: List[Optional[int]] = []
    resolved_attack_infos: List[Tuple[str, dict, str]] = []

    if uses_runtime_spec:
        item_attack_seeds = [
            _compute_attack_seed(
                attack_seed=attack_seed,
                batch_idx=batch_idx,
                global_attack_idx=global_attack_idx,
                img_idx=img_idx,
            )
            for img_idx in range(batch_size)
        ]
        resolved_attack_infos = [
            _resolve_attack_identity(
                attack=attack,
                item=item,
                attack_seed=attack_seed,
                batch_idx=batch_idx,
                global_attack_idx=global_attack_idx,
                img_idx=img_idx,
                device=device,
            )[:3]
            for img_idx, item in enumerate(batch_items)
        ]

        t0 = time.time()
        batch_results = []
        for item, item_attack_seed in zip(batch_items, item_attack_seeds):
            if item_attack_seed is not None:
                with _temporary_seed(item_attack_seed):
                    batch_results.append(
                        attack.apply_to_item(item=item, seed=item_attack_seed, device=device)
                    )
            else:
                batch_results.append(
                    attack.apply_to_item(item=item, seed=item_attack_seed, device=device)
                )

        total_attack_time_ms = (time.time() - t0) * 1000
        avg_attack_time_ms = total_attack_time_ms / batch_size if batch_size else 0.0

        for item_attack_seed, attack_result in zip(item_attack_seeds, batch_results):
            if "is_geometry_modifying" not in attack_result.metadata:
                attack_result.metadata["is_geometry_modifying"] = attack.is_geometry_modifying
            attack_results_list.append((attack_result, avg_attack_time_ms))
            attacked_images_list.append(attack_result.attacked)
            actual_attack_seeds.append(item_attack_seed)

    # Apply attack to each image in the batch
    # Check if this attack supports batching (e.g., SD inpainting)
    elif uses_batched_attack:
        # Batched attack processing
        print(f"[Runner] Detected batched attack support for {attack.__class__.__name__}, batch_size={batch_size}")
        watermarked_images = torch.stack([item["watermarked_single"] for item in batch_items])

        # Stack localization maps if they exist
        localization_maps = None
        if batch_items[0]["original_mask_tensor"] is not None:
            localization_maps = torch.stack([item["original_mask_tensor"] for item in batch_items])

        t0 = time.time()
        if batch_attack_seed is not None:
            with _temporary_seed(batch_attack_seed):
                batch_results = attack.__call__batch__(watermarked_images, localization_maps)
        else:
            batch_results = attack.__call__batch__(watermarked_images, localization_maps)

        total_attack_time_ms = (time.time() - t0) * 1000
        avg_attack_time_ms = total_attack_time_ms / batch_size

        # Populate results lists
        attack_results_list = [(result, avg_attack_time_ms) for result in batch_results]
        attacked_images_list = [result.attacked for result in batch_results]
        actual_attack_seeds = [batch_attack_seed] * batch_size
        resolved_attack_infos = [
            _resolve_attack_identity(
                attack=attack,
                item=item,
                attack_seed=attack_seed,
                batch_idx=batch_idx,
                global_attack_idx=global_attack_idx,
                img_idx=img_idx,
                device=device,
            )[:3]
            for img_idx, item in enumerate(batch_items)
        ]

        # Add geometry metadata if not already present
        for attack_result, _ in attack_results_list:
            if "is_geometry_modifying" not in attack_result.metadata:
                attack_result.metadata["is_geometry_modifying"] = attack.is_geometry_modifying

    else:
        # Original single-image attack processing
        for img_idx in range(batch_size):
            item = batch_items[img_idx]
            item_attack_seed = _compute_attack_seed(
                attack_seed=attack_seed,
                batch_idx=batch_idx,
                global_attack_idx=global_attack_idx,
                img_idx=img_idx,
            )
            attack_name, config_normalized, config_json, _ = _resolve_attack_identity(
                attack=attack,
                item=item,
                attack_seed=attack_seed,
                batch_idx=batch_idx,
                global_attack_idx=global_attack_idx,
                img_idx=img_idx,
                device=device,
            )

            t0 = time.time()

            if item_attack_seed is not None:
                with _temporary_seed(item_attack_seed):
                    attack_result = attack(
                        item["watermarked_single"],
                        localization_map=item["original_mask_tensor"]
                    )
            else:
                # If no seed is provided, we can't guarantee stackability for random attacks
                attack_result = attack(
                    item["watermarked_single"],
                    localization_map=item["original_mask_tensor"]
                )

            if "is_geometry_modifying" not in attack_result.metadata:
                # For simple attacks (Rotation, Noise etc.), this pulls the static property.
                # For CompositeAttack, the key is dynamically calculated and already exists
                attack_result.metadata["is_geometry_modifying"] = attack.is_geometry_modifying

            attack_time_ms = (time.time() - t0) * 1000
            attack_results_list.append((attack_result, attack_time_ms))
            attacked_images_list.append(attack_result.attacked)
            actual_attack_seeds.append(item_attack_seed)
            resolved_attack_infos.append((attack_name, config_normalized, config_json))

    return attacked_images_list, attack_results_list, actual_attack_seeds, resolved_attack_infos


def _decode_batch_and_compute_attack_metrics(
    model: WatermarkModel,
    attacked_images_list: List[torch.Tensor],
    batch_items: List[dict],
    attack_results_list: List[Tuple[Any, float]],
    attack_name: str,
    attacks: List[AbstractAttack],
    device: str,
) -> Tuple[List[DecodeResult], List[float], List[dict]]:
    """Decode watermark from attacked images and compute attack metrics.

    Args:
        model: Watermark model
        attacked_images_list: List of attacked images
        batch_items: List of batch item dicts
        attack_results_list: List of (attack_result, time_ms) tuples
        attack_name: Name of attack (for error messages)
        attacks: List of all attacks (for metric computation)
        device: Device for tensors

    Returns:
        Tuple of:
        - decode_results_list: Per-image decode results
        - decode_time_ms_list: Per-image decode times in ms
        - attack_metrics_list: List of attack metrics per image
    """
    batch_size = len(batch_items)
    attacked_shapes = [tuple(img.shape) for img in attacked_images_list]
    shapes_are_uniform = len(set(attacked_shapes)) == 1

    def _compute_attack_metrics_list_individually() -> List[dict]:
        metrics_list: List[dict] = []
        for img_idx in range(batch_size):
            watermarked_image = batch_items[img_idx]["watermarked_single"]
            attacked_image = attacked_images_list[img_idx]
            attack_result = attack_results_list[img_idx][0]
            if tuple(watermarked_image.shape) != tuple(attacked_image.shape):
                metrics_list.append({})
                continue
            metrics_list.append(
                compute_attack_metrics(
                    watermarked_image,
                    attack_result,
                    {"attacks": attacks},
                )
            )
        return metrics_list

    if shapes_are_uniform:
        attacked_batch = torch.stack(attacked_images_list)

        t0 = time.time()
        decode_result_batch = model.decode(attacked_batch.to(device=device))
        total_decode_time_ms = (time.time() - t0) * 1000
        avg_decode_time_ms = total_decode_time_ms / batch_size
        decode_time_ms_list = [avg_decode_time_ms] * batch_size

        decode_results_list: List[DecodeResult] = []
        for img_idx in range(batch_size):
            decoded_msg = (
                decode_result_batch.watermark_data.message[img_idx].to(device=device)
                if decode_result_batch.watermark_data.message is not None
                else None
            )
            decoded_loc = (
                decode_result_batch.watermark_data.localization_map[img_idx].to(device=device)
                if decode_result_batch.watermark_data.localization_map is not None
                else None
            )
            decode_results_list.append(
                DecodeResult(
                    watermark_data=WatermarkData(
                        message=decoded_msg,
                        localization_map=decoded_loc,
                    ),
                    decode_time_ms=avg_decode_time_ms,
                    metadata=decode_result_batch.metadata,
                )
            )

        watermarked_batch_for_attack = torch.stack([item["watermarked_single"] for item in batch_items])
        is_geometry_modifying_list = [
            attack_results_list[img_idx][0].metadata.get("is_geometry_modifying", False)
            for img_idx in range(batch_size)
        ]
        if tuple(attacked_batch.shape) == tuple(watermarked_batch_for_attack.shape):
            attack_metrics_list = compute_attack_metrics_batch(
                watermarked_batch_for_attack,
                attacked_batch,
                is_geometry_modifying_list,
                {"attacks": attacks},
            )
        else:
            attack_metrics_list = _compute_attack_metrics_list_individually()
        return decode_results_list, decode_time_ms_list, attack_metrics_list

    print(
        f"[Runner] Attack {attack_name} produced mixed image sizes within the batch "
        f"{attacked_shapes}; falling back to per-image decode."
    )
    decode_results_list = []
    decode_time_ms_list = []
    attack_metrics_list = []
    for img_idx, attacked_image in enumerate(attacked_images_list):
        t0 = time.time()
        decode_result_single_batch = model.decode(attacked_image.unsqueeze(0).to(device=device))
        decode_time_ms = (time.time() - t0) * 1000
        decode_time_ms_list.append(decode_time_ms)

        decoded_msg = (
            decode_result_single_batch.watermark_data.message[0].to(device=device)
            if decode_result_single_batch.watermark_data.message is not None
            else None
        )
        decoded_loc = (
            decode_result_single_batch.watermark_data.localization_map[0].to(device=device)
            if decode_result_single_batch.watermark_data.localization_map is not None
            else None
        )
        decode_results_list.append(
            DecodeResult(
                watermark_data=WatermarkData(
                    message=decoded_msg,
                    localization_map=decoded_loc,
                ),
                decode_time_ms=decode_time_ms,
                metadata=decode_result_single_batch.metadata,
            )
        )

        watermarked_image = batch_items[img_idx]["watermarked_single"]
        if tuple(watermarked_image.shape) != tuple(attacked_image.shape):
            attack_metrics_list.append({})
        else:
            attack_metrics_list.append(
                compute_attack_metrics(
                    watermarked_image,
                    attack_results_list[img_idx][0],
                    {"attacks": attacks},
                )
            )

    return decode_results_list, decode_time_ms_list, attack_metrics_list


def _create_safe_attack_name(attack_name: str, attack_config: dict) -> str:
    """Create filesystem-safe attack name with params.

    Args:
        attack_name: Original attack name
        attack_config: Attack configuration dict

    Returns:
        Safe filename string with attack name and params
    """
    safe_name = attack_name.replace(":", "_").replace(" ", "_")
    param_suffix = _sanitize_params(attack_config)
    if param_suffix:
        return f"{safe_name}_{param_suffix}"
    return safe_name


def _process_and_log_attack_results(
    attack: AbstractAttack,
    batch_items: List[dict],
    attack_results_list: List[Tuple[Any, float]],
    resolved_attack_infos: List[Tuple[str, dict, str]],
    decode_results_list: List[DecodeResult],
    decode_time_ms_list: List[float],
    attack_metrics_list: List[dict],
    embed_time_ms: float,
    actual_attack_seeds: List[Optional[int]],
    actual_mask_seed: Optional[int],
    actual_model_seed: Optional[int],
    skip_set: Optional[set],
    logger: BenchmarkLogger,
    device: str,
) -> None:
    """Save images/masks and log benchmark results for each image in batch.

    Args:
        attack: Attack that was applied
        batch_items: List of batch item dicts
        attack_results_list: List of (attack_result, time_ms) tuples
        resolved_attack_infos: Resolved (attack_name, config, config_json) per image
        decode_results_list: Per-image decode results
        decode_time_ms_list: Per-image decode times
        attack_metrics_list: List of attack metrics per image
        embed_time_ms: Total embed time for batch
        actual_attack_seeds: Attack seeds used, one per image
        actual_mask_seed: Mask seed used
        actual_model_seed: Model seed used
        skip_set: Set of completed (image, attack, config, seed) tuples (for resume)
        logger: Logger for saving and logging
        device: Device for tensors
    """
    batch_size = len(batch_items)

    for img_idx in range(batch_size):
        item = batch_items[img_idx]
        attack_result, attack_time_ms = attack_results_list[img_idx]
        attack_name, attack_config_normalized, attack_config_json = resolved_attack_infos[img_idx]
        base_name = item["base_name"]
        actual_attack_seed = actual_attack_seeds[img_idx]

        # Check if this combination should be skipped
        # Note: This check happens AFTER attack execution, so it only prevents
        # saving duplicate results, not wasted computation.
        # The batch-level check (before attack execution) is the optimization.
        if skip_set is not None:
            skip_key = (
                item["img_name"],
                attack_name,
                attack_config_json,
                actual_attack_seed,
            )

            if skip_key in skip_set:
                # Already completed - skip this image-attack combination
                continue

        decode_result_single = decode_results_list[img_idx]
        decoded_msg = (
            decode_result_single.watermark_data.message.to(device=device)
            if decode_result_single.watermark_data.message is not None
            else None
        )
        decoded_loc = (
            decode_result_single.watermark_data.localization_map.to(device=device)
            if decode_result_single.watermark_data.localization_map is not None
            else None
        )

        attack_metrics = attack_metrics_list[img_idx]

        transformed_watermark_data = WatermarkData(
            message=item["original_message"].to(device=device),
            localization_map=attack_result.localization_map.to(device=device),
        )

        decode_metrics = compute_decode_metrics(
            transformed_watermark_data,
            decode_result_single,
            {"log_messages": logger.log_messages},
        )

        safe_attack = _create_safe_attack_name(attack_name, attack_config_normalized)

        attacked_path = logger.save_image(
            attack_result.attacked, "attacked", f"{base_name}_{safe_attack}"
        )

        visualization_path = None
        visualization_decode_path = None
        if logger.save_visualizations and attack_result.localization_map is not None:
            from markbench.utils.visualization import visualize_mask_transformation
            viz = visualize_mask_transformation(
                original_image=item["watermarked_single"],
                original_masks=item["original_mask_tensor"],
                attacked_image=attack_result.attacked,
                transformed_masks=attack_result.localization_map
            )
            visualization_path = logger.save_image(
                viz, "visualizations", f"{base_name}_{safe_attack}_mask_viz"
            )

            if decoded_loc is not None and decoded_loc.numel() > 0:
                viz_decode = visualize_mask_transformation(
                    original_image=attack_result.attacked,
                    original_masks=attack_result.localization_map,
                    attacked_image=attack_result.attacked,
                    transformed_masks=decoded_loc
                )
                visualization_decode_path = logger.save_image(
                    viz_decode, "visualizations", f"{base_name}_{safe_attack}_decode_viz"
                )

        attacked_mask_path = None
        decoded_mask_path = None

        if attack_result.localization_map is not None:
            attacked_mask = _ensure_4d_mask(attack_result.localization_map)
            attacked_mask_path = logger.save_tensor(attacked_mask, f"{base_name}_{safe_attack}_attacked_mask")

        if decoded_loc is not None:
            decoded_mask = _ensure_4d_mask(decoded_loc)
            decoded_mask_path = logger.save_tensor(decoded_mask, f"{base_name}_{safe_attack}_decoded_mask")

        mask_paths = {
            "original": item["original_mask_path"],
            "attacked": attacked_mask_path,
            "decoded": decoded_mask_path,
        }
        original_mask_stats = _compute_mask_stats(item["original_mask_tensor"])
        attacked_mask_stats = _compute_mask_stats(attack_result.localization_map)

        # Build attack_params with base config + post-run metadata
        attack_params_with_metadata = attack_config_normalized.copy()
        if attack_result.metadata:
            # Only include metadata keys not already in base config
            post_run = {k: v for k, v in attack_result.metadata.items()
                       if k not in attack_config_normalized}
            if post_run:
                attack_params_with_metadata["post_run"] = post_run

        logger.log(
            BenchmarkRow(
                image_name=item["img_name"],
                attack_name=attack_name,
                attack_params=json.dumps(attack_params_with_metadata),
                embed_time_ms=embed_time_ms / batch_size,
                embed_metrics=item["embed_metrics"],
                attack_time_ms=attack_time_ms,
                attack_metrics=attack_metrics,
                decode_time_ms=decode_time_ms_list[img_idx],
                decode_metrics=decode_metrics,
                created_at=datetime.now().isoformat(),
                original_path=item["orig_path"],
                watermarked_path=item["wm_path"],
                attacked_path=attacked_path,
                visualization_path=visualization_path,
                visualization_decode_path=visualization_decode_path,
                mask_paths=mask_paths,
                mask_area=original_mask_stats["mask_area"],
                mask_bbox_area=original_mask_stats["mask_bbox_area"],
                mask_bbox=original_mask_stats["mask_bbox"],
                attacked_mask_area=attacked_mask_stats["mask_area"],
                attacked_mask_bbox_area=attacked_mask_stats["mask_bbox_area"],
                attacked_mask_bbox=attacked_mask_stats["mask_bbox"],
                attack_seed=actual_attack_seed,
                mask_seed=actual_mask_seed,
                model_seed=actual_model_seed,
            )
        )


def run_benchmark(
    dataset: ImageDataset,
    model: WatermarkModel,
    attacks: List[AbstractAttack],
    logger: BenchmarkLogger,
    resize_images: bool = True,
    device: str = "",
    attack_seed: int = None,
    mask_seed: int = None,
    model_seed: int = None,
    batch_size: int = 4,
    num_workers: int = 0,
    skip_set: Optional[set] = None,
    attack_global_indices: Optional[List[Tuple[int, AbstractAttack]]] = None,
    annotation_min_area: float = 0.02,
) -> None:
    """Run full benchmark pipeline.

    All metric computation should happen here.

    For each image:
      1. Embed watermark
      2. Compute embed metrics
      3. Save original + watermarked images
      4. For each attack:
         a. Apply attack
         b. Compute attack metrics
         c. Decode watermark
         e. Compute decode metrics
         f. Save attacked image
         g. Log result row

    Args:
        dataset: Image dataset to process
        model: Watermark model implementing embed() and decode()
        attacks: List of attack instances to apply
        logger: Logger for writing results
        device: Device for model inference (default: auto-detect)
        attack_seed: Base seed for attack randomness (optional)
        mask_seed: Base seed for random mask generation (optional)
        model_seed: Base seed for model message generation (optional)
        batch_size: Batch size for processing (default: 4)
        num_workers: Number of worker processes for data loading (default: 0)
        annotation_min_area: Minimum annotation mask area ratio. Images whose
            annotation masks all fall below this threshold are skipped.
    """
    device = get_device(device)

    # Backwards compatibility: create trivial mapping if not provided
    if attack_global_indices is None:
        attack_global_indices = [(idx, attack) for idx, attack in enumerate(attacks)]

    for _, attack in attack_global_indices:
        if _is_runtime_attack_spec(attack):
            attack.set_runtime_context(dataset)

    collator = CustomCollator(mode="resize" if resize_images else "pad", target_size=(512, 512))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator)

    for batch_idx, (image_names, samples) in enumerate(
        tqdm(dataloader, desc="Batches", position=0)
    ):
        actual_mask_seed = mask_seed + batch_idx if mask_seed is not None else None
        actual_model_seed = model_seed + batch_idx if model_seed is not None else None
        dataset_indices = list(range(batch_idx * batch_size, (batch_idx * batch_size) + len(image_names)))

        # Prepare batch data
        original_batch, original, dataset_masks, n_masks, supports_multiple = _load_and_prepare_batch(
            samples=samples,
            model=model,
            device=device,
        )
        B = original_batch.shape[0]

        # Handle masks (generate or select from dataset)
        localization_map_batch, valid_image_flags = _prepare_localization_masks(
            dataset_masks=dataset_masks,
            supports_multiple=supports_multiple,
            model=model,
            original=original,
            batch_size=B,
            n_masks=n_masks,
            batch_idx=batch_idx,
            mask_seed=mask_seed,
            annotation_min_area=annotation_min_area,
            device=device,
        )

        if valid_image_flags is not None and not torch.all(valid_image_flags):
            skipped_count = int((~valid_image_flags).sum().item())
            kept_count = int(valid_image_flags.sum().item())
            print(
                f"[Runner] Batch {batch_idx}: skipping {skipped_count} image(s) "
                f"without annotation masks meeting min area {annotation_min_area:.4f}; "
                f"keeping {kept_count}."
            )

            if kept_count == 0:
                continue

            image_names, dataset_indices, original_batch, original, localization_map_batch = _filter_batch_to_valid_images(
                image_names=image_names,
                dataset_indices=dataset_indices,
                original_batch=original_batch,
                original=original,
                localization_map_batch=localization_map_batch,
                valid_image_flags=valid_image_flags,
            )
            B = original_batch.shape[0]

        # Generate message outside model for reproducibility if supported
        # Determine number of masks for message generation
        if localization_map_batch is not None:
            n_masks_msg = localization_map_batch.shape[1]
        else:
            n_masks_msg = 1

        message_batch = _generate_messages(
            model=model,
            batch_size=B,
            n_masks=n_masks_msg,
            device=original.device,
            model_seed=actual_model_seed,
        )

        # Embed watermark
        embed_result, embed_time_ms = _embed_watermark(
            model=model,
            original_batch=original_batch,
            message_batch=message_batch,
            localization_map_batch=localization_map_batch,
        )

        # Prepare batch items: save images and cache data
        batch_items, embed_metrics_list = _compute_embed_metrics_and_save_images(
            image_names=image_names,
            dataset_indices=dataset_indices,
            original_batch=original_batch,
            embed_result=embed_result,
            embed_time_ms=embed_time_ms,
            logger=logger,
            device=device,
        )

        # 2. Iterate over Attacks
        for local_idx, (global_attack_idx, attack) in enumerate(
            tqdm(attack_global_indices, desc="Attacks", position=1, leave=False)
        ):
            # Check if all images in this batch are already complete for this attack
            if _should_skip_batch_attack(batch_items, attack, batch_idx, global_attack_idx, attack_seed, skip_set):
                continue

            # Apply attack to batch
            attacked_images_list, attack_results_list, actual_attack_seeds, resolved_attack_infos = _apply_attacks_to_batch(
                attack=attack,
                batch_items=batch_items,
                batch_idx=batch_idx,
                global_attack_idx=global_attack_idx,
                attack_seed=attack_seed,
                device=device,
            )

            batch_size = len(batch_items)

            # Decode batch and compute attack metrics
            decode_results_list, decode_time_ms_list, attack_metrics_list = _decode_batch_and_compute_attack_metrics(
                model=model,
                attacked_images_list=attacked_images_list,
                batch_items=batch_items,
                attack_results_list=attack_results_list,
                attack_name=attack.name,
                attacks=attacks,
                device=device,
            )

            # Save images/masks and log results per image
            _process_and_log_attack_results(
                attack=attack,
                batch_items=batch_items,
                attack_results_list=attack_results_list,
                resolved_attack_infos=resolved_attack_infos,
                decode_results_list=decode_results_list,
                decode_time_ms_list=decode_time_ms_list,
                attack_metrics_list=attack_metrics_list,
                embed_time_ms=embed_time_ms,
                actual_attack_seeds=actual_attack_seeds,
                actual_mask_seed=actual_mask_seed,
                actual_model_seed=actual_model_seed,
                skip_set=skip_set,
                logger=logger,
                device=device,
            )


def _extract_attack_params(attack: AbstractAttack) -> dict:
    """Extract parameters from attack instance."""
    return attack.get_config()

def _sanitize_params(params: dict) -> str:
    """Create filename-safe suffix from attack params."""
    if not params:
        return ""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, (int, float, str)):
            items.append(f"{k}{v}")
        # elif isinstance(v, str):
        #     items.append(f"{k}{v}")
        elif isinstance(v, list) and len(v) <= 2:
            items.append(f"{k}{'_'.join(map(str, v))}")
    return "_".join(items) if items else ""

class CustomCollator:
    """
    Collator that handles variable resolution images by either:
    1. 'resize': Resizing everything to a fixed target size (Default).
    2. 'pad': Padding everything to the largest image in the batch.
    """
    def __init__(self, mode: str = "resize", target_size: tuple = (512, 512)):
        self.mode = mode
        self.target_size = target_size

    def __call__(self, batch):
        names, datapoints = zip(*batch)
        
        padded_images = []
        padded_masks = []
        has_masks = datapoints[0].masks is not None

        if self.mode == "resize":
            # Find max number of masks across all datapoints
            if has_masks:
                max_n_masks = max([dp.masks.shape[0] if dp.masks is not None else 0 for dp in datapoints])

            for dp in datapoints:
                img_resized = F.interpolate(
                    dp.image.unsqueeze(0),
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                padded_images.append(img_resized)

                if has_masks:
                    if dp.masks is not None:
                        mask_resized = F.interpolate(
                            dp.masks.unsqueeze(0),
                            size=self.target_size,
                            mode='nearest'
                        ).squeeze(0)

                        # Pad the number of masks dimension if needed
                        n_masks = mask_resized.shape[0]
                        if n_masks < max_n_masks:
                            pad_n_masks = max_n_masks - n_masks
                            mask_resized = F.pad(mask_resized, (0, 0, 0, 0, 0, pad_n_masks), mode='constant', value=0)

                        padded_masks.append(mask_resized)
                    else:
                        # Create a zero tensor for missing masks to maintain batch consistency
                        h, w = self.target_size
                        zero_masks = torch.zeros((max_n_masks, h, w), dtype=torch.float32)
                        padded_masks.append(zero_masks)

        elif self.mode == "pad":
            max_h = max([dp.image.shape[1] for dp in datapoints])
            max_w = max([dp.image.shape[2] for dp in datapoints])

            # Find max number of masks across all datapoints
            if has_masks:
                max_n_masks = max([dp.masks.shape[0] if dp.masks is not None else 0 for dp in datapoints])

            for dp in datapoints:
                c, h, w = dp.image.shape
                pad_w = max_w - w
                pad_h = max_h - h
                padding = (0, pad_w, 0, pad_h)

                img_padded = F.pad(dp.image, padding, mode='constant', value=0)
                padded_images.append(img_padded)

                if has_masks:
                    if dp.masks is not None:
                        mask_padded = F.pad(dp.masks, padding, mode='constant', value=0)

                        n_masks = mask_padded.shape[0]
                        if n_masks < max_n_masks:
                            pad_n_masks = max_n_masks - n_masks
                            # F.pad works from last dim backwards, so for [n_masks, H, W]:
                            # we need to add padding (0, 0, 0, 0, 0, pad_n_masks)
                            mask_padded = F.pad(mask_padded, (0, 0, 0, 0, 0, pad_n_masks), mode='constant', value=0)

                        padded_masks.append(mask_padded)
                    else:
                        # Create a zero tensor for missing masks to maintain batch consistency
                        zero_masks = torch.zeros((max_n_masks, max_h, max_w), dtype=torch.float32)
                        padded_masks.append(zero_masks)
        
        batched_images = torch.stack(padded_images)
        batched_masks = torch.stack(padded_masks) if has_masks else None
        
        batched_dp = DataPoint(image=batched_images, masks=batched_masks)
        return names, batched_dp
