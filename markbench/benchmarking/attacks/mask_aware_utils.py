import random
from typing import Tuple
import torch


def calculate_mask_metrics(
    loc_map: torch.Tensor,
    region_x: int,
    region_y: int,
    region_w: int,
    region_h: int,
    operation: str = "delete"
) -> Tuple[float, float]:
    total_mask = loc_map.sum().item()
    if total_mask == 0:
        return 0.0, 1.0

    region_mask = loc_map[:, region_y:region_y+region_h, region_x:region_x+region_w]
    overlap = region_mask.sum().item() / total_mask

    if operation == "delete":
        retention = 1.0 - overlap
    else:
        retention = overlap

    return overlap, retention


def find_mask_aware_region(
    localization_map: torch.Tensor,
    region_w: int,
    region_h: int,
    img_h: int,
    img_w: int,
    min_retention: float = 0.25,
    num_candidates: int = 100,
    operation: str = "delete"
) -> Tuple[int, int, float, float]:
    if region_w > img_w or region_h > img_h:
        return 0, 0, 0.0, 0.0

    combined_mask = localization_map.sum(dim=0, keepdim=True) if localization_map.dim() == 3 else localization_map.unsqueeze(0)

    candidates = []
    for _ in range(num_candidates):
        x = random.randint(0, img_w - region_w)
        y = random.randint(0, img_h - region_h)

        overlap, retention = calculate_mask_metrics(
            combined_mask, x, y, region_w, region_h, operation
        )

        candidates.append((x, y, overlap, retention))

    valid_candidates = [c for c in candidates if c[3] >= min_retention]

    if valid_candidates:
        best = max(valid_candidates, key=lambda c: c[2])
    else:
        best = max(candidates, key=lambda c: c[3])

    return best


def find_mask_aware_splice_position(
    localization_map: torch.Tensor,
    splice_w: int,
    splice_h: int,
    bg_h: int,
    bg_w: int,
    current_retention: float,
    min_retention: float = 0.25,
    num_candidates: int = 50
) -> Tuple[int, int]:
    if splice_w > bg_w or splice_h > bg_h:
        return 0, 0

    combined_mask = localization_map.sum(dim=0, keepdim=True) if localization_map.dim() == 3 else localization_map.unsqueeze(0)

    candidates = []
    for _ in range(num_candidates):
        x = random.randint(0, bg_w - splice_w)
        y = random.randint(0, bg_h - splice_h)

        overlap, retention_after = calculate_mask_metrics(
            combined_mask, x, y, splice_w, splice_h, "delete"
        )

        actual_retention = current_retention * retention_after

        candidates.append((x, y, overlap, actual_retention))

    valid_candidates = [c for c in candidates if c[3] >= min_retention]

    if valid_candidates:
        best = max(valid_candidates, key=lambda c: c[2])
        return best[0], best[1]
    else:
        best = max(candidates, key=lambda c: c[3])
        return best[0], best[1]
