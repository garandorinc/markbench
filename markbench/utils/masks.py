import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def mask_to_bbox(
    mask: Float[Tensor, "height width"]
) -> tuple[int, int, int, int]:
    """Convert binary mask to bounding box with exclusive upper bounds.

    Args:
        mask: Binary mask tensor of shape (H, W)

    Returns:
        Tuple (ymin, xmin, ymax, xmax).
        The box covers the region [ymin, ymax) and [xmin, xmax).
        Returns (0, 0, 0, 0) for empty masks.
    """
    mask_binary = (mask > 0).float()

    rows = mask_binary.sum(dim=1) > 0
    cols = mask_binary.sum(dim=0) > 0

    row_indices = torch.where(rows)[0]
    col_indices = torch.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return 0, 0, 0, 0

    ymin = row_indices[0].item()
    ymax = row_indices[-1].item() + 1
    
    xmin = col_indices[0].item()
    xmax = col_indices[-1].item() + 1

    return ymin, xmin, ymax, xmax


@jaxtyped(typechecker=typechecker)
def bbox_to_mask(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Float[Tensor, "height width"]:
    """Rasterize a bounding box into a binary mask.

    Args:
        bbox: Bounding box (ymin, xmin, ymax, xmax) with exclusive upper bounds.
        height: Output height.
        width: Output width.
        device: Optional tensor device.
        dtype: Output dtype.

    Returns:
        Binary mask tensor with ones inside the bbox and zeros elsewhere.
    """
    ymin, xmin, ymax, xmax = bbox
    mask = torch.zeros((height, width), dtype=dtype, device=device)
    if ymax <= ymin or xmax <= xmin:
        return mask

    ymin = max(0, min(int(ymin), height))
    ymax = max(0, min(int(ymax), height))
    xmin = max(0, min(int(xmin), width))
    xmax = max(0, min(int(xmax), width))
    if ymax <= ymin or xmax <= xmin:
        return mask

    mask[ymin:ymax, xmin:xmax] = 1.0
    return mask


@jaxtyped(typechecker=typechecker)
def mask_to_bbox_mask(
    mask: Float[Tensor, "height width"]
) -> Float[Tensor, "height width"]:
    """Convert a binary mask into its tight rectangular bbox mask."""
    height, width = mask.shape
    bbox = mask_to_bbox(mask)
    return bbox_to_mask(
        bbox,
        height,
        width,
        device=mask.device,
        dtype=mask.dtype if torch.is_floating_point(mask) else torch.float32,
    )


@jaxtyped(typechecker=typechecker)
def bbox_iou(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int]
) -> float:
    """Compute IoU between two bounding boxes (exclusive max coordinates).

    Args:
        bbox1: (ymin, xmin, ymax, xmax)
        bbox2: (ymin, xmin, ymax, xmax)

    Returns:
        IoU value in [0, 1]. Returns 0.0 if union is 0.
    """
    y1_min, x1_min, y1_max, x1_max = bbox1
    y2_min, x2_min, y2_max, x2_max = bbox2

    area1 = (y1_max - y1_min) * (x1_max - x1_min)
    area2 = (y2_max - y2_min) * (x2_max - x2_min)

    if area1 <= 0 or area2 <= 0:
        return 0.0

    inter_y_min = max(y1_min, y2_min)
    inter_x_min = max(x1_min, x2_min)
    inter_y_max = min(y1_max, y2_max)
    inter_x_max = min(x1_max, x2_max)

    inter_h = max(0, inter_y_max - inter_y_min)
    inter_w = max(0, inter_x_max - inter_x_min)
    
    inter_area = inter_h * inter_w
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area
