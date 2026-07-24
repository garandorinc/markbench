import cv2
import numpy as np
import torch


def extract_object_components(
    mask_tensor: torch.Tensor, image_tensor: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract individual objects from an image based on a binary mask.

    Args:
        mask_tensor: Binary mask (1, H, W) or (H, W).
        image_tensor: Source image (C, H, W).

    Returns:
        List of tuples containing (cropped_image_chunk, cropped_mask_chunk).
        Both tensors in each tuple have shape (C, h_new, w_new) and (1, h_new, w_new).
    """
    original_device = mask_tensor.device

    mask_2d = mask_tensor.squeeze(0) if mask_tensor.dim() == 3 else mask_tensor

    mask_np = mask_2d.detach().cpu().numpy()
    mask_uint8 = (mask_np > 0.5).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    components = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        labels_crop = labels[y : y + h, x : x + w]

        comp_mask_np = (labels_crop == i).astype(np.float32)

        comp_mask_tensor = (
            torch.from_numpy(comp_mask_np).to(original_device).unsqueeze(0)
        )

        comp_img_tensor = image_tensor[:, y : y + h, x : x + w].clone()
        comp_img_tensor = comp_img_tensor * comp_mask_tensor

        components.append((comp_img_tensor, comp_mask_tensor))

    return components
