from __future__ import annotations
import os
import random
from typing import Optional, Tuple, List, NamedTuple

import cv2
import numpy as np
import torch

from markbench.interface import AttackResult
from markbench.benchmarking.attacks import AbstractAttack
from markbench.data.random_mask_generator import RandomMaskGenerator
from markbench.utils.image import load_image, save_image
from ..registry import register_attack


class ComponentWithPosition(NamedTuple):
    """Component data with source position information."""
    image_chunk: torch.Tensor  # (C, h, w)
    mask_chunk: torch.Tensor   # (1, h, w)
    source_x: int              # Original x position
    source_y: int              # Original y position
    source_w: int              # Original width
    source_h: int              # Original height


@register_attack("reversed_splicing", aliases=["reverse_splice", "extract_splice"])
class ReversedSplicingAttack(AbstractAttack):

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        background_path: str,
        mask: Optional[torch.Tensor] = None,
        mask_mode: str = "auto",  # "auto", "box", "irregular", "segmentation", "user"
        num_splices: Optional[int] = None,
        resize_background: bool = True,
        mask_aware: bool = False,
        min_retention: float = 0.25,
    ):
        """
        Reversed splicing: Extract masked regions from original image and splice onto background.

        This is the reverse of normal splicing - instead of adding external content to the image,
        we extract parts of the original watermarked image and place them on a clean background.

        Args:
            background_path: Path to background image to splice onto
            mask: Optional user-provided mask tensor (H, W) or (1, H, W) in [0, 1]
            mask_mode: Mask generation mode:
                - "auto": Use RandomMaskGenerator with any mask type allowed
                - "box": Generate rectangular masks
                - "irregular": Generate irregular masks
                - "segmentation": Use segmentation-based masks
                - "user": Use provided mask parameter
            num_splices: Number of components to extract and splice:
                - None: Use all available components (default: None)
                - -1: Random number (between 1 and all available)
                - > 0: Use exactly that many components
            resize_background: Whether to resize background to match input image size (default: True)
        """
        super().__init__()
        self.background_path = background_path
        self.user_mask = mask
        self.mask_mode = mask_mode
        self.num_splices = num_splices
        self.resize_background = resize_background
        self.mask_aware = mask_aware
        self.min_retention = min_retention
        self._background: Optional[torch.Tensor] = None
        self._initialized = False

    def _initialize_background(self, device: torch.device, target_size: Optional[Tuple[int, int]] = None):
        """Load and optionally resize background image."""
        if self._initialized:
            return

        if not self.background_path or not os.path.exists(self.background_path):
            raise ValueError(f"Background image not found: {self.background_path}")

        try:
            # Load background image [0, 255]
            bg_tensor = load_image(self.background_path, device=str(device))

            # Resize if needed
            if self.resize_background and target_size is not None:
                h, w = target_size
                bg_h, bg_w = bg_tensor.shape[1], bg_tensor.shape[2]
                if (bg_h, bg_w) != (h, w):
                    # Resize using torch interpolate
                    bg_tensor = torch.nn.functional.interpolate(
                        bg_tensor.unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

            self._background = bg_tensor
        except Exception as e:
            raise ValueError(f"ReversedSplicingAttack initialization failed: {e}")

        self._initialized = True

    def _extract_components_with_positions(
        self,
        mask_tensor: torch.Tensor,
        image_tensor: torch.Tensor
    ) -> List[ComponentWithPosition]:
        """Extract components with their source positions tracked.

        Args:
            mask_tensor: Binary mask (1, H, W)
            image_tensor: Source image (C, H, W)

        Returns:
            List of ComponentWithPosition containing chunks and their original positions
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

            components.append(ComponentWithPosition(
                image_chunk=comp_img_tensor,
                mask_chunk=comp_mask_tensor,
                source_x=int(x),
                source_y=int(y),
                source_w=int(w),
                source_h=int(h)
            ))

        return components

    def _generate_or_get_mask(self, image: torch.Tensor, device: torch.device, localization_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate or use provided mask."""
        if self.mask_aware and localization_map is not None:
            # Generate mask targeting watermarked region with target retention
            combined_mask = localization_map.sum(dim=0, keepdim=True)
            target_mask_sum = combined_mask.sum().item() * self.min_retention

            # Start with the localization map and erode it to target retention
            extraction_mask = combined_mask.clone()

            # Simple approach: threshold at progressively higher values until we hit target
            for threshold in torch.linspace(0.1, 0.9, 20):
                candidate = (extraction_mask > threshold).float()
                if candidate.sum().item() <= target_mask_sum:
                    return candidate
            return (extraction_mask > 0.5).float()

        elif self.mask_mode == "user" and self.user_mask is not None:
            # Use user-provided mask
            return self._normalize_mask(self.user_mask.to(device))
        else:
            # Generate mask using RandomMaskGenerator
            try:
                mask_gen = RandomMaskGenerator(
                    height=image.shape[1],
                    width=image.shape[2],
                    device=str(device),
                )

                # Set mask generation parameters based on mode
                allow_box = self.mask_mode in ["auto", "box"]
                allow_irregular = self.mask_mode in ["auto", "irregular"]
                allow_segmentation = self.mask_mode in ["auto", "segmentation"]

                raw_mask = mask_gen.generate(
                    allow_box=allow_box,
                    allow_irregular=allow_irregular,
                    allow_segmentation=allow_segmentation,
                    image_tensor=image,
                )

                return self._normalize_mask(raw_mask)

            except Exception as e:
                print(f"Mask generation failed: {e}")
                # Fallback to random rectangular mask
                h, w = image.shape[1], image.shape[2]
                mask = torch.zeros(1, h, w, device=device)
                # Create a random rectangular mask
                mask_h = random.randint(h // 4, h // 2)
                mask_w = random.randint(w // 4, w // 2)
                y = random.randint(0, h - mask_h)
                x = random.randint(0, w - mask_w)
                mask[:, y:y+mask_h, x:x+mask_w] = 1.0
                return mask

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Extract masked regions from original image and splice onto background.

        Args:
            image: Original/foreground tensor (C, H, W) in [0, 255].
            localization_map: Optional localization map tensor (N, H, W).

        Returns:
            AttackResult: Background with spliced regions and transformed localization map.
        """
        device = image.device
        h, w = image.shape[1], image.shape[2]

        # Initialize background
        self._initialize_background(device, target_size=(h, w) if self.resize_background else None)

        if self._background is None:
            raise ValueError("Background not initialized")

        bg_h, bg_w = self._background.shape[1], self._background.shape[2]

        # Generate or get mask
        mask = self._generate_or_get_mask(image, device, localization_map)

        # Extract components from original image using mask with position tracking
        components = self._extract_components_with_positions(mask, image)

        if not components:
            # No components extracted, return background as-is
            return AttackResult(
                attacked=self._background.clone(),
                localization_map=torch.zeros(
                    localization_map.shape[0] if localization_map is not None else 1,
                    bg_h, bg_w,
                    device=device
                ) if localization_map is not None else None,
                metadata={
                    "num_splices": 0,
                    "mask_mode": self.mask_mode,
                    "max_splices": 0,
                    "available_components": 0,
                    "num_splices_param": self.num_splices,
                    "random_selection": self.num_splices == -1
                }
            )

        # Start with background
        out_img = self._background.clone()

        # Create new localization maps on background
        transformed_masks = None
        if localization_map is not None:
            transformed_masks = torch.zeros(
                localization_map.shape[0], bg_h, bg_w,
                dtype=localization_map.dtype,
                device=device
            )

        # Determine how many components to splice
        num_spliced = 0
        available_components = len(components)

        if self.num_splices is None:
            # Use all available components
            max_splices = available_components
        elif self.num_splices == -1:
            # Random number between 1 and all available
            max_splices = random.randint(1, max(1, available_components))
        else:
            # Use exactly num_splices (or all if fewer available)
            max_splices = min(self.num_splices, available_components)

        # Splice components onto background
        for component in components[:max_splices]:
            fg_chunk = component.image_chunk
            mask_chunk = component.mask_chunk
            src_x, src_y = component.source_x, component.source_y
            src_w, src_h = component.source_w, component.source_h

            fg_h, fg_w = fg_chunk.shape[1], fg_chunk.shape[2]

            # Skip if chunk is larger than background
            if fg_h > bg_h or fg_w > bg_w:
                continue

            # Get random position on background
            pos_y = random.randint(0, bg_h - fg_h)
            pos_x = random.randint(0, bg_w - fg_w)

            # Extract ROI from background
            bg_roi = out_img[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w]

            # Blend: (Foreground * Mask) + (Background * (1 - Mask))
            spliced_roi = fg_chunk * mask_chunk + bg_roi * (1.0 - mask_chunk)

            # Place on background
            out_img[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w] = spliced_roi

            # Transform localization maps
            if transformed_masks is not None and localization_map is not None:
                # Extract the corresponding localization map region from source position
                loc_chunk = localization_map[:, src_y : src_y + src_h, src_x : src_x + src_w]

                # Apply mask to localization chunk
                # Only transfer watermark information where the component mask is active
                masked_loc = loc_chunk * mask_chunk  # (N, H, W) * (1, H, W)

                # Place masked localization on background
                # Background localization is already zeros, so we can just add
                transformed_masks[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w] = masked_loc

            num_spliced += 1

        return AttackResult(
            attacked=out_img,
            localization_map=transformed_masks,
            metadata={
                "num_splices": num_spliced,
                "mask_mode": self.mask_mode,
                "max_splices": max_splices,
                "available_components": available_components,
                "num_splices_param": self.num_splices,
                "random_selection": self.num_splices == -1,
                "background_size": (bg_h, bg_w),
                "resize_background": self.resize_background
            }
        )

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Standardizes mask to shape (1, H, W) and float values in [0.0, 1.0]."""
        mask = mask.float()

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 3 and mask.shape[0] >= 3:
            # Convert RGB mask to grayscale using luminance formula
            lum = 0.299 * mask[0] + 0.587 * mask[1] + 0.114 * mask[2]
            mask = lum.unsqueeze(0)

        if mask.max() > 1.0:
            mask = mask / 255.0

        return mask.clamp(0, 1)


if __name__ == "__main__":
    # Test block
    from markbench.utils.device import get_device

    DEVICE = get_device()
    FG_PATH = "./foreground.jpg"  # Original watermarked image
    BG_PATH = "./background.jpg"  # Clean background

    print(f"Running Reversed Splicing Attack on {DEVICE}...")

    # Example 1: Random splices onto background
    attack_random = ReversedSplicingAttack(
        background_path=BG_PATH,
        mask_mode="irregular",
        num_splices=-1,  # Random number
        resize_background=True
    )

    # Example 2: Extract all components onto background
    attack_all = ReversedSplicingAttack(
        background_path=BG_PATH,
        mask_mode="auto",
        num_splices=None,  # All components
        resize_background=True
    )

    # Example 3: User-provided mask for extraction
    if os.path.exists(FG_PATH):
        fg_tensor = load_image(FG_PATH, device=DEVICE)

        # Create example mask
        user_mask = torch.zeros(fg_tensor.shape[1], fg_tensor.shape[2])
        h, w = user_mask.shape
        user_mask[h//4:3*h//4, w//4:3*w//4] = 1.0

        attack_user = ReversedSplicingAttack(
            background_path=BG_PATH,
            mask=user_mask,
            mask_mode="user",
            num_splices=1,
            resize_background=True
        )

        # Test with localization maps
        test_masks = torch.ones(2, fg_tensor.shape[1], fg_tensor.shape[2])

        result = attack_random(fg_tensor, localization_map=test_masks)
        save_image(result.attacked, "result_reversed_random.png")
        print(f"Random mode metadata: {result.metadata}")

        result = attack_all(fg_tensor, localization_map=test_masks)
        save_image(result.attacked, "result_reversed_all.png")
        print(f"All components metadata: {result.metadata}")
    else:
        print("Foreground image not found, skipping test run.")
