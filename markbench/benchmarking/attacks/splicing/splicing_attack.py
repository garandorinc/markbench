from __future__ import annotations
import os
import random
from typing import Optional, Tuple, List

# import cv2
# import numpy as np
import torch

from markbench.interface import AttackResult
from markbench.benchmarking.attacks import AbstractAttack
from markbench.data.random_mask_generator import RandomMaskGenerator
from markbench.utils.device import get_device
from markbench.utils.image import load_image, save_image
from markbench.benchmarking.utils import extract_object_components
from ..registry import register_attack


@register_attack("splicing", aliases=["splice"])
class SplicingAttack(AbstractAttack):
    def __init__(
        self,
        foreground_path: Optional[str] = None,
        mask: Optional[torch.Tensor] = None,
        mask_mode: str = "auto",  # "auto", "box", "irregular", "segmentation", "user"
        num_splices: Optional[int] = None,
        mask_aware: bool = False,
        min_retention: float = 0.25,
    ):
        """
        Args:
            foreground_path: Path to foreground image to splice
            mask: Optional user-provided mask tensor (H, W) or (1, H, W) in [0, 1]
            mask_mode: Mask generation mode:
                - "auto": Use RandomMaskGenerator with any mask type allowed
                - "box": Generate rectangular masks
                - "irregular": Generate irregular masks
                - "segmentation": Use segmentation-based masks
                - "user": Use provided mask parameter
            num_splices: Number of components to splice:
                - None: Use all available components (default: None)
                - -1: Random number (between 1 and all available)
                - > 0: Use exactly that many components
        """
        super().__init__()
        self.foreground_path = foreground_path
        self.user_mask = mask
        self.mask_mode = mask_mode
        self.num_splices = num_splices
        self.mask_aware = mask_aware
        self.min_retention = min_retention
        self._components: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._resolved_mask_mode = mask_mode
        self._initialized = False
        self._initialized_size: Optional[Tuple[int, int]] = None

    @staticmethod
    def _device_str(device: torch.device | str) -> str:
        return get_device(device)

    def _extract_with_mask(self, mask_tensor: torch.Tensor, fg_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return extract_object_components(self._normalize_mask(mask_tensor), fg_tensor)

    def _generate_components_for_mode(
        self,
        mask_gen: RandomMaskGenerator,
        fg_tensor: torch.Tensor,
        mask_mode: str,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        allow_box = mask_mode in ["auto", "box"]
        allow_irregular = mask_mode in ["auto", "irregular"]
        allow_segmentation = mask_mode in ["auto", "segmentation"]

        try:
            raw_mask = mask_gen.generate(
                allow_box=allow_box,
                allow_irregular=allow_irregular,
                allow_segmentation=allow_segmentation,
                image_tensor=fg_tensor,
                allow_invert=False,
                localize_single_irregular=True,
            )
        except Exception:
            return []
        return self._extract_with_mask(raw_mask, fg_tensor)

    def _center_box_components(self, fg_tensor: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        _, height, width = fg_tensor.shape
        box_h = max(16, int(round(height * 0.4)))
        box_w = max(16, int(round(width * 0.4)))
        box_h = min(box_h, height)
        box_w = min(box_w, width)
        y0 = max(0, (height - box_h) // 2)
        x0 = max(0, (width - box_w) // 2)
        mask = torch.zeros((1, height, width), device=fg_tensor.device, dtype=torch.float32)
        mask[:, y0:y0 + box_h, x0:x0 + box_w] = 1.0
        return self._extract_with_mask(mask, fg_tensor)

    def _initialize_components(self, device: torch.device, target_size: Tuple[int, int]):
        """Initialize components from foreground image if provided."""
        if self._initialized and self._initialized_size == target_size:
            return

        self._components = []
        self._initialized = False
        self._initialized_size = target_size

        if not self.foreground_path or not os.path.exists(self.foreground_path):
            self._initialized = True
            return

        device_str = self._device_str(device)
        try:
            # Load foreground image [0, 255]
            fg_tensor = load_image(
                self.foreground_path,
                device=device_str,
                size=(target_size[1], target_size[0]),
            )
            self._resolved_mask_mode = self.mask_mode

            # Generate or use provided mask
            if self.mask_mode == "user" and self.user_mask is not None:
                # Use user-provided mask
                self._components = self._extract_with_mask(self.user_mask.to(device), fg_tensor)
            else:
                # Generate mask using RandomMaskGenerator
                mask_gen = RandomMaskGenerator(
                    height=fg_tensor.shape[1],
                    width=fg_tensor.shape[2],
                    device=device_str,
                )

                self._components = self._generate_components_for_mode(mask_gen, fg_tensor, self.mask_mode)

                if not self._components and self.mask_mode != "box":
                    self._components = self._generate_components_for_mode(mask_gen, fg_tensor, "box")
                    if self._components:
                        self._resolved_mask_mode = f"{self.mask_mode}->box_fallback"

                if not self._components:
                    self._components = self._center_box_components(fg_tensor)
                    if self._components:
                        self._resolved_mask_mode = f"{self.mask_mode}->center_box_fallback"

        except Exception as e:
            raise RuntimeError(f"SplicingAttack initialization failed: {e}") from e

        self._initialized = True

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Splices objects onto the background image.

        Args:
            image: Destination/Background tensor (C, H, W) in [0, 255].
            localization_map: Optional localization map tensor (N, H, W).

        Returns:
            AttackResult: The spliced image with transformed localization map.
        """
        self._initialize_components(image.device, (image.shape[1], image.shape[2]))

        if not self._components:
            return AttackResult(
                attacked=image.clone(),
                localization_map=localization_map,
                metadata={
                    "num_splices": 0,
                    "mask_mode": self.mask_mode,
                    "resolved_mask_mode": self._resolved_mask_mode,
                    "max_splices": 0,
                    "available_components": 0,
                    "num_splices_param": self.num_splices,
                    "random_selection": self.num_splices == -1
                }
            )

        out_img = image.clone()
        bg_h, bg_w = image.shape[1], image.shape[2]

        # Track spliced regions to zero out in masks
        transformed_masks = None
        if localization_map is not None:
            transformed_masks = localization_map.clone()
        attack_mask = torch.zeros((1, bg_h, bg_w), dtype=image.dtype, device=image.device)

        # Determine how many components to splice based on num_splices parameter
        num_spliced = 0
        available_components = len(self._components)

        if self.num_splices is None:
            # Use all available components
            max_splices = available_components
        elif self.num_splices == -1:
            # Random number between 1 and all available
            max_splices = random.randint(1, max(1, available_components))
        else:
            # Use exactly num_splices (or all if fewer available)
            max_splices = min(self.num_splices, available_components)

        current_retention = 1.0

        for fg_chunk, mask_chunk in self._components[:max_splices]:
            fg_h, fg_w = fg_chunk.shape[1], fg_chunk.shape[2]

            # Skip if the chunk is larger than the background
            if fg_h > bg_h or fg_w > bg_w:
                continue

            # Get position
            if self.mask_aware and localization_map is not None:
                from ..mask_aware_utils import find_mask_aware_splice_position
                pos_x, pos_y = find_mask_aware_splice_position(
                    localization_map, fg_w, fg_h, bg_h, bg_w,
                    current_retention, self.min_retention
                )
            else:
                pos_y = random.randint(0, bg_h - fg_h)
                pos_x = random.randint(0, bg_w - fg_w)

            # Extract ROI from background
            bg_roi = out_img[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w]

            # Blend: (Foreground * Mask) + (Background * (1 - Mask))
            spliced_roi = fg_chunk * mask_chunk + bg_roi * (1.0 - mask_chunk)

            # Place back
            out_img[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w] = spliced_roi
            attack_mask[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w] = torch.maximum(
                attack_mask[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w],
                mask_chunk.to(dtype=attack_mask.dtype),
            )

            # Zero out masks in spliced region using actual mask shape
            if transformed_masks is not None:
                # Extract ROI from localization maps
                mask_roi = transformed_masks[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w]

                # Apply inverse of splicing mask: where mask_chunk=1 (spliced), set to 0
                # mask_chunk is (1, H, W), mask_roi is (N, H, W)
                mask_inverse = (1.0 - mask_chunk)  # (1, H, W)
                transformed_roi = mask_roi * mask_inverse  # (N, H, W) * (1, H, W)

                # Place back
                transformed_masks[:, pos_y : pos_y + fg_h, pos_x : pos_x + fg_w] = transformed_roi

                # Update retention for next splice
                if self.mask_aware:
                    from ..mask_aware_utils import calculate_mask_metrics
                    combined_mask = transformed_masks.sum(dim=0, keepdim=True)
                    total_original = localization_map.sum().item()
                    if total_original > 0:
                        current_retention = combined_mask.sum().item() / total_original

            num_spliced += 1

        return AttackResult(
            attacked=out_img,
            localization_map=transformed_masks,
            metadata={
                "num_splices": num_spliced,
                "mask_mode": self.mask_mode,
                "resolved_mask_mode": self._resolved_mask_mode,
                "max_splices": max_splices,
                "available_components": available_components,
                "num_splices_param": self.num_splices,
                "random_selection": self.num_splices == -1,
                "attack_mask_area": float((attack_mask > 0.5).float().mean().item()),
            },
            debug_tensors={"attack_mask": attack_mask},
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
    BG_PATH = "./background.jpg"
    FG_PATH = "./foreground.jpg"

    print(f"Running Splicing Attack on {DEVICE}...")

    # Example 1: Fixed number of splices (2 components)
    attack_fixed = SplicingAttack(
        foreground_path=FG_PATH,
        mask_mode="auto",
        num_splices=2
    )

    # Example 2: Random number of splices
    attack_random = SplicingAttack(
        foreground_path=FG_PATH,
        mask_mode="irregular",
        num_splices=-1  # Random number between 1 and all available
    )

    # Example 3: Unlimited splices (use all available)
    attack_unlimited = SplicingAttack(
        foreground_path=FG_PATH,
        mask_mode="box",
        num_splices=None  # Use all available components
    )

    # Example 4: User-provided mask with single splice
    user_mask = torch.ones(512, 512) * 0.5  # Example mask
    attack_user = SplicingAttack(
        foreground_path=FG_PATH,
        mask=user_mask,
        mask_mode="user",
        num_splices=1
    )

    # Needs a background image to run
    if os.path.exists(BG_PATH):
        bg_tensor = load_image(BG_PATH, device=DEVICE)

        # Test with localization maps
        test_masks = torch.ones(2, bg_tensor.shape[1], bg_tensor.shape[2])

        # Test fixed splices
        result = attack_fixed(bg_tensor, localization_map=test_masks)
        save_image(result.attacked, "result_fixed_splice.png")
        print(f"Fixed mode metadata: {result.metadata}")

        # Test random splices
        result = attack_random(bg_tensor, localization_map=test_masks)
        save_image(result.attacked, "result_random_splice.png")
        print(f"Random mode metadata: {result.metadata}")

        # Test unlimited splices
        result = attack_unlimited(bg_tensor, localization_map=test_masks)
        save_image(result.attacked, "result_unlimited_splice.png")
        print(f"Unlimited mode metadata: {result.metadata}")
    else:
        print("Background image not found, skipping test run.")
