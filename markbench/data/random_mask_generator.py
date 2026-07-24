import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    sam_model_registry = None
    SamAutomaticMaskGenerator = None

from markbench.utils.device import get_device


BENCHMARK_RANDOM_MASK_TARGET_AREA_RATIO_RANGE = (0.04, 0.14)
BENCHMARK_RANDOM_MASK_ASPECT_RATIO_RANGE = (0.6, 1.8)
BENCHMARK_RANDOM_MASK_IRREGULAR_FILL_RATIO_RANGE = (0.55, 0.8)


def get_benchmark_random_mask_kwargs() -> Dict[str, Any]:
    """Return the standard random-mask regime for datasets without annotations.

    This regime is used for DIV2K/MirFlickr in the benchmark pipeline. It aims
    to keep masks local while matching the effective annotation-backed area
    regime more closely than the historical 1-2% patches.
    """
    return {
        "allow_box": True,
        "allow_irregular": True,
        "allow_segmentation": False,
        "allow_invert": False,
        "target_area_ratio_range": BENCHMARK_RANDOM_MASK_TARGET_AREA_RATIO_RANGE,
        "aspect_ratio_range": BENCHMARK_RANDOM_MASK_ASPECT_RATIO_RANGE,
        "irregular_fill_ratio_range": BENCHMARK_RANDOM_MASK_IRREGULAR_FILL_RATIO_RANGE,
    }


class MaskType(Enum):
    BOX = "box"
    IRREGULAR = "irregular"
    SEGMENTATION = "segmentation"


class RandomMaskGenerator:
    def __init__(
        self,
        height: int,
        width: int,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: Optional[str] = "vit_b",
        device: str = "",
    ):
        self.height = height
        self.width = width
        self.device = get_device(device)

        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.__sam_mask_generator = None

    def generate(
        self,
        num_masks: int = 1,
        allow_box: bool = True,
        allow_irregular: bool = True,
        allow_segmentation: bool = False,
        image_tensor: Optional[torch.Tensor] = None,
        allow_invert: bool = True,
        localize_single_irregular: bool = False,
        target_area_ratio_range: Optional[Tuple[float, float]] = None,
        aspect_ratio_range: Tuple[float, float] = BENCHMARK_RANDOM_MASK_ASPECT_RATIO_RANGE,
        irregular_fill_ratio_range: Tuple[float, float] = BENCHMARK_RANDOM_MASK_IRREGULAR_FILL_RATIO_RANGE,
    ) -> torch.Tensor:
        """
        Generates a random mask tensor.

        Args:
            num_masks: The number of mask layers to generate.
                       If 1: Generates a single layer, potentially inverted.
                       If >1: Generates N disjoint layers, never inverted.
            image_tensor: Input image (C, H, W) required if allow_segmentation is True.
            allow_invert: Whether single-layer masks may be inverted.
            localize_single_irregular: If True, a single irregular mask is drawn
                inside a local box instead of across the full canvas.

        Returns:
            torch.Tensor: Binary mask of shape (num_masks, H, W).
        """
        choices = []
        if allow_box:
            choices.append(MaskType.BOX)
        if allow_irregular:
            choices.append(MaskType.IRREGULAR)
        if allow_segmentation:
            choices.append(MaskType.SEGMENTATION)

        if not choices:
            raise ValueError("No mask types enabled")

        mask_type = random.choice(choices)
        
        if target_area_ratio_range is not None:
            target_area_ratio_range = self.__normalize_ratio_range(
                target_area_ratio_range,
                min_value=1.0 / float(self.height * self.width),
                max_value=1.0,
                name="target_area_ratio_range",
            )
            aspect_ratio_range = self.__normalize_ratio_range(
                aspect_ratio_range,
                min_value=1e-3,
                max_value=1e3,
                name="aspect_ratio_range",
            )
            irregular_fill_ratio_range = self.__normalize_ratio_range(
                irregular_fill_ratio_range,
                min_value=1e-3,
                max_value=1.0,
                name="irregular_fill_ratio_range",
            )

        if mask_type == MaskType.BOX:
            if target_area_ratio_range is not None:
                mask = self.__area_controlled_box_mask(
                    num_masks=num_masks,
                    target_area_ratio_range=target_area_ratio_range,
                    aspect_ratio_range=aspect_ratio_range,
                )
            else:
                mask = self.__box_mask(
                    num_masks=num_masks,
                    min_w=random.randint(30, 50),
                    max_w=random.randint(100, 150),
                    min_h=random.randint(30, 50),
                    max_h=random.randint(100, 150)
                )

        elif mask_type == MaskType.IRREGULAR:
            if target_area_ratio_range is not None:
                mask = self.__area_controlled_irregular_mask(
                    num_masks=num_masks,
                    target_area_ratio_range=target_area_ratio_range,
                    aspect_ratio_range=aspect_ratio_range,
                    fill_ratio_range=irregular_fill_ratio_range,
                )
            else:
                mask = self.__irregular_mask(
                    num_masks=num_masks,
                    localize_single=localize_single_irregular,
                )

        elif mask_type == MaskType.SEGMENTATION:
            if image_tensor is None:
                raise ValueError("image_tensor must be provided for segmentation masks")

            if target_area_ratio_range is None:
                area_side = random.randint(50, 150)
                target_area = area_side * area_side
            else:
                target_area_ratio = random.uniform(*target_area_ratio_range)
                target_area = max(1, int(round(target_area_ratio * self.height * self.width)))
            
            mask = self.__generate_sam_masks(
                image_tensor=image_tensor,
                num_masks=num_masks,
                target_area=target_area
            )

        else:
            raise RuntimeError("Unhandled mask type")

        if num_masks == 1:
            return self.__maybe_invert(mask, allow=allow_invert)
        
        return mask

    def __normalize_ratio_range(
        self,
        value_range: Tuple[float, float],
        *,
        min_value: float,
        max_value: float,
        name: str,
    ) -> Tuple[float, float]:
        if len(value_range) != 2:
            raise ValueError(f"{name} must contain exactly two floats")

        lo = max(min(float(value_range[0]), float(value_range[1])), min_value)
        hi = min(max(float(value_range[0]), float(value_range[1])), max_value)
        if hi < lo:
            raise ValueError(f"{name} must intersect [{min_value}, {max_value}]")
        return lo, hi

    def __load_sam(self):
        if self.__sam_mask_generator is not None:
            return

        if not HAS_SAM:
            raise RuntimeError(
                "segment_anything is not installed. "
                "Install it with: pip install segment-anything"
            )

        if self.sam_checkpoint is None:
            raise RuntimeError("SAM checkpoint path not provided")

        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        self.__sam_mask_generator = SamAutomaticMaskGenerator(sam)

    def __generate_box_coordinates(
        self,
        num_boxes: int,
        min_w: int = 30,
        max_w: int = 100,
        min_h: int = 30,
        max_h: int = 100,
        margin: int = 10,
        max_attempts: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """
        Utility method to generate non-overlapping box coordinates.
        
        Returns:
            List of tuples (x0, y0, x1, y1).
            If collision checks fail repeatedly, the returned list length 
            may be smaller than num_boxes.
        """
        boxes = []

        for _ in range(num_boxes):
            for _attempt in range(max_attempts):
                w = random.randint(min_w, max_w)
                h = random.randint(min_h, max_h)

                max_x = self.width - w - margin
                max_y = self.height - h - margin
                
                if max_x < margin or max_y < margin:
                    continue  # Dimensions define a box too large for the image+margin

                x = random.randint(margin, max_x)
                y = random.randint(margin, max_y)
                
                new_box = (x, y, x + w, y + h)

                overlap = False
                for b in boxes:
                    if not (
                        new_box[2] <= b[0] or  # new is left of b
                        new_box[0] >= b[2] or  # new is right of b
                        new_box[3] <= b[1] or  # new is above b
                        new_box[1] >= b[3]     # new is below b
                    ):
                        overlap = True
                        break

                if not overlap:
                    boxes.append(new_box)
                    break 

        return boxes

    def __box_mask(
        self, 
        num_masks: int = 1, 
        min_w: int = 30, 
        max_w: int = 100, 
        min_h: int = 30, 
        max_h: int = 100
    ) -> torch.Tensor:
        """
        Generates a binary mask tensor with random rectangles.
        
        Args:
            num_masks: The number of masks (channels) to generate.
                       If 1: Generates 1 layer with multiple disjoint boxes.
                       If >1: Generates N layers with one box per layer.
        """
        
        if num_masks == 1:
            target_box_count = random.randint(1, 5)
        else:
            target_box_count = num_masks

        boxes = self.__generate_box_coordinates(
            num_boxes=target_box_count,
            min_w=min_w, max_w=max_w,
            min_h=min_h, max_h=max_h
        )

        mask = torch.zeros(
            (num_masks, self.height, self.width), 
            device=self.device, 
            dtype=torch.float32
        )

        if num_masks == 1:
            for (x0, y0, x1, y1) in boxes:
                mask[0, y0:y1, x0:x1] = 1.0
        else:
            for i, (x0, y0, x1, y1) in enumerate(boxes):
                mask[i, y0:y1, x0:x1] = 1.0

        return mask

    def __sample_box_dims_from_area_ratio(
        self,
        area_ratio_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
    ) -> Tuple[int, int]:
        total_area = float(self.height * self.width)
        target_ratio = random.uniform(*area_ratio_range)
        aspect_ratio = math.exp(random.uniform(math.log(aspect_ratio_range[0]), math.log(aspect_ratio_range[1])))
        target_pixels = max(1.0, target_ratio * total_area)

        box_w = max(1, min(self.width, int(round(math.sqrt(target_pixels * aspect_ratio)))))
        box_h = max(1, min(self.height, int(round(target_pixels / float(box_w)))))
        box_h = max(1, min(self.height, box_h))
        box_w = max(1, min(self.width, int(round(target_pixels / float(box_h)))))
        return box_h, box_w

    def __sample_box_position(
        self,
        box_h: int,
        box_w: int,
        existing_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        margin: int = 10,
        max_attempts: int = 50,
    ) -> Tuple[int, int, int, int]:
        existing_boxes = existing_boxes or []
        box_h = min(max(1, box_h), self.height)
        box_w = min(max(1, box_w), self.width)
        max_x = max(0, self.width - box_w)
        max_y = max(0, self.height - box_h)
        effective_margin_x = min(margin, max_x)
        effective_margin_y = min(margin, max_y)

        def _sample_without_overlap() -> Optional[Tuple[int, int, int, int]]:
            if max_x < effective_margin_x or max_y < effective_margin_y:
                return None
            x = random.randint(effective_margin_x, max_x) if max_x > 0 else 0
            y = random.randint(effective_margin_y, max_y) if max_y > 0 else 0
            new_box = (x, y, x + box_w, y + box_h)
            for b in existing_boxes:
                if not (
                    new_box[2] <= b[0]
                    or new_box[0] >= b[2]
                    or new_box[3] <= b[1]
                    or new_box[1] >= b[3]
                ):
                    return None
            return new_box

        for _ in range(max_attempts):
            candidate = _sample_without_overlap()
            if candidate is not None:
                return candidate

        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        return (x, y, x + box_w, y + box_h)

    def __area_controlled_box_mask(
        self,
        num_masks: int,
        target_area_ratio_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
    ) -> torch.Tensor:
        mask = torch.zeros(
            (num_masks, self.height, self.width),
            device=self.device,
            dtype=torch.float32,
        )
        placed_boxes: List[Tuple[int, int, int, int]] = []

        for i in range(num_masks):
            box_h, box_w = self.__sample_box_dims_from_area_ratio(
                target_area_ratio_range,
                aspect_ratio_range,
            )
            box = self.__sample_box_position(
                box_h=box_h,
                box_w=box_w,
                existing_boxes=placed_boxes if num_masks > 1 else None,
            )
            x0, y0, x1, y1 = box
            mask[i, y0:y1, x0:x1] = 1.0
            if num_masks > 1:
                placed_boxes.append(box)

        return mask

    def __make_local_irregular_patch(
        self,
        box_h: int,
        box_w: int,
        fill_ratio: float,
    ) -> torch.Tensor:
        coarse_h = max(4, min(16, box_h // 24 if box_h >= 24 else 4))
        coarse_w = max(4, min(16, box_w // 24 if box_w >= 24 else 4))
        noise = torch.rand((1, 1, coarse_h, coarse_w), device=self.device)
        noise = F.interpolate(
            noise,
            size=(box_h, box_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        threshold = torch.quantile(noise.flatten(), max(0.0, min(1.0, 1.0 - fill_ratio)))
        patch = (noise >= threshold).float()

        # Smooth the thresholded noise slightly so the final mask is local and
        # blob-like rather than a sparse pixel pattern.
        patch = F.max_pool2d(patch.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2)
        patch = (patch > 0).float().squeeze(0).squeeze(0)
        return patch

    def __area_controlled_irregular_mask(
        self,
        num_masks: int,
        target_area_ratio_range: Tuple[float, float],
        aspect_ratio_range: Tuple[float, float],
        fill_ratio_range: Tuple[float, float],
    ) -> torch.Tensor:
        mask = torch.zeros(
            (num_masks, self.height, self.width),
            device=self.device,
            dtype=torch.float32,
        )
        placed_boxes: List[Tuple[int, int, int, int]] = []

        for i in range(num_masks):
            target_area_ratio = random.uniform(*target_area_ratio_range)
            fill_ratio = random.uniform(*fill_ratio_range)
            bbox_area_ratio = min(1.0, max(target_area_ratio, target_area_ratio / max(fill_ratio, 1e-3)))
            box_h, box_w = self.__sample_box_dims_from_area_ratio(
                (bbox_area_ratio, bbox_area_ratio),
                aspect_ratio_range,
            )
            x0, y0, x1, y1 = self.__sample_box_position(
                box_h=box_h,
                box_w=box_w,
                existing_boxes=placed_boxes if num_masks > 1 else None,
            )

            local_patch = self.__make_local_irregular_patch(
                box_h=y1 - y0,
                box_w=x1 - x0,
                fill_ratio=fill_ratio,
            )
            mask[i, y0:y1, x0:x1] = local_patch
            if num_masks > 1:
                placed_boxes.append((x0, y0, x1, y1))

        return mask

    def __irregular_mask(
        self,
        num_masks: int = 1,
        localize_single: bool = False,
    ) -> torch.Tensor:
        """
        Generates random brush stroke masks.
        
        Args:
            num_masks: Number of mask layers to generate.
            
        Returns:
            Tensor of shape (num_masks, H, W) with binary values (0.0 or 1.0).
        """

        masks = torch.zeros(
            (num_masks, self.height, self.width), 
            device=self.device, 
            dtype=torch.float32
        )

        if num_masks == 1:
            # For dataset-level random masks we preserve the historical full-canvas
            # behavior. For splice masks we can force a local box to get an actual
            # patch-like region instead of a near-global scribble.
            if localize_single:
                boxes = self.__generate_box_coordinates(
                    num_boxes=1,
                    min_w=50, max_w=150,
                    min_h=50, max_h=150,
                )
                full_canvas_box = boxes[0] if boxes else (0, 0, self.width, self.height)
            else:
                full_canvas_box = (0, 0, self.width, self.height)

            num_strokes = random.randint(1, 5)
            self.__generate_strokes_on_tensor(masks[0], full_canvas_box, num_strokes)
            
        else:
            # Case 2: Use disjoint boxes for each layer
            boxes = self.__generate_box_coordinates(
                num_boxes=num_masks,
                min_w=50, max_w=150,
                min_h=50, max_h=150
            )

            # Note: If boxes < num_masks, remaining layers stay 0.0
            for i, box in enumerate(boxes):
                num_strokes = random.randint(1, 3)
                self.__generate_strokes_on_tensor(masks[i], box, num_strokes)

        return masks

    def __generate_strokes_on_tensor(
        self, 
        mask_slice: torch.Tensor, 
        box: Tuple[int, int, int, int], 
        num_strokes: int
    ):
        """
        Draws random strokes onto a specific (H, W) tensor slice, 
        strictly constrained within the given box (x0, y0, x1, y1).
        """
        x0, y0, x1, y1 = box

        segment_skeleton = torch.zeros(
            (1, 1, self.height, self.width),
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(num_strokes):
            num_vertices = random.randint(4, 12)
            
            start_x = random.randint(x0, x1 - 1)
            start_y = random.randint(y0, y1 - 1)
            angle = random.uniform(0, 2 * math.pi)

            for _ in range(num_vertices):
                
                angle += random.uniform(-4, 4) * math.pi / 180
                length = random.randint(20, 50)
                brush_width = random.randint(20, 50)

                end_x = int(max(x0, min(x1 - 1, start_x + length * torch.cos(torch.tensor(angle)).item())))
                end_y = int(max(y0, min(y1 - 1, start_y + length * torch.sin(torch.tensor(angle)).item())))

                segment_skeleton.zero_()
                
                self.__draw_line_tensor(
                    segment_skeleton[0, 0], start_x, start_y, end_x, end_y
                )

                radius = brush_width // 2
                k_size = radius * 2 + 1
                
                grid = torch.arange(k_size, device=self.device)
                y_grid, x_grid = torch.meshgrid(grid, grid, indexing="ij")
                center = radius
                dist_sq = (x_grid - center) ** 2 + (y_grid - center) ** 2
                kernel = (dist_sq <= radius**2).float().view(1, 1, k_size, k_size)

                thick_segment = F.conv2d(segment_skeleton, kernel, padding=radius)

                mask_slice[:] = torch.max(mask_slice, (thick_segment > 0).float().squeeze(0))

                start_x, start_y = end_x, end_y

        box_mask = torch.zeros_like(mask_slice)
        box_mask[y0:y1, x0:x1] = 1.0
        
        mask_slice[:] = mask_slice * box_mask

    def __draw_line_tensor(
        self, canvas: torch.Tensor, x0: int, y0: int, x1: int, y1: int
    ):
        """Draws a 1-pixel wide line on the canvas tensor."""
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            canvas[y0, x0] = 1.0
            return

        t = torch.linspace(0, 1, steps + 1, device=self.device)
        x_coords = torch.round(x0 + (x1 - x0) * t).long()
        y_coords = torch.round(y0 + (y1 - y0) * t).long()

        x_coords = x_coords.clamp(0, self.width - 1)
        y_coords = y_coords.clamp(0, self.height - 1)

        canvas[y_coords, x_coords] = 1.0

    def __maybe_invert(self, mask: torch.Tensor, allow: bool = True) -> torch.Tensor:
        if allow and random.random() < 0.5:
            return 1.0 - mask
        return mask

    def __generate_sam_masks(
        self, 
        image_tensor: torch.Tensor, 
        num_masks: int = 1,
        target_area: int = 10000
    ) -> torch.Tensor:
        """
        Generates SAM masks. 
        If num_masks=1, collapses multiple valid segments into a single mask layer.
        If num_masks>1, assigns one valid segment per layer.
        """
        self.__load_sam()

        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        img_cpu = image_tensor.detach().cpu()
        if img_cpu.max() <= 1.0:
            img_cpu = img_cpu * 255.0
        img_np = img_cpu.permute(1, 2, 0).numpy().astype(np.uint8)

        if num_masks == 1:
            masks_to_collect = random.randint(2, 5)
        else:
            masks_to_collect = num_masks

        raw_masks = self.__sam_mask_generator.generate(img_np)
        random.shuffle(raw_masks)

        final_masks_list = []
        lower_bound = target_area * 0.5
        upper_bound = target_area * 1.5

        for m in raw_masks:
            if len(final_masks_list) >= masks_to_collect:
                break

            seg_np = m["segmentation"]
            current_area = np.sum(seg_np)

            if current_area < lower_bound:
                continue

            elif current_area <= upper_bound:
                t_seg = torch.from_numpy(seg_np).to(self.device, dtype=torch.float32)
                final_masks_list.append(t_seg)

            else:
                valid_indices = np.argwhere(seg_np)
                if len(valid_indices) == 0: continue
                
                idx = random.randint(0, len(valid_indices) - 1)
                cy, cx = valid_indices[idx]

                ratio = random.uniform(0.5, 2.0)
                crop_w = int(math.sqrt(target_area * ratio))
                crop_h = int(math.sqrt(target_area / ratio))

                x0 = max(0, cx - crop_w // 2)
                y0 = max(0, cy - crop_h // 2)
                x1 = min(self.width, x0 + crop_w)
                y1 = min(self.height, y0 + crop_h)

                t_seg = torch.from_numpy(seg_np).to(self.device, dtype=torch.float32)
                box_mask = torch.zeros_like(t_seg)
                box_mask[y0:y1, x0:x1] = 1.0
                
                cropped_seg = t_seg * box_mask
                
                if cropped_seg.sum() > 0:
                    final_masks_list.append(cropped_seg)

        output_tensor = torch.zeros(
            (num_masks, self.height, self.width), 
            device=self.device, 
            dtype=torch.float32
        )

        if num_masks == 1:
            for mask in final_masks_list:
                output_tensor[0] += mask
        else:
            for i, mask in enumerate(final_masks_list):
                if i < num_masks:
                    output_tensor[i] = mask

        return output_tensor


if __name__ == "__main__":
    HEIGHT = 512
    WIDTH = 512
    
    generator = RandomMaskGenerator(
        height=HEIGHT,
        width=WIDTH,
        sam_checkpoint="./data/sam_checkpoints/sam_vit_b_01ec64.pth",
        sam_model_type="vit_b",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    try:
        image_np = cv2.imread("./test.png")
        if image_np is None: raise FileNotFoundError
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (WIDTH, HEIGHT))
        image_tensor = (
            torch.from_numpy(image_np).permute(2, 0, 1).float().to(generator.device) / 255.0
        )
    except (FileNotFoundError, AttributeError):
        print("Warning: 'test.png' not found. Creating random noise image for testing.")
        image_tensor = torch.rand((3, HEIGHT, WIDTH), device=generator.device)

    print("Generating Mask (N=1)...")
    singular_mask = generator.generate(
        num_masks=1,
        allow_box=False,
        allow_irregular=True,
        allow_segmentation=False,
        image_tensor=image_tensor,
    )
    
    s_mask_np = singular_mask.squeeze(0).detach().cpu().numpy() * 255.0
    cv2.imwrite("test_singular_mask.png", s_mask_np.astype(np.uint8))
    print(f"Saved 'test_singular_mask.png' with shape {singular_mask.shape}")

    print("Generating Multiple Masks (N=5)...")
    embedding_masks = generator.generate(
        num_masks=5,
        allow_box=False,
        allow_irregular=True,
        allow_segmentation=False,
        image_tensor=image_tensor,
    )
    
    print(f"Embedding Tensor Shape: {embedding_masks.shape}")
    
    combined_viz = torch.sum(embedding_masks, dim=0).detach().cpu().numpy() * 255.0
    cv2.imwrite("test_embedding_combined.png", combined_viz.astype(np.uint8))
    
    layer_0 = embedding_masks[0].detach().cpu().numpy() * 255.0
    cv2.imwrite("test_embedding_layer_0.png", layer_0.astype(np.uint8))
    
    print("Saved 'test_embedding_combined.png' and 'test_embedding_layer_0.png'")
