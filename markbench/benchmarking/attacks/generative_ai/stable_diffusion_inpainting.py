from typing import Optional, Tuple
import random

import torch
from PIL import Image as PILImage
import numpy as np
import torch.nn.functional as F

from diffusers import StableDiffusionInpaintPipeline

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.utils.image import tensor_to_pil, pil_to_tensor
from markbench.interface import AttackResult


@register_attack("sd_inpaint", aliases=["inpainting", "outpainting", "sd_outpaint"])
class StableDiffusionEditAttack(AbstractAttack):
    @property
    def name(self) -> str:
        if self.mode == "padding":
            return "outpainting"
        return "sd_inpaint"

    @property
    def is_geometry_modifying(self) -> bool:
        return self.mode == "padding"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        prompt: str = "",
        mode: str = "inpaint",  # one of: 'inpaint' or 'padding'
        mask: Optional[torch.Tensor] = None,
        min_mask_side_ratio: float = 0.05,
        max_mask_side_ratio: float = 0.5,
        pad_scale_range: Tuple[float, float] = (1.25, 2.0),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Args:
            device: torch.device
            prompt: optional prompt guiding the inpaint
            mode: 'inpaint' (modify a region inside the original image) or 'padding' (extrapolate canvas)
            mask: optional tensor to be used as mask for inpaint mode (white area replaced)
            min_mask_side_ratio: minimum ratio of rectangular mask width/height
            max_mask_side_ratio: maximum ratio of rectangular mask
            pad_scale_range: (min_scale, max_scale) the random range to expand canvas (padding mode)
            guidance_scale, num_inference_steps: pipeline sampling params
            seed: optional integer to make generation deterministic
        """
        self.verbose = bool(verbose)
        self._log(f"[SD Inpainting Init] device param: {device}, type: {type(device)}")

        # Convert string device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log(f"[SD Inpainting Init] Using device: {device}")

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable safety checker to avoid potential deadlocks
            requires_safety_checker=False
        )

        self._log(f"[SD Inpainting Init] Moving pipeline to {device}...")
        self.pipe = self.pipe.to(device)
        self._log(f"[SD Inpainting Init] Pipeline moved to device: {next(self.pipe.unet.parameters()).device}")

        # Disable attention slicing for better performance (we have enough VRAM)
        # Note: Attention slicing trades speed for memory - only use if running out of VRAM
        try:
            self.pipe.disable_attention_slicing()
        except AttributeError:
            pass  # Method might not exist in all versions

        self.device = device
        self.prompt = prompt
        self.mode = mode
        self.user_mask = mask
        self.min_mask_side_ratio = float(min_mask_side_ratio)
        self.max_mask_side_ratio = float(max_mask_side_ratio)
        self.pad_scale_range = pad_scale_range
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # Use centralized conversion functions from utils.image
    _tensor_to_pil = staticmethod(tensor_to_pil)
    _pil_to_tensor = staticmethod(pil_to_tensor)

    @staticmethod
    def _tensor_mask_to_pil(
        mask_img: torch.Tensor, target_size: Tuple[int, int]
    ) -> PILImage:
        arr = mask_img
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().float()
            if arr.ndim == 2:
                lum = arr
            elif arr.shape[0] >= 3:
                lum = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
            else:
                lum = arr[0]

            # Watermark/localization masks are usually in [0, 1]. Diffusers expects
            # white inpaint regions, so scale low-range masks up to [0, 255].
            if lum.max().item() <= 1.0:
                lum = lum * 255.0
            np_mask = lum.clamp(0, 255).byte().cpu().numpy().squeeze()
        else:
            np_mask = np.array(mask_img).astype(np.uint8)

        pil_mask = PILImage.fromarray(np_mask)

        if pil_mask.size != target_size:
            pil_mask = pil_mask.resize(target_size, resample=PILImage.NEAREST)

        pil_mask = pil_mask.convert("L")
        return pil_mask

    def _create_random_rect_mask(self, W: int, H: int) -> Tuple[PILImage, Tuple[int, int, int, int]]:
        min_w = max(1, int(round(self.min_mask_side_ratio * W)))
        min_h = max(1, int(round(self.min_mask_side_ratio * H)))
        max_w = max(1, int(self.max_mask_side_ratio * W))
        max_h = max(1, int(self.max_mask_side_ratio * H))
        min_w = min(min_w, max_w)
        min_h = min(min_h, max_h)
        w = random.randint(min_w, max_w)
        h = random.randint(min_h, max_h)
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y : y + h, x : x + w] = 255
        return PILImage.fromarray(mask), (x, y, w, h)

    def _create_padding_canvas_and_mask(
        self, orig_pil: PILImage.Image
    ) -> Tuple[PILImage.Image, PILImage.Image, int, int, int, int, int, int]:
        """
        Create padding canvas and return transformation parameters.

        Returns:
            (canvas, pil_mask, canvas_w, canvas_h, target_w, target_h, tx, ty)
        """
        W, H = orig_pil.size
        s = random.uniform(
            float(self.pad_scale_range[0]), float(self.pad_scale_range[1])
        )
        canvas_w = max(W + 1, int(W * s))
        canvas_h = max(H + 1, int(H * s))

        resize_factor = random.uniform(0.8, 1.2)
        target_w = max(1, int(W * resize_factor))
        target_h = max(1, int(H * resize_factor))
        if target_w > canvas_w:
            target_w = canvas_w
        if target_h > canvas_h:
            target_h = canvas_h

        max_x = canvas_w - target_w
        max_y = canvas_h - target_h
        tx = random.randint(0, max_x) if max_x > 0 else 0
        ty = random.randint(0, max_y) if max_y > 0 else 0

        canvas = PILImage.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        if (target_w, target_h) != (W, H):
            pasted = orig_pil.resize((target_w, target_h), resample=PILImage.LANCZOS)
        else:
            pasted = orig_pil
        canvas.paste(pasted, (tx, ty))

        mask = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
        mask[ty : ty + target_h, tx : tx + target_w] = 0
        pil_mask = PILImage.fromarray(mask)

        return canvas, pil_mask, canvas_w, canvas_h, target_w, target_h, tx, ty

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """
        Apply the Stable Diffusion inpainting/padding attack.

        Args:
            image: Input tensor (C, H, W) in [0, 255]
            localization_map: Optional ground truth localization maps (N, H, W)

        Returns:
            AttackResult with attacked image and transformed localization_map
        """
        self._log(f"[SD Inpainting] Processing single image with {self.num_inference_steps} steps...")
        src_pil = self._tensor_to_pil(image)
        W, H = src_pil.size

        generator = None
        if self.seed is not None:
            # IMPORTANT: Use CPU generator to avoid CUDA deadlocks
            # See: https://github.com/huggingface/diffusers/issues/2909
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.seed))

        # Track transformation parameters for masks
        inpaint_mask_tensor = None  # For inpaint mode: actual mask as tensor (H, W) in [0, 1]
        padding_mask_tensor = None  # For padding mode: pad-region mask as tensor (H, W) in [0, 1]
        padding_params = None  # For padding mode: (canvas_w, canvas_h, target_w, target_h, tx, ty)

        attack_bbox = None
        if self.mode == "inpaint":
            if self.user_mask is not None:
                pil_mask = self._tensor_mask_to_pil(self.user_mask, (W, H))
                pil_image_for_pipe = src_pil
            else:
                # Create random mask
                pil_mask, attack_bbox = self._create_random_rect_mask(W, H)
                pil_image_for_pipe = src_pil

            # Convert PIL mask to tensor for later use
            # White pixels (255) = regions to inpaint = should be zeroed in localization maps
            mask_array = np.array(pil_mask).astype(np.float32) / 255.0  # (H, W) in [0, 1]
            inpaint_mask_tensor = torch.from_numpy(mask_array)  # (H, W)
        elif self.mode == "padding":
            result = self._create_padding_canvas_and_mask(src_pil)
            pil_image_for_pipe, pil_mask, canvas_w, canvas_h, target_w, target_h, tx, ty = result
            padding_params = (canvas_w, canvas_h, target_w, target_h, tx, ty)
            mask_array = np.array(pil_mask).astype(np.float32) / 255.0
            padding_mask_tensor = torch.from_numpy(mask_array)
        else:
            raise ValueError(
                f"Unsupported mode: {self.mode}. Use 'inpaint' or 'padding'."
            )

        # Add progress callback to show inference steps
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            if step_index % 5 == 0 or step_index == self.num_inference_steps - 1:
                self._log(f"[SD Inpainting] Step {step_index + 1}/{self.num_inference_steps}")
            return callback_kwargs

        pipe_kwargs = dict(
            prompt=self.prompt,
            image=pil_image_for_pipe,
            mask_image=pil_mask,
            guidance_scale=float(self.guidance_scale),
            num_inference_steps=int(self.num_inference_steps),
            callback_on_step_end=progress_callback,
        )
        if generator is not None:
            pipe_kwargs["generator"] = generator

        out = self.pipe(**pipe_kwargs)
        pil_out = out.images[0]
        if self.mode == "inpaint" and pil_out.size != (W, H):
            pil_out = pil_out.resize((W, H), resample=PILImage.LANCZOS)

        attacked = self._pil_to_tensor(pil_out)

        # Transform localization_map if provided
        transformed_masks = None
        if localization_map is not None:
            if self.mode == "inpaint":
                # Zero out the inpainted regions using the actual mask shape
                # inpaint_mask_tensor: 1 where inpainted, 0 where preserved
                # Result: localization_map * (1 - inpaint_mask_tensor)
                if inpaint_mask_tensor is not None:
                    # Expand mask to match localization_map batch dimension
                    mask_inverse = (1 - inpaint_mask_tensor).unsqueeze(0).to(
                        device=localization_map.device,
                        dtype=localization_map.dtype,
                    )  # (1, H, W)
                    transformed_masks = localization_map * mask_inverse  # (N, H, W) * (1, H, W)
            elif self.mode == "padding" and padding_params is not None:
                # Apply same resize and translation to masks
                canvas_w, canvas_h, target_w, target_h, tx, ty = padding_params
                n_masks = localization_map.shape[0]

                # Resize masks to target size using nearest neighbor to maintain binary values
                masks_resized = F.interpolate(
                    localization_map.unsqueeze(1),  # (N, 1, H, W)
                    size=(target_h, target_w),
                    mode='nearest',
                ).squeeze(1)  # (N, H', W')

                # Create canvas and paste resized masks
                canvas_masks = torch.zeros(
                    (n_masks, canvas_h, canvas_w),
                    dtype=localization_map.dtype,
                    device=localization_map.device,
                )
                canvas_masks[:, ty:ty+target_h, tx:tx+target_w] = masks_resized
                transformed_masks = canvas_masks

        if transformed_masks is not None and transformed_masks.shape[-2:] != attacked.shape[-2:]:
            transformed_masks = F.interpolate(
                transformed_masks.unsqueeze(0),
                size=attacked.shape[-2:],
                mode='nearest',
            ).squeeze(0)

        metadata = {"mode": self.mode}
        debug_tensors = {}
        if inpaint_mask_tensor is not None:
            attack_mask = inpaint_mask_tensor.unsqueeze(0).to(dtype=attacked.dtype)
            debug_tensors["attack_mask"] = attack_mask
            metadata["attack_mask_area"] = float((attack_mask > 0.5).float().mean().item())
            if attack_bbox is not None:
                metadata["attack_bbox"] = {
                    "x": int(attack_bbox[0]),
                    "y": int(attack_bbox[1]),
                    "width": int(attack_bbox[2]),
                    "height": int(attack_bbox[3]),
                }
        elif padding_mask_tensor is not None:
            attack_mask = padding_mask_tensor.unsqueeze(0).to(dtype=attacked.dtype)
            debug_tensors["attack_mask"] = attack_mask
            metadata["attack_mask_area"] = float((attack_mask > 0.5).float().mean().item())
        if padding_params is not None:
            metadata["padding_params"] = {
                "canvas_size": (padding_params[0], padding_params[1]),
                "target_size": (padding_params[2], padding_params[3]),
                "position": (padding_params[4], padding_params[5])
            }

        return AttackResult(
            attacked=attacked,
            localization_map=transformed_masks,
            metadata=metadata,
            debug_tensors=debug_tensors,
        )

    def __call__batch__(
        self,
        images: torch.Tensor,
        localization_maps: Optional[torch.Tensor] = None
    ) -> list:
        """
        Apply the Stable Diffusion inpainting/padding attack to a batch of images.

        Args:
            images: Input tensor (B, C, H, W) in [0, 255]
            localization_maps: Optional ground truth localization maps (B, N, H, W)

        Returns:
            List of AttackResult, one per image in batch
        """
        import time as time_module
        batch_size = images.shape[0]
        self._log(f"[SD Inpainting] Processing batch of {batch_size} images with {self.num_inference_steps} steps...")
        batch_start_time = time_module.time()

        # Setup generator for deterministic results
        self._log(f"[SD Inpainting] Setting up generator...")
        generator = None
        if self.seed is not None:
            # IMPORTANT: Use CPU generator to avoid CUDA deadlocks
            # See: https://github.com/huggingface/diffusers/issues/2909
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.seed))

        # Convert all images to PIL
        self._log(f"[SD Inpainting] Converting {batch_size} images to PIL and creating masks...")
        pil_images = []
        pil_masks = []
        inpaint_mask_tensors = []  # For inpaint mode
        padding_mask_tensors = []  # For padding mode
        padding_params_list = []  # For padding mode
        attack_bboxes = []

        for b in range(batch_size):
            if b == 0 or b % 2 == 0:
                self._log(f"[SD Inpainting] Processing image {b+1}/{batch_size}...")
            src_pil = self._tensor_to_pil(images[b])
            W, H = src_pil.size

            inpaint_mask_tensor = None
            padding_mask_tensor = None
            padding_params = None

            attack_bbox = None
            if self.mode == "inpaint":
                if self.user_mask is not None:
                    pil_mask = self._tensor_mask_to_pil(self.user_mask, (W, H))
                    pil_image_for_pipe = src_pil
                else:
                    # Create random mask
                    pil_mask, attack_bbox = self._create_random_rect_mask(W, H)
                    pil_image_for_pipe = src_pil

                # Convert PIL mask to tensor for later use
                mask_array = np.array(pil_mask).astype(np.float32) / 255.0
                inpaint_mask_tensor = torch.from_numpy(mask_array)

            elif self.mode == "padding":
                result = self._create_padding_canvas_and_mask(src_pil)
                pil_image_for_pipe, pil_mask, canvas_w, canvas_h, target_w, target_h, tx, ty = result
                padding_params = (canvas_w, canvas_h, target_w, target_h, tx, ty)
                mask_array = np.array(pil_mask).astype(np.float32) / 255.0
                padding_mask_tensor = torch.from_numpy(mask_array)

            else:
                raise ValueError(
                    f"Unsupported mode: {self.mode}. Use 'inpaint' or 'padding'."
                )

            pil_images.append(pil_image_for_pipe)
            pil_masks.append(pil_mask)
            inpaint_mask_tensors.append(inpaint_mask_tensor)
            padding_mask_tensors.append(padding_mask_tensor)
            padding_params_list.append(padding_params)
            attack_bboxes.append(attack_bbox)

        # Check image sizes - diffusers requires all images to be the same size for efficient batching
        image_sizes = [img.size for img in pil_images]
        unique_sizes = set(image_sizes)
        if len(unique_sizes) > 1:
            self._log(f"[SD Inpainting] WARNING: Batch has images with different sizes: {unique_sizes}")
            self._log(f"[SD Inpainting] Diffusers will process these sequentially, not as a true batch")
        else:
            self._log(f"[SD Inpainting] All images same size {image_sizes[0]}, true batching enabled")

        # Single batched call to pipeline with progress callback
        self._log(f"[SD Inpainting] Calling diffusers pipeline...")
        self._log(f"[SD Inpainting] Pipeline device: {self.pipe.device}")
        self._log(f"[SD Inpainting] UNet device: {next(self.pipe.unet.parameters()).device}")
        self._log(f"[SD Inpainting] This may take 30-60 seconds for the first step with batch_size={batch_size}...")

        pipeline_start = time_module.time()
        step_times = []

        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            step_time = time_module.time() - pipeline_start
            step_times.append(step_time)
            if step_index == 0:
                self._log(f"[SD Inpainting] Step {step_index + 1}/{self.num_inference_steps} (first step took {step_time:.1f}s)")
            elif step_index % 5 == 0 or step_index == self.num_inference_steps - 1:
                avg_time = (step_time - step_times[0]) / step_index if step_index > 0 else 0
                self._log(f"[SD Inpainting] Step {step_index + 1}/{self.num_inference_steps} (avg {avg_time:.1f}s/step)")
            return callback_kwargs

        # The batch path uses one pipeline call; the single-image path keeps the
        # callback for optional verbose progress reporting.
        pipe_kwargs = dict(
            prompt=[self.prompt] * batch_size,
            image=pil_images,
            mask_image=pil_masks,
            guidance_scale=float(self.guidance_scale),
            num_inference_steps=int(self.num_inference_steps),
            # callback_on_step_end=progress_callback,  # Disabled to test hang
        )
        if generator is not None:
            pipe_kwargs["generator"] = generator

        self._log(f"[SD Inpainting] Starting pipeline inference now (no progress callback)...")
        self._log(f"[SD Inpainting] Expected time: ~{self.num_inference_steps * 2}s for batch_size={batch_size}")

        if self.verbose:
            import threading

            def timeout_warning():
                time_module.sleep(60)
                self._log(f"[SD Inpainting] WARNING: Still processing after 60s, GPU util should be high!")

            threading.Thread(target=timeout_warning, daemon=True).start()

        out = self.pipe(**pipe_kwargs)
        self._log(f"[SD Inpainting] Pipeline complete, converting results...")

        # Convert results back to tensors and create AttackResults
        results = []
        for b in range(batch_size):
            pil_out = out.images[b]
            if self.mode == "inpaint" and pil_out.size != image_sizes[b]:
                pil_out = pil_out.resize(image_sizes[b], resample=PILImage.LANCZOS)
            attacked = self._pil_to_tensor(pil_out)

            # Transform localization_map for this image
            transformed_masks = None
            if localization_maps is not None:
                localization_map = localization_maps[b]  # (N, H, W)

                if self.mode == "inpaint":
                    inpaint_mask_tensor = inpaint_mask_tensors[b]
                    if inpaint_mask_tensor is not None:
                        mask_inverse = (1 - inpaint_mask_tensor).unsqueeze(0).to(
                            device=localization_map.device,
                            dtype=localization_map.dtype,
                        )  # (1, H, W)
                        transformed_masks = localization_map * mask_inverse

                elif self.mode == "padding":
                    padding_params = padding_params_list[b]
                    if padding_params is not None:
                        canvas_w, canvas_h, target_w, target_h, tx, ty = padding_params
                        n_masks = localization_map.shape[0]

                        # Resize masks to target size
                        masks_resized = F.interpolate(
                            localization_map.unsqueeze(1),  # (N, 1, H, W)
                            size=(target_h, target_w),
                            mode='nearest',
                        ).squeeze(1)

                        # Create canvas and paste resized masks
                        canvas_masks = torch.zeros(
                            (n_masks, canvas_h, canvas_w),
                            dtype=localization_map.dtype,
                            device=localization_map.device,
                        )
                        canvas_masks[:, ty:ty+target_h, tx:tx+target_w] = masks_resized
                        transformed_masks = canvas_masks

            if transformed_masks is not None and transformed_masks.shape[-2:] != attacked.shape[-2:]:
                transformed_masks = F.interpolate(
                    transformed_masks.unsqueeze(0),
                    size=attacked.shape[-2:],
                    mode='nearest',
                ).squeeze(0)

            metadata = {"mode": self.mode}
            debug_tensors = {}
            if self.mode == "inpaint" and inpaint_mask_tensors[b] is not None:
                attack_mask = inpaint_mask_tensors[b].unsqueeze(0).to(dtype=attacked.dtype)
                debug_tensors["attack_mask"] = attack_mask
                metadata["attack_mask_area"] = float((attack_mask > 0.5).float().mean().item())
                if attack_bboxes[b] is not None:
                    metadata["attack_bbox"] = {
                        "x": int(attack_bboxes[b][0]),
                        "y": int(attack_bboxes[b][1]),
                        "width": int(attack_bboxes[b][2]),
                        "height": int(attack_bboxes[b][3]),
                    }
            elif self.mode == "padding" and padding_mask_tensors[b] is not None:
                attack_mask = padding_mask_tensors[b].unsqueeze(0).to(dtype=attacked.dtype)
                debug_tensors["attack_mask"] = attack_mask
                metadata["attack_mask_area"] = float((attack_mask > 0.5).float().mean().item())
            if self.mode == "padding" and padding_params_list[b] is not None:
                padding_params = padding_params_list[b]
                metadata["padding_params"] = {
                    "canvas_size": (padding_params[0], padding_params[1]),
                    "target_size": (padding_params[2], padding_params[3]),
                    "position": (padding_params[4], padding_params[5])
                }

            results.append(AttackResult(
                attacked=attacked,
                localization_map=transformed_masks,
                metadata=metadata,
                debug_tensors=debug_tensors,
            ))

        batch_time = time_module.time() - batch_start_time
        self._log(f"[SD Inpainting] Batch complete in {batch_time:.2f}s ({batch_time/batch_size:.2f}s per image)")
        return results


StableDiffusionInpaintAttack = StableDiffusionEditAttack
