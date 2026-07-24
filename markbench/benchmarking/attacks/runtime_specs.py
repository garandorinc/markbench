from __future__ import annotations

import inspect
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from markbench.interface import AttackResult
from markbench.utils.image import load_image

from .abstract_attack import AbstractAttack
from .registry import _ATTACK_REGISTRY, get_attack
from .splicing.splicing_attack import SplicingAttack


def _instantiate_attack(name: str, params: dict[str, Any], device: Optional[str] = None) -> AbstractAttack:
    """Instantiate an attack with the same YAML semantics as parse_attack_config."""
    attack_name = name.lower()
    if attack_name not in _ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack for compact spec: {name}")

    params = dict(params)

    if device is not None and "device" not in params:
        attack_class = _ATTACK_REGISTRY[attack_name]
        sig = inspect.signature(attack_class.__init__)
        if "device" in sig.parameters:
            params["device"] = device

    if attack_name == "chain" and "attacks" in params:
        parsed = []
        for entry in params["attacks"]:
            if isinstance(entry, str):
                from . import parse_attack_string

                parsed.append(parse_attack_string(entry))
            else:
                nested_name = entry.get("name") or entry.get("attack")
                nested_params = {
                    key: value for key, value in entry.items() if key not in {"name", "attack"}
                }
                parsed.append(_instantiate_attack(nested_name, nested_params, device=device))
        return get_attack("chain", attacks=parsed)

    if attack_name == "random" and "pool" in params:
        parsed_pool = []
        for entry in params["pool"]:
            if isinstance(entry, str):
                from . import parse_attack_string

                parsed_pool.append(parse_attack_string(entry))
            else:
                nested_name = entry.get("name") or entry.get("attack")
                nested_params = {
                    key: value for key, value in entry.items() if key not in {"name", "attack"}
                }
                parsed_pool.append(_instantiate_attack(nested_name, nested_params, device=device))
        count = params.get("count", 1)
        replacement = params.get("replacement", False)
        return get_attack("random", count=count, pool=parsed_pool, replacement=replacement)

    return get_attack(name, **params)


@contextmanager
def _temporary_attributes(obj: Any, **updates: Any):
    """Temporarily mutate object attributes while running an attack."""
    previous = {}
    for key, value in updates.items():
        previous[key] = getattr(obj, key)
        setattr(obj, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(obj, key, value)


class RuntimeAttackSpec(AbstractAttack):
    """Attack spec that resolves a concrete attack per image at runtime."""

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None,
    ) -> AttackResult:
        item = {
            "watermarked_single": image,
            "original_mask_tensor": localization_map,
            "dataset_idx": 0,
        }
        return self.apply_to_item(
            item=item,
            seed=None,
            device=str(image.device),
        )

    def resolve_config(
        self,
        item: Optional[dict[str, Any]],
        seed: Optional[int],
        device: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def apply_to_item(
        self,
        item: dict[str, Any],
        seed: Optional[int],
        device: str,
    ) -> AttackResult:
        raise NotImplementedError

    def apply_to_batch(
        self,
        batch_items: list[dict[str, Any]],
        seeds: list[Optional[int]],
        device: str,
    ) -> list[AttackResult]:
        return [
            self.apply_to_item(item=item, seed=seed, device=device)
            for item, seed in zip(batch_items, seeds)
        ]

    def set_runtime_context(self, dataset: Any) -> None:
        """Attach optional dataset context before benchmark execution."""


def _combine_localization_mask(localization_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if localization_map is None:
        return None

    if localization_map.dim() != 3:
        raise ValueError(
            f"Expected localization_map with shape (N, H, W), got {tuple(localization_map.shape)}"
        )

    combined = (localization_map.sum(dim=0, keepdim=True) > 0.5).float()
    if combined.sum().item() == 0:
        return None
    return combined


class VariantAttackSpec(RuntimeAttackSpec):
    """Generic compact attack spec that samples one preset bundle per image."""

    def __init__(
        self,
        family_name: str,
        variants: list[dict[str, Any]],
        device: Optional[str] = None,
    ):
        if not variants:
            raise ValueError(f"Compact attack '{family_name}' must provide at least one variant")
        self.family_name = family_name
        self.variants = variants
        self.device = device

    @property
    def name(self) -> str:
        return self.family_name

    def get_config(self) -> dict[str, Any]:
        return {"variants": self.variants}

    def _select_variant(self, seed: Optional[int]) -> dict[str, Any]:
        if len(self.variants) == 1:
            return dict(self.variants[0])
        if seed is None:
            return dict(random.choice(self.variants))
        return dict(random.Random(seed).choice(self.variants))

    def resolve_config(
        self,
        item: Optional[dict[str, Any]],
        seed: Optional[int],
        device: str,
    ) -> dict[str, Any]:
        return self._select_variant(seed)

    def apply_to_item(
        self,
        item: dict[str, Any],
        seed: Optional[int],
        device: str,
    ) -> AttackResult:
        resolved = self.resolve_config(item=item, seed=seed, device=device)
        concrete = _instantiate_attack(self.family_name, resolved, device=device or self.device)
        return concrete(
            item["watermarked_single"],
            localization_map=item["original_mask_tensor"],
        )


class InpaintingAttackSpec(RuntimeAttackSpec):
    """Shared-pipeline inpainting spec for targeted and untargeted runs."""

    def __init__(
        self,
        log_name: str,
        variants: list[dict[str, Any]],
        target_mode: str,
        target_mask_dilation_ratio: float = 0.03,
        min_mask_side_ratio: float = 0.05,
        prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        device: Optional[str] = None,
    ):
        if not variants:
            raise ValueError(f"{log_name} must provide at least one variant")
        self.log_name = log_name
        self.variants = variants
        self.target_mode = target_mode
        self.target_mask_dilation_ratio = float(target_mask_dilation_ratio)
        self.min_mask_side_ratio = float(min_mask_side_ratio)
        self.prompt = prompt
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)
        self.device = device
        self._attack = None

    @property
    def name(self) -> str:
        return self.log_name

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "variants": self.variants,
            "target_mode": self.target_mode,
            "min_mask_side_ratio": self.min_mask_side_ratio,
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }
        if self.target_mode == "targeted":
            config["target_mask_dilation_ratio"] = self.target_mask_dilation_ratio
        return config

    def _ensure_attack(self, device: str):
        if self._attack is not None:
            return self._attack
        from .generative_ai.stable_diffusion_inpainting import StableDiffusionInpaintAttack

        default_ratio = 0.5
        default_min_ratio = self.min_mask_side_ratio
        if self.variants:
            default_ratio = float(self.variants[0].get("max_mask_side_ratio", default_ratio))
            default_min_ratio = float(self.variants[0].get("min_mask_side_ratio", default_min_ratio))

        self._attack = StableDiffusionInpaintAttack(
            device=device or self.device,
            prompt=self.prompt,
            mode="inpaint",
            min_mask_side_ratio=default_min_ratio,
            max_mask_side_ratio=default_ratio,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        )
        return self._attack

    def _select_variant(self, seed: Optional[int]) -> dict[str, Any]:
        if len(self.variants) == 1:
            return dict(self.variants[0])
        if seed is None:
            return dict(random.choice(self.variants))
        return dict(random.Random(seed).choice(self.variants))

    def resolve_config(
        self,
        item: Optional[dict[str, Any]],
        seed: Optional[int],
        device: str,
    ) -> dict[str, Any]:
        resolved = self._select_variant(seed)
        resolved["target_mode"] = self.target_mode
        if self.target_mode == "targeted":
            resolved["target_mask_dilation_ratio"] = self.target_mask_dilation_ratio
        return resolved

    def _build_target_mask(self, localization_map: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        combined = _combine_localization_mask(localization_map)
        if combined is None:
            return None

        _, height, width = combined.shape
        min_side = min(height, width)
        kernel_size = max(3, int(round(min_side * self.target_mask_dilation_ratio)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        dilated = F.max_pool2d(
            combined.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).squeeze(0)
        return dilated

    def apply_to_item(
        self,
        item: dict[str, Any],
        seed: Optional[int],
        device: str,
    ) -> AttackResult:
        resolved = self.resolve_config(item=item, seed=seed, device=device)
        attack = self._ensure_attack(device)

        user_mask = None
        if self.target_mode == "targeted":
            user_mask = self._build_target_mask(item.get("original_mask_tensor"))
            if user_mask is None:
                return AttackResult(
                    attacked=item["watermarked_single"].clone(),
                    localization_map=item["original_mask_tensor"].clone()
                    if item.get("original_mask_tensor") is not None
                    else None,
                    metadata={
                        "target_mode": self.target_mode,
                        "skipped_no_target_mask": True,
                    },
                )

        max_mask_side_ratio = float(resolved.get("max_mask_side_ratio", attack.max_mask_side_ratio))
        min_mask_side_ratio = float(resolved.get("min_mask_side_ratio", attack.min_mask_side_ratio))
        guidance_scale = float(resolved.get("guidance_scale", attack.guidance_scale))
        num_inference_steps = int(resolved.get("num_inference_steps", attack.num_inference_steps))
        prompt = str(resolved.get("prompt", attack.prompt))
        mode = str(resolved.get("mode", "inpaint"))

        with _temporary_attributes(
            attack,
            user_mask=user_mask,
            min_mask_side_ratio=min_mask_side_ratio,
            max_mask_side_ratio=max_mask_side_ratio,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            mode=mode,
            seed=seed,
        ):
            result = attack(
                item["watermarked_single"],
                localization_map=item["original_mask_tensor"],
            )

        result.metadata.setdefault("target_mode", self.target_mode)
        if user_mask is not None:
            result.metadata["target_mask_area"] = float((user_mask > 0.5).float().mean().item())
        return result


class StableDiffusionMaskEditSpec(RuntimeAttackSpec):
    """Stable-Diffusion inpaint controls with explicit benchmark mask semantics."""

    _shared_attacks: dict[tuple[str, str, float, int], Any] = {}

    def __init__(
        self,
        log_name: str,
        variants: list[dict[str, Any]],
        mask_mode: str,
        copy_original_outside_attack_mask: bool = False,
        min_mask_side_ratio: float = 0.05,
        prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 25,
        device: Optional[str] = None,
    ):
        if not variants:
            raise ValueError(f"{log_name} must provide at least one variant")
        self.log_name = log_name
        self.variants = variants
        self.mask_mode = mask_mode
        self.copy_original_outside_attack_mask = bool(copy_original_outside_attack_mask)
        self.min_mask_side_ratio = float(min_mask_side_ratio)
        self.prompt = prompt
        self.guidance_scale = float(guidance_scale)
        self.num_inference_steps = int(num_inference_steps)
        self.device = device

    @property
    def name(self) -> str:
        return self.log_name

    def get_config(self) -> dict[str, Any]:
        return {
            "variants": self.variants,
            "mask_mode": self.mask_mode,
            "copy_original_outside_attack_mask": self.copy_original_outside_attack_mask,
            "min_mask_side_ratio": self.min_mask_side_ratio,
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

    def _ensure_attack(self, device: str):
        from .generative_ai.stable_diffusion_inpainting import StableDiffusionInpaintAttack

        key = (
            str(device or self.device),
            self.prompt,
            self.guidance_scale,
            self.num_inference_steps,
        )
        if key in self._shared_attacks:
            return self._shared_attacks[key]

        default_ratio = 0.5
        default_min_ratio = self.min_mask_side_ratio
        if self.variants:
            default_ratio = float(self.variants[0].get("max_mask_side_ratio", default_ratio))
            default_min_ratio = float(self.variants[0].get("min_mask_side_ratio", default_min_ratio))

        attack = StableDiffusionInpaintAttack(
            device=device or self.device,
            prompt=self.prompt,
            mode="inpaint",
            min_mask_side_ratio=default_min_ratio,
            max_mask_side_ratio=default_ratio,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
        )
        self._shared_attacks[key] = attack
        return attack

    def _select_variant(self, seed: Optional[int]) -> dict[str, Any]:
        if len(self.variants) == 1:
            return dict(self.variants[0])
        if seed is None:
            return dict(random.choice(self.variants))
        return dict(random.Random(seed).choice(self.variants))

    def resolve_config(
        self,
        item: Optional[dict[str, Any]],
        seed: Optional[int],
        device: str,
    ) -> dict[str, Any]:
        resolved = self._select_variant(seed)
        resolved["mask_mode"] = self.mask_mode
        resolved["copy_original_outside_attack_mask"] = self.copy_original_outside_attack_mask
        return resolved

    def _build_mask_outpainting_mask(
        self,
        localization_map: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        target_mask = _combine_localization_mask(localization_map)
        if target_mask is None:
            return None
        return (1.0 - target_mask).clamp(0.0, 1.0)

    def _apply_copy_control(self, item: dict[str, Any], result: AttackResult) -> AttackResult:
        attack_mask = result.debug_tensors.get("attack_mask")
        if attack_mask is None:
            return result

        base = item["watermarked_single"].to(device=result.attacked.device, dtype=result.attacked.dtype)
        mask = attack_mask.to(device=result.attacked.device, dtype=result.attacked.dtype).clamp(0.0, 1.0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        result.attacked = result.attacked * mask + base * (1.0 - mask)
        result.metadata["copied_original_outside_attack_mask"] = True
        return result

    def apply_to_item(
        self,
        item: dict[str, Any],
        seed: Optional[int],
        device: str,
    ) -> AttackResult:
        resolved = self.resolve_config(item=item, seed=seed, device=device)
        attack = self._ensure_attack(device)

        user_mask = None
        if self.mask_mode == "inverse_target_mask":
            user_mask = self._build_mask_outpainting_mask(item.get("original_mask_tensor"))
            if user_mask is None:
                return AttackResult(
                    attacked=item["watermarked_single"].clone(),
                    localization_map=item["original_mask_tensor"].clone()
                    if item.get("original_mask_tensor") is not None
                    else None,
                    metadata={
                        "mask_mode": self.mask_mode,
                        "skipped_no_target_mask": True,
                    },
                )
        elif self.mask_mode != "random_rect":
            raise ValueError(f"Unsupported mask_mode for {self.log_name}: {self.mask_mode}")

        max_mask_side_ratio = float(resolved.get("max_mask_side_ratio", attack.max_mask_side_ratio))
        min_mask_side_ratio = float(resolved.get("min_mask_side_ratio", attack.min_mask_side_ratio))
        guidance_scale = float(resolved.get("guidance_scale", attack.guidance_scale))
        num_inference_steps = int(resolved.get("num_inference_steps", attack.num_inference_steps))
        prompt = str(resolved.get("prompt", attack.prompt))
        mode = str(resolved.get("mode", "inpaint"))

        with _temporary_attributes(
            attack,
            user_mask=user_mask,
            min_mask_side_ratio=min_mask_side_ratio,
            max_mask_side_ratio=max_mask_side_ratio,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            mode=mode,
            seed=seed,
        ):
            result = attack(
                item["watermarked_single"],
                localization_map=item["original_mask_tensor"],
            )

        result.metadata.setdefault("mask_mode", self.mask_mode)
        if user_mask is not None:
            result.metadata["target_mask_area"] = float((1.0 - user_mask > 0.5).float().mean().item())
            result.metadata["attack_mask_area"] = float((user_mask > 0.5).float().mean().item())

        if self.copy_original_outside_attack_mask:
            result = self._apply_copy_control(item, result)
        return result


class SplicingAttackSpec(RuntimeAttackSpec):
    """Runtime splicing spec with same-dataset donor-image resolution."""

    def __init__(
        self,
        log_name: str,
        variants: list[dict[str, Any]],
        target_mode: str,
        donor_source: str = "same_dataset_random",
        device: Optional[str] = None,
    ):
        if not variants:
            raise ValueError(f"{log_name} must provide at least one variant")
        self.log_name = log_name
        self.variants = variants
        self.target_mode = target_mode
        self.donor_source = donor_source
        self.device = device
        self._dataset_paths: list[Path] = []
        self._dataset = None
        self._dataset_len = 0
        self._resolved_path_cache: dict[int, Path] = {}

    @property
    def name(self) -> str:
        return self.log_name

    def get_config(self) -> dict[str, Any]:
        return {
            "variants": self.variants,
            "target_mode": self.target_mode,
            "donor_source": self.donor_source,
        }

    def set_runtime_context(self, dataset: Any) -> None:
        self._dataset_paths = []
        self._dataset = None
        self._dataset_len = 0
        self._resolved_path_cache = {}

        if hasattr(dataset, "get_paths"):
            try:
                self._dataset_paths = list(dataset.get_paths())
                self._dataset_len = len(self._dataset_paths)
                return
            except NotImplementedError:
                pass
        if hasattr(dataset, "_resolve_index"):
            # SA-1B does not materialize every image path in memory. Keep the
            # dataset handle and resolve donor paths lazily per sampled index.
            self._dataset = dataset
            self._dataset_len = len(dataset)
            return

    def _select_variant(self, seed: Optional[int]) -> dict[str, Any]:
        if len(self.variants) == 1:
            variant = dict(self.variants[0])
        elif seed is None:
            variant = dict(random.choice(self.variants))
        else:
            variant = dict(random.Random(seed).choice(self.variants))

        num_splices = variant.get("num_splices")
        if isinstance(num_splices, (list, tuple)) and num_splices:
            rng = random.Random(seed) if seed is not None else random
            if len(num_splices) == 2 and all(isinstance(v, int) for v in num_splices):
                variant["num_splices"] = rng.randint(num_splices[0], num_splices[1])
            else:
                variant["num_splices"] = rng.choice(list(num_splices))
        return variant

    def _resolve_donor_path(self, item: dict[str, Any], seed: Optional[int]) -> Optional[Path]:
        donor_pool_size = len(self._dataset_paths) if self._dataset_paths else self._dataset_len
        if self.donor_source != "same_dataset_random" or donor_pool_size <= 1:
            return None

        dataset_idx = item.get("dataset_idx")
        if dataset_idx is None or dataset_idx < 0 or dataset_idx >= donor_pool_size:
            return None

        if seed is None:
            donor_idx = random.randrange(donor_pool_size - 1)
        else:
            donor_idx = random.Random(seed).randrange(donor_pool_size - 1)

        if donor_idx >= dataset_idx:
            donor_idx += 1

        if self._dataset_paths:
            return self._dataset_paths[donor_idx]

        if self._dataset is None:
            return None

        cached = self._resolved_path_cache.get(donor_idx)
        if cached is not None:
            return cached

        resolved = Path(self._dataset._resolve_index(donor_idx)[0])
        self._resolved_path_cache[donor_idx] = resolved
        return resolved

    def resolve_config(
        self,
        item: Optional[dict[str, Any]],
        seed: Optional[int],
        device: str,
    ) -> dict[str, Any]:
        resolved = self._select_variant(seed)
        resolved["target_mode"] = self.target_mode
        resolved["donor_source"] = self.donor_source
        return resolved

    def _apply_targeted_splice(
        self,
        item: dict[str, Any],
        donor_path: Path,
        device: str,
    ) -> AttackResult:
        target_mask = _combine_localization_mask(item.get("original_mask_tensor"))
        if target_mask is None:
            return AttackResult(
                attacked=item["watermarked_single"].clone(),
                localization_map=item["original_mask_tensor"].clone()
                if item.get("original_mask_tensor") is not None
                else None,
                metadata={
                    "target_mode": self.target_mode,
                    "donor_source": self.donor_source,
                    "skipped_no_target_mask": True,
                },
            )

        base = item["watermarked_single"]
        _, height, width = base.shape
        donor = load_image(donor_path, device=device, size=(width, height)).to(dtype=base.dtype)
        alpha = target_mask.to(device=base.device, dtype=base.dtype).clamp(0.0, 1.0)
        attacked = donor * alpha + base * (1.0 - alpha)

        localization_map = item.get("original_mask_tensor")
        transformed_masks = None
        if localization_map is not None:
            mask_inverse = (1.0 - alpha).to(dtype=localization_map.dtype, device=localization_map.device)
            transformed_masks = localization_map * mask_inverse

        attack_mask_area = float((alpha > 0.5).float().mean().item())
        return AttackResult(
            attacked=attacked,
            localization_map=transformed_masks,
            metadata={
                "target_mode": self.target_mode,
                "donor_source": self.donor_source,
                "donor_image": donor_path.name,
                "attack_mask_area": attack_mask_area,
                "target_mask_area": attack_mask_area,
                "num_splices": 1,
                "available_components": 1,
                "resolved_mask_mode": "watermark_mask",
            },
            debug_tensors={"attack_mask": alpha},
        )

    def apply_to_item(
        self,
        item: dict[str, Any],
        seed: Optional[int],
        device: str,
    ) -> AttackResult:
        resolved = self.resolve_config(item=item, seed=seed, device=device)
        donor_path = self._resolve_donor_path(item, seed)
        if donor_path is None:
            return AttackResult(
                attacked=item["watermarked_single"].clone(),
                localization_map=item["original_mask_tensor"].clone()
                if item.get("original_mask_tensor") is not None
                else None,
                metadata={
                    "target_mode": self.target_mode,
                    "donor_source": self.donor_source,
                    "skipped_no_donor": True,
                },
            )

        if self.target_mode == "targeted":
            return self._apply_targeted_splice(item=item, donor_path=donor_path, device=device)

        concrete_params = {
            key: value
            for key, value in resolved.items()
            if key not in {"target_mode", "donor_source"}
        }
        concrete_params["foreground_path"] = str(donor_path)
        concrete_params["mask_aware"] = self.target_mode == "targeted"
        if self.target_mode == "targeted":
            concrete_params["min_retention"] = 0.0

        attack = SplicingAttack(**concrete_params)
        result = attack(
            item["watermarked_single"],
            localization_map=item["original_mask_tensor"],
        )
        result.metadata.setdefault("target_mode", self.target_mode)
        result.metadata["donor_source"] = self.donor_source
        result.metadata["donor_image"] = donor_path.name
        return result


def build_runtime_attack_spec(
    name: str,
    params: dict[str, Any],
    device: Optional[str] = None,
) -> RuntimeAttackSpec:
    """Build a runtime-resolved attack spec from compact YAML."""
    variants = params.get("variants", [])

    if name == "inpainting_untargeted":
        return InpaintingAttackSpec(
            log_name=name,
            variants=variants,
            target_mode="untargeted",
            min_mask_side_ratio=float(params.get("min_mask_side_ratio", 0.05)),
            prompt=str(params.get("prompt", "")),
            guidance_scale=float(params.get("guidance_scale", 7.5)),
            num_inference_steps=int(params.get("num_inference_steps", 25)),
            device=device,
        )

    if name == "inpainting_targeted":
        return InpaintingAttackSpec(
            log_name=name,
            variants=variants,
            target_mode="targeted",
            target_mask_dilation_ratio=float(params.get("target_mask_dilation_ratio", 0.03)),
            min_mask_side_ratio=float(params.get("min_mask_side_ratio", 0.05)),
            prompt=str(params.get("prompt", "")),
            guidance_scale=float(params.get("guidance_scale", 7.5)),
            num_inference_steps=int(params.get("num_inference_steps", 25)),
            device=device,
        )

    if name == "mask_outpainting":
        return StableDiffusionMaskEditSpec(
            log_name=name,
            variants=variants,
            mask_mode="inverse_target_mask",
            copy_original_outside_attack_mask=False,
            min_mask_side_ratio=float(params.get("min_mask_side_ratio", 0.05)),
            prompt=str(params.get("prompt", "")),
            guidance_scale=float(params.get("guidance_scale", 7.5)),
            num_inference_steps=int(params.get("num_inference_steps", 25)),
            device=device,
        )

    if name == "inpainting_off_mask_copy_content":
        return StableDiffusionMaskEditSpec(
            log_name=name,
            variants=variants,
            mask_mode="random_rect",
            copy_original_outside_attack_mask=True,
            min_mask_side_ratio=float(params.get("min_mask_side_ratio", 0.05)),
            prompt=str(params.get("prompt", "")),
            guidance_scale=float(params.get("guidance_scale", 7.5)),
            num_inference_steps=int(params.get("num_inference_steps", 25)),
            device=device,
        )

    if name == "splicing_untargeted":
        return SplicingAttackSpec(
            log_name=name,
            variants=variants,
            target_mode="untargeted",
            donor_source=str(params.get("donor_source", "same_dataset_random")),
            device=device,
        )

    if name == "splicing_targeted":
        return SplicingAttackSpec(
            log_name=name,
            variants=variants,
            target_mode="targeted",
            donor_source=str(params.get("donor_source", "same_dataset_random")),
            device=device,
        )

    return VariantAttackSpec(family_name=name, variants=variants, device=device)
