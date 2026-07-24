import random
from typing import Optional, Tuple, Union

import torch

from ..abstract_attack import AbstractAttack
from ..registry import register_attack
from markbench.interface import AttackResult


@register_attack("salt_pepper", aliases=["sp_noise", "salt_and_pepper"])
class SaltAndPepperNoiseAttack(AbstractAttack):
    """
    Salt and pepper noise attack.

    Supports randomization for amount parameter.
    """

    def __init__(
        self,
        amount: Union[float, Tuple[float, float]] = 0.01,
        salt_vs_pepper: float = 0.5,
    ):
        """
        Args:
            amount: Proportion of pixels to affect (0 to 1).
                    Can be float or tuple (min, max) for random range.
            salt_vs_pepper: Ratio of salt to total (0.5 = equal salt and pepper).
        """
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def _resolve_amount(self) -> float:
        """Resolve amount to a concrete value, randomizing if given a range."""
        if isinstance(self.amount, (list, tuple)):
            return random.uniform(self.amount[0], self.amount[1])
        return self.amount

    def _get_unique_indices(
        self, num_needed: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Get unique random pixel indices using oversampling + unique filtering."""
        total_pixels = height * width
        num_needed = min(num_needed, total_pixels)
        if num_needed <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        # Oversample by 20% to account for collisions, then take unique
        oversample_factor = 1.2
        num_samples = min(int(num_needed * oversample_factor) + 100, total_pixels)

        # Generate random linear indices
        indices = torch.randint(0, total_pixels, (num_samples,), device=device)
        unique_indices = torch.unique(indices)

        # If we still don't have enough (rare for sparse noise), retry with more
        if len(unique_indices) < num_needed:
            # Fall back to randperm for dense noise (rare case)
            unique_indices = torch.randperm(total_pixels, device=device)[:num_needed]
        else:
            # Randomly sample from unique indices to avoid bias toward low indices
            # (torch.unique returns sorted, so [:num_needed] would bias to top of image)
            perm = torch.randperm(len(unique_indices), device=device)
            unique_indices = unique_indices[perm[:num_needed]]

        return unique_indices

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        """Apply salt and pepper noise to image (masks unchanged)."""
        img = image.clone()

        # Get amount value (possibly random)
        amount = self._resolve_amount()

        # Get image dimensions (C, H, W)
        _, height, width = img.shape
        total_pixels = height * width

        num_salt = int(amount * total_pixels * self.salt_vs_pepper)
        num_pepper = int(amount * total_pixels * (1.0 - self.salt_vs_pepper))

        # Get unique non-overlapping indices for salt and pepper
        total_noise = num_salt + num_pepper
        all_indices = self._get_unique_indices(total_noise, height, width, img.device)

        # Shuffle indices to ensure random distribution of salt vs pepper
        shuffle_perm = torch.randperm(len(all_indices), device=img.device)
        all_indices = all_indices[shuffle_perm]

        salt_indices = all_indices[:num_salt]
        pepper_indices = all_indices[num_salt : num_salt + num_pepper]

        # Apply salt noise
        if len(salt_indices) > 0:
            salt_h = salt_indices // width
            salt_w = salt_indices % width
            img[:, salt_h, salt_w] = 255

        # Apply pepper noise
        if len(pepper_indices) > 0:
            pepper_h = pepper_indices // width
            pepper_w = pepper_indices % width
            img[:, pepper_h, pepper_w] = 0

        # Shape-preserving - masks unchanged
        return AttackResult(
            attacked=img,
            localization_map=localization_map,
            metadata={
                "amount": amount,
                "salt_vs_pepper": self.salt_vs_pepper,
                "num_salt": num_salt,
                "num_pepper": num_pepper
            }
        )
