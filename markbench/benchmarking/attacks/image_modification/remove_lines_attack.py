import random
from typing import Optional, Tuple, Union

import torch

from markbench.interface import AttackResult
from ..abstract_attack import AbstractAttack
from ..registry import register_attack


@register_attack("remove_lines")
class RemoveLinesAttack(AbstractAttack):
    """
    Remove random rows and/or columns from an image.

    Supports randomization by passing tuples (min, max) for remove_rows/cols.
    """

    @property
    def is_geometry_modifying(self) -> bool:
        return True

    def __init__(
        self,
        remove_rows: Union[int, Tuple[int, int]] = 0,
        remove_cols: Union[int, Tuple[int, int]] = 0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            remove_rows: Number of rows to remove. Can be int or tuple (min, max).
            remove_cols: Number of columns to remove. Can be int or tuple (min, max).
            seed: Random seed for reproducibility
        """
        self.remove_rows = remove_rows
        self.remove_cols = remove_cols
        if seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator().manual_seed(seed)

    def _resolve_count(self, count: Union[int, Tuple[int, int]]) -> int:
        """Resolve a count to a concrete value, randomizing if given a range."""
        if isinstance(count, (list, tuple)):
            return random.randint(count[0], count[1])
        return count

    def _get_indices_to_keep(self, total: int, num_to_remove: int) -> torch.Tensor:
        """Get indices to keep after randomly removing some."""
        perm = torch.randperm(total) if self.generator is None else torch.randperm(
            total, generator=self.generator
        )
        indices_to_remove = perm[:num_to_remove]
        mask = torch.ones(total, dtype=torch.bool)
        mask[indices_to_remove] = False
        return torch.where(mask)[0]

    def __call__(
        self,
        image: torch.Tensor,
        localization_map: Optional[torch.Tensor] = None
    ) -> AttackResult:
        img = image.float()

        # Image format is (C, H, W)
        _, h, w = img.shape

        # Get number of rows/cols to remove
        rows = self._resolve_count(self.remove_rows)
        cols = self._resolve_count(self.remove_cols)

        # Validate removal counts
        remove_rows = min(rows, h - 1)
        remove_cols = min(cols, w - 1)

        # Compute indices to keep ONCE
        rows_to_keep = None
        cols_to_keep = None

        # Randomly select rows to keep
        if remove_rows > 0:
            rows_to_keep = self._get_indices_to_keep(h, remove_rows)
            img = img[:, rows_to_keep, :]

        # Randomly select cols to keep
        if remove_cols > 0:
            cols_to_keep = self._get_indices_to_keep(w, remove_cols)
            img = img[:, :, cols_to_keep]

        # Apply SAME indexing to masks
        transformed_masks = None
        if localization_map is not None:
            masks = localization_map.clone()
            if remove_rows > 0 and rows_to_keep is not None:
                masks = masks[:, rows_to_keep, :]
            if remove_cols > 0 and cols_to_keep is not None:
                masks = masks[:, :, cols_to_keep]
            transformed_masks = masks

        return AttackResult(
            attacked=img,
            localization_map=transformed_masks,
            metadata={
                "rows_removed": remove_rows,
                "cols_removed": remove_cols
            }
        )
