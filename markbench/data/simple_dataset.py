"""Simple image dataset for general-purpose image loading."""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, Union
import torch

from .image_dataset import ImageDataset, MaskMode


class SimpleImageDataset(ImageDataset):
    """Simple image dataset for loading images from any directory.

    This is a minimal concrete implementation of ImageDataset that:
    - Loads images from any directory (no special structure required)
    - Does not support annotation-based masks (has_annotations=False)
    - Supports random mask generation and no-mask modes

    Use this dataset when you just want to load images from a directory
    for testing purposes.
    """

    def __init__(
        self,
        root: str,
        seed: int = 42,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_mode: Union[MaskMode, str] = MaskMode.NONE,
        mask_config: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
    ):
        super().__init__(
            root=root,
            seed=seed,
            max_samples=max_samples,
            shuffle=shuffle,
            transform=transform,
            mask_mode=mask_mode,
            mask_config=mask_config,
            recursive=recursive,
        )

        if self.mask_mode == MaskMode.ANNOTATION:
            raise ValueError(
                "SimpleImageDataset does not support MaskMode.ANNOTATION. "
                "Use MaskMode.RANDOM or MaskMode.NONE instead."
            )

    @property
    def has_annotations(self) -> bool:
        """SimpleImageDataset does not support annotations."""
        return False

    def _load_annotation(self, idx: int, image_path: Path) -> torch.Tensor:
        """Not implemented - SimpleImageDataset has no annotations."""
        raise NotImplementedError(
            "SimpleImageDataset does not support annotations. "
            "This method should never be called."
        )