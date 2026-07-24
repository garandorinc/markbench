import os
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union
import torch

from .image_dataset import ImageDataset, MaskMode

class MirFlickrDataset(ImageDataset):
    """
    MIRFLICKR-25K dataset wrapper.
            
    Note: The annotations provided in this dataset are image-level labels (classification tags),
    not pixel-level segmentation masks. Therefore, MaskMode.ANNOTATION is not supported
    for generating spatial masks. Use MaskMode.RANDOM or MaskMode.NONE.
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
    ):
        super().__init__(
            root=root,
            seed=seed,
            max_samples=max_samples,
            shuffle=shuffle,
            transform=transform,
            mask_mode=mask_mode,
            mask_config=mask_config,
            recursive=False
        )

        if self.mask_mode == MaskMode.ANNOTATION:
             raise ValueError(
                 "MirFlickrDataset does not support MaskMode.ANNOTATION because the dataset "
                 "only contains image-level classification labels, not pixel-level segmentation masks."
             )

    @property
    def has_annotations(self) -> bool:
        # Returns False because we cannot provide spatial masks
        return False

    def _find_images(self) -> List[Path]:
        """
        Scans the 'mirflickr' subfolder for images named imX.jpg.
        Sorts them numerically by the index X (im1.jpg, im2.jpg, ...).
        """
        img_dir = self.root / "mirflickr"
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        image_paths = []
        
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(tuple(self.EXTENSIONS)):
                    image_paths.append(Path(entry.path))

        def sort_key(p: Path):
            try:
                stem = p.stem
                if stem.lower().startswith('im'):
                    return int(stem[2:])
                return stem
            except ValueError:
                return p.name

        image_paths.sort(key=sort_key)
        
        print(f"Found {len(image_paths)} images in MIRFLICKR-25K.")
        return image_paths

    def _load_annotation(self, idx: int, image_path: Path) -> torch.Tensor:
        """
        Not implemented because has_annotations is False.
        """
        raise NotImplementedError("MirFlickrDataset does not support spatial mask annotations.")