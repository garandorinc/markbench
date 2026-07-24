import os
import torch
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Union

from .image_dataset import ImageDataset, MaskMode

class Div2kDataset(ImageDataset):
    """
    DIV2K dataset wrapper.
    Combines 'DIV2K_train_HR' and 'DIV2K_valid_HR' into a single dataset.
    This dataset contains no ground truth annotations (masks).
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
            raise ValueError("Div2kDataset does not support MaskMode.ANNOTATION (no ground truth masks).")

    @property
    def has_annotations(self) -> bool:
        return False

    def _find_images(self) -> List[Path]:
        """
        Scans specifically DIV2K_train_HR and DIV2K_valid_HR subfolders.
        """
        subfolders = ["DIV2K_train_HR", "DIV2K_valid_HR"]
        image_paths = []

        print(f"Scanning DIV2K folders in {self.root}...")

        for sub in subfolders:
            folder_path = self.root / sub
            if not folder_path.exists():
                print(f"Warning: Subfolder {sub} not found in {self.root}")
                continue

            with os.scandir(folder_path) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(tuple(self.EXTENSIONS)):
                        image_paths.append(Path(entry.path))

        image_paths.sort(key=lambda p: p.name)
        
        if not image_paths:
            print("Warning: No images found in DIV2K directories.")
        else:
            print(f"Found {len(image_paths)} images in DIV2K.")

        return image_paths

    def _load_annotation(self, idx: int, image_path: Path) -> torch.Tensor:
        """
        This method should not be called because has_annotations returns False,
        and the base class guards against it.
        """
        raise NotImplementedError("DIV2K dataset does not contain annotations.")