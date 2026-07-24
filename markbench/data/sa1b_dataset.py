import json
import os
import bisect
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

from .image_dataset import ImageDataset, MaskMode
from markbench.utils.image import load_image
from markbench.interface import DataPoint

class Sa1bDataset(ImageDataset):
    """
    SA-1B dataset wrapper for parts sa_000000 to sa_000009.
    """

    def __init__(
        self,
        root: str,
        seed: int = 42,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_mode: Union[MaskMode, str] = MaskMode.ANNOTATION,
        mask_config: Optional[Dict[str, Any]] = None,
    ):
        self.root = Path(root)
        self.seed = seed
        self.transform = transform
        self.recursive = False
        
        if isinstance(mask_mode, str):
            try:
                mask_mode = MaskMode(mask_mode.lower())
            except ValueError:
                raise ValueError(f"Invalid mask_mode '{mask_mode}'. Options: {list(MaskMode)}")
        self.mask_mode = mask_mode
        self.mask_config = mask_config or {}

        if not self.root.exists():
            raise ValueError(f"Dataset directory does not exist: {self.root}")

        if self.mask_mode == MaskMode.ANNOTATION and mask_utils is None:
            print("Warning: 'pycocotools' not found. SA-1B annotation loading may fail.")

        self.part_names = [f"sa_{i:06d}" for i in range(10)]
        self.cumulative_counts: List[int] = []
        self.total_images = 0
        
        self._scan_dataset_structure()
        
        if max_samples is not None and max_samples < self.total_images:
            self.total_images = max_samples
        
        self.indices = None
        if shuffle:
            rng = np.random.default_rng(self.seed)
            self.indices = np.arange(self.total_images)
            rng.shuffle(self.indices)

        self.mask_generator = None
        if self.mask_mode == MaskMode.RANDOM:
            self._init_mask_generator()

    @property
    def has_annotations(self) -> bool:
        return True

    def _scan_dataset_structure(self):
        """
        Populates self.cumulative_counts by counting files in each folder.
        """
        print(f"Scanning SA-1B structure ({len(self.part_names)} parts)...")
        
        running_count = 0
        
        for part_name in self.part_names:
            part_dir = self.root / part_name
            
            if not part_dir.exists():
                self.cumulative_counts.append(running_count)
                continue

            with os.scandir(part_dir) as it:
                file_count = sum(1 for entry in it if entry.is_file())
            
            image_count = file_count // 2
            running_count += image_count
            self.cumulative_counts.append(running_count)

        self.total_images = running_count
        print(f"Total images found: {self.total_images}")

    def _resolve_index(self, idx: int) -> Tuple[Path, int]:
        """
        Converts a dataset index (0..N-1) into:
        1. The path to the image file.
        2. The index of the part (folder) it belongs to.
        
        Uses binary search on cumulative_counts.
        """
        if idx < 0 or idx >= self.total_images:
            raise IndexError(f"Index {idx} out of range (0..{self.total_images-1})")

        effective_idx = idx if self.indices is None else self.indices[idx]

        part_idx = bisect.bisect_right(self.cumulative_counts, effective_idx)
        
        file_id = effective_idx + 1
        
        part_name = self.part_names[part_idx]
        filename = f"sa_{file_id}.jpg"
        
        return self.root / part_name / filename, part_idx

    def _find_images(self) -> List[Path]:
        """
        Overridden/Stubbed. 
        We do not generate a list of paths in memory for this class. 
        Calling get_paths() will raise an error or return empty to save memory.
        """
        return []

    def __len__(self) -> int:
        return self.total_images

    def _load_annotation(self, idx: int, image_path: Path) -> Optional[torch.Tensor]:
        json_path = image_path.with_suffix('.json')
        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            anns = data.get('annotations', [])
            if not anns: return None
            
            masks_list = []
            for ann in anns:
                rle = ann.get('segmentation')
                if not rle: continue
                
                if mask_utils:
                    m = mask_utils.decode(rle)
                    masks_list.append(torch.from_numpy(m).float())
            
            if not masks_list: return None
            return torch.stack(masks_list)

        except Exception as e:
            print(f"Error loading annotation for {image_path.name}: {e}")
            return None

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_path, _ = self._resolve_index(idx)
        
        image = load_image(image_path, transform=self.transform)
        masks: Optional[torch.Tensor] = None

        if self.mask_mode == MaskMode.ANNOTATION:
            masks = self._load_annotation(idx, image_path)
        elif self.mask_mode == MaskMode.RANDOM:
            masks = self.get_random_masks(image, idx)

        if masks is not None:
            img_h, img_w = image.shape[-2:]
            mask_h, mask_w = masks.shape[-2:]

            if img_h != mask_h or img_w != mask_w:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0),
                    size=(img_h, img_w),
                    mode='nearest'
                ).squeeze(0)

        return image_path.name, DataPoint(image=image, masks=masks)

    def get_paths(self) -> List[Path]:
        raise NotImplementedError("Sa1bDataset does not store paths in memory. Iterate the dataset to access paths.")
