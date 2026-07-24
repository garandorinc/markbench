"""Dataset utilities for markbench."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Dict, Union, Any
from enum import Enum
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from markbench.utils.image import load_image
from markbench.data.random_mask_generator import RandomMaskGenerator
from markbench.interface import DataPoint 

class MaskMode(Enum):
    NONE = "none"
    RANDOM = "random"
    ANNOTATION = "annotation"

class ImageDataset(Dataset, ABC):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

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
        super().__init__()
        self.root = Path(root)
        self.seed = seed
        self.transform = transform
        self.recursive = recursive
        
        if isinstance(mask_mode, str):
            try:
                mask_mode = MaskMode(mask_mode.lower())
            except ValueError:
                raise ValueError(f"Invalid mask_mode '{mask_mode}'. Options: {list(MaskMode)}")
        self.mask_mode = mask_mode
        self.mask_config = mask_config or {}
        
        if not self.root.exists():
            raise ValueError(f"Dataset directory does not exist: {self.root}")

        self.image_paths = self._find_images()

        rng = random.Random(self.seed)
        if shuffle:
            rng.shuffle(self.image_paths)
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]

        self.mask_generator: Optional[RandomMaskGenerator] = None
        if self.mask_mode == MaskMode.RANDOM:
            self._init_mask_generator()

    @property
    @abstractmethod
    def has_annotations(self) -> bool:
        pass

    def _init_mask_generator(self):
        # Check if we're in a DataLoader worker process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Force CPU when in DataLoader worker
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mask_generator = RandomMaskGenerator(
            height=512, width=512,
            device=device
        )

    def _find_images(self) -> List[Path]:
        if self.recursive:
            files = []
            for ext in self.EXTENSIONS:
                files.extend(self.root.rglob(f"*{ext}"))
                files.extend(self.root.rglob(f"*{ext.upper()}"))
            return sorted(set(files))
        else:
            return sorted(
                [f for f in self.root.iterdir() if f.suffix.lower() in self.EXTENSIONS]
            )

    def _seed_worker(self, idx: int):
        worker_seed = self.seed + idx
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def get_random_masks(self, image_tensor: torch.Tensor, idx: int) -> torch.Tensor:
        if self.mask_generator is None:
            self._init_mask_generator()
        _, h, w = image_tensor.shape
        self.mask_generator.height = h
        self.mask_generator.width = w
        self._seed_worker(idx)
        return self.mask_generator.generate(image_tensor=image_tensor, **self.mask_config)

    @abstractmethod
    def _load_annotation(self, idx: int, image_path: Path) -> torch.Tensor:
        pass

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, DataPoint]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range.")

        path = self.image_paths[idx]
        
        image = load_image(path, transform=self.transform)
        
        masks: Optional[torch.Tensor] = None

        if self.mask_mode == MaskMode.ANNOTATION:
            if not self.has_annotations:
                raise ValueError(f"Dataset {self.__class__.__name__} does not support annotations.")
            masks = self._load_annotation(idx, path)

        elif self.mask_mode == MaskMode.RANDOM:
            masks = self.get_random_masks(image, idx)

        if masks is not None:
            img_h, img_w = image.shape[-2:]
            mask_h, mask_w = masks.shape[-2:]

            if img_h != mask_h or img_w != mask_w:
                masks = F.interpolate(
                    masks.unsqueeze(0),
                    size=(img_h, img_w),
                    mode='nearest'
                ).squeeze(0)

        return path.name, DataPoint(image=image, masks=masks)

    def get_paths(self) -> List[Path]:
        return self.image_paths.copy()
