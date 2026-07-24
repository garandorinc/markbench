"""Data loading utilities for markbench."""

from typing import Dict, Any, List
from .image_dataset import ImageDataset
from .simple_dataset import SimpleImageDataset
from .coco_dataset import CocoDataset
from .div2k_dataset import Div2kDataset
from .sa1b_dataset import Sa1bDataset
from .mirflickr_dataset import MirFlickrDataset

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "simple": {
        "class": SimpleImageDataset,
        "default_root": "data/images",
    },
    "coco": {
        "class": CocoDataset,
        "default_root": "data/coco2017",
    },
    "div2k": {
        "class": Div2kDataset,
        "default_root": "data/div2k",
    },
    "sa1b": {
        "class": Sa1bDataset,
        "default_root": "data/sa1b",
    },
    "mirflickr": {
        "class": MirFlickrDataset,
        "default_root": "data/mirflickr25k",
    },
}

def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())

__all__ = [
    "ImageDataset",
    "SimpleImageDataset",
    "CocoDataset",
    "Div2kDataset",
    "Sa1bDataset",
    "MirFlickrDataset",
    "DATASET_REGISTRY",
    "list_datasets",
]
