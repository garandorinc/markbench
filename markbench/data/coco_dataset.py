import json
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

from .image_dataset import ImageDataset, MaskMode

class CocoDataset(ImageDataset):
    """
    COCO dataset wrapper.
    """

    def __init__(
        self,
        root: str,
        load_train: bool = True,
        seed: int = 42,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_mode: Union[MaskMode, str] = MaskMode.ANNOTATION,
        mask_config: Optional[Dict[str, Any]] = None,
    ):
        self.load_train = load_train
        self.split_name = "train2017" if load_train else "val2017"
        
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

        self.annotation_dir = self.root / "annotations"
        self.annotations_map: Dict[str, List[Dict[str, Any]]] = {}

        if self.mask_mode == MaskMode.ANNOTATION:
            self._parse_coco_json()

    @property
    def has_annotations(self) -> bool:
        return True

    def _find_images(self) -> List[Path]:

        img_dir = self.root / self.split_name
        if not img_dir.exists():
            if (self.root / f"{self.split_name}.jpg").exists():
                img_dir = self.root
            else:
                print(f"Warning: Directory {img_dir} not found.")
                return []

        image_paths = []
        
        with os.scandir(img_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(tuple(self.EXTENSIONS)):
                    image_paths.append(Path(entry.path))
        
        
        image_paths.sort()
        return image_paths

    def _parse_coco_json(self):

        json_name = f"instances_{self.split_name}.json"
        json_path = self.annotation_dir / json_name
        
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation file missing: {json_path}")

        print(f"Loading metadata from {json_name}...")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            relevant_filenames = {p.name for p in self.image_paths}
            
            img_id_to_name = {}
            for img_info in data.get("images", []):
                fname = img_info["file_name"]
                if fname in relevant_filenames:
                    img_id_to_name[img_info["id"]] = fname
                    self.annotations_map[fname] = []

            count = 0
            for ann in data.get("annotations", []):
                img_id = ann["image_id"]
                if ann.get("iscrowd", 0) == 1:
                    continue
                
                if img_id in img_id_to_name:
                    filename = img_id_to_name[img_id]
                    self.annotations_map[filename].append(ann)
                    count += 1
            
            print(f"Parsed {count} annotations for {len(self.annotations_map)} images.")

        except Exception as e:
            raise RuntimeError(f"Failed to parse COCO annotations: {e}")

    def _load_annotation(self, idx: int, image_path: Path) -> Optional[torch.Tensor]:
        filename = image_path.name
        raw_anns = self.annotations_map.get(filename, [])
        if not raw_anns:
            return None

        img_temp = cv2.imread(str(image_path))
        if img_temp is None: return None
        h, w, _ = img_temp.shape

        masks_list = []
        for ann in raw_anns:
            seg = ann.get("segmentation", [])
            if not seg or isinstance(seg, dict): continue
            
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in seg:
                pts = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
            masks_list.append(torch.from_numpy(mask).float())

        if not masks_list: return None

        return torch.stack(masks_list)

    def __getitem__(self, idx: int):
        name, datapoint = super().__getitem__(idx)
        
        if datapoint.masks is not None:
            img_h, img_w = datapoint.image.shape[-2:]
            mask_h, mask_w = datapoint.masks.shape[-2:]
            
            if (img_h != mask_h) or (img_w != mask_w):
                datapoint.masks = F.interpolate(
                    datapoint.masks.unsqueeze(0),
                    size=(img_h, img_w),
                    mode='nearest'
                ).squeeze(0)
                
        return name, datapoint
