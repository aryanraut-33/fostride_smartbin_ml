import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, List

class CocoSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Roboflow COCO Segmentation data.
    Expected format is a directory containing images and a single `_annotations.coco.json` file.
    """
    # These are the only valid object classes (excluding the supercategory)
    VALID_CLASS_IDS = {1, 2, 3}  # metal, paper, plastic
    
    def __init__(self, root_dir: str, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        annotation_path = os.path.join(self.root_dir, "_annotations.coco.json")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotations file not found at {annotation_path}")
            
        with open(annotation_path, "r") as f:
            self.coco_data = json.load(f)
            
        # Map category names to IDs (exclude the supercategory 'metal-paper-plastic')
        self.categories = {
            cat["id"]: cat["name"]
            for cat in self.coco_data["categories"]
            if cat["id"] in self.VALID_CLASS_IDS
        }
        
        # Build image records (only images with valid annotations)
        self.images = []
        self.img_to_anns = {}
        
        for ann in self.coco_data["annotations"]:
            # Skip annotations for the supercategory (category_id=0)
            if ann["category_id"] not in self.VALID_CLASS_IDS:
                continue
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        for img in self.coco_data["images"]:
            if img["id"] in self.img_to_anns:
                self.images.append(img)
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_id = img_info["id"]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # Bounding box: COCO format is [x_min, y_min, width, height]
            # PyTorch expects [x_min, y_min, x_max, y_max]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
            
            # Segmentation mask: COCO format is a list of polygons [x1, y1, x2, y2, ...]
            # We need to convert this to a 2D binary numpy array matching the image dimensions
            mask = self._poly_to_mask(ann["segmentation"], img_info["height"], img_info["width"])
            masks.append(mask)
            
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle empty annotation lists (e.g. background images)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            # IMPORTANT: For object detection/segmentation, transforms must update both 
            # the image AND the bounding boxes/masks simultaneously (e.g., albumentations or torchvision v2)
            img, target = self.transforms(img, target)
            
        return img, target
        
    def _poly_to_mask(self, polygons, height, width):
        """Convert COCO polygon segmentation to a 2D binary numpy array."""
        from PIL import ImageDraw
        # COCO segmentation can be a list of lists if there are multiple polygons for one object
        mask = Image.new('L', (width, height), 0)
        for poly in polygons:
            # For simplicity, using PIL drawing to create masks
            ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
        return np.array(mask)
