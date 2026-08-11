from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class Instance2D:
    source_class: str
    target_class: str | None
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray


COCO_TO_TJ4D = {
    "car": "Car",
    "truck": "Truck",
    "bus": "Truck",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}


class YoloSegmenter:
    def __init__(self, weights: str = "yolo26s-seg.pt", device: str = "cuda:0", image_size: int = 960) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.image_size = image_size

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray, confidence: float = 0.15) -> list[Instance2D]:
        result = self.model.predict(
            image_rgb,
            device=self.device,
            imgsz=self.image_size,
            conf=confidence,
            verbose=False,
            retina_masks=True,
        )[0]
        if result.boxes is None or result.masks is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy() > 0.5
        instances = []
        for box, class_id, score, mask in zip(boxes, classes, scores, masks, strict=True):
            source_class = result.names[int(class_id)]
            target_class = COCO_TO_TJ4D.get(source_class)
            if target_class is not None:
                instances.append(Instance2D(source_class, target_class, float(score), box, mask))
        return instances


class DepthAnythingMetric:
    MODEL_CONFIG = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        repository: str | Path,
        checkpoint: str | Path,
        encoder: str = "vits",
        max_depth: float = 80.0,
        device: str = "cuda:0",
    ) -> None:
        metric_root = Path(repository) / "metric_depth"
        if not metric_root.is_dir():
            raise FileNotFoundError(f"Depth Anything V2 metric_depth not found: {metric_root}")
        sys.path.insert(0, str(metric_root))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        finally:
            sys.path.pop(0)
        self.device = torch.device(device)
        self.model = DepthAnythingV2(**{**self.MODEL_CONFIG[encoder], "max_depth": max_depth})
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray, input_size: int = 518) -> np.ndarray:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.model.infer_image(image_bgr, input_size=input_size).astype(np.float32)
