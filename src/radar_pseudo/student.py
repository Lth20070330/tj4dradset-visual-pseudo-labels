from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .dataset import KittiObject, TJ4DRadSet
from .geometry import kitti_box_corners_camera, rectified_camera_to_radar


CLASSES = ("Car", "Truck", "Pedestrian", "Cyclist", "Other")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class BEVConfig:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = -40.0
    y_max: float = 40.0
    resolution: float = 0.5

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


def radar_to_bev(radar: np.ndarray, config: BEVConfig) -> np.ndarray:
    x, y = radar[:, 0], radar[:, 1]
    row = np.floor((x - config.x_min) / config.resolution).astype(int)
    col = np.floor((y - config.y_min) / config.resolution).astype(int)
    valid = (row >= 0) & (row < config.height) & (col >= 0) & (col < config.width)
    row, col, points = row[valid], col[valid], radar[valid]
    count = np.zeros((config.height, config.width), dtype=np.float32)
    sum_z = np.zeros_like(count); sum_vr = np.zeros_like(count); sum_power = np.zeros_like(count)
    max_power = np.full_like(count, -30.0)
    np.add.at(count, (row, col), 1)
    np.add.at(sum_z, (row, col), points[:, 2])
    np.add.at(sum_vr, (row, col), points[:, 3])
    np.add.at(sum_power, (row, col), points[:, 5])
    np.maximum.at(max_power, (row, col), points[:, 5])
    denominator = np.maximum(count, 1)
    features = np.stack(
        (
            np.clip(np.log1p(count) / np.log(10.0), 0, 1),
            np.clip((sum_z / denominator + 3.0) / 6.0, 0, 1),
            np.clip((sum_vr / denominator + 20.0) / 40.0, 0, 1),
            np.clip((sum_power / denominator + 10.0) / 40.0, 0, 1),
            np.clip((max_power + 10.0) / 40.0, 0, 1),
        ),
        axis=0,
    )
    features[:, count == 0] = 0
    return features.astype(np.float32)


def draw_gaussian(heatmap: np.ndarray, row: int, col: int, radius: int = 2) -> None:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * max(radius / 2, 0.5) ** 2))
    top, bottom = max(0, row - radius), min(heatmap.shape[0], row + radius + 1)
    left, right = max(0, col - radius), min(heatmap.shape[1], col + radius + 1)
    kernel_top, kernel_left = top - (row - radius), left - (col - radius)
    patch = kernel[kernel_top:kernel_top + bottom - top, kernel_left:kernel_left + right - left]
    np.maximum(heatmap[top:bottom, left:right], patch, out=heatmap[top:bottom, left:right])


def load_label_file(path: Path) -> list[KittiObject]:
    return [KittiObject.from_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def object_to_radar_parameters(obj: KittiObject, calibration) -> tuple[np.ndarray, float]:
    center = rectified_camera_to_radar(obj.location_camera[None, :], calibration)[0]
    corners = rectified_camera_to_radar(kitti_box_corners_camera(obj), calibration)
    length_axis = corners[0, :2] - corners[3, :2]
    yaw = float(np.arctan2(length_axis[1], length_axis[0]))
    return center, yaw


class RadarDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        frame_ids: list[str],
        label_dir: str | Path | None = None,
        metadata_dir: str | Path | None = None,
        ignore_label_dir: str | Path | None = None,
        bev_config: BEVConfig = BEVConfig(),
    ) -> None:
        self.dataset = TJ4DRadSet(dataset_root)
        self.frame_ids = frame_ids
        self.label_dir = Path(label_dir) if label_dir else self.dataset.label_dir
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.ignore_label_dir = Path(ignore_label_dir) if ignore_label_dir else None
        self.config = bev_config

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int):
        frame_id = self.frame_ids[index]
        radar = self.dataset.load_radar(frame_id)
        calibration = self.dataset.load_calibration(frame_id)
        objects = load_label_file(self.label_dir / f"{frame_id}.txt")
        quality_records = None
        if self.metadata_dir is not None:
            import json
            quality_records = json.loads((self.metadata_dir / f"{frame_id}.json").read_text(encoding="utf-8"))
        heatmap = np.zeros((len(CLASSES), self.config.height, self.config.width), dtype=np.float32)
        regression = np.zeros((8, self.config.height, self.config.width), dtype=np.float32)
        weight = np.zeros((8, self.config.height, self.config.width), dtype=np.float32)
        positive_weight = np.zeros_like(heatmap)
        classification_weight = np.ones((1, self.config.height, self.config.width), dtype=np.float32)
        if self.ignore_label_dir is not None:
            import cv2

            ignored_objects = load_label_file(self.ignore_label_dir / f"{frame_id}.txt")
            for ignored in ignored_objects:
                if ignored.category not in CLASS_TO_INDEX:
                    continue
                ignored_center, ignored_yaw = object_to_radar_parameters(ignored, calibration)
                _, ignored_width, ignored_length = ignored.dimensions_hwl
                local = np.array(
                    [[ignored_length / 2, ignored_width / 2], [ignored_length / 2, -ignored_width / 2],
                     [-ignored_length / 2, -ignored_width / 2], [-ignored_length / 2, ignored_width / 2]],
                    dtype=np.float64,
                )
                cosine, sine = np.cos(ignored_yaw), np.sin(ignored_yaw)
                rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
                corners = local @ rotation.T + ignored_center[:2]
                pixels = np.column_stack(
                    ((corners[:, 1] - self.config.y_min) / self.config.resolution,
                     (corners[:, 0] - self.config.x_min) / self.config.resolution)
                ).round().astype(np.int32)
                cv2.fillConvexPoly(classification_weight[0], pixels, 0.0)
        for object_index, obj in enumerate(objects):
            if obj.category not in CLASS_TO_INDEX:
                continue
            center, yaw = object_to_radar_parameters(obj, calibration)
            row_float = (center[0] - self.config.x_min) / self.config.resolution
            col_float = (center[1] - self.config.y_min) / self.config.resolution
            row, col = int(np.floor(row_float)), int(np.floor(col_float))
            if not (0 <= row < self.config.height and 0 <= col < self.config.width):
                continue
            class_index = CLASS_TO_INDEX[obj.category]
            draw_gaussian(heatmap[class_index], row, col)
            h, w, length = obj.dimensions_hwl
            regression[:, row, col] = (
                row_float - row, col_float - col, center[2],
                np.log(max(length, 1e-3)), np.log(max(w, 1e-3)), np.log(max(h, 1e-3)),
                np.sin(yaw), np.cos(yaw),
            )
            if quality_records is not None and object_index < len(quality_records):
                record = quality_records[object_index]
                legacy_quality = float(record.get("quality", 1.0))
                class_quality = float(record.get("class_quality", legacy_quality))
                center_quality = float(record.get("center_quality", legacy_quality))
                size_quality = float(record.get("size_quality", legacy_quality))
                yaw_quality = float(record.get("yaw_quality", legacy_quality))
            else:
                class_quality = center_quality = size_quality = yaw_quality = 1.0
            positive_weight[class_index, row, col] = class_quality
            weight[0:3, row, col] = np.sqrt(class_quality * center_quality)
            weight[3:6, row, col] = np.sqrt(class_quality * size_quality)
            weight[6:8, row, col] = np.sqrt(class_quality * yaw_quality)
        return {
            "frame_id": frame_id,
            "features": torch.from_numpy(radar_to_bev(radar, self.config)),
            "heatmap": torch.from_numpy(heatmap),
            "regression": torch.from_numpy(regression),
            "weight": torch.from_numpy(weight),
            "positive_weight": torch.from_numpy(positive_weight),
            "classification_weight": torch.from_numpy(classification_weight),
        }


def conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
    )


class RadarBEVDetector(nn.Module):
    def __init__(self, input_channels: int = 5, classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.stem = conv_block(input_channels, 24)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), conv_block(24, 48))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), conv_block(48, 96))
        self.up1 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.decode1 = conv_block(96, 48)
        self.up2 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.decode2 = conv_block(48, 24)
        self.heatmap_head = nn.Conv2d(24, classes, 1)
        self.regression_head = nn.Conv2d(24, 8, 1)
        nn.init.constant_(self.heatmap_head.bias, -3.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        level0 = self.stem(features)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        decoded1 = self.decode1(torch.cat((self.up1(level2), level1), dim=1))
        decoded0 = self.decode2(torch.cat((self.up2(decoded1), level0), dim=1))
        return {"heatmap": self.heatmap_head(decoded0), "regression": self.regression_head(decoded0)}


def detection_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["heatmap"]
    target = batch["heatmap"]
    probabilities = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = target.eq(1).float()
    negatives = target.lt(1).float()
    negative_weights = (1 - target).pow(4)
    positive_weights = batch.get("positive_weight", positives)
    classification_weights = batch.get("classification_weight", torch.ones_like(target[:, :1]))
    positive_loss = -(probabilities.log()) * (1 - probabilities).pow(2) * positives * positive_weights
    negative_loss = -((1 - probabilities).log()) * probabilities.pow(2) * negative_weights * negatives * classification_weights
    number_of_positives = (positives * positive_weights).sum().clamp_min(1.0)
    heatmap_loss = (positive_loss.sum() + negative_loss.sum()) / number_of_positives
    mask = batch["weight"]
    regression_error = torch.nn.functional.smooth_l1_loss(outputs["regression"], batch["regression"], reduction="none")
    regression_loss = (regression_error * mask).sum() / mask.sum().clamp_min(1.0)
    total = heatmap_loss + regression_loss
    return total, {"heatmap": float(heatmap_loss.detach()), "regression": float(regression_loss.detach())}


@dataclass(frozen=True)
class DetectionBEV:
    category: str
    score: float
    center_xyz: np.ndarray
    dimensions_lwh: np.ndarray
    yaw: float


@torch.inference_mode()
def decode_detections(
    outputs: dict[str, torch.Tensor],
    config: BEVConfig = BEVConfig(),
    score_threshold: float = 0.05,
    top_k: int = 100,
) -> list[list[DetectionBEV]]:
    heatmap = outputs["heatmap"].sigmoid()
    pooled = torch.nn.functional.max_pool2d(heatmap, 3, stride=1, padding=1)
    heatmap = heatmap * heatmap.eq(pooled)
    regression = outputs["regression"]
    decoded_batch = []
    for batch_index in range(heatmap.shape[0]):
        scores, flat_indices = torch.topk(heatmap[batch_index].flatten(), k=min(top_k, heatmap[batch_index].numel()))
        detections = []
        for score_tensor, flat_index_tensor in zip(scores, flat_indices, strict=True):
            score = float(score_tensor)
            if score < score_threshold:
                break
            flat_index = int(flat_index_tensor)
            spatial_size = config.height * config.width
            class_index = flat_index // spatial_size
            spatial_index = flat_index % spatial_size
            row, col = spatial_index // config.width, spatial_index % config.width
            values = regression[batch_index, :, row, col].detach().cpu().numpy()
            center = np.array(
                [
                    config.x_min + (row + values[0]) * config.resolution,
                    config.y_min + (col + values[1]) * config.resolution,
                    values[2],
                ],
                dtype=np.float64,
            )
            length, width, height = np.exp(np.clip(values[3:6], -3, 4))
            yaw = float(np.arctan2(values[6], values[7]))
            detections.append(DetectionBEV(CLASSES[class_index], score, center, np.array([length, width, height]), yaw))
        decoded_batch.append(detections)
    return decoded_batch
