from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dataset import Calibration
from .geometry import project_camera_points, radar_to_rectified_camera
from .vision import Instance2D


CLASS_DIMENSIONS_HWL = {
    "Car": np.array([1.55, 1.80, 4.30], dtype=np.float64),
    "Truck": np.array([3.00, 2.50, 8.00], dtype=np.float64),
    "Pedestrian": np.array([1.70, 0.60, 0.60], dtype=np.float64),
    "Cyclist": np.array([1.70, 0.80, 1.80], dtype=np.float64),
    "Other": np.array([1.80, 1.80, 4.00], dtype=np.float64),
}


@dataclass(frozen=True)
class PseudoBox3D:
    category: str
    dimensions_hwl: np.ndarray
    location_camera: np.ndarray
    rotation_y: float
    visual_confidence: float
    visual_depth_m: float
    radar_depth_m: float | None
    radar_points: int
    quality: float
    position_source: str
    bbox_2d: np.ndarray

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        for key in ("dimensions_hwl", "location_camera", "bbox_2d"):
            record[key] = record[key].tolist()
        return record


def backproject_pixel(u: float, v: float, depth: float, p2: np.ndarray) -> np.ndarray:
    fx, fy = p2[0, 0], p2[1, 1]
    cx, cy = p2[0, 2], p2[1, 2]
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=np.float64)


def densest_depth_cluster(depths: np.ndarray, gap_m: float = 2.0) -> np.ndarray:
    depths = np.sort(np.asarray(depths, dtype=np.float64))
    if len(depths) == 0:
        return depths
    split_indices = np.flatnonzero(np.diff(depths) > gap_m) + 1
    clusters = np.split(depths, split_indices)
    return max(clusters, key=lambda values: (len(values), -np.median(values)))


def radar_points_for_instance(
    radar: np.ndarray,
    calibration: Calibration,
    instance: Instance2D,
    visual_depth_m: float,
    relative_gate: float = 0.35,
    minimum_gate_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    camera = radar_to_rectified_camera(radar[:, :3], calibration)
    pixels, depth = project_camera_points(camera, calibration.p2)
    u = np.rint(pixels[:, 0]).astype(int)
    v = np.rint(pixels[:, 1]).astype(int)
    height, width = instance.mask.shape
    valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    in_mask = np.zeros(len(radar), dtype=bool)
    in_mask[valid] = instance.mask[v[valid], u[valid]]
    gate = max(minimum_gate_m, relative_gate * visual_depth_m)
    candidates = in_mask & (np.abs(depth - visual_depth_m) <= gate)
    if not candidates.any():
        return camera[:0], radar[:0]
    candidate_indices = np.flatnonzero(candidates)
    cluster = densest_depth_cluster(depth[candidates])
    low, high = cluster.min() - 1e-6, cluster.max() + 1e-6
    chosen = candidate_indices[(depth[candidate_indices] >= low) & (depth[candidate_indices] <= high)]
    return camera[chosen], radar[chosen]


def estimate_pseudo_box(
    instance: Instance2D,
    depth_map_m: np.ndarray,
    radar: np.ndarray,
    calibration: Calibration,
) -> PseudoBox3D:
    mask_depths = depth_map_m[instance.mask]
    valid_depths = mask_depths[np.isfinite(mask_depths) & (mask_depths > 0)]
    if len(valid_depths) < 10:
        raise ValueError("Instance mask does not contain enough valid depth pixels")
    visual_depth = float(np.median(valid_depths))
    camera_hits, _ = radar_points_for_instance(radar, calibration, instance, visual_depth)
    dimensions = CLASS_DIMENSIONS_HWL[instance.target_class or "Other"].copy()
    x1, _, x2, y2 = instance.bbox_xyxy
    u_center = (x1 + x2) / 2

    if len(camera_hits):
        radar_depth = float(np.median(camera_hits[:, 2]))
        center_depth = radar_depth + 0.30 * dimensions[2]
        source = "radar_corrected"
    else:
        radar_depth = None
        center_depth = visual_depth
        source = "visual_depth"
    location = backproject_pixel(u_center, y2, center_depth, calibration.p2)

    depth_agreement = 0.0 if radar_depth is None else float(np.exp(-abs(visual_depth - radar_depth) / 12.0))
    radar_support = min(1.0, len(camera_hits) / 5.0)
    quality = float(np.clip(0.55 * instance.confidence + 0.25 * depth_agreement + 0.20 * radar_support, 0, 1))
    return PseudoBox3D(
        category=instance.target_class or "Other",
        dimensions_hwl=dimensions,
        location_camera=location,
        rotation_y=-np.pi / 2,
        visual_confidence=instance.confidence,
        visual_depth_m=visual_depth,
        radar_depth_m=radar_depth,
        radar_points=len(camera_hits),
        quality=quality,
        position_source=source,
        bbox_2d=instance.bbox_xyxy.copy(),
    )
