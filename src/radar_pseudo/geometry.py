from __future__ import annotations

import numpy as np

from .dataset import Calibration, KittiObject


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=points.dtype)))
    return (homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def radar_to_rectified_camera(points_radar: np.ndarray, calibration: Calibration) -> np.ndarray:
    camera = transform_points(points_radar, calibration.radar_to_camera)
    return transform_points(camera, calibration.r0_rect)


def rectified_camera_to_radar(points_camera: np.ndarray, calibration: Calibration) -> np.ndarray:
    unrectified = transform_points(points_camera, np.linalg.inv(calibration.r0_rect))
    return transform_points(unrectified, calibration.camera_to_radar)


def project_camera_points(points_camera: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=points.dtype)))
    projected = homogeneous @ np.asarray(p2, dtype=np.float64).T
    depth = projected[:, 2]
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-6
    pixels[valid] = projected[valid, :2] / depth[valid, None]
    return pixels, depth


def kitti_box_corners_camera(obj: KittiObject) -> np.ndarray:
    """Return eight corners; KITTI location is the bottom-center of the box."""
    h, w, length = obj.dimensions_hwl
    x = np.array([length / 2, length / 2, -length / 2, -length / 2,
                  length / 2, length / 2, -length / 2, -length / 2])
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w / 2, -w / 2, -w / 2, w / 2,
                  w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.column_stack((x, y, z)) @ rotation.T + obj.location_camera


def points_in_kitti_box_camera(points_camera: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Return a mask for points inside a KITTI camera-coordinate 3D box."""
    points = np.asarray(points_camera, dtype=np.float64)
    h, w, length = obj.dimensions_hwl
    translated = points - obj.location_camera
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    local = translated @ rotation
    tolerance = 1e-8
    return (
        (np.abs(local[:, 0]) <= length / 2 + tolerance)
        & (local[:, 1] <= tolerance)
        & (local[:, 1] >= -h - tolerance)
        & (np.abs(local[:, 2]) <= w / 2 + tolerance)
    )
