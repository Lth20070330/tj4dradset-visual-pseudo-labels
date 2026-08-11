from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .dataset import TJ4DRadSet
from .geometry import (
    kitti_box_corners_camera,
    project_camera_points,
    radar_to_rectified_camera,
    rectified_camera_to_radar,
)


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
CLASS_COLORS = {
    "Car": (255, 92, 92),
    "Truck": (255, 190, 60),
    "Pedestrian": (70, 230, 90),
    "Cyclist": (80, 180, 255),
    "Other": (220, 100, 255),
}


def _draw_box_edges(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray, color: tuple[int, int, int]) -> None:
    for first, second in BOX_EDGES:
        if valid[first] and valid[second]:
            p1 = tuple(np.rint(pixels[first]).astype(int))
            p2 = tuple(np.rint(pixels[second]).astype(int))
            cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)


def render_camera_overlay(dataset: TJ4DRadSet, frame_id: str) -> np.ndarray:
    rgb = dataset.load_image(frame_id).copy()
    calibration = dataset.load_calibration(frame_id)
    radar = dataset.load_radar(frame_id)
    camera_points = radar_to_rectified_camera(radar[:, :3], calibration)
    pixels, depth = project_camera_points(camera_points, calibration.p2)
    height, width = rgb.shape[:2]
    valid = (
        (depth > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    if valid.any():
        distances = radar[valid, 4]
        normalized = np.clip(distances / 120.0, 0, 1)
        color_values = np.uint8(255 * (1 - normalized))
        heat = cv2.applyColorMap(color_values[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
        for (u, v), color in zip(np.rint(pixels[valid]).astype(int), heat, strict=True):
            cv2.circle(rgb, (u, v), 2, tuple(int(x) for x in color), -1, cv2.LINE_AA)

    for obj in dataset.load_labels(frame_id):
        if obj.category == "DontCare":
            continue
        corners = kitti_box_corners_camera(obj)
        box_pixels, box_depth = project_camera_points(corners, calibration.p2)
        color = CLASS_COLORS.get(obj.category, (255, 255, 255))
        _draw_box_edges(rgb, box_pixels, box_depth > 0, color)
        x, y = np.rint(box_pixels[0]).astype(int)
        cv2.putText(rgb, obj.category, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return rgb


def render_bev(dataset: TJ4DRadSet, frame_id: str, x_limits=(0.0, 120.0), y_limits=(-50.0, 50.0), scale=8) -> np.ndarray:
    width = int((y_limits[1] - y_limits[0]) * scale)
    height = int((x_limits[1] - x_limits[0]) * scale)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)

    def to_pixel(points_xy: np.ndarray) -> np.ndarray:
        px = (points_xy[:, 1] - y_limits[0]) * scale
        py = height - (points_xy[:, 0] - x_limits[0]) * scale
        return np.column_stack((px, py)).astype(int)

    radar = dataset.load_radar(frame_id)
    inside = (
        (radar[:, 0] >= x_limits[0]) & (radar[:, 0] <= x_limits[1])
        & (radar[:, 1] >= y_limits[0]) & (radar[:, 1] <= y_limits[1])
    )
    for u, v in to_pixel(radar[inside, :2]):
        cv2.circle(canvas, (u, v), 1, (70, 70, 70), -1)

    calibration = dataset.load_calibration(frame_id)
    for obj in dataset.load_labels(frame_id):
        if obj.category == "DontCare":
            continue
        corners_camera = kitti_box_corners_camera(obj)
        corners_radar = rectified_camera_to_radar(corners_camera, calibration)
        polygon = to_pixel(corners_radar[[0, 1, 2, 3], :2])
        color = CLASS_COLORS.get(obj.category, (255, 255, 255))
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
    center_x = int((0 - y_limits[0]) * scale)
    cv2.arrowedLine(canvas, (center_x, height - 5), (center_x, height - 80), (20, 20, 20), 2, tipLength=0.15)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TJ4DRadSet camera and BEV overlays")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--output-dir", default="outputs/visualizations")
    args = parser.parse_args()
    dataset = TJ4DRadSet(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_path = output_dir / f"{args.frame_id}_camera.png"
    bev_path = output_dir / f"{args.frame_id}_bev.png"
    cv2.imwrite(str(camera_path), cv2.cvtColor(render_camera_overlay(dataset, args.frame_id), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(bev_path), cv2.cvtColor(render_bev(dataset, args.frame_id), cv2.COLOR_RGB2BGR))
    print(camera_path.resolve())
    print(bev_path.resolve())


if __name__ == "__main__":
    main()
