from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import KittiObject, TJ4DRadSet
from .geometry import kitti_box_corners_camera, project_camera_points, rectified_camera_to_radar
from .student import load_label_file
from .visualize import BOX_EDGES, CLASS_COLORS


GT_COLOR = "#ff4d5a"
PSEUDO_COLOR = "#00bcd4"


def draw_camera_boxes(image_rgb: np.ndarray, objects: list[KittiObject], calibration, color: tuple[int, int, int], dashed: bool = False) -> np.ndarray:
    canvas = image_rgb.copy()
    for obj in objects:
        corners = kitti_box_corners_camera(obj)
        pixels, depth = project_camera_points(corners, calibration.p2)
        for first, second in BOX_EDGES:
            if depth[first] <= 0 or depth[second] <= 0:
                continue
            p1, p2 = tuple(np.rint(pixels[first]).astype(int)), tuple(np.rint(pixels[second]).astype(int))
            if dashed:
                for t0 in np.arange(0, 1, 0.18):
                    t1 = min(t0 + 0.10, 1.0)
                    a = tuple(np.rint((1 - t0) * np.array(p1) + t0 * np.array(p2)).astype(int))
                    b = tuple(np.rint((1 - t1) * np.array(p1) + t1 * np.array(p2)).astype(int))
                    cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
            else:
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
    return canvas


def radar_corners(objects: list[KittiObject], calibration) -> list[tuple[str, np.ndarray]]:
    return [(obj.category, rectified_camera_to_radar(kitti_box_corners_camera(obj), calibration)) for obj in objects]


def draw_bev(ax, radar: np.ndarray, gt_boxes, pseudo_boxes) -> None:
    mask = (radar[:, 0] >= 0) & (radar[:, 0] <= 100) & (np.abs(radar[:, 1]) <= 40)
    ax.scatter(radar[mask, 1], radar[mask, 0], s=1.2, c=radar[mask, 5], cmap="gray_r", alpha=.42, linewidths=0)
    for _, corners in gt_boxes:
        polygon = corners[[0, 1, 2, 3, 0], :2]
        ax.plot(polygon[:, 1], polygon[:, 0], color=GT_COLOR, linewidth=1.8)
    for _, corners in pseudo_boxes:
        polygon = corners[[0, 1, 2, 3, 0], :2]
        ax.plot(polygon[:, 1], polygon[:, 0], color=PSEUDO_COLOR, linewidth=1.8, linestyle="--")
    ax.set_xlim(-40, 40); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_xlabel("Lateral y (m)"); ax.set_ylabel("Forward x (m)")
    ax.grid(alpha=.16); ax.set_title("BEV overlay: GT solid / pseudo dashed")


def draw_3d(ax, radar: np.ndarray, boxes, color: str, title: str) -> None:
    mask = (radar[:, 0] >= 0) & (radar[:, 0] <= 80) & (np.abs(radar[:, 1]) <= 35) & (radar[:, 2] > -4) & (radar[:, 2] < 6)
    points = radar[mask]
    if len(points) > 5000:
        points = points[:: max(1, len(points) // 5000)]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.0, c=points[:, 5], cmap="gray_r", alpha=.35)
    for category, corners in boxes:
        for first, second in BOX_EDGES:
            ax.plot(*zip(corners[first], corners[second]), color=color, linewidth=1.4)
        center = corners.mean(axis=0)
        ax.text(center[0], center[1], center[2] + .5, category, color=color, fontsize=7)
    ax.set_xlim(0, 80); ax.set_ylim(-35, 35); ax.set_zlim(-4, 6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=24, azim=-125); ax.set_title(title)


def render_comparison(dataset: TJ4DRadSet, pseudo_label_dir: Path, frame_id: str, output: Path) -> None:
    image = dataset.load_image(frame_id)
    radar = dataset.load_radar(frame_id)
    calibration = dataset.load_calibration(frame_id)
    gt = dataset.load_labels(frame_id)
    pseudo = load_label_file(pseudo_label_dir / f"{frame_id}.txt")
    camera = draw_camera_boxes(image, gt, calibration, (255, 77, 90), dashed=False)
    camera = draw_camera_boxes(camera, pseudo, calibration, (0, 188, 212), dashed=True)
    gt_radar, pseudo_radar = radar_corners(gt, calibration), radar_corners(pseudo, calibration)

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 1.08))
    camera_ax = figure.add_subplot(grid[0, 0])
    camera_ax.imshow(camera); camera_ax.axis("off")
    camera_ax.set_title(f"Frame {frame_id}: image-space 3D boxes\nGT solid red / pseudo dashed cyan")
    bev_ax = figure.add_subplot(grid[0, 1]); draw_bev(bev_ax, radar, gt_radar, pseudo_radar)
    gt_ax = figure.add_subplot(grid[1, 0], projection="3d"); draw_3d(gt_ax, radar, gt_radar, GT_COLOR, f"Radar point cloud + GT ({len(gt)} boxes)")
    pseudo_ax = figure.add_subplot(grid[1, 1], projection="3d"); draw_3d(pseudo_ax, radar, pseudo_radar, PSEUDO_COLOR, f"Radar point cloud + pseudo labels ({len(pseudo)} boxes)")
    figure.suptitle("TJ4DRadSet visual pseudo-label comparison", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create camera/BEV/3D GT-vs-pseudo comparison figures")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pseudo-label-dir", required=True)
    parser.add_argument("--frame-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", default="docs/assets")
    args = parser.parse_args()
    dataset = TJ4DRadSet(args.dataset_root)
    for frame_id in args.frame_ids:
        output = Path(args.output_dir) / f"pseudo-comparison-{frame_id}.png"
        render_comparison(dataset, Path(args.pseudo_label_dir), frame_id, output)
        print(output.resolve())


if __name__ == "__main__":
    main()
