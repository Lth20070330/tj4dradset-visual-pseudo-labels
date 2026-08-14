from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import TJ4DRadSet
from .student import load_label_file
from .visualize_comparison import GT_COLOR, PSEUDO_COLOR, draw_3d, draw_camera_boxes, radar_corners


BASELINE_COLOR = "#f5a623"


def render_ablation(
    dataset: TJ4DRadSet,
    baseline_dir: Path,
    mgu_dir: Path,
    frame_id: str,
    output: Path,
) -> None:
    image = dataset.load_image(frame_id)
    radar = dataset.load_radar(frame_id)
    calibration = dataset.load_calibration(frame_id)
    ground_truth = dataset.load_labels(frame_id)
    baseline = load_label_file(baseline_dir / f"{frame_id}.txt")
    mgu = load_label_file(mgu_dir / f"{frame_id}.txt")

    camera = draw_camera_boxes(image, ground_truth, calibration, (255, 77, 90))
    camera = draw_camera_boxes(camera, mgu, calibration, (0, 188, 212), dashed=True)
    gt_radar = radar_corners(ground_truth, calibration)
    baseline_radar = radar_corners(baseline, calibration)
    mgu_radar = radar_corners(mgu, calibration)

    figure = plt.figure(figsize=(17, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    camera_ax = figure.add_subplot(grid[0, 0])
    camera_ax.imshow(camera)
    camera_ax.axis("off")
    camera_ax.set_title(f"Frame {frame_id}: GT red / MGU-PL cyan dashed")
    gt_ax = figure.add_subplot(grid[0, 1], projection="3d")
    draw_3d(gt_ax, radar, gt_radar, GT_COLOR, f"Ground truth ({len(ground_truth)} boxes)")
    baseline_ax = figure.add_subplot(grid[1, 0], projection="3d")
    draw_3d(baseline_ax, radar, baseline_radar, BASELINE_COLOR, f"B0 single-cluster baseline ({len(baseline)} boxes)")
    mgu_ax = figure.add_subplot(grid[1, 1], projection="3d")
    draw_3d(mgu_ax, radar, mgu_radar, PSEUDO_COLOR, f"MGU-PL selected labels ({len(mgu)} boxes)")
    figure.suptitle("TJ4DRadSet 3D pseudo-label ablation: B0 vs MGU-PL", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render B0-vs-MGU-vs-GT camera and 3D point-cloud figures")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--baseline-label-dir", required=True)
    parser.add_argument("--mgu-label-dir", required=True)
    parser.add_argument("--frame-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", default="docs/assets")
    args = parser.parse_args()
    dataset = TJ4DRadSet(args.dataset_root)
    for frame_id in args.frame_ids:
        output = Path(args.output_dir) / f"ablation-b0-mgu-{frame_id}.png"
        render_ablation(dataset, Path(args.baseline_label_dir), Path(args.mgu_label_dir), frame_id, output)
        print(output.resolve())


if __name__ == "__main__":
    main()
