from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .dataset import TJ4DRadSet
from .geometry import (
    kitti_box_corners_camera,
    points_in_kitti_box_camera,
    radar_to_rectified_camera,
    rectified_camera_to_radar,
)


def inspect_frame(dataset_root: str, frame_id: str | None) -> dict[str, object]:
    dataset = TJ4DRadSet(dataset_root)
    ids = dataset.frame_ids
    if not ids:
        raise RuntimeError("No complete radar/calibration/label frames found")
    selected = frame_id or ids[0]
    if selected not in ids:
        raise ValueError(f"Frame {selected} is unavailable; range is {ids[0]}..{ids[-1]}")

    radar = dataset.load_radar(selected)
    calibration = dataset.load_calibration(selected)
    labels = dataset.load_labels(selected)
    radar_camera = radar_to_rectified_camera(radar[:, :3], calibration)
    boxes_radar = [
        rectified_camera_to_radar(kitti_box_corners_camera(obj), calibration)
        for obj in labels if obj.category != "DontCare"
    ]
    ranges = radar[:, 4]
    return {
        "dataset_root": str(dataset.root),
        "available_frames": len(ids),
        "frame_id": selected,
        "has_image": dataset.has_image(selected),
        "radar_points": int(len(radar)),
        "radar_feature_names": dataset.RADAR_FEATURES,
        "range_m": {"min": float(np.min(ranges)), "median": float(np.median(ranges)), "max": float(np.max(ranges))},
        "labels": len(labels),
        "classes": dict(Counter(obj.category for obj in labels)),
        "radar_points_per_label": [
            int(points_in_kitti_box_camera(radar_camera, obj).sum()) for obj in labels
        ],
        "radar_box_bounds": [
            {"min_xyz": corners.min(axis=0).round(4).tolist(), "max_xyz": corners.max(axis=0).round(4).tolist()}
            for corners in boxes_radar
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one TJ4DRadSet frame")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frame-id")
    args = parser.parse_args()
    print(json.dumps(inspect_frame(args.dataset_root, args.frame_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
