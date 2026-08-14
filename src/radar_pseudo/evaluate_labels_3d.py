from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import KittiObject, TJ4DRadSet
from .geometry import rectified_camera_to_radar
from .metrics3d import (
    ScoredObject,
    aligned_scale_error,
    bev_iou,
    center_distance_bev,
    evaluate_ranked_detections,
    iou_3d,
    orientation_error,
)


DEFAULT_CLASSES = ("Car", "Truck", "Pedestrian", "Cyclist")
DEFAULT_IOU = {"Car": 0.5, "Truck": 0.5, "Pedestrian": 0.25, "Cyclist": 0.25}


def read_objects(path: Path) -> list[KittiObject]:
    if not path.is_file():
        return []
    return [KittiObject.from_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_scores(path: Path | None, count: int) -> list[float]:
    if path is None or not path.is_file():
        return [1.0] * count
    records = json.loads(path.read_text(encoding="utf-8"))
    return [float(record.get("quality", record.get("score", 1.0))) for record in records[:count]] + [1.0] * max(0, count - len(records))


def radar_range(obj: KittiObject, calibration) -> float:
    center = rectified_camera_to_radar(obj.location_camera[None, :], calibration)[0]
    return float(np.linalg.norm(center[:2]))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def evaluate_label_directory(
    dataset: TJ4DRadSet,
    frame_ids: list[str],
    prediction_dir: Path,
    metadata_dir: Path | None,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    maximum_range: float = 70.0,
) -> dict[str, object]:
    predictions_by_class: dict[str, list[ScoredObject]] = defaultdict(list)
    ground_truth_by_class: dict[str, dict[str, list[KittiObject]]] = {category: {} for category in classes}
    diagnostics: dict[str, dict[str, list[float]]] = {
        category: {"translation": [], "scale": [], "orientation": []} for category in classes
    }
    center_thresholds = (0.5, 1.0, 2.0, 4.0)
    center_hits = {category: {str(threshold): 0 for threshold in center_thresholds} for category in classes}

    for frame_id in frame_ids:
        calibration = dataset.load_calibration(frame_id)
        ground_truth = [
            obj for obj in dataset.load_labels(frame_id)
            if obj.category in classes and radar_range(obj, calibration) <= maximum_range
        ]
        predicted = read_objects(prediction_dir / f"{frame_id}.txt")
        scores = read_scores(metadata_dir / f"{frame_id}.json" if metadata_dir else None, len(predicted))
        predicted_with_scores = [
            (obj, score) for obj, score in zip(predicted, scores, strict=True)
            if obj.category in classes and radar_range(obj, calibration) <= maximum_range
        ]
        for category in classes:
            category_ground_truth = [obj for obj in ground_truth if obj.category == category]
            category_predictions = [(obj, score) for obj, score in predicted_with_scores if obj.category == category]
            ground_truth_by_class[category][frame_id] = category_ground_truth
            predictions_by_class[category].extend(ScoredObject(frame_id, obj, score) for obj, score in category_predictions)

            used_ground_truth: set[int] = set()
            for prediction, _ in sorted(category_predictions, key=lambda item: item[1], reverse=True):
                candidates = [
                    (index, center_distance_bev(prediction, truth))
                    for index, truth in enumerate(category_ground_truth) if index not in used_ground_truth
                ]
                if not candidates:
                    continue
                truth_index, distance = min(candidates, key=lambda item: item[1])
                if distance <= 2.0:
                    used_ground_truth.add(truth_index)
                    truth = category_ground_truth[truth_index]
                    diagnostics[category]["translation"].append(distance)
                    diagnostics[category]["scale"].append(aligned_scale_error(prediction, truth))
                    diagnostics[category]["orientation"].append(orientation_error(prediction, truth))
                    for threshold in center_thresholds:
                        center_hits[category][str(threshold)] += int(distance <= threshold)

    per_class: dict[str, object] = {}
    for category in classes:
        threshold = DEFAULT_IOU[category]
        bev_result = evaluate_ranked_detections(
            predictions_by_class[category], ground_truth_by_class[category], threshold, bev_iou
        )
        result_3d = evaluate_ranked_detections(
            predictions_by_class[category], ground_truth_by_class[category], threshold, iou_3d
        )
        values = diagnostics[category]
        per_class[category] = {
            "iou_threshold": threshold,
            "bev": bev_result,
            "3d": result_3d,
            "tp_diagnostics_at_2m": {
                "matches": len(values["translation"]),
                "ate_mean_m": float(np.mean(values["translation"])) if values["translation"] else None,
                "ate_median_m": percentile(values["translation"], 50),
                "ate_p90_m": percentile(values["translation"], 90),
                "ase_mean": float(np.mean(values["scale"])) if values["scale"] else None,
                "aoe_mean_rad": float(np.mean(values["orientation"])) if values["orientation"] else None,
                "center_hits": center_hits[category],
            },
        }
    map_bev = float(np.mean([per_class[category]["bev"]["ap_r40"] for category in classes]))
    map_3d = float(np.mean([per_class[category]["3d"]["ap_r40"] for category in classes]))
    return {
        "protocol": {
            "classes": list(classes),
            "maximum_range_m": maximum_range,
            "vehicle_iou": 0.5,
            "vru_iou": 0.25,
            "ap": "R40 interpolated",
        },
        "frames": len(frame_ids),
        "mAP_BEV_R40": map_bev,
        "mAP_3D_R40": map_3d,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KITTI-format pseudo labels with TJ4DRadSet 3D/BEV metrics")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--metadata-dir")
    parser.add_argument("--maximum-range", type=float, default=70.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = TJ4DRadSet(args.dataset_root)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate_label_directory(
        dataset,
        frame_ids,
        Path(args.prediction_dir),
        Path(args.metadata_dir) if args.metadata_dir else None,
        maximum_range=args.maximum_range,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
