from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import KittiObject, TJ4DRadSet
from .evaluate_labels_3d import DEFAULT_CLASSES, DEFAULT_IOU, radar_range, read_objects
from .metrics3d import ScoredObject, bev_iou, center_distance_bev, iou_3d


def ranked_outcomes(
    predictions: list[ScoredObject],
    ground_truth: dict[str, list[KittiObject]],
    match_function,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    ranked = sorted(predictions, key=lambda item: item.score, reverse=True)
    matched = {frame_id: np.zeros(len(objects), dtype=bool) for frame_id, objects in ground_truth.items()}
    scores = np.asarray([item.score for item in ranked], dtype=np.float64)
    outcomes = np.zeros(len(ranked), dtype=np.float64)
    for index, prediction in enumerate(ranked):
        candidates = ground_truth.get(prediction.frame_id, [])
        if not candidates:
            continue
        values = np.asarray([match_function(prediction.obj, truth) for truth in candidates], dtype=np.float64)
        if match_function in (bev_iou, iou_3d):
            values[matched[prediction.frame_id]] = -1.0
            best = int(np.argmax(values))
            valid = values[best] >= threshold
        else:
            values[matched[prediction.frame_id]] = np.inf
            best = int(np.argmin(values))
            valid = values[best] <= threshold
        if valid:
            outcomes[index] = 1.0
            matched[prediction.frame_id][best] = True
    return scores, outcomes


def calibration_statistics(scores: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> dict[str, object]:
    records = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (scores >= lower) & (scores < upper if index < bins - 1 else scores <= upper)
        count = int(mask.sum())
        if count:
            mean_score = float(scores[mask].mean())
            accuracy = float(outcomes[mask].mean())
            ece += count / max(len(scores), 1) * abs(mean_score - accuracy)
        else:
            mean_score = accuracy = None
        records.append({"lower": lower, "upper": upper, "count": count, "mean_score": mean_score, "accuracy": accuracy})
    return {
        "samples": len(scores),
        "accuracy": float(outcomes.mean()) if len(outcomes) else 0.0,
        "brier": float(np.mean((scores - outcomes) ** 2)) if len(outcomes) else None,
        "ece": float(ece),
        "bins": records,
    }


def threshold_for_precision(
    scores: np.ndarray,
    outcomes: np.ndarray,
    total_ground_truth: int,
    target_precision: float,
) -> dict[str, float | int | None]:
    if not len(scores):
        return {"threshold": None, "precision": 0.0, "recall": 0.0, "accepted": 0, "true_positives": 0}
    cumulative_tp = np.cumsum(outcomes)
    accepted = np.arange(1, len(scores) + 1)
    precision = cumulative_tp / accepted
    valid = np.flatnonzero(precision >= target_precision)
    if not len(valid):
        return {"threshold": None, "precision": 0.0, "recall": 0.0, "accepted": 0, "true_positives": 0}
    best = int(valid[np.argmax(cumulative_tp[valid])])
    return {
        "threshold": float(scores[best]),
        "precision": float(precision[best]),
        "recall": float(cumulative_tp[best] / max(total_ground_truth, 1)),
        "accepted": best + 1,
        "true_positives": int(cumulative_tp[best]),
    }


def threshold_for_best_f1(
    scores: np.ndarray,
    outcomes: np.ndarray,
    total_ground_truth: int,
) -> dict[str, float | int | None]:
    if not len(scores):
        return {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accepted": 0, "true_positives": 0}
    cumulative_tp = np.cumsum(outcomes)
    accepted = np.arange(1, len(scores) + 1)
    precision = cumulative_tp / accepted
    recall = cumulative_tp / max(total_ground_truth, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return {
        "threshold": float(scores[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
        "accepted": best + 1,
        "true_positives": int(cumulative_tp[best]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and calibrate pseudo-label quality on sequence-held-out GT")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument(
        "--score-field",
        choices=("quality", "class_quality", "center_quality", "size_quality", "yaw_quality", "joint_center", "joint_box"),
        default="quality",
    )
    parser.add_argument("--maximum-range", type=float, default=70.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = TJ4DRadSet(args.dataset_root)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions: dict[str, list[ScoredObject]] = defaultdict(list)
    ground_truth: dict[str, dict[str, list[KittiObject]]] = {category: {} for category in DEFAULT_CLASSES}
    prediction_dir, metadata_dir = Path(args.prediction_dir), Path(args.metadata_dir)
    for frame_id in frame_ids:
        calibration = dataset.load_calibration(frame_id)
        frame_predictions = read_objects(prediction_dir / f"{frame_id}.txt")
        records = json.loads((metadata_dir / f"{frame_id}.json").read_text(encoding="utf-8"))
        scores = []
        for record in records[:len(frame_predictions)]:
            legacy = float(record.get("quality", 1.0))
            class_quality = float(record.get("class_quality", legacy))
            center_quality = float(record.get("center_quality", legacy))
            size_quality = float(record.get("size_quality", legacy))
            yaw_quality = float(record.get("yaw_quality", legacy))
            if args.score_field == "joint_center":
                score = float(np.sqrt(class_quality * center_quality))
            elif args.score_field == "joint_box":
                score = float((class_quality * center_quality * size_quality * max(yaw_quality, 1e-6)) ** 0.25)
            else:
                score = float(record.get(args.score_field, legacy))
            scores.append(score)
        scores += [1.0] * max(0, len(frame_predictions) - len(scores))
        frame_ground_truth = [obj for obj in dataset.load_labels(frame_id) if radar_range(obj, calibration) <= args.maximum_range]
        for category in DEFAULT_CLASSES:
            ground_truth[category][frame_id] = [obj for obj in frame_ground_truth if obj.category == category]
        for obj, score in zip(frame_predictions, scores, strict=True):
            if obj.category in DEFAULT_CLASSES and radar_range(obj, calibration) <= args.maximum_range:
                predictions[obj.category].append(ScoredObject(frame_id, obj, score))

    result: dict[str, object] = {
        "frames": len(frame_ids),
        "target_precision": args.target_precision,
        "score_field": args.score_field,
        "split_file": args.split_file,
        "per_class": {},
    }
    for category in DEFAULT_CLASSES:
        total_gt = sum(len(items) for items in ground_truth[category].values())
        criteria = {
            "center_2m": (center_distance_bev, 2.0),
            "bev_standard_iou": (bev_iou, DEFAULT_IOU[category]),
            "3d_standard_iou": (iou_3d, DEFAULT_IOU[category]),
        }
        category_result = {"ground_truth": total_gt, "predictions": len(predictions[category]), "criteria": {}}
        for name, (function, threshold) in criteria.items():
            scores, outcomes = ranked_outcomes(predictions[category], ground_truth[category], function, threshold)
            category_result["criteria"][name] = {
                "match_threshold": threshold,
                "calibration": calibration_statistics(scores, outcomes),
                "recommended_operating_point": threshold_for_precision(
                    scores, outcomes, total_gt, args.target_precision
                ),
                "best_f1_operating_point": threshold_for_best_f1(scores, outcomes, total_gt),
            }
        result["per_class"][category] = category_result
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
