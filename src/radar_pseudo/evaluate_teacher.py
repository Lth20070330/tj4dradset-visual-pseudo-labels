from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .dataset import KittiObject, TJ4DRadSet
from .pseudo_label import PseudoBox3D, backproject_pixel, estimate_pseudo_box
from .vision import DepthAnythingMetric, YoloSegmenter


def box_iou_2d(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_matches(predictions: list[PseudoBox3D], ground_truth: list[KittiObject], threshold: float = 0.3) -> list[tuple[int, int, float]]:
    candidates = sorted(
        ((pi, gi, box_iou_2d(pred.bbox_2d, gt.bbox_2d)) for pi, pred in enumerate(predictions) for gi, gt in enumerate(ground_truth)),
        key=lambda item: item[2],
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches = []
    for pi, gi, iou in candidates:
        if iou < threshold:
            break
        if pi not in used_predictions and gi not in used_ground_truth:
            used_predictions.add(pi)
            used_ground_truth.add(gi)
            matches.append((pi, gi, iou))
    return matches


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def evaluate(
    dataset: TJ4DRadSet,
    frame_ids: list[str],
    segmenter: YoloSegmenter,
    depth_model: DepthAnythingMetric,
    confidence: float,
) -> dict[str, object]:
    gt_count = prediction_count = match_count = correct_class = radar_corrected = 0
    quality_thresholds = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    quality_stats = {threshold: {"predictions": 0, "matches": 0, "correct_class": 0} for threshold in quality_thresholds}
    visual_errors: list[float] = []
    final_errors: list[float] = []
    frame_records = []
    for frame_id in tqdm(frame_ids, desc="teacher evaluation"):
        image = dataset.load_image(frame_id)
        calibration = dataset.load_calibration(frame_id)
        radar = dataset.load_radar(frame_id)
        ground_truth = [obj for obj in dataset.load_labels(frame_id) if obj.category != "DontCare"]
        instances = segmenter.predict(image, confidence=confidence)
        depth = depth_model.predict(image)
        predictions = [estimate_pseudo_box(instance, depth, radar, calibration) for instance in instances]
        matches = greedy_matches(predictions, ground_truth)
        gt_count += len(ground_truth)
        prediction_count += len(predictions)
        match_count += len(matches)
        frame_correct = 0
        for prediction_index, gt_index, _ in matches:
            prediction = predictions[prediction_index]
            gt = ground_truth[gt_index]
            if prediction.category == gt.category:
                correct_class += 1
                frame_correct += 1
            x1, _, x2, y2 = prediction.bbox_2d
            visual_location = backproject_pixel((x1 + x2) / 2, y2, prediction.visual_depth_m, calibration.p2)
            visual_errors.append(float(np.linalg.norm(visual_location - gt.location_camera)))
            final_errors.append(float(np.linalg.norm(prediction.location_camera - gt.location_camera)))
            radar_corrected += int(prediction.position_source == "radar_corrected")
        frame_records.append({"frame_id": frame_id, "gt": len(ground_truth), "predictions": len(predictions), "matches": len(matches), "correct_class": frame_correct})
        for threshold in quality_thresholds:
            filtered = [prediction for prediction in predictions if prediction.quality >= threshold]
            filtered_matches = greedy_matches(filtered, ground_truth)
            quality_stats[threshold]["predictions"] += len(filtered)
            quality_stats[threshold]["matches"] += len(filtered_matches)
            quality_stats[threshold]["correct_class"] += sum(
                filtered[prediction_index].category == ground_truth[gt_index].category
                for prediction_index, gt_index, _ in filtered_matches
            )
    precision = match_count / prediction_count if prediction_count else 0.0
    recall = match_count / gt_count if gt_count else 0.0
    quality_curve = {}
    for threshold, stats in quality_stats.items():
        predictions_at_threshold = stats["predictions"]
        matches_at_threshold = stats["matches"]
        quality_curve[str(threshold)] = {
            **stats,
            "precision": matches_at_threshold / predictions_at_threshold if predictions_at_threshold else 0.0,
            "recall": matches_at_threshold / gt_count if gt_count else 0.0,
            "class_accuracy": stats["correct_class"] / matches_at_threshold if matches_at_threshold else 0.0,
        }
    return {
        "frames": len(frame_ids),
        "ground_truth_objects": gt_count,
        "predictions": prediction_count,
        "matches_iou_0.3": match_count,
        "localization_precision": precision,
        "localization_recall": recall,
        "class_accuracy_on_matches": correct_class / match_count if match_count else 0.0,
        "radar_corrected_matches": radar_corrected,
        "visual_center_error_m": {"median": percentile(visual_errors, 50), "p90": percentile(visual_errors, 90)},
        "final_center_error_m": {"median": percentile(final_errors, 50), "p90": percentile(final_errors, 90)},
        "quality_curve": quality_curve,
        "frame_records": frame_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the visual pseudo-label teacher on a frame subset")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--output", default="outputs/evaluation/teacher_subset.json")
    parser.add_argument("--segmenter", default="yolo26s-seg.pt")
    parser.add_argument("--depth-repository", default="third_party/Depth-Anything-V2")
    parser.add_argument("--depth-checkpoint", default="models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth")
    args = parser.parse_args()
    dataset = TJ4DRadSet(args.dataset_root)
    if args.split_file:
        all_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        all_ids = dataset.frame_ids
    indices = np.linspace(0, len(all_ids) - 1, min(args.max_frames, len(all_ids)), dtype=int)
    frame_ids = [all_ids[index] for index in indices]
    result = evaluate(
        dataset,
        frame_ids,
        YoloSegmenter(args.segmenter, image_size=args.image_size),
        DepthAnythingMetric(args.depth_repository, args.depth_checkpoint),
        args.confidence,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "frame_records"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
