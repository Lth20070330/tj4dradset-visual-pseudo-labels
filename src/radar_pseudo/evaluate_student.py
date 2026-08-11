from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .student import (
    CLASSES,
    DetectionBEV,
    RadarBEVDetector,
    RadarDetectionDataset,
    decode_detections,
    load_label_file,
    object_to_radar_parameters,
)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def evaluate_at_distance(
    predictions: dict[str, list[DetectionBEV]],
    ground_truth: dict[str, list[tuple[str, np.ndarray]]],
    distance_threshold: float,
) -> dict[str, object]:
    per_class = {}
    for category in CLASSES:
        gt_by_frame = {frame: [center for name, center in objects if name == category] for frame, objects in ground_truth.items()}
        total_gt = sum(len(centers) for centers in gt_by_frame.values())
        ranked = sorted(
            ((prediction.score, frame, prediction) for frame, items in predictions.items() for prediction in items if prediction.category == category),
            reverse=True,
            key=lambda item: item[0],
        )
        matched = {frame: np.zeros(len(centers), dtype=bool) for frame, centers in gt_by_frame.items()}
        tp = np.zeros(len(ranked)); fp = np.zeros(len(ranked))
        for index, (_, frame, prediction) in enumerate(ranked):
            centers = gt_by_frame.get(frame, [])
            if not centers:
                fp[index] = 1; continue
            distances = np.array([np.linalg.norm(prediction.center_xyz[:2] - center[:2]) for center in centers])
            distances[matched[frame]] = np.inf
            nearest = int(np.argmin(distances))
            if distances[nearest] <= distance_threshold:
                tp[index] = 1; matched[frame][nearest] = True
            else:
                fp[index] = 1
        cumulative_tp = np.cumsum(tp); cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / max(total_gt, 1)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
        per_class[category] = {
            "ground_truth": total_gt,
            "predictions": len(ranked),
            "ap": average_precision(recall, precision) if total_gt else None,
            "precision": float(precision[-1]) if len(precision) else 0.0,
            "recall": float(recall[-1]) if len(recall) else 0.0,
        }
    valid_aps = [record["ap"] for record in per_class.values() if record["ap"] is not None]
    return {"mAP": float(np.mean(valid_aps)), "per_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a radar student on official GT validation labels")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_frames: frame_ids = frame_ids[:args.max_frames]
    dataset = RadarDetectionDataset(args.dataset_root, frame_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RadarBEVDetector().to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    predictions: dict[str, list[DetectionBEV]] = {}
    with torch.inference_mode():
        for batch in tqdm(loader, desc="student evaluation"):
            outputs = model(batch["features"].to(device, non_blocking=True))
            decoded = decode_detections(outputs, score_threshold=args.score_threshold)
            predictions.update(zip(batch["frame_id"], decoded, strict=True))
    ground_truth = {}
    for frame_id in frame_ids:
        calibration = dataset.dataset.load_calibration(frame_id)
        objects = load_label_file(dataset.dataset.label_dir / f"{frame_id}.txt")
        ground_truth[frame_id] = [(obj.category, object_to_radar_parameters(obj, calibration)[0]) for obj in objects if obj.category in CLASSES]
    result = {
        "checkpoint": args.checkpoint,
        "frames": len(frame_ids),
        "distance_2m": evaluate_at_distance(predictions, ground_truth, 2.0),
        "distance_4m": evaluate_at_distance(predictions, ground_truth, 4.0),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
