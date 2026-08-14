from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .batch_generate import atomic_write_text, to_kitti_line
from .dataset import Calibration, KittiObject
from .evaluate_teacher import box_iou_2d
from .geometry import kitti_box_corners_camera, project_camera_points
from .pseudo_label import CLASS_DIMENSIONS_HWL, PseudoBox3D


def reprojection_geometry(obj: KittiObject, calibration: Calibration) -> tuple[np.ndarray, float, float]:
    prior = CLASS_DIMENSIONS_HWL[obj.category].copy()
    pixel_height = max(1.0, obj.bbox_2d[3] - obj.bbox_2d[1])
    estimated_height = pixel_height * max(obj.location_camera[2], 1.0) / calibration.p2[1, 1]
    height = float(np.clip(0.55 * prior[0] + 0.45 * estimated_height, 0.75 * prior[0], 1.30 * prior[0]))
    candidates: list[tuple[float, float, np.ndarray]] = []
    for width_scale in (0.9, 1.0, 1.1):
        for length_scale in (0.9, 1.0, 1.1):
            dimensions = np.array([height, prior[1] * width_scale, prior[2] * length_scale], dtype=np.float64)
            prior_penalty = 0.025 * ((width_scale - 1.0) ** 2 + (length_scale - 1.0) ** 2)
            for yaw in np.linspace(-np.pi / 2, np.pi / 2, 72, endpoint=False):
                candidate = KittiObject(
                    obj.category, obj.truncated, obj.occluded, obj.alpha, obj.bbox_2d,
                    dimensions, obj.location_camera, float(yaw),
                )
                pixels, depth = project_camera_points(kitti_box_corners_camera(candidate), calibration.p2)
                if np.any(depth <= 0) or not np.all(np.isfinite(pixels)):
                    continue
                projected = np.array(
                    [pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(), pixels[:, 1].max()], dtype=np.float64
                )
                road_prior = 0.012 * abs(float(yaw + np.pi / 2)) / np.pi
                score = box_iou_2d(projected, obj.bbox_2d) - prior_penalty - road_prior
                candidates.append((float(score), float(yaw), dimensions.copy()))
    if not candidates:
        return prior, -np.pi / 2, 0.0
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    separated = [candidate for candidate in candidates[1:] if abs(candidate[1] - best[1]) > np.deg2rad(10)]
    second_score = separated[0][0] if separated else candidates[1][0]
    confidence = float(np.clip((best[0] - second_score) / 0.08, 0.0, 1.0))
    return best[2], best[1], confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Create geometry ablations from saved MGU pseudo labels")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "class_prior",
            "reprojection",
            "conditional_reprojection",
            "selective_reprojection",
            "selective_reprojection_direct",
        ),
        default="class_prior",
    )
    parser.add_argument("--dataset-root", help="Required for reprojection geometry")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument(
        "--reprojection-classes",
        nargs="+",
        default=["Truck"],
        help="Classes allowed to retain reprojection geometry in selective_reprojection mode",
    )
    args = parser.parse_args()
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    input_root, output_root = Path(args.input_root), Path(args.output_root)
    if args.mode in ("reprojection", "selective_reprojection_direct") and not args.dataset_root:
        raise ValueError(f"--dataset-root is required for {args.mode} mode")
    calibration_dir = Path(args.dataset_root) / "training" / "calib" if args.dataset_root else None
    for frame_id in frame_ids:
        label_path = input_root / "label_2" / f"{frame_id}.txt"
        metadata_path = input_root / "metadata" / f"{frame_id}.json"
        objects = [KittiObject.from_line(line) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_calibration = (
            Calibration.from_file(calibration_dir / f"{frame_id}.txt")
            if args.mode in ("reprojection", "selective_reprojection_direct")
            else None
        )
        transformed = []
        transformed_metadata = []
        for obj, record in zip(objects, metadata, strict=True):
            retain_saved_reprojection = (
                args.mode == "conditional_reprojection"
                and float(record.get("geometry_confidence", 0.0)) >= args.confidence_threshold
            ) or (
                args.mode == "selective_reprojection"
                and obj.category in args.reprojection_classes
                and float(record.get("geometry_confidence", 0.0)) >= args.confidence_threshold
            )
            direct_reprojection_candidate = (
                args.mode == "selective_reprojection_direct" and obj.category in args.reprojection_classes
            )
            if args.mode == "reprojection" or direct_reprojection_candidate:
                dimensions, rotation_y, reprojection_confidence = reprojection_geometry(obj, frame_calibration)
                if direct_reprojection_candidate and reprojection_confidence < args.confidence_threshold:
                    dimensions = CLASS_DIMENSIONS_HWL[obj.category].copy()
                    rotation_y = -np.pi / 2
                    reprojection_confidence = 0.0
            elif retain_saved_reprojection:
                dimensions = obj.dimensions_hwl.copy()
                rotation_y = obj.rotation_y
                reprojection_confidence = float(record.get("geometry_confidence", 0.0))
            else:
                dimensions = CLASS_DIMENSIONS_HWL[obj.category].copy()
                rotation_y = -np.pi / 2
                reprojection_confidence = 0.0
            box = PseudoBox3D(
                category=obj.category,
                dimensions_hwl=dimensions,
                location_camera=obj.location_camera,
                rotation_y=rotation_y,
                visual_confidence=float(record["visual_confidence"]),
                visual_depth_m=float(record["visual_depth_m"]),
                radar_depth_m=record.get("radar_depth_m"),
                radar_points=int(record["radar_points"]),
                quality=float(record["quality"]),
                position_source=(
                    str(record["position_source"]) + "_reprojection_geometry"
                    if reprojection_confidence > 0
                    else str(record["position_source"]) + "_prior_geometry"
                ),
                bbox_2d=obj.bbox_2d,
                class_quality=float(record.get("class_quality", record["visual_confidence"])),
                center_quality=float(record.get("center_quality", record["quality"])),
                size_quality=(
                    float(np.sqrt(float(record.get("center_quality", record["quality"])) * reprojection_confidence))
                    if args.mode == "reprojection" or retain_saved_reprojection or reprojection_confidence > 0
                    else float(0.55 * float(record.get("center_quality", record["quality"])))
                ),
                yaw_quality=(
                    float(np.sqrt(float(record.get("center_quality", record["quality"])) * reprojection_confidence))
                    if args.mode == "reprojection" or retain_saved_reprojection or reprojection_confidence > 0
                    else float(0.25 * float(record.get("center_quality", record["quality"])))
                ),
                cluster_candidates=int(record.get("cluster_candidates", 0)),
                cluster_score=float(record.get("cluster_score", 0.0)),
                candidate_margin=float(record.get("candidate_margin", 0.0)),
                geometry_confidence=(
                    reprojection_confidence
                    if args.mode == "reprojection" or retain_saved_reprojection or reprojection_confidence > 0
                    else 0.0
                ),
            )
            transformed.append(to_kitti_line(box))
            transformed_metadata.append(box.to_dict())
        atomic_write_text(output_root / "label_2" / f"{frame_id}.txt", "\n".join(transformed))
        atomic_write_text(
            output_root / "metadata" / f"{frame_id}.json",
            json.dumps(transformed_metadata, ensure_ascii=False, indent=2),
        )
    atomic_write_text(
        output_root / "manifest.json",
        json.dumps(
            {
                "source": args.input_root,
                "mode": args.mode,
                "confidence_threshold": args.confidence_threshold,
                "reprojection_classes": args.reprojection_classes,
                "frames": len(frame_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
