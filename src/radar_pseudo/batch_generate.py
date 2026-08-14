from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from tqdm import tqdm

from .dataset import TJ4DRadSet
from .pseudo_label import PseudoBox3D, estimate_pseudo_box_by_method
from .vision import DepthAnythingMetric, YoloSegmenter


def to_kitti_line(box: PseudoBox3D) -> str:
    x1, y1, x2, y2 = box.bbox_2d
    h, w, length = box.dimensions_hwl
    x, y, z = box.location_camera
    return (
        f"{box.category} 0 0 0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
        f"{h:.6f} {w:.6f} {length:.6f} {x:.6f} {y:.6f} {z:.6f} {box.rotation_y:.6f}"
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def generate_frame(
    dataset: TJ4DRadSet,
    frame_id: str,
    segmenter: YoloSegmenter,
    depth_model: DepthAnythingMetric,
    confidence: float,
    quality_threshold: float,
    method: str = "b0",
) -> list[PseudoBox3D]:
    image = dataset.load_image(frame_id)
    calibration = dataset.load_calibration(frame_id)
    radar = dataset.load_radar(frame_id)
    instances = segmenter.predict(image, confidence=confidence)
    depth = depth_model.predict(image)
    boxes = [estimate_pseudo_box_by_method(method, instance, depth, radar, calibration) for instance in instances]
    return [box for box in boxes if box.quality >= quality_threshold]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate TJ4DRadSet pseudo labels with resume support")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--quality-threshold", type=float, default=0.6)
    parser.add_argument("--method", choices=("b0", "mgu", "mgu_pca"), default="b0")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--segmenter", default="yolo26s-seg.pt")
    parser.add_argument("--depth-repository", default="third_party/Depth-Anything-V2")
    parser.add_argument("--depth-checkpoint", default="models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth")
    args = parser.parse_args()

    dataset = TJ4DRadSet(args.dataset_root)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]
    output_root = Path(args.output_root)
    label_dir = output_root / "label_2"
    metadata_dir = output_root / "metadata"
    pending = [
        frame_id for frame_id in frame_ids
        if args.overwrite or not ((label_dir / f"{frame_id}.txt").is_file() and (metadata_dir / f"{frame_id}.json").is_file())
    ]
    print(json.dumps({"frames": len(frame_ids), "pending": len(pending), "quality_threshold": args.quality_threshold}, ensure_ascii=False))
    if not pending:
        return
    segmenter = YoloSegmenter(args.segmenter, image_size=args.image_size)
    depth_model = DepthAnythingMetric(args.depth_repository, args.depth_checkpoint)
    failures = []
    for frame_id in tqdm(pending, desc="pseudo-label generation"):
        try:
            boxes = generate_frame(dataset, frame_id, segmenter, depth_model, args.confidence, args.quality_threshold, args.method)
            atomic_write_text(label_dir / f"{frame_id}.txt", "\n".join(to_kitti_line(box) for box in boxes))
            atomic_write_text(metadata_dir / f"{frame_id}.json", json.dumps([box.to_dict() for box in boxes], ensure_ascii=False, indent=2))
        except Exception as error:
            failures.append({"frame_id": frame_id, "error": repr(error)})
            atomic_write_text(output_root / "failures.json", json.dumps(failures, ensure_ascii=False, indent=2))
    manifest = {
        "dataset_root": args.dataset_root,
        "split_file": args.split_file,
        "frames": len(frame_ids),
        "quality_threshold": args.quality_threshold,
        "method": args.method,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "segmenter": args.segmenter,
        "depth_checkpoint": args.depth_checkpoint,
        "failures": failures,
    }
    atomic_write_text(output_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"completed": len(pending) - len(failures), "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
