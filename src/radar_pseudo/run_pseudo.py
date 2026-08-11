from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import TJ4DRadSet
from .pseudo_label import estimate_pseudo_box
from .vision import DepthAnythingMetric, YoloSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual/radar pseudo 3D boxes for one frame")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--output-dir", default="outputs/pseudo_labels")
    parser.add_argument("--segmenter", default="yolo26s-seg.pt")
    parser.add_argument("--depth-repository", default="third_party/Depth-Anything-V2")
    parser.add_argument("--depth-checkpoint", default="models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth")
    args = parser.parse_args()

    dataset = TJ4DRadSet(args.dataset_root)
    image = dataset.load_image(args.frame_id)
    calibration = dataset.load_calibration(args.frame_id)
    radar = dataset.load_radar(args.frame_id)
    instances = YoloSegmenter(args.segmenter).predict(image)
    depth = DepthAnythingMetric(args.depth_repository, args.depth_checkpoint).predict(image)
    boxes = [estimate_pseudo_box(instance, depth, radar, calibration) for instance in instances]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{args.frame_id}.json"
    target.write_text(json.dumps([box.to_dict() for box in boxes], ensure_ascii=False, indent=2), encoding="utf-8")
    print(target.resolve())
    print(json.dumps([box.to_dict() for box in boxes], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
