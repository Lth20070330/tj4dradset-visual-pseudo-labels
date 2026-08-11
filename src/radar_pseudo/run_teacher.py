from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import TJ4DRadSet
from .vision import DepthAnythingMetric, YoloSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2D segmentation and metric-depth teachers")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frame-id", default="000000")
    parser.add_argument("--output-dir", default="outputs/teacher")
    parser.add_argument("--segmenter", default="yolo26s-seg.pt")
    parser.add_argument("--depth-repository", default="third_party/Depth-Anything-V2")
    parser.add_argument("--depth-checkpoint", default="models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth")
    args = parser.parse_args()

    dataset = TJ4DRadSet(args.dataset_root)
    image = dataset.load_image(args.frame_id)
    segmenter = YoloSegmenter(args.segmenter)
    depth_model = DepthAnythingMetric(args.depth_repository, args.depth_checkpoint)
    instances = segmenter.predict(image)
    depth = depth_model.predict(image)

    output_dir = Path(args.output_dir) / args.frame_id
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "depth_m.npy", depth)
    depth_color = cv2.applyColorMap(np.uint8(255 * np.clip(depth / 80.0, 0, 1)), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_dir / "depth_m.png"), depth_color)
    overlay = image.copy()
    records = []
    for index, instance in enumerate(instances):
        color = np.array([30, 220, 80], dtype=np.uint8)
        overlay[instance.mask] = (0.55 * overlay[instance.mask] + 0.45 * color).astype(np.uint8)
        x1, y1, x2, y2 = np.rint(instance.bbox_xyxy).astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(x) for x in color), 2)
        cv2.putText(overlay, f"{instance.target_class} {instance.confidence:.2f}", (x1, max(20, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(int(x) for x in color), 2)
        mask_path = output_dir / f"mask_{index:03d}.png"
        cv2.imwrite(str(mask_path), np.uint8(instance.mask) * 255)
        values = depth[instance.mask]
        records.append({
            "index": index,
            "source_class": instance.source_class,
            "target_class": instance.target_class,
            "confidence": instance.confidence,
            "bbox_xyxy": instance.bbox_xyxy.tolist(),
            "depth_median_m": float(np.median(values)) if len(values) else None,
            "mask_pixels": int(instance.mask.sum()),
        })
    cv2.imwrite(str(output_dir / "instances.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    (output_dir / "instances.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frame_id": args.frame_id, "instances": records, "depth_min_m": float(depth.min()), "depth_median_m": float(np.median(depth)), "depth_max_m": float(depth.max())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
