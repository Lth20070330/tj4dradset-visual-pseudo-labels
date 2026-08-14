from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .batch_generate import atomic_write_text
from .dataset import KittiObject


FROZEN_CENTER_THRESHOLDS = {
    "Car": 0.6033604214562833,
    "Truck": 0.4092081090243417,
    "Pedestrian": 0.4486718960765075,
    "Cyclist": 0.36705212455404057,
}


def joint_center_score(record: dict[str, object]) -> float:
    legacy = float(record.get("quality", 1.0))
    class_quality = float(record.get("class_quality", legacy))
    center_quality = float(record.get("center_quality", legacy))
    return float(np.sqrt(max(class_quality, 0.0) * max(center_quality, 0.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter pseudo labels with frozen class-specific center-quality thresholds")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
        help="Override a frozen threshold; may be repeated",
    )
    args = parser.parse_args()
    thresholds = dict(FROZEN_CENTER_THRESHOLDS)
    for item in args.threshold:
        category, value = item.split("=", 1)
        thresholds[category] = float(value)

    input_root, output_root = Path(args.input_root), Path(args.output_root)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    input_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    for frame_id in frame_ids:
        label_path = input_root / "label_2" / f"{frame_id}.txt"
        metadata_path = input_root / "metadata" / f"{frame_id}.json"
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        objects = [KittiObject.from_line(line) for line in lines]
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        kept_lines: list[str] = []
        kept_records: list[dict[str, object]] = []
        for line, obj, record in zip(lines, objects, records, strict=True):
            input_counts[obj.category] += 1
            score = joint_center_score(record)
            if score < thresholds.get(obj.category, 1.0):
                continue
            record = dict(record)
            record["selection_score"] = score
            record["selection_threshold"] = thresholds[obj.category]
            kept_lines.append(line)
            kept_records.append(record)
            output_counts[obj.category] += 1
        atomic_write_text(output_root / "label_2" / f"{frame_id}.txt", "\n".join(kept_lines))
        atomic_write_text(
            output_root / "metadata" / f"{frame_id}.json",
            json.dumps(kept_records, ensure_ascii=False, indent=2),
        )
    manifest = {
        "source": str(input_root),
        "split_file": args.split_file,
        "frames": len(frame_ids),
        "score": "sqrt(class_quality * center_quality)",
        "thresholds": thresholds,
        "input_counts": dict(input_counts),
        "output_counts": dict(output_counts),
        "input_total": sum(input_counts.values()),
        "output_total": sum(output_counts.values()),
    }
    atomic_write_text(output_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
