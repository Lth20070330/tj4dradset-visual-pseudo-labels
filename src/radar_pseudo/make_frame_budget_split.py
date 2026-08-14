from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .batch_generate import atomic_write_text


PRIMARY_CLASSES = ("Car", "Truck", "Pedestrian", "Cyclist")


def count_labels(path: Path) -> Counter[str]:
    return Counter(
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split()[0] in PRIMARY_CLASSES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select complete GT frames under an object-annotation budget")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--target-objects", type=int, required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    label_dir = Path(args.dataset_root) / "training" / "label_2"
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(frame_ids)
    counts_by_frame = {frame_id: count_labels(label_dir / f"{frame_id}.txt") for frame_id in frame_ids}

    selected: list[str] = []
    counts: Counter[str] = Counter()
    total = 0
    for frame_id in frame_ids:
        frame_total = sum(counts_by_frame[frame_id].values())
        if total + frame_total <= args.target_objects:
            selected.append(frame_id)
            counts.update(counts_by_frame[frame_id])
            total += frame_total

    # Add one final frame only if it makes the annotation budget closer.
    remaining = [frame_id for frame_id in frame_ids if frame_id not in set(selected)]
    if remaining:
        closest = min(remaining, key=lambda frame_id: abs(total + sum(counts_by_frame[frame_id].values()) - args.target_objects))
        closest_total = sum(counts_by_frame[closest].values())
        if abs(total + closest_total - args.target_objects) < abs(total - args.target_objects):
            selected.append(closest)
            counts.update(counts_by_frame[closest])
            total += closest_total

    selected.sort()
    atomic_write_text(Path(args.output_split), "\n".join(selected))
    manifest = {
        "strategy": "complete_frames_random_budget",
        "seed": args.seed,
        "target_objects": args.target_objects,
        "selected_objects": total,
        "selected_frames": len(selected),
        "class_counts": dict(counts),
        "split_file": args.output_split,
        "label_dir": str(label_dir),
    }
    atomic_write_text(Path(args.output_manifest), json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
