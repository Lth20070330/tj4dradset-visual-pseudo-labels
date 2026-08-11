from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .batch_generate import atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic GT subset with pseudo-label class counts")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--pseudo-label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    gt_dir = root / "training" / "label_2"
    pseudo_dir = Path(args.pseudo_label_dir)
    output_dir = Path(args.output_dir)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    pseudo_counts = Counter()
    for frame_id in frame_ids:
        for line in (pseudo_dir / f"{frame_id}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip(): pseudo_counts[line.split()[0]] += 1
    candidates = defaultdict(list)
    for frame_id in frame_ids:
        for line_index, line in enumerate((gt_dir / f"{frame_id}.txt").read_text(encoding="utf-8").splitlines()):
            if line.strip(): candidates[line.split()[0]].append((frame_id, line_index, line))
    rng = random.Random(args.seed)
    selected_by_frame = defaultdict(list)
    for category, target_count in pseudo_counts.items():
        choices = candidates[category]
        if target_count > len(choices):
            raise ValueError(f"Pseudo count for {category} exceeds GT: {target_count} > {len(choices)}")
        for frame_id, line_index, line in rng.sample(choices, target_count):
            selected_by_frame[frame_id].append((line_index, line))
    for frame_id in frame_ids:
        lines = [line for _, line in sorted(selected_by_frame[frame_id])]
        atomic_write_text(output_dir / f"{frame_id}.txt", "\n".join(lines))
    manifest = {"seed": args.seed, "frames": len(frame_ids), "class_counts": dict(pseudo_counts), "objects": sum(pseudo_counts.values())}
    atomic_write_text(output_dir.parent / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
