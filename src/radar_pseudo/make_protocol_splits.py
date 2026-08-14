from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .batch_generate import atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sequence-disjoint train-core and teacher-calibration splits")
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", help="If set, stratify development sequences by four-class GT distribution")
    parser.add_argument("--development-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame_ids = [line.strip() for line in Path(args.source_split).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = sorted({frame_id[:2] for frame_id in frame_ids})
    development_count = max(1, round(len(groups) * args.development_fraction))
    if args.dataset_root:
        classes = ("Car", "Truck", "Pedestrian", "Cyclist")
        label_dir = Path(args.dataset_root) / "training" / "label_2"
        group_vectors = {group: np.zeros(len(classes) + 1, dtype=np.float64) for group in groups}
        class_to_index = {name: index for index, name in enumerate(classes)}
        for frame_id in frame_ids:
            vector = group_vectors[frame_id[:2]]
            vector[-1] += 1
            for line in (label_dir / f"{frame_id}.txt").read_text(encoding="utf-8").splitlines():
                if line.strip() and line.split()[0] in class_to_index:
                    vector[class_to_index[line.split()[0]]] += 1
        total = sum(group_vectors.values(), start=np.zeros(len(classes) + 1, dtype=np.float64))
        target = total * args.development_fraction
        best_groups: tuple[str, ...] | None = None
        best_score = float("inf")
        combinations = math.comb(len(groups), development_count)
        candidates = itertools.combinations(groups, development_count)
        if combinations > 2_000_000:
            rng = random.Random(args.seed)
            sampled = {tuple(sorted(rng.sample(groups, development_count))) for _ in range(200_000)}
            candidates = iter(sampled)
        for candidate in candidates:
            selected = sum((group_vectors[group] for group in candidate), start=np.zeros_like(total))
            relative_error = (selected - target) / np.maximum(target, 1.0)
            score = float(np.mean(relative_error**2))
            if score < best_score:
                best_score, best_groups = score, candidate
        development_groups = set(best_groups or ())
        stratification_score = best_score
    else:
        rng = random.Random(args.seed)
        development_groups = set(rng.sample(groups, development_count))
        stratification_score = None
    development = [frame_id for frame_id in frame_ids if frame_id[:2] in development_groups]
    train_core = [frame_id for frame_id in frame_ids if frame_id[:2] not in development_groups]
    output_root = Path(args.output_root)
    atomic_write_text(output_root / "train_core.txt", "\n".join(train_core))
    atomic_write_text(output_root / "teacher_calibration.txt", "\n".join(development))
    manifest = {
        "source_split": args.source_split,
        "seed": args.seed,
        "group_key": "first two frame-id digits (TJ4DRadSet sequence)",
        "groups": groups,
        "train_core_groups": sorted(set(groups) - development_groups),
        "teacher_calibration_groups": sorted(development_groups),
        "train_core_frames": len(train_core),
        "teacher_calibration_frames": len(development),
        "stratification_score": stratification_score,
        "overlap_frames": len(set(train_core) & set(development)),
        "source_group_counts": dict(Counter(frame_id[:2] for frame_id in frame_ids)),
    }
    atomic_write_text(output_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
