from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "center_2m_map": ("distance_2m", "mAP"),
    "center_4m_map": ("distance_4m", "mAP"),
    "bev_map_r40": ("standard_70m", "mAP_BEV_R40"),
    "3d_map_r40": ("standard_70m", "mAP_3D_R40"),
}
CLASSES = ("Car", "Truck", "Pedestrian", "Cyclist")


def statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": len(values),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed student evaluation JSON files")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_per_class: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    seeds: dict[str, list[int]] = defaultdict(list)
    for path in sorted(Path(args.input_dir).glob("*_seed*.json")):
        match = re.fullmatch(r"(.+)_seed(\d+)", path.stem)
        if not match:
            continue
        variant, seed_text = match.groups()
        record = json.loads(path.read_text(encoding="utf-8"))
        if not all(section in record for section, _ in METRICS.values()):
            continue
        seeds[variant].append(int(seed_text))
        for metric, (section, key) in METRICS.items():
            grouped[variant][metric].append(float(record[section][key]))
        for category in CLASSES:
            grouped_per_class[variant][category]["center_2m_ap"].append(
                float(record["distance_2m"]["per_class"][category]["ap"])
            )
            grouped_per_class[variant][category]["center_4m_ap"].append(
                float(record["distance_4m"]["per_class"][category]["ap"])
            )
            grouped_per_class[variant][category]["bev_ap_r40"].append(
                float(record["standard_70m"]["per_class"][category]["bev"]["ap_r40"])
            )
            grouped_per_class[variant][category]["3d_ap_r40"].append(
                float(record["standard_70m"]["per_class"][category]["3d"]["ap_r40"])
            )

    summary = {
        variant: {
            "seeds": sorted(seeds[variant]),
            "metrics": {metric: statistics(values) for metric, values in metrics.items()},
            "per_class": {
                category: {metric: statistics(values) for metric, values in category_metrics.items()}
                for category, category_metrics in grouped_per_class[variant].items()
            },
        }
        for variant, metrics in sorted(grouped.items())
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 最终学生多种子结果",
        "",
        "数值为均值 ± 样本标准差。",
        "",
        "| 训练标签 | 2 m 中心 mAP | 4 m 中心 mAP | BEV mAP R40 | 3D mAP R40 | 种子 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    labels = {"b0": "B0 单簇基线", "mgu": "MGU-PL", "gt_equal": "对象数等量真值", "gt_complete": "完整帧预算真值", "gt_full": "train_core 全部真值"}
    for variant in ("b0", "mgu", "gt_equal", "gt_complete", "gt_full"):
        if variant not in summary:
            continue
        metrics = summary[variant]["metrics"]
        cells = []
        for name in METRICS:
            item = metrics[name]
            cells.append(f"{100 * item['mean']:.2f} ± {100 * item['std']:.2f}")
        lines.append(f"| {labels.get(variant, variant)} | {' | '.join(cells)} | {', '.join(map(str, summary[variant]['seeds']))} |")
    markdown = output.with_suffix(".md")
    if "b0" in summary and "mgu" in summary:
        lines.extend(
            [
                "",
                "## B0 与 MGU-PL 逐类别",
                "",
                "| 类别 | 方法 | 2 m 中心 AP | 4 m 中心 AP | BEV AP R40 | 3D AP R40 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for category in CLASSES:
            for variant, label in (("b0", "B0"), ("mgu", "MGU-PL")):
                values = summary[variant]["per_class"][category]
                cells = [
                    f"{100 * values[name]['mean']:.2f} ± {100 * values[name]['std']:.2f}"
                    for name in ("center_2m_ap", "center_4m_ap", "bev_ap_r40", "3d_ap_r40")
                ]
                lines.append(f"| {category} | {label} | {' | '.join(cells)} |")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
