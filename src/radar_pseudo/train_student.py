from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .student import RadarBEVDetector, RadarDetectionDataset, detection_loss


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the common radar BEV student on GT or pseudo labels")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-dir")
    parser.add_argument("--metadata-dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    seed_everything(args.seed)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_frames: frame_ids = frame_ids[:args.max_frames]
    dataset = RadarDetectionDataset(args.dataset_root, frame_ids, args.label_dir, args.metadata_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RadarBEVDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; parts = {"heatmap": 0.0, "regression": 0.0}
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            for key in ("features", "heatmap", "regression", "weight"):
                batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_parts = detection_loss(model(batch["features"]), batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            running += float(loss.detach()); parts["heatmap"] += loss_parts["heatmap"]; parts["regression"] += loss_parts["regression"]
            progress.set_postfix(loss=f"{float(loss.detach()):.3f}")
        scheduler.step(); batches=max(1,len(loader))
        record={"epoch":epoch,"loss":running/batches,"heatmap_loss":parts['heatmap']/batches,"regression_loss":parts['regression']/batches,"learning_rate":scheduler.get_last_lr()[0]}
        history.append(record)
        (output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        torch.save({"model":model.state_dict(),"epoch":epoch,"args":vars(args)},output_dir/"last.pt")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
