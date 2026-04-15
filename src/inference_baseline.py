from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from config import get_default_config
from dataset import load_gli_splits
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean GLI-only binary inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/clean_gli_binary_baseline")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    return parser.parse_args()


def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + 1e-6) / (den + 1e-6))


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg.patch_size = tuple(args.patch_size)

    train_cases, val_cases = load_gli_splits(cfg.repo_root, cfg.data_root, cfg.train_list, cfg.val_list)
    cases = train_cases if args.split == "train" else val_cases
    cases = cases[: args.max_cases]

    if not cases:
        raise RuntimeError("No cases available for inference.")

    ds = Dataset(
        data=cases,
        transform=build_transforms(
            modalities=cfg.modalities,
            patch_size=cfg.patch_size,
            min_fg_ratio=cfg.min_fg_ratio,
            max_tries=cfg.max_sample_tries,
            margin=cfg.tumor_margin,
            training=False,
        ),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=4, out_channels=2).to(device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (cfg.repo_root / checkpoint_path).resolve()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (cfg.repo_root / out_dir).resolve()
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in loader:
            case_id = str(batch["case_id"][0])
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = sliding_window_inference(image, tuple(cfg.patch_size), 1, model, overlap=0.25)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1, keepdim=True)

            # Save binary label map (0 or 1)
            np.save(pred_dir / f"{case_id}_pred.npy", pred[0, 0].detach().cpu().numpy().astype(np.uint8))

            dice = dice_binary(pred.float(), label.float())
            rows.append({"case_id": case_id, "dice": dice})

    summary = {
        "num_cases": len(rows),
        "mean_dice": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        "cases": rows,
    }
    with (out_dir / "inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Predictions saved to: {pred_dir}")
    print(f"Mean Dice: {summary['mean_dice']:.4f}")


if __name__ == "__main__":
    main()
