from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from config import get_default_config
from dataset import load_gli_train_val_test_strict
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI baseline NEW (strict train/val/test splits)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gli_4mod_binary_new")
    parser.add_argument("--results-dir", type=str, default="results/gli_4mod_binary_new")
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
    cfg.batch_size = args.batch_size
    cfg.patch_size = tuple(args.patch_size)
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.seed = args.seed
    cfg.checkpoint_dir = (cfg.repo_root / Path(args.checkpoint_dir)).resolve()
    cfg.results_dir = (cfg.repo_root / Path(args.results_dir)).resolve()

    train_list = cfg.repo_root / "patient_lists/gli_train.txt"
    val_list = cfg.repo_root / "patient_lists/gli_val.txt"
    test_list = cfg.repo_root / "patient_lists/gli_test.txt"

    set_determinism(seed=cfg.seed)

    train_cases, val_cases, test_cases = load_gli_train_val_test_strict(
        data_root=cfg.data_root,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
    )

    if not train_cases:
        raise RuntimeError("No train cases loaded from gli_train.txt")
    if not val_cases:
        raise RuntimeError("No val cases loaded from gli_val.txt")

    print("Mode: FULL TRAINING NEW")
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)} | Test cases: {len(test_cases)}")
    print(f"Modalities: {cfg.modalities} (4 channels)")
    print(f"Patch size: {cfg.patch_size}")
    print(f"LR: {cfg.learning_rate} | Weight decay: {cfg.weight_decay}")
    print(f"Epochs: {args.epochs} | Val interval: {args.val_interval}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Results dir: {cfg.results_dir}")
    print()

    train_ds = Dataset(
        data=train_cases,
        transform=build_transforms(
            modalities=cfg.modalities,
            patch_size=cfg.patch_size,
            min_fg_ratio=cfg.min_fg_ratio,
            max_tries=cfg.max_sample_tries,
            margin=cfg.tumor_margin,
            training=True,
        ),
    )
    val_ds = Dataset(
        data=val_cases,
        transform=build_transforms(
            modalities=cfg.modalities,
            patch_size=cfg.patch_size,
            min_fg_ratio=cfg.min_fg_ratio,
            max_tries=cfg.max_sample_tries,
            margin=cfg.tumor_margin,
            training=False,
        ),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=4, out_channels=2).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_dice = -1.0
    last_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            steps += 1
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, label.long())
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(steps, 1)

        run_val = (epoch % max(1, args.val_interval) == 0) or (epoch == args.epochs)
        if run_val:
            model.eval()
            dice_scores = []
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)
                    logits = sliding_window_inference(image, cfg.patch_size, 1, model, overlap=0.25)
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1, keepdim=True)
                    dice_scores.append(dice_binary(pred.float(), label.float()))

            last_val_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
            if last_val_dice > best_dice:
                best_dice = last_val_dice
                torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")
            print(f"Epoch {epoch:03d}/{args.epochs} | loss={epoch_loss:.4f} | val_dice={last_val_dice:.4f}")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs} | loss={epoch_loss:.4f} | val_dice=skip")

        history.append({"epoch": epoch, "loss": epoch_loss, "dice": last_val_dice, "validated": run_val})

    torch.save(model.state_dict(), cfg.checkpoint_dir / "last.pt")
    with (cfg.checkpoint_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    split_meta = {
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
        "train_list": str(train_list),
        "val_list": str(val_list),
        "test_list": str(test_list),
    }
    with (cfg.results_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"\nBest Dice: {best_dice:.4f}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
