from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from config import get_default_config
from dataset import load_gli_splits
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI binary baseline trainer")
    parser.add_argument("--overfit-cases", type=int, default=0, help="If > 0, use only this many cases for debugging")
    parser.add_argument("--overfit-epochs", type=int, default=60, help="Epochs for overfit mode")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=125, help="Epochs for full training")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio used only when val list is empty")
    parser.add_argument("--seed", type=int, default=42)
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

    set_determinism(seed=cfg.seed)

    train_cases, val_cases = load_gli_splits(cfg.repo_root, cfg.data_root, cfg.train_list, cfg.val_list)
    
    # Handle overfit mode (small training set for debugging)
    if args.overfit_cases > 0:
        train_cases = train_cases[: args.overfit_cases]
        val_cases = train_cases.copy()
        num_epochs = args.overfit_epochs
        mode_str = f"OVERFIT ({args.overfit_cases} cases, {args.overfit_epochs} epochs)"
    else:
        # Full training on all available cases
        if len(val_cases) == 0:
            if len(train_cases) < 2:
                raise RuntimeError("Need at least 2 training cases to create a validation split.")
            rng = np.random.default_rng(cfg.seed)
            perm = rng.permutation(len(train_cases))
            val_count = max(1, int(len(train_cases) * args.val_ratio))
            val_idx = set(perm[:val_count].tolist())
            split_train = []
            split_val = []
            for i, case in enumerate(train_cases):
                if i in val_idx:
                    split_val.append(case)
                else:
                    split_train.append(case)
            train_cases = split_train
            val_cases = split_val
            print(f"Val list empty: created split with {len(train_cases)} train / {len(val_cases)} val cases")

        num_epochs = args.epochs
        mode_str = f"FULL TRAINING ({len(train_cases)} cases, {args.epochs} epochs)"

    print(f"Mode: {mode_str}")
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")
    print(f"Modalities: {cfg.modalities} (4 channels)")
    print(f"Patch size: {cfg.patch_size}")
    print(f"LR: {cfg.learning_rate} | Weight decay: {cfg.weight_decay}")
    print("Label setup: binary tumor vs background")
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

    for epoch in range(1, num_epochs + 1):
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

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        history.append({"epoch": epoch, "loss": epoch_loss, "dice": mean_dice})

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")

        print(f"Epoch {epoch:03d}/{num_epochs} | loss={epoch_loss:.4f} | dice={mean_dice:.4f}")

    torch.save(model.state_dict(), cfg.checkpoint_dir / "last.pt")
    with (cfg.checkpoint_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest Dice: {best_dice:.4f}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
