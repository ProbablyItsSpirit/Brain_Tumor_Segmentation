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
from dataset import load_gli_train_val_test_cases
from model import build_model
from transforms import build_multiclass_transforms


CLASS_NAMES = {1: "TC", 2: "ED", 3: "ET"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI multiclass baseline trainer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gli_4mod_multiclass_new")
    parser.add_argument("--results-dir", type=str, default="results/gli_4mod_multiclass_new")
    return parser.parse_args()


def dice_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> float:
    pred_mask = pred == class_id
    target_mask = target == class_id
    inter = torch.logical_and(pred_mask, target_mask).sum().item()
    den = pred_mask.sum().item() + target_mask.sum().item()
    return float((2.0 * inter + 1e-6) / (den + 1e-6))


def evaluate_multiclass(model: torch.nn.Module, loader: DataLoader, device: torch.device, patch_size: tuple[int, int, int]) -> dict[str, float]:
    per_class_values: dict[int, list[float]] = {1: [], 2: [], 3: []}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device).long().squeeze(1)
            logits = sliding_window_inference(image, patch_size, 1, model, overlap=0.25)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            for class_id in (1, 2, 3):
                per_class_values[class_id].append(dice_for_class(pred, label, class_id))

    dice_tc = float(np.mean(per_class_values[1])) if per_class_values[1] else 0.0
    dice_ed = float(np.mean(per_class_values[2])) if per_class_values[2] else 0.0
    dice_et = float(np.mean(per_class_values[3])) if per_class_values[3] else 0.0
    mean_dice = float(np.mean([dice_tc, dice_ed, dice_et]))
    return {"dice_tc": dice_tc, "dice_ed": dice_ed, "dice_et": dice_et, "mean_dice": mean_dice}


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

    train_cases, val_cases, test_cases = load_gli_train_val_test_cases(
        data_root=cfg.data_root,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
    )

    train_cases = [case for case in train_cases if "label" in case]
    val_cases = [case for case in val_cases if "label" in case]

    raw_val_cases = list(val_cases)
    if not raw_val_cases:
        if len(train_cases) < 2:
            raise RuntimeError("Need at least 2 labeled train cases to create a multiclass validation split.")
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
        print(f"gli_val.txt is image-only; created holdout val split with {len(train_cases)} train / {len(val_cases)} val cases")
    else:
        print(f"Loaded {len(raw_val_cases)} labeled val cases")

    if not train_cases:
        raise RuntimeError("No labeled train cases available for multiclass training")

    print("Mode: MULTICLASS TRAINING")
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)} | Test cases: {len(test_cases)}")
    print(f"Modalities: {cfg.modalities} (4 channels)")
    print(f"Classes: background, TC, ED, ET")
    print(f"Patch size: {cfg.patch_size}")
    print(f"LR: {cfg.learning_rate} | Weight decay: {cfg.weight_decay}")
    print(f"Epochs: {args.epochs} | Val interval: {args.val_interval}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Results dir: {cfg.results_dir}")
    print()

    train_ds = Dataset(
        data=train_cases,
        transform=build_multiclass_transforms(
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
        transform=build_multiclass_transforms(
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
    model = build_model(in_channels=4, out_channels=4).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_mean_dice = -1.0
    last_metrics = {"dice_tc": 0.0, "dice_ed": 0.0, "dice_et": 0.0, "mean_dice": 0.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            steps += 1
            image = batch["image"].to(device)
            label = batch["label"].to(device).long()

            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(steps, 1)

        run_val = (epoch % max(1, args.val_interval) == 0) or (epoch == args.epochs)
        if run_val:
            last_metrics = evaluate_multiclass(model, val_loader, device, tuple(cfg.patch_size))
            if last_metrics["mean_dice"] > best_mean_dice:
                best_mean_dice = last_metrics["mean_dice"]
                torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")
            print(
                f"Epoch {epoch:03d}/{args.epochs} | loss={epoch_loss:.4f} | "
                f"mean_dice={last_metrics['mean_dice']:.4f} | TC={last_metrics['dice_tc']:.4f} | "
                f"ED={last_metrics['dice_ed']:.4f} | ET={last_metrics['dice_et']:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d}/{args.epochs} | loss={epoch_loss:.4f} | val=skip")

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "validated": run_val,
                **last_metrics,
            }
        )

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
        "best_mean_dice": best_mean_dice,
        "class_names": CLASS_NAMES,
    }
    with (cfg.results_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"\nBest mean Dice: {best_mean_dice:.4f}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
