from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.utils import set_determinism

from config import get_default_config
from dataset import load_mixed_train_val_test_cases
from model import build_model_with_name
from transforms import build_region_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI+PED+MEN multimodal region-target trainer (WT/TC/ET, balanced sampling)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mixed_gli_ped_men_region")
    parser.add_argument("--results-dir", type=str, default="results/mixed_gli_ped_men_region")
    parser.add_argument("--model-name", type=str, default="unet", choices=["unet", "swinunetr"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.15)
    parser.add_argument("--wt-focal-weight", type=float, default=1.0)
    parser.add_argument("--tc-focal-weight", type=float, default=2.0)
    parser.add_argument("--et-focal-weight", type=float, default=2.5)
    parser.add_argument("--wt-threshold", type=float, default=0.45)
    parser.add_argument("--tc-threshold", type=float, default=0.50)
    parser.add_argument("--et-threshold", type=float, default=0.55)
    parser.add_argument("--min-fg-ratio", type=float, default=0.03)
    parser.add_argument("--max-sample-tries", type=int, default=45)
    parser.add_argument("--tumor-margin", type=int, default=28)
    parser.add_argument("--min-component-size", type=int, default=75)
    parser.add_argument("--disable-keep-largest-wt", action="store_true")
    parser.add_argument("--min-et-component-size", type=int, default=75)
    return parser.parse_args()


def create_dataset_weights(cases: list[dict]) -> list[float]:
    """Compute inverse-size weights for balanced sampling across datasets."""
    dataset_counts = {}
    for case in cases:
        ds = case.get("dataset", "GLI")
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    
    weights = []
    for case in cases:
        ds = case.get("dataset", "GLI")
        weight = 1.0 / dataset_counts[ds]
        weights.append(weight)
    
    print(f"\nDataset counts: {dataset_counts}")
    print(f"Inverse weights: {[(ds, 1.0/c) for ds, c in dataset_counts.items()]}")
    
    return weights


def dice_for_binary(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    inter = torch.logical_and(pred_mask, target_mask).sum().item()
    den = pred_mask.sum().item() + target_mask.sum().item()
    return float((2.0 * inter + 1e-6) / (den + 1e-6))


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0 or not np.any(mask):
        return mask

    visited = np.zeros(mask.shape, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    d, h, w = mask.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    coords = np.argwhere(mask)
    for z, y, x in coords:
        z = int(z)
        y = int(y)
        x = int(x)
        if visited[z, y, x]:
            continue

        comp = []
        q = deque([(z, y, x)])
        visited[z, y, x] = True

        while q:
            cz, cy, cx = q.pop()
            comp.append((cz, cy, cx))
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if nz < 0 or ny < 0 or nx < 0 or nz >= d or ny >= h or nx >= w:
                    continue
                if visited[nz, ny, nx] or not mask[nz, ny, nx]:
                    continue
                visited[nz, ny, nx] = True
                q.append((nz, ny, nx))

        if len(comp) >= min_size:
            zz, yy, xx = zip(*comp)
            out[np.asarray(zz), np.asarray(yy), np.asarray(xx)] = True

    return out


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask

    visited = np.zeros(mask.shape, dtype=bool)
    d, h, w = mask.shape
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    largest_comp = []
    for z, y, x in np.argwhere(mask):
        z = int(z)
        y = int(y)
        x = int(x)
        if visited[z, y, x]:
            continue

        comp = []
        q = deque([(z, y, x)])
        visited[z, y, x] = True

        while q:
            cz, cy, cx = q.pop()
            comp.append((cz, cy, cx))
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if nz < 0 or ny < 0 or nx < 0 or nz >= d or ny >= h or nx >= w:
                    continue
                if visited[nz, ny, nx] or not mask[nz, ny, nx]:
                    continue
                visited[nz, ny, nx] = True
                q.append((nz, ny, nx))

        if len(comp) > len(largest_comp):
            largest_comp = comp

    out = np.zeros_like(mask, dtype=bool)
    if largest_comp:
        zz, yy, xx = zip(*largest_comp)
        out[np.asarray(zz), np.asarray(yy), np.asarray(xx)] = True
    return out


def postprocess_regions(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
    min_component_size: int,
    keep_largest_wt: bool,
    min_et_component_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt = _remove_small_components(wt, min_component_size)
    tc = _remove_small_components(tc, min_component_size)
    et = _remove_small_components(et, min_et_component_size)

    if keep_largest_wt:
        wt = _largest_component_mask(wt)

    # Enforce hierarchy ET ⊆ TC ⊆ WT.
    tc = np.logical_and(tc, wt)
    et = np.logical_and(et, tc)

    return wt, tc, et


def evaluate_regions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    wt_thr: float,
    tc_thr: float,
    et_thr: float,
    min_component_size: int,
    keep_largest_wt: bool,
    min_et_component_size: int,
) -> dict[str, float]:
    values_wt: list[float] = []
    values_tc: list[float] = []
    values_et: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            logits = sliding_window_inference(image, patch_size, 1, model, overlap=0.25)
            probs = torch.sigmoid(logits)

            wt = (probs[:, 0] > wt_thr)
            tc = (probs[:, 1] > tc_thr)
            et = (probs[:, 2] > et_thr)

            wt_np = wt[0].detach().cpu().numpy().astype(bool)
            tc_np = tc[0].detach().cpu().numpy().astype(bool)
            et_np = et[0].detach().cpu().numpy().astype(bool)

            wt_np, tc_np, et_np = postprocess_regions(
                wt_np,
                tc_np,
                et_np,
                min_component_size=min_component_size,
                keep_largest_wt=keep_largest_wt,
                min_et_component_size=min_et_component_size,
            )

            pred_wt = torch.from_numpy(wt_np).unsqueeze(0).to(device=device)
            pred_tc = torch.from_numpy(tc_np).unsqueeze(0).to(device=device)
            pred_et = torch.from_numpy(et_np).unsqueeze(0).to(device=device)

            tgt_wt = target[:, 0] > 0.5
            tgt_tc = target[:, 1] > 0.5
            tgt_et = target[:, 2] > 0.5

            values_wt.append(dice_for_binary(pred_wt, tgt_wt))
            values_tc.append(dice_for_binary(pred_tc, tgt_tc))
            values_et.append(dice_for_binary(pred_et, tgt_et))

    dice_wt = float(np.mean(values_wt)) if values_wt else 0.0
    dice_tc = float(np.mean(values_tc)) if values_tc else 0.0
    dice_et = float(np.mean(values_et)) if values_et else 0.0
    mean_dice = float(np.mean([dice_wt, dice_tc, dice_et]))
    return {"dice_wt": dice_wt, "dice_tc": dice_tc, "dice_et": dice_et, "mean_dice": mean_dice}


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg.batch_size = args.batch_size
    cfg.patch_size = tuple(args.patch_size)
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.seed = args.seed
    cfg.min_fg_ratio = args.min_fg_ratio
    cfg.max_sample_tries = args.max_sample_tries
    cfg.tumor_margin = args.tumor_margin
    cfg.checkpoint_dir = (cfg.repo_root / Path(args.checkpoint_dir)).resolve()
    cfg.results_dir = (cfg.repo_root / Path(args.results_dir)).resolve()

    set_determinism(seed=cfg.seed)

    # Load GLI, PED, MEN
    gli_data_root = cfg.repo_root / "BraTS-2024-Complete/BraTS-GLI"
    ped_data_root = cfg.repo_root / "BraTS-2024-Complete/BraTS-PED"
    men_data_root = cfg.repo_root / "BraTS-2024-Complete/BraTS-MEN-RT"

    gli_train_list = cfg.repo_root / "patient_lists/gli_train.txt"
    gli_val_list = cfg.repo_root / "patient_lists/gli_val.txt"
    gli_test_list = cfg.repo_root / "patient_lists/gli_test.txt"
    
    ped_train_list = cfg.repo_root / "patient_lists/ped_train.txt"
    ped_val_list = cfg.repo_root / "patient_lists/ped_val.txt"
    ped_test_list = cfg.repo_root / "patient_lists/ped_test.txt"
    
    men_train_list = cfg.repo_root / "patient_lists/men_train.txt"
    men_val_list = cfg.repo_root / "patient_lists/men_val.txt"
    men_test_list = cfg.repo_root / "patient_lists/men_test.txt"

    train_cases, val_cases, test_cases = load_mixed_train_val_test_cases(
        repo_root=cfg.repo_root,
        gli_data_root=gli_data_root,
        ped_data_root=ped_data_root,
        men_data_root=men_data_root,
        gli_train_list=gli_train_list,
        gli_val_list=gli_val_list,
        gli_test_list=gli_test_list,
        ped_train_list=ped_train_list,
        ped_val_list=ped_val_list,
        ped_test_list=ped_test_list,
        men_train_list=men_train_list,
        men_val_list=men_val_list,
        men_test_list=men_test_list,
    )

    train_cases = [case for case in train_cases if "label" in case]
    val_cases = [case for case in val_cases if "label" in case]

    print("Mode: MULTIMODAL REGION TARGET TRAINING (GLI+PED+MEN, balanced)")
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)} | Test cases: {len(test_cases)}")
    
    # Count by dataset
    train_counts = {}
    for case in train_cases:
        ds = case.get("dataset", "GLI")
        train_counts[ds] = train_counts.get(ds, 0) + 1
    print(f"Training split by dataset: {train_counts}")

    train_ds = Dataset(
        data=train_cases,
        transform=build_region_transforms(
            modalities=["t1n", "t1c", "t2w", "t2f"],
            patch_size=cfg.patch_size,
            min_fg_ratio=cfg.min_fg_ratio,
            max_tries=cfg.max_sample_tries,
            margin=cfg.tumor_margin,
            training=True,
        ),
    )

    val_ds = Dataset(
        data=val_cases,
        transform=build_region_transforms(
            modalities=["t1n", "t1c", "t2w", "t2f"],
            patch_size=cfg.patch_size,
            min_fg_ratio=0.0,
            max_tries=1,
            margin=0,
            training=False,
        ),
    )

    # Create weighted sampler for balanced dataset sampling
    weights = create_dataset_weights(train_cases)
    sampler = WeightedRandomSampler(weights, len(train_cases), replacement=True)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_with_name(args.model_name, in_channels=4, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Setup loss
    loss_fn = DiceFocalLoss(
        sigmoid=True,
        weight=[args.wt_focal_weight, args.tc_focal_weight, args.et_focal_weight],
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
    )

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = model(image)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % args.val_interval == 0:
            val_metrics = evaluate_regions(
                model,
                val_loader,
                device,
                cfg.patch_size,
                args.wt_threshold,
                args.tc_threshold,
                args.et_threshold,
                args.min_component_size,
                not args.disable_keep_largest_wt,
                args.min_et_component_size,
            )

            history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                **val_metrics,
            })

            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice_WT={val_metrics['dice_wt']:.4f}, "
                  f"Dice_TC={val_metrics['dice_tc']:.4f}, Dice_ET={val_metrics['dice_et']:.4f}, "
                  f"Mean={val_metrics['mean_dice']:.4f}")

            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]
                torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")

    # Save history
    with open(cfg.results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save split summary
    split_summary = {
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
        "best_mean_dice": float(best_dice),
        "targets": ["WT", "TC", "ET"],
        "thresholds": {
            "wt": args.wt_threshold,
            "tc": args.tc_threshold,
            "et": args.et_threshold,
        },
        "dataset_split": train_counts,
    }
    with open(cfg.results_dir / "split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    print(f"\n✓ Training complete. Best Mean Dice: {best_dice:.4f}")
    print(f"  Checkpoints: {cfg.checkpoint_dir}")
    print(f"  Results: {cfg.results_dir}")


if __name__ == "__main__":
    main()
