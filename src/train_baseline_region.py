from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.utils import set_determinism

from config import get_default_config
from dataset import load_gli_train_val_test_cases
from model import build_model
from transforms import build_region_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI region-target baseline trainer (WT/TC/ET)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/gli_4mod_region_new")
    parser.add_argument("--results-dir", type=str, default="results/gli_4mod_region_new")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
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

    if not val_cases:
        if len(train_cases) < 2:
            raise RuntimeError("Need at least 2 labeled train cases to create a validation split.")
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
        print(f"Loaded {len(val_cases)} labeled val cases")

    print("Mode: REGION TARGET TRAINING (WT/TC/ET)")
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)} | Test cases: {len(test_cases)}")

    train_ds = Dataset(
        data=train_cases,
        transform=build_region_transforms(
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
        transform=build_region_transforms(
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
    model = build_model(in_channels=4, out_channels=3).to(device)

    class_weights = [args.wt_focal_weight, args.tc_focal_weight, args.et_focal_weight]
    loss_kwargs: dict[str, object] = {
        "sigmoid": True,
        "to_onehot_y": False,
        "gamma": args.focal_gamma,
        "lambda_dice": args.lambda_dice,
        "lambda_focal": args.lambda_focal,
    }
    sig = inspect.signature(DiceFocalLoss.__init__).parameters
    if "focal_weight" in sig:
        loss_kwargs["focal_weight"] = class_weights
    elif "weight" in sig:
        loss_kwargs["weight"] = class_weights
    elif "class_weight" in sig:
        loss_kwargs["class_weight"] = class_weights
    else:
        print("Warning: DiceFocalLoss in this MONAI version has no class-weight arg; running unweighted.")

    loss_fn = DiceFocalLoss(**loss_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_mean_dice = -1.0
    last_metrics = {"dice_wt": 0.0, "dice_tc": 0.0, "dice_et": 0.0, "mean_dice": 0.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            steps += 1
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(steps, 1)

        run_val = (epoch % max(1, args.val_interval) == 0) or (epoch == args.epochs)
        if run_val:
            last_metrics = evaluate_regions(
                model,
                val_loader,
                device,
                tuple(cfg.patch_size),
                wt_thr=args.wt_threshold,
                tc_thr=args.tc_threshold,
                et_thr=args.et_threshold,
                min_component_size=args.min_component_size,
                keep_largest_wt=not args.disable_keep_largest_wt,
                min_et_component_size=args.min_et_component_size,
            )
            if last_metrics["mean_dice"] > best_mean_dice:
                best_mean_dice = last_metrics["mean_dice"]
                torch.save(model.state_dict(), cfg.checkpoint_dir / "best.pt")
            print(
                f"Epoch {epoch:03d}/{args.epochs} | loss={epoch_loss:.4f} | "
                f"mean_dice={last_metrics['mean_dice']:.4f} | WT={last_metrics['dice_wt']:.4f} | "
                f"TC={last_metrics['dice_tc']:.4f} | ET={last_metrics['dice_et']:.4f}"
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

    meta = {
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
        "best_mean_dice": best_mean_dice,
        "targets": ["WT", "TC", "ET"],
        "thresholds": {"wt": args.wt_threshold, "tc": args.tc_threshold, "et": args.et_threshold},
    }
    with (cfg.results_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest mean Dice: {best_mean_dice:.4f}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
