from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from config import get_default_config
from dataset import load_gli_train_val_test_cases
from model import build_model
from transforms import build_multiclass_inference_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI multiclass inference with TTA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases in split")
    parser.add_argument("--output-dir", type=str, default="results/gli_4mod_multiclass_new_tta")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--min-component-size", type=int, default=75)
    parser.add_argument("--disable-keep-largest-tumor", action="store_true")
    return parser.parse_args()


def dice_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> float:
    pred_mask = pred == class_id
    target_mask = target == class_id
    inter = torch.logical_and(pred_mask, target_mask).sum().item()
    den = pred_mask.sum().item() + target_mask.sum().item()
    return float((2.0 * inter + 1e-6) / (den + 1e-6))


def tta_predict_probs(model: torch.nn.Module, image: torch.Tensor, patch_size: tuple[int, int, int], enabled: bool = True) -> torch.Tensor:
    if not enabled:
        logits = sliding_window_inference(image, patch_size, 1, model, overlap=0.25)
        return torch.softmax(logits, dim=1)

    flip_axes = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    prob_sum = None

    for axes in flip_axes:
        x = torch.flip(image, dims=axes) if axes else image
        logits = sliding_window_inference(x, patch_size, 1, model, overlap=0.25)
        probs = torch.softmax(logits, dim=1)
        if axes:
            probs = torch.flip(probs, dims=axes)
        prob_sum = probs if prob_sum is None else prob_sum + probs

    return prob_sum / float(len(flip_axes))


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask
    if not np.any(mask):
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

        if len(comp) > len(largest_comp):
            largest_comp = comp

    out = np.zeros_like(mask, dtype=bool)
    if largest_comp:
        zz, yy, xx = zip(*largest_comp)
        out[np.asarray(zz), np.asarray(yy), np.asarray(xx)] = True
    return out


def postprocess_prediction(pred_3d: np.ndarray, min_component_size: int, keep_largest_tumor: bool) -> np.ndarray:
    out = np.zeros_like(pred_3d, dtype=np.uint8)

    for class_id in (1, 2, 3):
        class_mask = pred_3d == class_id
        class_mask = _remove_small_components(class_mask, min_component_size)
        out[class_mask] = class_id

    if keep_largest_tumor:
        tumor = out > 0
        keep_mask = _largest_component_mask(tumor)
        out = np.where(keep_mask, out, 0).astype(np.uint8)

    return out


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg.patch_size = tuple(args.patch_size)

    train_list = cfg.repo_root / "patient_lists/gli_train.txt"
    val_list = cfg.repo_root / "patient_lists/gli_val.txt"
    test_list = cfg.repo_root / "patient_lists/gli_test.txt"

    train_cases, val_cases, test_cases = load_gli_train_val_test_cases(
        data_root=cfg.data_root,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
    )

    if args.split == "train":
        cases = train_cases
    elif args.split == "val":
        cases = val_cases
    else:
        cases = test_cases

    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        raise RuntimeError(f"No cases available for split={args.split}")

    include_label = args.split == "train" or any("label" in case for case in cases)

    ds = Dataset(
        data=cases,
        transform=build_multiclass_inference_transforms(modalities=cfg.modalities, include_label=include_label),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=4, out_channels=4).to(device)
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
            label = batch.get("label")
            has_label = label is not None
            if has_label:
                label = label.to(device).long().squeeze(1)

            probs = tta_predict_probs(model, image, tuple(cfg.patch_size), enabled=not args.disable_tta)
            pred = torch.argmax(probs, dim=1, keepdim=True)
            pred_np = pred[0, 0].detach().cpu().numpy().astype(np.uint8)
            pred_np = postprocess_prediction(
                pred_np,
                min_component_size=args.min_component_size,
                keep_largest_tumor=not args.disable_keep_largest_tumor,
            )
            np.save(pred_dir / f"{case_id}_pred.npy", pred_np)

            if has_label:
                pred_t = torch.from_numpy(pred_np).unsqueeze(0).to(device=device, dtype=torch.long)
                dice_tc = dice_for_class(pred_t, label, 1)
                dice_ed = dice_for_class(pred_t, label, 2)
                dice_et = dice_for_class(pred_t, label, 3)
                mean_dice = float(np.mean([dice_tc, dice_ed, dice_et]))
                rows.append(
                    {
                        "case_id": case_id,
                        "dice_tc": dice_tc,
                        "dice_ed": dice_ed,
                        "dice_et": dice_et,
                        "mean_dice": mean_dice,
                    }
                )
            else:
                rows.append(
                    {
                        "case_id": case_id,
                        "dice_tc": None,
                        "dice_ed": None,
                        "dice_et": None,
                        "mean_dice": None,
                    }
                )

    valid_rows = [r for r in rows if r["mean_dice"] is not None]
    summary = {
        "split": args.split,
        "num_cases": len(rows),
        "mean_dice": float(np.mean([r["mean_dice"] for r in valid_rows])) if valid_rows else None,
        "dice_tc": float(np.mean([r["dice_tc"] for r in valid_rows])) if valid_rows else None,
        "dice_ed": float(np.mean([r["dice_ed"] for r in valid_rows])) if valid_rows else None,
        "dice_et": float(np.mean([r["dice_et"] for r in valid_rows])) if valid_rows else None,
        "tta": not args.disable_tta,
        "min_component_size": args.min_component_size,
        "keep_largest_tumor": not args.disable_keep_largest_tumor,
        "cases": rows,
    }

    with (out_dir / "inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Predictions saved to: {pred_dir}")
    if summary["mean_dice"] is None:
        print("Mean Dice: N/A (no labels available for this split)")
    else:
        print(
            f"Mean Dice: {summary['mean_dice']:.4f} | TC={summary['dice_tc']:.4f} | "
            f"ED={summary['dice_ed']:.4f} | ET={summary['dice_et']:.4f}"
        )


if __name__ == "__main__":
    main()
