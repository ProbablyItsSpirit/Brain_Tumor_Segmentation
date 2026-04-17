from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from config import get_default_config
from dataset import load_gli_train_val_test_strict
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI binary inference NEW with TTA + post-processing")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases in split")
    parser.add_argument("--output-dir", type=str, default="results/gli_4mod_binary_new_tta")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--min-component-size", type=int, default=100)
    parser.add_argument("--disable-post", action="store_true")
    return parser.parse_args()


def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + 1e-6) / (den + 1e-6))


def tta_predict_probs(model: torch.nn.Module, image: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    flip_axes = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
    prob_sum = None

    for axes in flip_axes:
        x = torch.flip(image, dims=axes) if axes else image
        logits = sliding_window_inference(x, patch_size, 1, model, overlap=0.25)
        probs = torch.softmax(logits, dim=1)
        if axes:
            probs = torch.flip(probs, dims=axes)
        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum = prob_sum + probs

    return prob_sum / float(len(flip_axes))


def post_process_binary_mask(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    if mask.max() == 0:
        return mask.astype(np.uint8)

    try:
        from scipy.ndimage import label
    except Exception:
        return mask.astype(np.uint8)

    labeled, num = label(mask > 0)
    if num <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    component_sizes: list[tuple[int, int]] = []
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        size = int(comp.sum())
        if size >= min_component_size:
            component_sizes.append((comp_id, size))

    if not component_sizes:
        return np.zeros_like(mask, dtype=np.uint8)

    largest_id, _ = max(component_sizes, key=lambda x: x[1])
    cleaned[labeled == largest_id] = 1
    return cleaned


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg.patch_size = tuple(args.patch_size)

    train_list = cfg.repo_root / "patient_lists/gli_train.txt"
    val_list = cfg.repo_root / "patient_lists/gli_val.txt"
    test_list = cfg.repo_root / "patient_lists/gli_test.txt"

    train_cases, val_cases, test_cases = load_gli_train_val_test_strict(
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

            probs = tta_predict_probs(model, image, tuple(cfg.patch_size))
            pred = torch.argmax(probs, dim=1, keepdim=True)
            pred_np = pred[0, 0].detach().cpu().numpy().astype(np.uint8)

            if not args.disable_post:
                pred_np = post_process_binary_mask(pred_np, min_component_size=args.min_component_size)

            np.save(pred_dir / f"{case_id}_pred.npy", pred_np)

            pred_for_metric = torch.from_numpy(pred_np).to(label.device).unsqueeze(0).unsqueeze(0).float()
            dice = dice_binary(pred_for_metric, label.float())
            rows.append({"case_id": case_id, "dice": dice})

    summary = {
        "split": args.split,
        "num_cases": len(rows),
        "mean_dice": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        "min_component_size": args.min_component_size,
        "post_processing": not args.disable_post,
        "cases": rows,
    }
    with (out_dir / "inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Predictions saved to: {pred_dir}")
    print(f"Mean Dice: {summary['mean_dice']:.4f}")


if __name__ == "__main__":
    main()
