from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset

from config import get_default_config
from dataset import load_gli_train_val_test_cases
from inference_baseline_region import (
    dice_for_binary,
    postprocess_regions,
    reconstruct_multiclass_map,
    tta_predict_probs,
)
from model import build_model_with_name
from transforms import build_region_inference_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GLI region-target ensemble inference (WT/TC/ET)")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases in split")
    parser.add_argument("--output-dir", type=str, default="results/gli_4mod_region_v2_boundary_ensemble_3model_tta")
    parser.add_argument("--model-name", type=str, default="unet", choices=["unet", "swinunetr"])
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--disable-tta", action="store_true")
    parser.add_argument("--wt-threshold", type=float, default=0.45)
    parser.add_argument("--tc-threshold", type=float, default=0.50)
    parser.add_argument("--et-threshold", type=float, default=0.55)
    parser.add_argument("--min-component-size", type=int, default=75)
    parser.add_argument("--disable-keep-largest-wt", action="store_true")
    parser.add_argument("--min-et-component-size", type=int, default=75)
    return parser.parse_args()


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
        transform=build_region_inference_transforms(modalities=cfg.modalities, include_label=include_label),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    resolved_checkpoints: list[str] = []
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (cfg.repo_root / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = build_model_with_name(
            model_name=args.model_name,
            in_channels=4,
            out_channels=3,
            patch_size=tuple(cfg.patch_size),
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        models.append(model)
        resolved_checkpoints.append(str(ckpt_path))

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
                label = label.to(device)

            prob_sum = None
            for model in models:
                probs = tta_predict_probs(model, image, tuple(cfg.patch_size), enabled=not args.disable_tta)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            probs = prob_sum / float(len(models))

            wt = probs[:, 0] > args.wt_threshold
            tc = probs[:, 1] > args.tc_threshold
            et = probs[:, 2] > args.et_threshold

            wt_np = wt[0].detach().cpu().numpy().astype(bool)
            tc_np = tc[0].detach().cpu().numpy().astype(bool)
            et_np = et[0].detach().cpu().numpy().astype(bool)
            wt_np, tc_np, et_np = postprocess_regions(
                wt_np,
                tc_np,
                et_np,
                min_component_size=args.min_component_size,
                keep_largest_wt=not args.disable_keep_largest_wt,
                min_et_component_size=args.min_et_component_size,
            )

            pred_map = reconstruct_multiclass_map(wt_np, tc_np, et_np)
            np.save(pred_dir / f"{case_id}_pred.npy", pred_map)

            if has_label:
                tgt_wt = label[:, 0] > 0.5
                tgt_tc = label[:, 1] > 0.5
                tgt_et = label[:, 2] > 0.5

                pred_wt = torch.from_numpy(wt_np).unsqueeze(0).to(device=device)
                pred_tc = torch.from_numpy(tc_np).unsqueeze(0).to(device=device)
                pred_et = torch.from_numpy(et_np).unsqueeze(0).to(device=device)

                dice_wt = dice_for_binary(pred_wt, tgt_wt)
                dice_tc = dice_for_binary(pred_tc, tgt_tc)
                dice_et = dice_for_binary(pred_et, tgt_et)
                mean_dice = float(np.mean([dice_wt, dice_tc, dice_et]))
                rows.append(
                    {
                        "case_id": case_id,
                        "dice_wt": dice_wt,
                        "dice_tc": dice_tc,
                        "dice_et": dice_et,
                        "mean_dice": mean_dice,
                    }
                )
            else:
                rows.append(
                    {
                        "case_id": case_id,
                        "dice_wt": None,
                        "dice_tc": None,
                        "dice_et": None,
                        "mean_dice": None,
                    }
                )

    valid_rows = [r for r in rows if r["mean_dice"] is not None]
    summary = {
        "split": args.split,
        "num_cases": len(rows),
        "model_name": args.model_name,
        "ensemble_size": len(models),
        "checkpoints": resolved_checkpoints,
        "mean_dice": float(np.mean([r["mean_dice"] for r in valid_rows])) if valid_rows else None,
        "dice_wt": float(np.mean([r["dice_wt"] for r in valid_rows])) if valid_rows else None,
        "dice_tc": float(np.mean([r["dice_tc"] for r in valid_rows])) if valid_rows else None,
        "dice_et": float(np.mean([r["dice_et"] for r in valid_rows])) if valid_rows else None,
        "tta": not args.disable_tta,
        "thresholds": {"wt": args.wt_threshold, "tc": args.tc_threshold, "et": args.et_threshold},
        "min_component_size": args.min_component_size,
        "keep_largest_wt": not args.disable_keep_largest_wt,
        "min_et_component_size": args.min_et_component_size,
        "cases": rows,
    }

    with (out_dir / "inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Predictions saved to: {pred_dir}")
    if summary["mean_dice"] is None:
        print("Mean Dice: N/A (no labels available for this split)")
    else:
        print(
            f"Mean Dice: {summary['mean_dice']:.4f} | WT={summary['dice_wt']:.4f} | "
            f"TC={summary['dice_tc']:.4f} | ET={summary['dice_et']:.4f}"
        )


if __name__ == "__main__":
    main()
