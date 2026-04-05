from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch visual debug for GT vs prediction")
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="results/inference_stage_b/predictions",
        help="Prediction directory containing *_pred.npy files",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of cases to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visual_debug",
        help="Directory to save PNG panels and summary JSON",
    )
    return parser.parse_args()


def find_case_files(repo_root: Path, case_id: str) -> tuple[Path, Path]:
    dataset_specs = [
        ("BraTS-GLI", "-t1c.nii.gz", "-seg.nii.gz"),
        ("BraTS-PED", "-t1c.nii.gz", "-seg.nii.gz"),
        ("BraTS-MEN-RT", "_t1c.nii.gz", "_gtv.nii.gz"),
    ]

    for dataset_name, image_suffix, label_suffix in dataset_specs:
        data_root = repo_root / "BraTS-2024-Complete" / dataset_name
        candidates = [
            data_root / "train" / case_id,
            data_root / "val" / case_id,
            data_root / "train_additional" / case_id,
        ]
        case_dir = next((p for p in candidates if p.exists()), None)
        if case_dir is None:
            continue

        img_path = case_dir / f"{case_id}{image_suffix}"
        gt_path = case_dir / f"{case_id}{label_suffix}"
        if img_path.exists() and gt_path.exists():
            return img_path, gt_path

    raise FileNotFoundError(f"Could not find image/label files for case: {case_id}")


def classify_pattern(pred_fg_ratio: float, gt_fg_ratio: float) -> str:
    if pred_fg_ratio < 1e-5 and gt_fg_ratio > 1e-4:
        return "predicts_nothing"
    if pred_fg_ratio > 0.2 and gt_fg_ratio < 0.02:
        return "predicts_everywhere"
    if abs(pred_fg_ratio - gt_fg_ratio) > 0.1:
        return "strong_mismatch"
    return "looks_reasonable"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    pred_dir = (script_dir / args.pred_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*_pred.npy"))
    if not pred_files:
        raise RuntimeError(f"No prediction files found in: {pred_dir}")

    selected = pred_files[: max(1, int(args.num_cases))]

    report: list[dict[str, Any]] = []
    for pred_file in selected:
        case_id = pred_file.stem.replace("_pred", "")
        img_path, gt_path = find_case_files(repo_root, case_id)

        img = nib.load(str(img_path)).get_fdata()
        gt = nib.load(str(gt_path)).get_fdata()
        pred = np.load(pred_file)

        z = img.shape[2] // 2

        gt_fg_ratio = float((gt > 0).mean())
        pred_fg_ratio = float((pred > 0).mean())
        pattern = classify_pattern(pred_fg_ratio, gt_fg_ratio)

        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        ax[0].imshow(img[:, :, z], cmap="gray")
        ax[0].set_title("T1ce")
        ax[0].axis("off")

        ax[1].imshow(gt[:, :, z], cmap="nipy_spectral", vmin=0, vmax=max(3, int(gt.max())))
        ax[1].set_title("Ground Truth")
        ax[1].axis("off")

        ax[2].imshow(pred[:, :, z], cmap="nipy_spectral", vmin=0, vmax=max(3, int(pred.max())))
        ax[2].set_title("Prediction")
        ax[2].axis("off")

        fig.suptitle(
            f"{case_id} | gt_fg={gt_fg_ratio:.4f} | pred_fg={pred_fg_ratio:.4f} | {pattern}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{case_id}_debug.png", dpi=140)
        plt.close(fig)

        report.append(
            {
                "case_id": case_id,
                "gt_fg_ratio": gt_fg_ratio,
                "pred_fg_ratio": pred_fg_ratio,
                "pattern": pattern,
            }
        )

    summary = {
        "num_cases": len(report),
        "cases": report,
    }
    with (output_dir / "visual_debug_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved debug panels to: {output_dir}")
    print(f"Saved report to: {output_dir / 'visual_debug_report.json'}")


if __name__ == "__main__":
    main()
