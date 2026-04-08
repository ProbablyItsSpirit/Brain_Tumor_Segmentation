from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save 10-case visual comparisons (image/GT/pred) and failure pattern report."
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing *_pred.npy predictions from inference.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baseline_t1ce_multiclass/results/visual_validation_10",
        help="Directory to save PNG comparisons and summary JSON",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Optional comma-separated case IDs; if empty, uses first N prediction files",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="How many cases to validate when --cases is not provided",
    )
    return parser.parse_args()


def find_case_files(repo_root: Path, case_id: str) -> Tuple[Path, Path]:
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


def choose_slice(gt: np.ndarray, pred: np.ndarray) -> int:
    gt_fg = (gt > 0).astype(np.uint8)
    pred_fg = (pred > 0).astype(np.uint8)
    signal = gt_fg.sum(axis=(0, 1)) + pred_fg.sum(axis=(0, 1))
    if signal.max() > 0:
        return int(np.argmax(signal))
    return int(gt.shape[2] // 2)


def dice_no_bg(pred: np.ndarray, gt: np.ndarray, num_classes: int = 4) -> float:
    vals: List[float] = []
    for c in range(1, num_classes):
        gt_c = (gt == c)
        pred_c = (pred == c)
        if gt_c.sum() == 0:
            continue
        den = float(gt_c.sum() + pred_c.sum())
        if den == 0.0:
            continue
        inter = float((gt_c & pred_c).sum())
        vals.append((2.0 * inter) / den)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def classify_failure(pred: np.ndarray, gt: np.ndarray) -> str:
    pred_fg = int((pred > 0).sum())
    gt_fg = int((gt > 0).sum())

    if gt_fg > 0 and pred_fg == 0:
        return "predicts nothing"

    total = int(np.prod(pred.shape))
    pred_ratio = pred_fg / max(total, 1)
    if pred_ratio > 0.15:
        return "predicts everywhere"

    d = dice_no_bg(pred, gt)
    if d >= 0.5:
        return "almost correct"

    return "noisy blobs"


def save_case_figure(case_id: str, img: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_png: Path) -> None:
    z = choose_slice(gt, pred)

    img2d = img[:, :, z]
    gt2d = gt[:, :, z]
    pred2d = pred[:, :, z]

    err = np.zeros_like(gt2d, dtype=np.uint8)
    err[(gt2d > 0) & (pred2d == 0)] = 1
    err[(gt2d == 0) & (pred2d > 0)] = 2

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    ax[0].imshow(img2d, cmap="gray")
    ax[0].set_title("T1ce")
    ax[0].axis("off")

    ax[1].imshow(img2d, cmap="gray")
    ax[1].imshow(np.ma.masked_where(gt2d == 0, gt2d), cmap="nipy_spectral", alpha=0.55)
    ax[1].set_title("GT overlay")
    ax[1].axis("off")

    ax[2].imshow(img2d, cmap="gray")
    ax[2].imshow(np.ma.masked_where(pred2d == 0, pred2d), cmap="nipy_spectral", alpha=0.55)
    ax[2].set_title("Prediction overlay")
    ax[2].axis("off")

    ax[3].imshow(img2d, cmap="gray")
    ax[3].imshow(np.ma.masked_where(err == 0, err), cmap="coolwarm", alpha=0.6, vmin=0, vmax=2)
    ax[3].set_title("Errors (miss=blue, fp=red)")
    ax[3].axis("off")

    fig.suptitle(f"{case_id} | z={z}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    pred_dir = (repo_root / Path(args.pred_dir)).resolve()
    output_dir = (repo_root / Path(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    if args.cases.strip():
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        pred_files = sorted(pred_dir.glob("*_pred.npy"))
        case_ids = [p.stem.replace("_pred", "") for p in pred_files[: args.num_cases]]

    if not case_ids:
        raise RuntimeError("No cases selected. Ensure prediction files exist in --pred-dir.")

    results: List[Dict[str, Any]] = []

    for case_id in case_ids:
        pred_path = pred_dir / f"{case_id}_pred.npy"
        if not pred_path.exists():
            results.append({
                "case_id": case_id,
                "status": "missing_prediction",
                "failure_pattern": "n/a",
            })
            continue

        try:
            img_path, gt_path = find_case_files(repo_root, case_id)
            img = nib.load(str(img_path)).get_fdata()
            gt = nib.load(str(gt_path)).get_fdata().astype(np.int64)
            pred = np.load(pred_path).astype(np.int64)

            d = dice_no_bg(pred, gt)
            pattern = classify_failure(pred, gt)

            out_png = output_dir / f"{case_id}_compare.png"
            save_case_figure(case_id, img, gt, pred, out_png)

            results.append(
                {
                    "case_id": case_id,
                    "status": "ok",
                    "dice_no_bg": d,
                    "failure_pattern": pattern,
                    "image": str(img_path),
                    "gt": str(gt_path),
                    "pred": str(pred_path),
                    "png": str(out_png),
                }
            )
        except Exception as ex:
            results.append(
                {
                    "case_id": case_id,
                    "status": "error",
                    "error": str(ex),
                    "failure_pattern": "n/a",
                }
            )

    summary_path = output_dir / "validation_10_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    counts: Dict[str, int] = {}
    for row in results:
        fp = row.get("failure_pattern", "n/a")
        counts[fp] = counts.get(fp, 0) + 1

    print(f"Saved visual comparisons to: {output_dir}")
    print(f"Saved summary JSON: {summary_path}")
    print("Failure pattern counts:")
    for k, v in sorted(counts.items(), key=lambda x: x[0]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
