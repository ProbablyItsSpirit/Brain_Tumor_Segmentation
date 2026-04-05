import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple GT vs prediction slice viewer")
    parser.add_argument("--case-id", type=str, default="BraTS-GLI-02243-101")
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="experiments/baseline_t1ce_multiclass/results/inference_stage_b/predictions",
        help="Directory containing *_pred.npy files",
    )
    return parser.parse_args()


args = parse_args()
case_id = args.case_id

repo_root = Path(__file__).resolve().parents[1]
pred_dir = repo_root / Path(args.pred_dir)
pred_path = pred_dir / f"{case_id}_pred.npy"

if not pred_path.exists():
    available_preds = sorted(pred_dir.glob("*_pred.npy"))
    if not available_preds:
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}\n"
            "No prediction files exist in the predictions folder.\n"
            "Run inference with prediction saving enabled first:\n"
            "python experiments/baseline_t1ce_multiclass/inference.py "
            "--config experiments/baseline_t1ce_multiclass/config.yaml "
            "--checkpoint experiments/baseline_t1ce_multiclass/checkpoints/baseline_t1ce_multiclass_mixed_dicefocal_3to1/stage_b_best.pt "
            "--label-setup 4c --save-predictions"
        )

    pred_path = available_preds[0]
    case_id = pred_path.stem.replace("_pred", "")
    print(f"Requested case not found. Using available case: {case_id}")


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


img_path, gt_path = find_case_files(repo_root, case_id)

img = nib.load(str(img_path)).get_fdata()
gt = nib.load(str(gt_path)).get_fdata()
pred = np.load(pred_path)

# Pick middle slice in z
z = img.shape[2] // 2

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img[:, :, z], cmap="gray")
ax[0].set_title("T1ce")
ax[0].axis("off")

ax[1].imshow(gt[:, :, z], cmap="nipy_spectral", vmin=0, vmax=max(3, int(gt.max())))
ax[1].set_title("Ground Truth")
ax[1].axis("off")

ax[2].imshow(pred[:, :, z], cmap="nipy_spectral", vmin=0, vmax=max(3, int(pred.max())))
ax[2].set_title("Prediction")
ax[2].axis("off")

plt.tight_layout()
plt.show()