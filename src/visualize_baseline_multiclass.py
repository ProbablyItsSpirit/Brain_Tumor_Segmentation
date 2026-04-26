from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from monai.data import DataLoader, Dataset

from config import get_default_config
from dataset import load_gli_train_val_test_cases
from transforms import build_multiclass_inference_transforms


CLASS_CMAP = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),  # background transparent
    (0.90, 0.20, 0.20, 0.65),  # TC - red
    (0.20, 0.80, 0.30, 0.65),  # ED - green
    (0.20, 0.45, 0.95, 0.65),  # ET - blue
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GLI multiclass predictions")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--predictions-dir", type=str, default="results/gli_4mod_multiclass_new_tta/predictions")
    parser.add_argument("--output-dir", type=str, default="results/gli_4mod_multiclass_new_tta/visuals")
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--case-index", type=int, default=0)
    return parser.parse_args()


def pick_slice(mask: np.ndarray) -> int:
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    if counts.max() <= 0:
        return mask.shape[0] // 2
    return int(counts.argmax())


def plot_multiclass_panel(axis, base_image: np.ndarray, mask: np.ndarray, title: str) -> None:
    axis.imshow(base_image, cmap="gray")
    axis.imshow(mask, cmap=CLASS_CMAP, vmin=0, vmax=3)
    axis.set_title(title)
    axis.axis("off")


def main() -> None:
    args = parse_args()
    cfg = get_default_config()

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

    if not cases:
        raise RuntimeError(f"No cases available for split={args.split}")

    start_index = max(0, args.case_index)
    end_index = min(len(cases), start_index + max(1, args.num_cases))
    selected_cases = cases[start_index:end_index]

    pred_dir = Path(args.predictions_dir)
    if not pred_dir.is_absolute():
        pred_dir = (cfg.repo_root / pred_dir).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (cfg.repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    include_label = args.split == "train" or any("label" in case for case in selected_cases)
    ds = Dataset(
        data=selected_cases,
        transform=build_multiclass_inference_transforms(modalities=cfg.modalities, include_label=include_label),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    for batch in loader:
        case_id = str(batch["case_id"][0])
        image = batch["image"][0].detach().cpu().numpy()
        label = batch["label"][0, 0].detach().cpu().numpy() if "label" in batch else None

        pred_path = pred_dir / f"{case_id}_pred.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction not found for {case_id}: {pred_path}")

        pred = np.load(pred_path)
        reference_mask = pred if pred.max() > 0 else (label if label is not None else np.zeros_like(pred))
        mid = pick_slice(reference_mask)

        t1c_slice = image[1, mid]

        if label is not None:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=160)
            plot_multiclass_panel(axes[0], t1c_slice, label[mid], "Ground Truth")
            plot_multiclass_panel(axes[1], t1c_slice, pred[mid], "Prediction overlay")
            axes[2].imshow(label[mid], cmap=CLASS_CMAP, vmin=0, vmax=3)
            axes[2].set_title("GT mask")
            axes[2].axis("off")
            axes[3].imshow(pred[mid], cmap=CLASS_CMAP, vmin=0, vmax=3)
            axes[3].set_title("Pred mask")
            axes[3].axis("off")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
            axes[0].imshow(t1c_slice, cmap="gray")
            axes[0].set_title("Input (T1c)")
            axes[0].axis("off")
            axes[1].imshow(pred[mid], cmap=CLASS_CMAP, vmin=0, vmax=3)
            axes[1].set_title("Pred mask")
            axes[1].axis("off")
            axes[2].imshow(t1c_slice, cmap="gray")
            axes[2].imshow(pred[mid], cmap=CLASS_CMAP, vmin=0, vmax=3)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

        fig.suptitle(case_id)
        fig.tight_layout()
        fig.savefig(out_dir / f"{case_id}_slice{mid}.png")
        plt.close(fig)
        print(f"Saved visualization for {case_id} to: {out_dir}")


if __name__ == "__main__":
    main()
