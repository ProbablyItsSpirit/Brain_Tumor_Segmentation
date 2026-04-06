from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged, NormalizeIntensityd, Orientationd


DATASET_SPECS = {
    "GLI": {
        "folder": "BraTS-GLI",
        "image_suffix": "-t1c.nii.gz",
        "label_suffix": "-seg.nii.gz",
    },
    "PED": {
        "folder": "BraTS-PED",
        "image_suffix": "-t1c.nii.gz",
        "label_suffix": "-seg.nii.gz",
    },
    "MEN": {
        "folder": "BraTS-MEN-RT",
        "image_suffix": "_t1c.nii.gz",
        "label_suffix": "_gtv.nii.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit labels and image-label alignment across datasets")
    parser.add_argument("--samples-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-setup",
        type=str,
        choices=["4c", "3c"],
        default="4c",
        help="Apply same remap logic used in training before post-transform checks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit_outputs",
        help="Output folder under experiments/",
    )
    return parser.parse_args()


def remap_labels(label: Any, mapping: dict[int, int]):
    if torch.is_tensor(label):
        remapped = label.clone()
        for src, dst in mapping.items():
            remapped[label == src] = dst
        return remapped.to(dtype=torch.int64)

    arr = np.asarray(label).copy()
    for src, dst in mapping.items():
        arr[arr == src] = dst
    return arr.astype(np.int64)


def get_label_mapping(label_setup: str) -> dict[int, int]:
    if label_setup == "3c":
        return {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}
    return {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}


def collect_cases(repo_root: Path, dataset_name: str) -> list[dict[str, str]]:
    spec = DATASET_SPECS[dataset_name]
    dataset_root = repo_root / "BraTS-2024-Complete" / spec["folder"]
    if not dataset_root.exists():
        return []

    out: list[dict[str, str]] = []
    for split in ["train", "val", "train_additional"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for case_dir in split_dir.iterdir():
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            image_path = case_dir / f"{case_id}{spec['image_suffix']}"
            label_path = case_dir / f"{case_id}{spec['label_suffix']}"
            if image_path.exists() and label_path.exists():
                out.append({"case_id": case_id, "image": str(image_path), "label": str(label_path), "split": split})
    return out


def save_overlay(img: np.ndarray, lbl: np.ndarray, save_path: Path, title: str) -> None:
    z = img.shape[2] // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(img[:, :, z], cmap="gray")
    plt.imshow(lbl[:, :, z], alpha=0.45, cmap="nipy_spectral")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    out_dir = script_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = get_label_mapping(args.label_setup)

    # Post-transform check mirrors training preprocessing intent.
    xform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=lambda x: remap_labels(x, mapping)),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
        ]
    )

    report: dict[str, Any] = {"label_setup": args.label_setup, "datasets": {}}

    for ds_name in ["GLI", "PED", "MEN"]:
        cases = collect_cases(repo_root, ds_name)
        random.shuffle(cases)
        selected = cases[: min(args.samples_per_dataset, len(cases))]

        ds_out = out_dir / ds_name
        ds_out.mkdir(parents=True, exist_ok=True)

        ds_items: list[dict[str, Any]] = []
        print(f"\n[{ds_name}] selected {len(selected)} case(s)")

        for item in selected:
            case_id = item["case_id"]
            img = nib.load(item["image"]).get_fdata()
            lbl = nib.load(item["label"]).get_fdata()

            raw_unique = np.unique(lbl).tolist()
            raw_nonzero_ratio = float((lbl > 0).sum() / lbl.size)
            shape_match = tuple(img.shape) == tuple(lbl.shape)

            transformed = xform({"image": item["image"], "label": item["label"]})
            t_lbl = transformed["label"]
            if torch.is_tensor(t_lbl):
                t_unique = torch.unique(t_lbl).detach().cpu().numpy().tolist()
            else:
                t_unique = np.unique(t_lbl).tolist()

            save_overlay(
                img,
                lbl,
                ds_out / f"{case_id}_overlay_raw.png",
                title=f"{ds_name} {case_id} RAW overlay",
            )

            ds_items.append(
                {
                    "case_id": case_id,
                    "split": item["split"],
                    "image_shape": tuple(img.shape),
                    "label_shape": tuple(lbl.shape),
                    "shape_match": shape_match,
                    "raw_unique_labels": raw_unique,
                    "raw_nonzero_ratio": raw_nonzero_ratio,
                    "post_transform_unique_labels": t_unique,
                }
            )

            print(
                f"{case_id} | shape_match={shape_match} | raw_unique={raw_unique} | "
                f"raw_nonzero={raw_nonzero_ratio:.6f} | post_unique={t_unique}"
            )

        report["datasets"][ds_name] = ds_items

    report_path = out_dir / "data_label_audit_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved audit report: {report_path}")
    print(f"Saved overlays under: {out_dir}")


if __name__ == "__main__":
    main()
