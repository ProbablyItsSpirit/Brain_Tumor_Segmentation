from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByLabelClassesd,
    SpatialPadd,
)


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
    parser = argparse.ArgumentParser(description="Check tumor-focused patch sampling quality")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--num-patches", type=int, default=6)
    parser.add_argument("--pos", type=float, default=3.0)
    parser.add_argument("--neg", type=float, default=1.0)
    parser.add_argument(
        "--ratios",
        type=str,
        default="0,2,2,2",
        help="Class sampling ratios as CSV for [bg,c1,c2,c3], e.g. 0,1,1,1",
    )
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=0.001,
        help="If foreground voxel ratio is below this, patch is flagged as bad",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="sampling_debug_outputs",
        help="Output folder (relative to experiments directory)",
    )
    return parser.parse_args()


def collect_cases_for_dataset(repo_root: Path, dataset_name: str) -> list[dict[str, str]]:
    spec = DATASET_SPECS[dataset_name]
    dataset_root = repo_root / "BraTS-2024-Complete" / spec["folder"]
    if not dataset_root.exists():
        return []

    cases: list[dict[str, str]] = []
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
                cases.append({"image": str(image_path), "label": str(label_path), "case_id": case_id})

    return cases


def get_transforms(patch_size: tuple[int, int, int], ratios: list[float]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                ratios=ratios,
                num_classes=4,
                num_samples=1,
            ),
        ]
    )


def to_3d(t: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


def save_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    full_image: np.ndarray,
    full_label: np.ndarray,
    save_path: Path,
) -> None:
    image_3d = to_3d(image)
    label_3d = to_3d(label)

    z_patch = image_3d.shape[2] // 2
    z_full = full_image.shape[2] // 2

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(full_image[:, :, z_full], cmap="gray")
    plt.title("Full Image (mid-z)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(full_label[:, :, z_full], cmap="nipy_spectral")
    plt.title(f"Full Label | unique: {np.unique(full_label).tolist()}")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(image_3d[:, :, z_patch], cmap="gray")
    plt.title("Sampled Patch (image)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(label_3d[:, :, z_patch], cmap="nipy_spectral")
    plt.title(f"Sampled Patch (label) | unique: {np.unique(label_3d).tolist()}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def process_dataset(
    name: str,
    data: list[dict[str, str]],
    save_dir: Path,
    patch_size: tuple[int, int, int],
    ratios: list[float],
    bad_threshold: float,
    num_patches: int,
    num_workers: int,
) -> None:
    print(f"\nChecking {name}")
    if not data:
        print(f"{name}: no valid image/label pairs found, skipping")
        return

    random.shuffle(data)
    selected = data[: min(num_patches, len(data))]

    transforms = get_transforms(patch_size=patch_size, ratios=ratios)
    ds = Dataset(data=selected, transform=transforms)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    save_folder = save_dir / name
    save_folder.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        img = batch["image"]
        lbl = batch["label"]
        case_id = batch.get("case_id", [f"sample_{i}"])
        if isinstance(case_id, (list, tuple)):
            case_id = str(case_id[0])

        tumor_ratio = float((lbl > 0).sum().item() / lbl.numel())
        unique = torch.unique(lbl)
        status = "BAD PATCH" if tumor_ratio < bad_threshold else "OK"

        print(
            f"{name} | Patch {i} | Case: {case_id} | Tumor %: {tumor_ratio:.4f} | "
            f"Labels: {unique.tolist()} | {status}"
        )

        if tumor_ratio < bad_threshold:
            print("  BAD PATCH (almost empty foreground)")

        full_case = next((x for x in selected if x["case_id"] == case_id), selected[i])
        full_image = nib_load(full_case["image"])
        full_label = nib_load(full_case["label"])

        save_path = save_folder / f"{name}_patch_{i}_{case_id}.png"
        save_patch(img, lbl, full_image=full_image, full_label=full_label, save_path=save_path)


def nib_load(path: str) -> np.ndarray:
    import nibabel as nib

    return nib.load(path).get_fdata()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiments_dir = Path(__file__).resolve().parent
    repo_root = experiments_dir.parent

    save_dir = experiments_dir / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    patch_size = tuple(args.patch_size)
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]
    if len(ratios) != 4:
        raise ValueError("--ratios must contain 4 values: bg,c1,c2,c3")

    for dataset_name in ["GLI", "PED", "MEN"]:
        data = collect_cases_for_dataset(repo_root=repo_root, dataset_name=dataset_name)
        process_dataset(
            name=dataset_name,
            data=data,
            save_dir=save_dir,
            patch_size=patch_size,
            ratios=ratios,
            bad_threshold=float(args.bad_threshold),
            num_patches=args.num_patches,
            num_workers=args.num_workers,
        )

    print(f"\nSampling check completed. See outputs in: {save_dir}")


if __name__ == "__main__":
    main()