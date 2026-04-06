from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
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
    parser.add_argument("--margin", type=int, default=20, help="Tumor bbox expansion margin")
    parser.add_argument(
        "--tumor-center-prob",
        type=float,
        default=0.7,
        help="Probability of tumor-centered sampling (remaining probability uses random patch)",
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


def get_transforms(patch_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ]
    )


def get_tumor_bbox(label_3d: np.ndarray) -> list[int] | None:
    coords = np.where(label_3d > 0)
    if len(coords[0]) == 0:
        return None

    zmin, ymin, xmin = np.min(coords, axis=1)
    zmax, ymax, xmax = np.max(coords, axis=1)
    return [int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)]


def random_centered_crop(
    image_3d: np.ndarray,
    label_3d: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    sz, sy, sx = patch_size
    zmax = max(label_3d.shape[0] - sz, 0)
    ymax = max(label_3d.shape[1] - sy, 0)
    xmax = max(label_3d.shape[2] - sx, 0)

    z1 = np.random.randint(0, zmax + 1) if zmax > 0 else 0
    y1 = np.random.randint(0, ymax + 1) if ymax > 0 else 0
    x1 = np.random.randint(0, xmax + 1) if xmax > 0 else 0

    z2, y2, x2 = z1 + sz, y1 + sy, x1 + sx
    return image_3d[z1:z2, y1:y2, x1:x2], label_3d[z1:z2, y1:y2, x1:x2]


def tumor_centered_crop(
    image_3d: np.ndarray,
    label_3d: np.ndarray,
    patch_size: tuple[int, int, int],
    margin: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    bbox = get_tumor_bbox(label_3d)
    if bbox is None:
        return None, None

    zmin, zmax, ymin, ymax, xmin, xmax = bbox

    zmin = max(0, zmin - margin)
    ymin = max(0, ymin - margin)
    xmin = max(0, xmin - margin)
    zmax = min(label_3d.shape[0] - 1, zmax + margin)
    ymax = min(label_3d.shape[1] - 1, ymax + margin)
    xmax = min(label_3d.shape[2] - 1, xmax + margin)

    cz = np.random.randint(zmin, zmax + 1)
    cy = np.random.randint(ymin, ymax + 1)
    cx = np.random.randint(xmin, xmax + 1)

    sz, sy, sx = patch_size
    z1 = max(0, cz - sz // 2)
    y1 = max(0, cy - sy // 2)
    x1 = max(0, cx - sx // 2)

    z2 = z1 + sz
    y2 = y1 + sy
    x2 = x1 + sx

    if z2 > label_3d.shape[0]:
        z1 = label_3d.shape[0] - sz
        z2 = label_3d.shape[0]
    if y2 > label_3d.shape[1]:
        y1 = label_3d.shape[1] - sy
        y2 = label_3d.shape[1]
    if x2 > label_3d.shape[2]:
        x1 = label_3d.shape[2] - sx
        x2 = label_3d.shape[2]

    z1 = max(0, z1)
    y1 = max(0, y1)
    x1 = max(0, x1)

    return image_3d[z1:z2, y1:y2, x1:x2], label_3d[z1:z2, y1:y2, x1:x2]


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
    margin: int,
    tumor_center_prob: float,
    bad_threshold: float,
    num_patches: int,
) -> None:
    print(f"\nChecking {name}")
    if not data:
        print(f"{name}: no valid image/label pairs found, skipping")
        return

    random.shuffle(data)
    selected = data[: min(num_patches, len(data))]

    transforms = get_transforms(patch_size=patch_size)

    save_folder = save_dir / name
    save_folder.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(selected):
        batch = transforms(item)
        img_full = to_3d(batch["image"])
        lbl_full = to_3d(batch["label"])
        case_id = str(item.get("case_id", f"sample_{i}"))
        full_unique = np.unique(lbl_full)

        if i == 0:
            print(f"{name} | Full-label unique values (first case): {full_unique.tolist()}")
            if len(full_unique) == 1 and full_unique[0] == 0:
                print(f"{name} | WARNING: labels appear all-zero; verify label path/remap for this dataset")

        use_tumor_center = np.random.rand() < tumor_center_prob
        strategy = "tumor_centered" if use_tumor_center else "random"

        if use_tumor_center:
            img_patch, lbl_patch = tumor_centered_crop(
                image_3d=img_full,
                label_3d=lbl_full,
                patch_size=patch_size,
                margin=margin,
            )
            if img_patch is None or lbl_patch is None:
                strategy = "random_fallback"
                img_patch, lbl_patch = random_centered_crop(img_full, lbl_full, patch_size)
        else:
            img_patch, lbl_patch = random_centered_crop(img_full, lbl_full, patch_size)

        img = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        lbl = torch.tensor(lbl_patch, dtype=torch.int64).unsqueeze(0).unsqueeze(0)

        tumor_ratio = float((lbl > 0).sum().item() / lbl.numel())
        unique = torch.unique(lbl)
        status = "BAD PATCH" if tumor_ratio < bad_threshold else "OK"

        print(
            f"{name} | Patch {i} | Case: {case_id} | Tumor %: {tumor_ratio:.4f} | "
            f"Labels: {unique.tolist()} | {status} | strategy={strategy}"
        )

        if tumor_ratio < bad_threshold:
            print("  BAD PATCH (almost empty foreground)")

        save_path = save_folder / f"{name}_patch_{i}_{case_id}.png"
        save_patch(img, lbl, full_image=img_full, full_label=lbl_full, save_path=save_path)


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

    if not (0.0 <= args.tumor_center_prob <= 1.0):
        raise ValueError("--tumor-center-prob must be between 0 and 1")

    for dataset_name in ["GLI", "PED", "MEN"]:
        data = collect_cases_for_dataset(repo_root=repo_root, dataset_name=dataset_name)
        process_dataset(
            name=dataset_name,
            data=data,
            save_dir=save_dir,
            patch_size=patch_size,
            margin=int(args.margin),
            tumor_center_prob=float(args.tumor_center_prob),
            bad_threshold=float(args.bad_threshold),
            num_patches=args.num_patches,
        )

    print(f"\nSampling check completed. See outputs in: {save_dir}")


if __name__ == "__main__":
    main()