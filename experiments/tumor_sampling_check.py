import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
)
from monai.data import Dataset, DataLoader

# ---------------- CONFIG ----------------
PATCH_SIZE = (128, 128, 128)
NUM_PATCHES = 6
POS_NEG_RATIO = (3, 1)

SAVE_DIR = "sampling_debug_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# 👉 UPDATE THESE PATHS
DATASETS = {
    "GLI": {
        "images": "path_to_gli_images",
        "labels": "path_to_gli_labels"
    },
    "PED": {
        "images": "path_to_ped_images",
        "labels": "path_to_ped_labels"
    },
    "MEN": {
        "images": "path_to_men_images",
        "labels": "path_to_men_labels"
    }
}

# ----------------------------------------

def get_file_pairs(img_dir, lbl_dir):
    imgs = sorted(os.listdir(img_dir))
    data = []
    for f in imgs:
        img_path = os.path.join(img_dir, f)
        lbl_path = os.path.join(lbl_dir, f)
        if os.path.exists(lbl_path):
            data.append({"image": img_path, "label": lbl_path})
    return data


def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=POS_NEG_RATIO[0],
            neg=POS_NEG_RATIO[1],
            num_samples=1
        ),
    ])


def save_patch(image, label, save_path):
    image = image[0].cpu().numpy()
    label = label[0].cpu().numpy()

    z = image.shape[2] // 2

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, z], cmap="gray")
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(label[:, :, z])
    plt.title(f"Label | unique: {np.unique(label)}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_dataset(name, paths):
    print(f"\n🔍 Checking {name}")

    data = get_file_pairs(paths["images"], paths["labels"])
    random.shuffle(data)
    data = data[:NUM_PATCHES]

    transforms = get_transforms()
    ds = Dataset(data=data, transform=transforms)
    loader = DataLoader(ds, batch_size=1)

    save_folder = os.path.join(SAVE_DIR, name)
    os.makedirs(save_folder, exist_ok=True)

    for i, batch in enumerate(loader):
        img = batch["image"]
        lbl = batch["label"]

        tumor_ratio = (lbl > 0).sum().item() / lbl.numel()
        unique = torch.unique(lbl)

        print(f"{name} | Patch {i} | Tumor %: {tumor_ratio:.4f} | Labels: {unique.tolist()}")

        save_path = os.path.join(save_folder, f"{name}_patch_{i}.png")
        save_patch(img, lbl, save_path)


def main():
    for name, paths in DATASETS.items():
        process_dataset(name, paths)

    print("\n✅ Sampling check completed. See saved images.")


if __name__ == "__main__":
    main()