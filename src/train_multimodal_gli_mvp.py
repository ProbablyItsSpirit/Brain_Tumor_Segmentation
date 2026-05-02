"""
Optimized Phase 1 multimodal GLI MVP (src version)

Features added:
- Mixed precision (AMP) via `torch.cuda.amp`
- DataLoader performance flags (`num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`)
- Gradient accumulation (`accum_steps`)
- cuDNN benchmark enabled
- Optional channels-last 3d memory format
- Training history plotting (loss vs val Dice)
- Paper-style visual sample outputs saved per-epoch
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import nibabel as nib
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


class MultimodalGLIDataset(Dataset):
    """Load 4-channel multimodal data (T1, T1ce, T2, FLAIR).

    Returns tensors shaped (C, H, W, D) as float32.
    """

    def __init__(self, cases: List[dict], patch_size=(96, 96, 96)):
        self.cases = cases
        self.patch_size = patch_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        t1 = nib.load(case["t1"]).get_fdata()
        t1ce = nib.load(case["t1ce"]).get_fdata()
        t2 = nib.load(case["t2"]).get_fdata()
        flair = nib.load(case["flair"]).get_fdata()

        image = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)
        label = nib.load(case["seg"]).get_fdata().astype(np.float32)

        # Per-channel normalization (ignore zeros as background)
        for ch in range(image.shape[0]):
            chdata = image[ch]
            mask = chdata > 0
            if mask.sum() > 0:
                m = chdata[mask].mean()
                s = chdata[mask].std()
                image[ch] = (chdata - m) / (s + 1e-8)
            else:
                image[ch] = 0.0

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float().unsqueeze(0)

        return {"image": image, "label": label}


def build_gli_case_dicts(data_root: str, list_file: str):
    data_root = resolve_repo_path(data_root)
    list_file = resolve_repo_path(list_file)
    cases = []
    with open(list_file, "r") as f:
        patient_ids = [l.strip() for l in f if l.strip()]

    for patient_id in patient_ids:
        case_dir = os.path.join(data_root, "BraTS-GLI", "train", patient_id)
        case = {
            "t1": os.path.join(case_dir, f"{patient_id}_t1.nii.gz"),
            "t1ce": os.path.join(case_dir, f"{patient_id}_t1ce.nii.gz"),
            "t2": os.path.join(case_dir, f"{patient_id}_t2.nii.gz"),
            "flair": os.path.join(case_dir, f"{patient_id}_flair.nii.gz"),
            "seg": os.path.join(case_dir, f"{patient_id}_seg.nii.gz"),
        }
        if all(os.path.exists(p) for p in case.values()):
            cases.append(case)
        else:
            print(f"⚠️  Skipped {patient_id}: missing files")
    return cases


def build_model(in_channels=4):
    from monai.networks.nets import UNet

    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    return model


class DiceFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_sig = torch.sigmoid(pred)
        dice = 1 - (2 * (pred_sig * target).sum() + 1e-5) / ((pred_sig + target).sum() + 1e-5)
        return dice


def save_history_plots(history: dict, results_dir: Path):
    plt.figure(figsize=(6, 4), dpi=200)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "train_loss.png")
    plt.close()

    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(epochs, history["val_dice"], label="Val Dice", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "val_dice.png")
    plt.close()


def save_visual_samples(model, loader, device, results_dir: Path, n_samples=3):
    model.eval()
    imgs_saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(images))

        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        for i in range(images.shape[0]):
            img = images[i]  # (C, H, W, D)
            lbl = labels[i, 0]
            pr = preds[i]

            # central axial slice
            _, H, W, D = img.shape
            z = D // 2

            fig, axes = plt.subplots(1, 5, figsize=(15, 4), dpi=200)
            modality_titles = ["T1", "T1ce", "T2", "FLAIR", "Overlay"]
            for m in range(4):
                axes[m].imshow(img[m, :, :, z].T, cmap="gray", origin="lower")
                axes[m].set_title(modality_titles[m])
                axes[m].axis("off")

            base = img[3, :, :, z].T  # FLAIR as base for overlay
            axes[4].imshow(base, cmap="gray", origin="lower")
            # ground truth overlay (green)
            axes[4].imshow(np.ma.masked_where(lbl[:, :, z].T <= 0.5, lbl[:, :, z].T), cmap="Greens", alpha=0.4, origin="lower")
            # prediction overlay (red) from combined channels
            combined_pred = pr.max(axis=0)
            axes[4].imshow(np.ma.masked_where(combined_pred[:, :, z].T <= 0.5, combined_pred[:, :, z].T), cmap="Reds", alpha=0.4, origin="lower")
            axes[4].set_title("GT (green) / Pred (red)")
            axes[4].axis("off")

            fig.tight_layout()
            fname = results_dir / f"visual_sample_{imgs_saved:03d}.png"
            fig.savefig(fname)
            plt.close(fig)

            imgs_saved += 1
            if imgs_saved >= n_samples:
                return


def train_multimodal_gli(
    data_root: str = "BraTS-2024-Complete",
    gli_list_train: str = "patient_lists/gli_train.txt",
    gli_list_val: str = "patient_lists/gli_val.txt",
    epochs: int = 50,
    batch_size: int = 1,
    lr: float = 5e-5,
    device: str = "cuda:0",
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/gli_multimodal_mvp_seed42",
    results_dir: str = "results/gli_multimodal_mvp_seed42",
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    amp: bool = True,
    accum_steps: int = 1,
    channels_last: bool = False,
):
    device = torch.device(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint_dir = Path(resolve_repo_path(checkpoint_dir))
    results_dir = Path(resolve_repo_path(results_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    print("[1] Loading data...")
    train_cases = build_gli_case_dicts(data_root, gli_list_train)
    val_cases = build_gli_case_dicts(data_root, gli_list_val)

    train_ds = MultimodalGLIDataset(train_cases)
    val_ds = MultimodalGLIDataset(val_cases)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # Quick sanity batch
    batch = next(iter(train_loader))
    print("Image shape:", batch["image"].shape)

    model = build_model(in_channels=4)
    if channels_last:
        model.to(memory_format=torch.channels_last_3d)
    model = model.to(device)

    if channels_last:
        print("Using channels_last_3d memory format for model and inputs")

    optimizer = Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = DiceFocalLoss()

    history = {"train_loss": [], "val_dice": [], "best_val_dice": 0.0, "best_epoch": -1}

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_losses = []
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if channels_last:
                images = images.to(memory_format=torch.channels_last_3d)

            with torch.cuda.amp.autocast(enabled=amp):
                preds = model(images)
                loss = loss_fn(preds, labels) / float(accum_steps)

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_losses.append(float(loss.item()) * accum_steps)
            global_step += 1

        train_loss = float(np.mean(running_losses)) if running_losses else 0.0

        # Validation (simple)
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val E{epoch+1}"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                if channels_last:
                    images = images.to(memory_format=torch.channels_last_3d)
                preds = torch.sigmoid(model(images))
                all_preds.append(preds.cpu().numpy())
                all_trues.append(labels.cpu().numpy())

        pred = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 3, 96, 96, 96))
        true = np.concatenate(all_trues, axis=0) if all_trues else np.zeros((0, 1, 96, 96, 96))
        val_dice = 2 * (pred * true).sum() / ((pred + true).sum() + 1e-5) if pred.size and true.size else 0.0

        history["train_loss"].append(train_loss)
        history["val_dice"].append(float(val_dice))

        # Save best
        if val_dice > history["best_val_dice"]:
            history["best_val_dice"] = float(val_dice)
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")

        # Save plots and sample visuals every epoch
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        save_history_plots(history, results_dir)
        if (epoch + 1) % 5 == 0:
            save_visual_samples(model, val_loader, device, results_dir, n_samples=4)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, ValDice={val_dice:.4f} (best={history['best_val_dice']:.4f})")

    print("Training complete. Best valDice=", history["best_val_dice"]) 
    return history


if __name__ == "__main__":
    # Simple CLI fallback when run from src/
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="BraTS-2024-Complete")
    parser.add_argument("--gli-list-train", default="patient_lists/gli_train.txt")
    parser.add_argument("--gli-list-val", default="patient_lists/gli_val.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-dir", default="checkpoints/gli_multimodal_mvp_seed42")
    parser.add_argument("--results-dir", default="results/gli_multimodal_mvp_seed42")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--channels-last", action="store_true")

    args = parser.parse_args()

    train_multimodal_gli(
        data_root=args.data_root,
        gli_list_train=args.gli_list_train,
        gli_list_val=args.gli_list_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        amp=args.amp,
        accum_steps=args.accum_steps,
        channels_last=args.channels_last,
    )
