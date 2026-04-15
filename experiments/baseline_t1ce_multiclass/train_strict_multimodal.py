from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, FocalLoss
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, Orientationd
from monai.utils import set_determinism

from dataset_loader import build_case_list, region_channels_from_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean strict multimodal GLI+PED training")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--overfit-cases", type=int, default=0, help="Use N train samples and validate on same set")
    parser.add_argument("--overfit-epochs", type=int, default=0, help="Epoch override for overfit mode")
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RegionLabeld:
    def __init__(self, key: str, et_labels: Sequence[int]):
        self.key = key
        self.et_labels = tuple(int(x) for x in et_labels)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        wt, tc, et = region_channels_from_label(d[self.key], et_labels=self.et_labels)
        d[self.key] = np.stack([wt, tc, et], axis=0).astype(np.float32)
        return d


class StackModalitiesd:
    def __init__(self, image_keys: Sequence[str]):
        self.image_keys = list(image_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d["image"] = torch.cat([d[k] for k in self.image_keys], dim=0)
        for k in self.image_keys:
            d.pop(k, None)
        return d


class StrictTumorSampler:
    def __init__(self, patch_size: Sequence[int], min_fg_ratio: float, max_tries: int, margin: int):
        self.patch = tuple(int(x) for x in patch_size)
        self.min_fg_ratio = float(min_fg_ratio)
        self.max_tries = int(max_tries)
        self.margin = int(margin)

    @staticmethod
    def _pad(img: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
        _, d, h, w = img.shape
        td, th, tw = target
        pd = max(0, td - d)
        ph = max(0, th - h)
        pw = max(0, tw - w)
        if pd == ph == pw == 0:
            return img
        return F.pad(img, (0, pw, 0, ph, 0, pd), mode="constant", value=0)

    @staticmethod
    def _crop(img: torch.Tensor, z: int, y: int, x: int, patch: tuple[int, int, int]) -> torch.Tensor:
        pd, ph, pw = patch
        return img[:, z : z + pd, y : y + ph, x : x + pw]

    @staticmethod
    def _rand_start(shape: tuple[int, int, int], patch: tuple[int, int, int]):
        d, h, w = shape
        pd, ph, pw = patch
        z = np.random.randint(0, max(d - pd, 0) + 1) if d > pd else 0
        y = np.random.randint(0, max(h - ph, 0) + 1) if h > ph else 0
        x = np.random.randint(0, max(w - pw, 0) + 1) if w > pw else 0
        return int(z), int(y), int(x)

    def _tumor_start(self, wt_region: torch.Tensor):
        fg = torch.nonzero(wt_region > 0, as_tuple=False)
        if fg.numel() == 0:
            return self._rand_start(tuple(wt_region.shape), self.patch)

        zmin = max(0, int(fg[:, 0].min()) - self.margin)
        ymin = max(0, int(fg[:, 1].min()) - self.margin)
        xmin = max(0, int(fg[:, 2].min()) - self.margin)
        zmax = min(wt_region.shape[0] - 1, int(fg[:, 0].max()) + self.margin)
        ymax = min(wt_region.shape[1] - 1, int(fg[:, 1].max()) + self.margin)
        xmax = min(wt_region.shape[2] - 1, int(fg[:, 2].max()) + self.margin)

        cz = np.random.randint(zmin, zmax + 1)
        cy = np.random.randint(ymin, ymax + 1)
        cx = np.random.randint(xmin, xmax + 1)

        pd, ph, pw = self.patch
        d, h, w = wt_region.shape
        z = max(0, min(cz - pd // 2, d - pd))
        y = max(0, min(cy - ph // 2, h - ph))
        x = max(0, min(cx - pw // 2, w - pw))
        return int(z), int(y), int(x)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        image = self._pad(d["image"], self.patch)
        label = self._pad(d["label"], self.patch)

        best_img = image
        best_lbl = label
        best_ratio = -1.0

        for _ in range(self.max_tries):
            z, y, x = self._tumor_start(label[0])
            ci = self._crop(image, z, y, x, self.patch)
            cl = self._crop(label, z, y, x, self.patch)
            fg_ratio = float((cl[0] > 0).float().mean().item())

            if fg_ratio > best_ratio:
                best_ratio = fg_ratio
                best_img = ci
                best_lbl = cl

            if fg_ratio >= self.min_fg_ratio:
                d["image"] = ci
                d["label"] = cl
                return d

        d["image"] = best_img
        d["label"] = best_lbl
        return d


def build_transforms(cfg: Dict[str, Any], training: bool) -> Compose:
    mods = list(cfg["data"]["modalities"])
    image_keys = [f"image_{m}" for m in mods]
    keys = image_keys + ["label"]

    et_labels = tuple(int(x) for x in cfg["labels"].get("et_labels", [3, 4]))

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        RegionLabeld(key="label", et_labels=et_labels),
        EnsureTyped(keys=image_keys, dtype=torch.float32),
        EnsureTyped(keys="label", dtype=torch.float32),
        StackModalitiesd(image_keys=image_keys),
    ]

    if training:
        p = cfg["patch"]
        transforms.append(
            StrictTumorSampler(
                patch_size=p["size"],
                min_fg_ratio=p["min_fg_ratio"],
                max_tries=p["max_sample_tries"],
                margin=p["tumor_margin"],
            )
        )

    return Compose(transforms)


def build_model(cfg: Dict[str, Any]) -> UNet:
    m = cfg["model"]
    return UNet(
        spatial_dims=3,
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        channels=tuple(int(x) for x in m["channels"]),
        strides=tuple(int(x) for x in m["strides"]),
        num_res_units=int(m["num_res_units"]),
    )


def dice_per_region(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> Dict[str, float]:
    names = ["WT", "TC", "ET"]
    out: Dict[str, float] = {}
    eps = 1e-6
    for c, name in enumerate(names):
        p = pred_bin[:, c]
        t = target_bin[:, c]
        inter = (p * t).sum().item()
        den = p.sum().item() + t.sum().item()
        out[name] = float((2.0 * inter + eps) / (den + eps))
    return out


def run_validation(model, val_loader, cfg, device):
    model.eval()
    losses = []
    dice_rows = []

    dloss = DiceLoss(sigmoid=True)
    floss = FocalLoss(gamma=float(cfg["training"]["loss"]["focal_gamma"]))
    ld = float(cfg["training"]["loss"]["lambda_dice"])
    lf = float(cfg["training"]["loss"]["lambda_focal"])

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            logits = sliding_window_inference(
                inputs=image,
                roi_size=tuple(int(x) for x in cfg["patch"]["size"]),
                sw_batch_size=1,
                predictor=model,
                overlap=float(cfg["inference"]["overlap"]),
            )

            loss = ld * dloss(logits, target) + lf * floss(logits, target)
            losses.append(float(loss.item()))

            probs = torch.sigmoid(logits)
            pred = (probs > float(cfg["inference"]["threshold"])).float()
            gt = (target > 0.5).float()
            dice_rows.append(dice_per_region(pred, gt))

    if not losses:
        return {"loss": 0.0, "WT": 0.0, "TC": 0.0, "ET": 0.0, "mean": 0.0}

    wt = float(np.mean([d["WT"] for d in dice_rows]))
    tc = float(np.mean([d["TC"] for d in dice_rows]))
    et = float(np.mean([d["ET"] for d in dice_rows]))
    return {
        "loss": float(np.mean(losses)),
        "WT": wt,
        "TC": tc,
        "ET": et,
        "mean": float((wt + tc + et) / 3.0),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 42)) if args.seed < 0 else int(args.seed)
    set_determinism(seed=seed)

    train_cases = build_case_list(cfg, config_path.parent, split="train")
    val_cases = build_case_list(cfg, config_path.parent, split="val")

    if args.overfit_cases > 0:
        n = min(args.overfit_cases, len(train_cases))
        train_cases = train_cases[:n]
        val_cases = train_cases.copy()
        if args.overfit_epochs > 0:
            cfg["training"]["epochs"] = int(args.overfit_epochs)

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    if not ckpt_dir.is_absolute():
        ckpt_dir = (config_path.parent / ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (ckpt_dir / "strict_val_cases.json").open("w", encoding="utf-8") as f:
        json.dump(val_cases, f, indent=2)

    train_ds = Dataset(data=train_cases, transform=build_transforms(cfg, training=True))
    val_ds = Dataset(data=val_cases, transform=build_transforms(cfg, training=False))

    nw = int(cfg["dataloader"]["num_workers"])
    if sys.platform.startswith("win"):
        nw = 0
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["dataloader"]["batch_size"]),
        shuffle=bool(cfg["dataloader"]["shuffle"]),
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    dloss = DiceLoss(sigmoid=True)
    floss = FocalLoss(gamma=float(cfg["training"]["loss"]["focal_gamma"]))
    ld = float(cfg["training"]["loss"]["lambda_dice"])
    lf = float(cfg["training"]["loss"]["lambda_focal"])

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    best_mean = -1.0
    history = []
    epochs = int(cfg["training"]["epochs"])
    log_every = int(cfg["training"].get("log_every", 10))

    print(f"Train samples: {len(train_cases)} | Val samples: {len(val_cases)}")
    print(f"Epochs: {epochs} | Overfit mode: {args.overfit_cases > 0}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            steps += 1
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(image)
            loss = ld * dloss(logits, target) + lf * floss(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss = train_loss / max(steps, 1)

        if epoch % int(cfg["training"]["val_interval"]) == 0:
            vm = run_validation(model, val_loader, cfg, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": vm["loss"],
                "dice_WT": vm["WT"],
                "dice_TC": vm["TC"],
                "dice_ET": vm["ET"],
                "dice_mean": vm["mean"],
            }
            history.append(row)

            if vm["mean"] > best_mean:
                best_mean = vm["mean"]
                torch.save(model.state_dict(), ckpt_dir / "stage_b_best.pt")

            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                    f"val_mean={vm['mean']:.4f} (WT={vm['WT']:.4f}, TC={vm['TC']:.4f}, ET={vm['ET']:.4f})"
                )

    torch.save(model.state_dict(), ckpt_dir / "stage_b_last.pt")
    with (ckpt_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Done. Best val mean Dice: {best_mean:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
