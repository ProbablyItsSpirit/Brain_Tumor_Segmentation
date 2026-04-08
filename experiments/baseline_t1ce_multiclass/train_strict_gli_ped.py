from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SqueezeDimd,
)
from monai.utils import set_determinism


def load_train_b_module():
    this_file = Path(__file__).resolve()
    train_b_path = this_file.with_name("train_B.py")
    spec = importlib.util.spec_from_file_location("train_B_module", train_b_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {train_b_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict 4c GLI+PED training with tumor-centered rejection sampling (MEN excluded)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="strict4c_gli_ped",
        help="Suffix appended to checkpoint_dir",
    )
    parser.add_argument(
        "--max-val-cases",
        type=int,
        default=30,
        help="Max number of validation cases to evaluate each epoch (0 means all)",
    )
    parser.add_argument(
        "--min-fg-ratio",
        type=float,
        default=0.01,
        help="Minimum tumor fraction required inside sampled training patch",
    )
    parser.add_argument(
        "--max-sample-tries",
        type=int,
        default=20,
        help="Number of rejection attempts before fallback to best available tumor patch",
    )
    parser.add_argument(
        "--tumor-margin",
        type=int,
        default=16,
        help="Random center margin around tumor bounding box",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["dicece", "dicefocal"],
        default="dicece",
        help="Loss function",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="",
        help="Optional comma-separated class weights for 4 classes",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Enable debug overfit mode: 20 cases, 50 epochs",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume from",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="If set, optimizer state from resume checkpoint is ignored",
    )
    return parser.parse_args()


def build_strict_transforms(
    cfg: Dict[str, Any],
    min_fg_ratio: float,
    max_sample_tries: int,
    tumor_margin: int,
):
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}
    patch_size = tuple(int(x) for x in cfg["patch"]["size"])

    def label_mapper(lbl: Any):
        remapped = torch.as_tensor(lbl).clone()
        for src, dst in mapping.items():
            remapped[remapped == src] = dst
        return remapped.to(dtype=torch.int64)

    class StrictTumorSampler:
        def __init__(self):
            self.patch = patch_size
            self.min_fg_ratio = float(min_fg_ratio)
            self.max_tries = int(max_sample_tries)
            self.margin = int(tumor_margin)

        @staticmethod
        def _pad_to_size(t: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
            _, d, h, w = t.shape
            td, th, tw = target_size
            pd = max(0, td - d)
            ph = max(0, th - h)
            pw = max(0, tw - w)
            if pd == 0 and ph == 0 and pw == 0:
                return t
            return F.pad(t, (0, pw, 0, ph, 0, pd), mode="constant", value=0)

        @staticmethod
        def _crop(
            img: torch.Tensor,
            lbl: torch.Tensor,
            z1: int,
            y1: int,
            x1: int,
            patch: tuple[int, int, int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            pd, ph, pw = patch
            return (
                img[:, z1 : z1 + pd, y1 : y1 + ph, x1 : x1 + pw],
                lbl[:, z1 : z1 + pd, y1 : y1 + ph, x1 : x1 + pw],
            )

        @staticmethod
        def _random_crop_coords(shape: tuple[int, int, int], patch: tuple[int, int, int]):
            d, h, w = shape
            pd, ph, pw = patch
            z1 = np.random.randint(0, max(d - pd, 0) + 1) if d > pd else 0
            y1 = np.random.randint(0, max(h - ph, 0) + 1) if h > ph else 0
            x1 = np.random.randint(0, max(w - pw, 0) + 1) if w > pw else 0
            return int(z1), int(y1), int(x1)

        def _tumor_center_coords(self, label_3d: torch.Tensor):
            fg = torch.nonzero(label_3d > 0, as_tuple=False)
            if fg.numel() == 0:
                return self._random_crop_coords(tuple(label_3d.shape), self.patch), False

            zmin = max(0, int(fg[:, 0].min().item()) - self.margin)
            ymin = max(0, int(fg[:, 1].min().item()) - self.margin)
            xmin = max(0, int(fg[:, 2].min().item()) - self.margin)
            zmax = min(label_3d.shape[0] - 1, int(fg[:, 0].max().item()) + self.margin)
            ymax = min(label_3d.shape[1] - 1, int(fg[:, 1].max().item()) + self.margin)
            xmax = min(label_3d.shape[2] - 1, int(fg[:, 2].max().item()) + self.margin)

            cz = np.random.randint(zmin, zmax + 1)
            cy = np.random.randint(ymin, ymax + 1)
            cx = np.random.randint(xmin, xmax + 1)

            pd, ph, pw = self.patch
            d, h, w = label_3d.shape
            z1 = max(0, min(int(cz) - pd // 2, d - pd))
            y1 = max(0, min(int(cy) - ph // 2, h - ph))
            x1 = max(0, min(int(cx) - pw // 2, w - pw))
            return (z1, y1, x1), True

        def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
            d = dict(data)
            img = self._pad_to_size(d["image"], self.patch)
            lbl = self._pad_to_size(d["label"], self.patch)

            best_img = None
            best_lbl = None
            best_ratio = -1.0

            for _ in range(self.max_tries):
                (z1, y1, x1), had_fg = self._tumor_center_coords(lbl[0])
                if not had_fg:
                    z1, y1, x1 = self._random_crop_coords(tuple(lbl[0].shape), self.patch)

                c_img, c_lbl = self._crop(img, lbl, z1, y1, x1, self.patch)
                fg_ratio = float((c_lbl > 0).float().mean().item())

                if fg_ratio > best_ratio:
                    best_ratio = fg_ratio
                    best_img = c_img
                    best_lbl = c_lbl

                if fg_ratio >= self.min_fg_ratio:
                    d["image"] = c_img
                    d["label"] = c_lbl
                    return d

            # Rejection fallback: choose the best tumor-containing patch found.
            d["image"] = best_img if best_img is not None else img
            d["label"] = best_lbl if best_lbl is not None else lbl
            return d

    sampler = StrictTumorSampler()

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=label_mapper),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
            sampler,
            SqueezeDimd(keys="label", dim=0),
        ]
    )


def main() -> None:
    tb = load_train_b_module()
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = tb.load_config(config_path)
    tb.apply_local_path_overrides(cfg, config_path.parent)

    # Force target setup from requested plan: 4c + GLI + PED only.
    num_classes = tb.apply_label_setup(cfg, "4c")
    cfg["splits"]["train"] = {"GLI": "train", "PED": "train"}

    # Keep validation trying GLI test first, then fallback if labels are unavailable.
    val_mapping = {"GLI": "test"}

    # Force patch size requested in plan.
    cfg["patch"]["size"] = [128, 128, 128]

    if args.overfit:
        cfg["training"]["epochs"] = 50

    if args.checkpoint_suffix:
        base_ckpt_dir = str(cfg["training"]["checkpoint_dir"]).rstrip("/\\")
        cfg["training"]["checkpoint_dir"] = f"{base_ckpt_dir}_{args.checkpoint_suffix}"

    set_determinism(seed=int(cfg.get("seed", 42)))

    cfg_for_loading = dict(cfg)
    cfg_for_loading["splits"] = dict(cfg["splits"])
    cfg_for_loading["splits"]["val"] = val_mapping

    all_dataset_dicts = tb.build_dataset_dicts(cfg_for_loading, config_path.parent)
    train_files = tb.select_split_files(
        all_dataset_dicts,
        split_datasets=cfg["splits"]["train"],
        split_name="train",
    )
    test_files = tb.select_split_files(
        all_dataset_dicts,
        split_datasets=cfg["splits"]["test"],
        split_name="test",
    )
    val_files = tb.select_split_files(
        all_dataset_dicts,
        split_datasets=val_mapping,
        split_name="val",
    )

    if len(val_files) == 0:
        print("[train_strict_gli_ped] No usable GLI test labels found. Falling back to GLI train for validation.")
        val_files = tb.select_split_files(
            all_dataset_dicts,
            split_datasets={"GLI": "train"},
            split_name="val",
        )

    if args.overfit:
        overfit_n = min(20, len(train_files))
        train_files = train_files[:overfit_n]
        val_files = train_files.copy()
        print(f"[overfit mode] Using {len(train_files)} train cases and same set for validation.")

    if args.max_val_cases > 0:
        val_files = val_files[: args.max_val_cases]

    print(f"\nTotal train samples: {len(train_files)}")
    print(f"Total test samples: {len(test_files)}")
    print(f"Total val samples: {len(val_files)}")
    print("Training mode: strict GLI+PED")
    print("Label setup: 4c")
    print(f"Patch size: {tuple(cfg['patch']['size'])}")
    print(
        "Strict sampling: "
        f"min_fg_ratio={args.min_fg_ratio}, "
        f"max_sample_tries={args.max_sample_tries}, "
        f"tumor_margin={args.tumor_margin}"
    )

    if len(train_files) == 0:
        raise RuntimeError("No training samples found for GLI+PED.")

    train_transforms = build_strict_transforms(
        cfg=cfg,
        min_fg_ratio=args.min_fg_ratio,
        max_sample_tries=args.max_sample_tries,
        tumor_margin=args.tumor_margin,
    )
    train_ds = Dataset(data=train_files, transform=train_transforms)

    num_workers = int(cfg["dataloader"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["dataloader"]["batch_size"]),
        shuffle=bool(cfg["dataloader"].get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    print("\nVerifying one training batch...")
    tb.verify_batch(train_loader)

    parsed_class_weights = tb.parse_class_weights(args.class_weights, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sanity_loss_fn = tb.build_loss_function(args.loss_type, parsed_class_weights, num_classes, cfg, device)

    print("\nVerifying model forward pass and loss...")
    tb.run_model_forward_check(train_loader, sanity_loss_fn, out_channels=num_classes)

    val_loader = None
    if len(val_files) > 0:
        val_ds = Dataset(data=val_files, transform=tb.build_eval_transforms(cfg))
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    print("\nRunning strict Stage B training...")
    tb.run_stage_b_training(
        loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        config_dir=config_path.parent,
        train_samples=len(train_files),
        test_samples=len(test_files),
        val_samples=len(val_files),
        loss_type=args.loss_type,
        class_weights=parsed_class_weights,
        num_classes=num_classes,
        resume_checkpoint=args.resume_checkpoint,
        reset_optimizer=args.reset_optimizer,
    )

    print("Strict GLI+PED training completed.")


if __name__ == "__main__":
    main()
