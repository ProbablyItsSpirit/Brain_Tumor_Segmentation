from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SpatialPadd,
    SqueezeDimd,
)
from monai.utils import set_determinism


def load_train_b_module():
    this_file = Path(__file__).resolve()
    train_b_path = this_file.parent.parent / "train_B.py"
    spec = importlib.util.spec_from_file_location("train_B_module", train_b_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {train_b_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple FLAIR-only GLI baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="flair_gli_baseline",
        help="Suffix appended to checkpoint_dir",
    )
    parser.add_argument(
        "--val-holdout-count",
        type=int,
        default=30,
        help="Number of GLI train cases to hold out for validation",
    )
    parser.add_argument(
        "--overfit-cases",
        type=int,
        default=0,
        help="If >0, train/validate on a tiny subset for debugging",
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=0,
        help="Optional epoch override for overfit mode (default 100 when overfit-cases > 0)",
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


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_case_ids(list_file: Path) -> List[str]:
    case_ids: List[str] = []
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            case_id = line.strip()
            if case_id and not case_id.startswith("#"):
                case_ids.append(case_id)
    return case_ids


def find_case_dir(dataset_root: Path, case_id: str) -> Path | None:
    candidates = [
        dataset_root / "train" / case_id,
        dataset_root / "val" / case_id,
        dataset_root / "train_additional" / case_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_case_dicts(cfg: Dict[str, Any], config_dir: Path) -> List[Dict[str, str]]:
    data_cfg = cfg["data"]
    data_root = resolve_path(config_dir, data_cfg["root"])
    patient_lists_dir = resolve_path(config_dir, data_cfg["patient_lists_dir"])
    modality = data_cfg.get("modality", "t2f")

    repo_root = config_dir.parent.parent.resolve()
    default_data_root = repo_root / "BraTS-2024-Complete"
    default_patient_lists_dir = repo_root / "patient_lists"

    if not data_root.exists() and default_data_root.exists():
        print(f"[path override] data.root not found: {data_root}")
        print(f"[path override] using local repo data root: {default_data_root}")
        data_root = default_data_root

    if not patient_lists_dir.exists() and default_patient_lists_dir.exists():
        print(f"[path override] data.patient_lists_dir not found: {patient_lists_dir}")
        print(f"[path override] using local repo patient lists: {default_patient_lists_dir}")
        patient_lists_dir = default_patient_lists_dir

    dataset_info = data_cfg["datasets"]["GLI"]
    dataset_root = data_root / dataset_info["folder"]
    list_path = patient_lists_dir / dataset_info["list_files"]["train"]
    case_ids = read_case_ids(list_path)

    loaded: List[Dict[str, str]] = []
    missing_count = 0
    for case_id in case_ids:
        case_dir = find_case_dir(dataset_root, case_id)
        if case_dir is None:
            missing_count += 1
            continue

        image_path = case_dir / f"{case_id}-{modality}.nii.gz"
        label_path = case_dir / f"{case_id}{dataset_info['label_suffix']}"
        if not image_path.exists() or not label_path.exists():
            missing_count += 1
            continue

        loaded.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "dataset": "GLI",
                "case_id": case_id,
            }
        )

    print(f"GLI train: {len(loaded)} cases loaded (missing/skipped: {missing_count})")
    return loaded


def remap_labels(label: Any, mapping: Dict[int, int]):
    if torch.is_tensor(label):
        remapped = label.clone()
        for src, dst in mapping.items():
            remapped[label == src] = dst
        return remapped.to(dtype=torch.int64)

    remapped = np.asarray(label).copy()
    for src, dst in mapping.items():
        remapped[remapped == src] = dst
    return remapped.astype(np.int64)


def build_train_transforms(cfg: Dict[str, Any]) -> Compose:
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}
    patch_size = tuple(int(x) for x in cfg["patch"]["size"])

    def label_mapper(lbl: Any):
        return remap_labels(lbl, mapping)

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=label_mapper),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
            # Remove empty margins before cropping so the fixed-size training patch
            # stays focused on the brain instead of mostly background voxels.
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
            SqueezeDimd(keys="label", dim=0),
        ]
    )


def build_inference_transforms(cfg: Dict[str, Any]) -> Compose:
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}

    def label_mapper(lbl: Any):
        return remap_labels(lbl, mapping)

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=label_mapper),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
        ]
    )


def main() -> None:
    tb = load_train_b_module()
    args = parse_args()

    config_path = Path(args.config).resolve()
    cfg = tb.load_config(config_path)
    tb.apply_local_path_overrides(cfg, config_path.parent)

    num_classes = tb.apply_label_setup(cfg, "4c")
    cfg["splits"]["train"] = {"GLI": "train"}
    cfg["patch"]["size"] = [128, 128, 128]

    if args.overfit_cases > 0:
        cfg["training"]["epochs"] = int(args.overfit_epochs) if args.overfit_epochs > 0 else 100
    elif args.overfit_epochs > 0:
        cfg["training"]["epochs"] = int(args.overfit_epochs)

    if args.checkpoint_suffix:
        base_ckpt_dir = str(cfg["training"]["checkpoint_dir"]).rstrip("/\\")
        cfg["training"]["checkpoint_dir"] = f"{base_ckpt_dir}_{args.checkpoint_suffix}"

    set_determinism(seed=int(cfg.get("seed", 42)))

    train_files = build_case_dicts(cfg, config_path.parent)
    rng = random.Random(int(cfg.get("seed", 42)))
    rng.shuffle(train_files)

    holdout_count = min(args.val_holdout_count, len(train_files))
    val_files = copy.deepcopy(train_files[:holdout_count])
    train_files = train_files[holdout_count:]

    if args.overfit_cases > 0:
        overfit_n = min(args.overfit_cases, len(train_files))
        train_files = train_files[:overfit_n]
        val_files = copy.deepcopy(train_files)
        print(f"[overfit mode] Using {len(train_files)} train cases and same set for validation.")

    print(f"\nTotal train samples: {len(train_files)}")
    print(f"Total val samples: {len(val_files)}")
    print("Training mode: FLAIR-only GLI baseline")
    print(f"Patch size: {tuple(cfg['patch']['size'])}")
    print(f"Label setup: 4c (num_classes={num_classes})")

    checkpoint_dir = resolve_path(config_path.parent, cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    holdout_file = checkpoint_dir / "holdout_cases.json"
    with holdout_file.open("w", encoding="utf-8") as f:
        json.dump(val_files, f, indent=2)
    print(f"Saved holdout case list: {holdout_file}")

    train_transforms = build_train_transforms(cfg)
    val_transforms = build_inference_transforms(cfg)

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    num_workers = int(cfg["dataloader"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["dataloader"]["batch_size"]),
        shuffle=bool(cfg["dataloader"].get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    print("\nVerifying one training batch...")
    batch = next(iter(train_loader))
    print(f"image shape: {tuple(batch['image'].shape)}")
    print(f"label shape: {tuple(batch['label'].shape)}")
    print(f"label unique values: {torch.unique(batch['label']).tolist()}")

    parsed_class_weights = tb.parse_class_weights("", num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sanity_loss_fn = tb.build_loss_function("dicece", parsed_class_weights, num_classes, cfg, device)

    print("\nVerifying model forward pass and loss...")
    tb.run_model_forward_check(train_loader, sanity_loss_fn, out_channels=num_classes)

    print("\nRunning FLAIR-only GLI Stage B training...")
    tb.run_stage_b_training(
        loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        config_dir=config_path.parent,
        train_samples=len(train_files),
        test_samples=0,
        val_samples=len(val_files),
        loss_type="dicece",
        class_weights=parsed_class_weights,
        num_classes=num_classes,
        resume_checkpoint=args.resume_checkpoint,
        reset_optimizer=args.reset_optimizer,
    )

    print("FLAIR-only GLI baseline training completed.")


if __name__ == "__main__":
    main()
