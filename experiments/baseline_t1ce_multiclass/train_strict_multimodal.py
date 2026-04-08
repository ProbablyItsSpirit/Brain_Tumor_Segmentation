from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet
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


MODALITIES = ["t1c", "t2w", "t2f"]
DATASET_IMAGE_SUFFIXES = {
    "GLI": {
        "t1c": "-t1c.nii.gz",
        "t2w": "-t2w.nii.gz",
        "t2f": "-t2f.nii.gz",
    },
    "PED": {
        "t1c": "-t1c.nii.gz",
        "t2w": "-t2w.nii.gz",
        "t2f": "-t2f.nii.gz",
    },
}


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
        description="Strict multimodal 4c GLI+PED training with tumor-centered rejection sampling (MEN excluded)."
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
        default="strictmm_4c_gli_ped",
        help="Suffix appended to checkpoint_dir",
    )
    parser.add_argument(
        "--val-holdout-count",
        type=int,
        default=30,
        help="Number of GLI train cases to hold out for validation",
    )
    parser.add_argument(
        "--min-fg-ratio",
        type=float,
        default=0.02,
        help="Minimum tumor fraction required inside sampled training patch",
    )
    parser.add_argument(
        "--max-sample-tries",
        type=int,
        default=30,
        help="Number of rejection attempts before fallback to best available tumor patch",
    )
    parser.add_argument(
        "--tumor-margin",
        type=int,
        default=24,
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
        "--overfit-cases",
        type=int,
        default=0,
        help="If >0, train/validate on a tiny subset for pipeline debugging",
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=0,
        help="Optional epoch override for overfit mode (default 50 when overfit-cases > 0)",
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


def remap_with_mapping(label: Any, mapping: Dict[int, int]):
    return remap_labels(label, mapping)


def apply_label_setup(cfg: Dict[str, Any], label_setup: str) -> int:
    if label_setup == "3c":
        cfg["data"]["label_mapping"] = {
            0: 0,
            1: 1,
            2: 2,
            3: 2,
            4: 2,
        }
    else:
        cfg["data"]["label_mapping"] = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 3,
        }
    return int(max(cfg["data"]["label_mapping"].values())) + 1


def build_multimodal_case_dicts(
    cfg: Dict[str, Any],
    config_dir: Path,
    modalities: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    data_cfg = cfg["data"]
    data_root = resolve_path(config_dir, data_cfg["root"])
    patient_lists_dir = resolve_path(config_dir, data_cfg["patient_lists_dir"])
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

    all_cases: Dict[str, List[Dict[str, Any]]] = {}

    for dataset_name in ("GLI", "PED"):
        dataset_info = data_cfg["datasets"][dataset_name]
        dataset_root = data_root / dataset_info["folder"]
        list_path = patient_lists_dir / dataset_info["list_files"]["train"]
        case_ids = read_case_ids(list_path)
        missing_count = 0
        loaded: List[Dict[str, Any]] = []

        for case_id in case_ids:
            case_dir = find_case_dir(dataset_root, case_id)
            if case_dir is None:
                missing_count += 1
                continue

            image_paths: Dict[str, str] = {}
            missing_image = False
            for modality in modalities:
                suffix = DATASET_IMAGE_SUFFIXES[dataset_name].get(modality)
                if suffix is None:
                    missing_image = True
                    break
                image_path = case_dir / f"{case_id}{suffix}"
                if not image_path.exists():
                    missing_image = True
                    break
                image_paths[f"image_{modality}"] = str(image_path)

            label_suffix = dataset_info["label_suffix"]
            label_path = case_dir / f"{case_id}{label_suffix}"
            if missing_image or not label_path.exists():
                missing_count += 1
                continue

            loaded.append(
                {
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "label": str(label_path),
                    **image_paths,
                }
            )

        print(f"{dataset_name} train: {len(loaded)} cases loaded (missing/skipped: {missing_count})")
        all_cases[dataset_name] = loaded

    return all_cases


class StackModalitiesd:
    def __init__(self, image_keys: Sequence[str]):
        self.image_keys = list(image_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d["image"] = torch.cat([d[key] for key in self.image_keys], dim=0)
        for key in self.image_keys:
            d.pop(key, None)
        return d


class StrictTumorSampler:
    def __init__(self, patch_size: tuple[int, int, int], min_fg_ratio: float, max_tries: int, tumor_margin: int):
        self.patch = patch_size
        self.min_fg_ratio = float(min_fg_ratio)
        self.max_tries = int(max_tries)
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

        d["image"] = best_img if best_img is not None else img
        d["label"] = best_lbl if best_lbl is not None else lbl
        return d


def build_transforms(
    cfg: Dict[str, Any],
    modalities: Sequence[str],
    min_fg_ratio: float,
    max_sample_tries: int,
    tumor_margin: int,
    training: bool = True,
) -> Compose:
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}
    image_keys = [f"image_{modality}" for modality in modalities]
    all_image_keys = list(image_keys) + ["label"]

    def label_mapper(lbl: Any):
        remapped = torch.as_tensor(lbl).clone()
        for src, dst in mapping.items():
            remapped[remapped == src] = dst
        return remapped.to(dtype=torch.int64)

    transforms = [
        LoadImaged(keys=all_image_keys),
        EnsureChannelFirstd(keys=all_image_keys),
        Orientationd(keys=all_image_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
        Lambdad(keys="label", func=label_mapper),
        EnsureTyped(keys=image_keys, dtype=torch.float32),
        EnsureTyped(keys="label", dtype=torch.int64),
        StackModalitiesd(image_keys),
    ]

    if training:
        transforms.extend(
            [
                StrictTumorSampler(
                    patch_size=tuple(int(x) for x in cfg["patch"]["size"]),
                    min_fg_ratio=min_fg_ratio,
                    max_tries=max_sample_tries,
                    tumor_margin=tumor_margin,
                ),
                SqueezeDimd(keys="label", dim=0),
            ]
        )

    return Compose(transforms)


def build_model(out_channels: int, in_channels: int) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def main() -> None:
    tb = load_train_b_module()
    args = parse_args()
    modalities = MODALITIES

    config_path = Path(args.config).resolve()
    cfg = tb.load_config(config_path)
    tb.apply_local_path_overrides(cfg, config_path.parent)

    num_classes = apply_label_setup(cfg, "4c")
    cfg["splits"]["train"] = {"GLI": "train", "PED": "train"}
    cfg["patch"]["size"] = [128, 128, 128]

    if args.overfit_cases > 0:
        cfg["training"]["epochs"] = int(args.overfit_epochs) if args.overfit_epochs > 0 else 50
    elif args.overfit_epochs > 0:
        cfg["training"]["epochs"] = int(args.overfit_epochs)

    if args.checkpoint_suffix:
        base_ckpt_dir = str(cfg["training"]["checkpoint_dir"]).rstrip("/\\")
        cfg["training"]["checkpoint_dir"] = f"{base_ckpt_dir}_{args.checkpoint_suffix}"

    set_determinism(seed=int(cfg.get("seed", 42)))

    all_cases = build_multimodal_case_dicts(cfg, config_path.parent, modalities)
    gli_cases = list(all_cases["GLI"])
    ped_cases = list(all_cases["PED"])

    rng = random.Random(int(cfg.get("seed", 42)))
    rng.shuffle(gli_cases)

    holdout_count = min(args.val_holdout_count, len(gli_cases))
    val_cases = gli_cases[:holdout_count]
    train_cases = gli_cases[holdout_count:] + ped_cases

    if args.overfit_cases > 0:
        overfit_n = min(args.overfit_cases, len(train_cases))
        train_cases = train_cases[:overfit_n]
        val_cases = train_cases.copy()
        print(f"[overfit mode] Using {len(train_cases)} train cases and same set for validation.")

    print(f"\nTotal train samples: {len(train_cases)}")
    print(f"Total val samples: {len(val_cases)}")
    print(f"Training mode: strict multimodal {'+'.join(modalities)}; GLI+PED only")
    print(f"Label setup: 4c (num_classes={num_classes})")
    print(f"Patch size: {tuple(cfg['patch']['size'])}")
    print(
        "Strict sampling: "
        f"min_fg_ratio={args.min_fg_ratio}, "
        f"max_sample_tries={args.max_sample_tries}, "
        f"tumor_margin={args.tumor_margin}"
    )

    checkpoint_dir = resolve_path(config_path.parent, cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_case_file = checkpoint_dir / "strict_val_cases.json"
    with val_case_file.open("w", encoding="utf-8") as f:
        json.dump(val_cases, f, indent=2)
    print(f"Saved validation case list: {val_case_file}")

    train_transforms = build_transforms(
        cfg=cfg,
        modalities=modalities,
        min_fg_ratio=args.min_fg_ratio,
        max_sample_tries=args.max_sample_tries,
        tumor_margin=args.tumor_margin,
        training=True,
    )
    val_transforms = build_transforms(
        cfg=cfg,
        modalities=modalities,
        min_fg_ratio=args.min_fg_ratio,
        max_sample_tries=args.max_sample_tries,
        tumor_margin=args.tumor_margin,
        training=False,
    )

    train_ds = Dataset(data=train_cases, transform=train_transforms)
    val_ds = Dataset(data=val_cases, transform=val_transforms)

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

    parsed_class_weights = tb.parse_class_weights(args.class_weights, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def multimodal_build_model(out_channels: int):
        return build_model(out_channels=out_channels, in_channels=len(modalities)).to(device)

    tb.build_model = multimodal_build_model  # type: ignore[assignment]

    sanity_loss_fn = tb.build_loss_function(args.loss_type, parsed_class_weights, num_classes, cfg, device)
    print("\nVerifying model forward pass and loss...")
    tb.run_model_forward_check(train_loader, sanity_loss_fn, out_channels=num_classes)

    print("\nRunning strict multimodal Stage B training...")
    tb.run_stage_b_training(
        loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        config_dir=config_path.parent,
        train_samples=len(train_cases),
        test_samples=len(val_cases),
        val_samples=len(val_cases),
        loss_type=args.loss_type,
        class_weights=parsed_class_weights,
        num_classes=num_classes,
        resume_checkpoint=args.resume_checkpoint,
        reset_optimizer=args.reset_optimizer,
    )

    print("Strict multimodal GLI+PED training completed.")


if __name__ == "__main__":
    main()
