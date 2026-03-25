from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    SqueezeDimd,
)
from monai.utils import set_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B: multi-epoch baseline training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to experiment config file",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def remap_labels(label: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    remapped = np.array(label, copy=True)
    for src, dst in mapping.items():
        remapped[label == src] = dst
    return remapped.astype(np.int64)


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


def get_required_source_splits(cfg: Dict[str, Any]) -> Dict[str, set[str]]:
    required: Dict[str, set[str]] = {}
    for split_cfg in cfg["splits"].values():
        if isinstance(split_cfg, list):
            for dataset_name in split_cfg:
                required.setdefault(dataset_name, set()).add("train")
        elif isinstance(split_cfg, dict):
            for dataset_name, source_split in split_cfg.items():
                required.setdefault(dataset_name, set()).add(str(source_split))
        else:
            raise ValueError("Each split entry must be a list or dataset->source-split mapping")
    return required


def build_dataset_dicts(
    cfg: Dict[str, Any],
    config_dir: Path,
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    data_cfg = cfg["data"]
    data_root = resolve_path(config_dir, data_cfg["root"])
    patient_lists_dir = resolve_path(config_dir, data_cfg["patient_lists_dir"])
    modality = data_cfg["modality"]
    required_source_splits = get_required_source_splits(cfg)

    all_dataset_dicts: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

    for dataset_name, dataset_info in data_cfg["datasets"].items():
        dataset_root = data_root / dataset_info["folder"]
        image_suffix = dataset_info.get("image_suffix", f"-{modality}.nii.gz")
        label_suffix = dataset_info["label_suffix"]
        list_files = dataset_info["list_files"]

        all_dataset_dicts[dataset_name] = {}
        dataset_required_splits = required_source_splits.get(dataset_name, {"train"})

        for split_name in dataset_required_splits:
            if split_name not in list_files:
                raise ValueError(
                    f"Missing list file config for dataset '{dataset_name}' split '{split_name}'"
                )

            all_dataset_dicts[dataset_name][split_name] = []
            list_path = patient_lists_dir / list_files[split_name]
            case_ids = read_case_ids(list_path)
            missing_count = 0

            for case_id in case_ids:
                case_dir = find_case_dir(dataset_root, case_id)
                if case_dir is None:
                    missing_count += 1
                    continue

                image_path = case_dir / f"{case_id}{image_suffix}"
                label_path = case_dir / f"{case_id}{label_suffix}"

                if not image_path.exists() or not label_path.exists():
                    missing_count += 1
                    continue

                all_dataset_dicts[dataset_name][split_name].append(
                    {
                        "image": str(image_path),
                        "label": str(label_path),
                        "dataset": dataset_name,
                        "case_id": case_id,
                    }
                )

            print(
                f"{dataset_name} {split_name}: "
                f"{len(all_dataset_dicts[dataset_name][split_name])} cases loaded"
                f" (missing/skipped: {missing_count})"
            )

    return all_dataset_dicts


def select_split_files(
    all_dataset_dicts: Dict[str, Dict[str, List[Dict[str, str]]]],
    split_datasets: Union[List[str], Dict[str, str]],
    split_name: str,
) -> List[Dict[str, str]]:
    files: List[Dict[str, str]] = []
    if isinstance(split_datasets, list):
        for dataset_name in split_datasets:
            if dataset_name not in all_dataset_dicts:
                raise ValueError(f"Unknown dataset '{dataset_name}' in splits.{split_name}")
            if "train" not in all_dataset_dicts[dataset_name]:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not have required source split 'train'"
                )
            files.extend(all_dataset_dicts[dataset_name]["train"])
    elif isinstance(split_datasets, dict):
        for dataset_name, source_split in split_datasets.items():
            if dataset_name not in all_dataset_dicts:
                raise ValueError(f"Unknown dataset '{dataset_name}' in splits.{split_name}")
            if source_split not in all_dataset_dicts[dataset_name]:
                raise ValueError(
                    f"Unknown source split '{source_split}' for dataset '{dataset_name}'"
                )
            files.extend(all_dataset_dicts[dataset_name][source_split])
    else:
        raise ValueError(
            f"splits.{split_name} must be a list of datasets or a dataset->split mapping"
        )
    return files


def build_transforms(cfg: Dict[str, Any]) -> Compose:
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}
    patch_size = tuple(cfg["patch"]["size"])
    num_samples = int(cfg["patch"].get("num_samples", 1))

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=lambda x: remap_labels(x, mapping)),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            SqueezeDimd(keys="label", dim=0),
        ]
    )


def verify_batch(loader: DataLoader) -> None:
    batch = next(iter(loader))
    image = batch["image"]
    label = batch["label"]

    print(f"image shape: {tuple(image.shape)}")
    print(f"label shape: {tuple(label.shape)}")
    print(f"label unique values: {torch.unique(label).tolist()}")


def build_model() -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    return images, labels


def prepare_labels_for_dicece(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4:
        return labels.unsqueeze(1)
    return labels


def run_model_forward_check(loader: DataLoader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    batch = next(iter(loader))
    images, labels = prepare_batch(batch, device)

    outputs = model(images)
    print(f"model output shape: {tuple(outputs.shape)}")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    loss = loss_fn(outputs, prepare_labels_for_dicece(labels))
    print(f"loss: {loss.item():.6f}")


def save_stage_b_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    epoch_mean_loss: float,
    cfg: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_loss": epoch_mean_loss,
            "config": cfg,
        },
        checkpoint_path,
    )


def run_stage_b_training(
    loader: DataLoader,
    cfg: Dict[str, Any],
    config_dir: Path,
    train_samples: int,
    test_samples: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    learning_rate = float(cfg["training"]["learning_rate"])
    num_epochs = int(cfg["training"].get("epochs", 10))
    log_every = int(cfg["training"].get("log_every", 20))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_dir = resolve_path(config_dir, cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = checkpoint_dir / "stage_b_latest.pt"
    best_ckpt_path = checkpoint_dir / "stage_b_best.pt"
    history_path = checkpoint_dir / "stage_b_metrics_history.json"

    best_mean_loss = float("inf")
    history: List[Dict[str, Any]] = []

    print(f"Checkpoint directory: {checkpoint_dir}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_steps = len(loader)

        for step, batch in enumerate(loader, start=1):
            images, labels = prepare_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_fn(outputs, prepare_labels_for_dicece(labels))
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            running_loss += loss_value

            if step == 1 or step % log_every == 0 or step == total_steps:
                print(
                    f"[epoch {epoch}/{num_epochs} | step {step}/{total_steps}] "
                    f"loss: {loss_value:.6f}"
                )

        epoch_mean_loss = running_loss / max(total_steps, 1)
        print(f"[epoch {epoch}/{num_epochs}] mean loss: {epoch_mean_loss:.6f}")

        save_stage_b_checkpoint(
            checkpoint_path=latest_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            epoch_mean_loss=epoch_mean_loss,
            cfg=cfg,
        )

        is_best = epoch_mean_loss < best_mean_loss
        if is_best:
            best_mean_loss = epoch_mean_loss
            save_stage_b_checkpoint(
                checkpoint_path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                epoch_mean_loss=epoch_mean_loss,
                cfg=cfg,
            )

        history.append(
            {
                "epoch": epoch,
                "mean_loss": epoch_mean_loss,
                "best_mean_loss_so_far": best_mean_loss,
                "is_best": is_best,
                "device": str(device),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "total_steps": total_steps,
            }
        )

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"Saved latest checkpoint: {latest_ckpt_path}")
        if is_best:
            print(f"Saved best checkpoint: {best_ckpt_path}")
        print(f"Updated history: {history_path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    set_determinism(seed=int(cfg.get("seed", 42)))

    all_dataset_dicts = build_dataset_dicts(cfg, config_path.parent)
    train_files = select_split_files(
        all_dataset_dicts,
        split_datasets=cfg["splits"]["train"],
        split_name="train",
    )
    test_files = select_split_files(
        all_dataset_dicts,
        split_datasets=cfg["splits"]["test"],
        split_name="test",
    )

    print(f"\nTotal train samples: {len(train_files)}")
    print(f"Total test samples: {len(test_files)}")

    if len(train_files) == 0:
        raise RuntimeError("No training samples were found. Check data.root and patient lists.")

    train_transforms = build_transforms(cfg)
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
    verify_batch(train_loader)

    print("\nVerifying model forward pass and loss...")
    run_model_forward_check(train_loader)

    print("\nRunning Stage B: multi-epoch baseline training...")
    run_stage_b_training(
        loader=train_loader,
        cfg=cfg,
        config_dir=config_path.parent,
        train_samples=len(train_files),
        test_samples=len(test_files),
    )

    print("Stage B training completed.")


if __name__ == "__main__":
    main()
