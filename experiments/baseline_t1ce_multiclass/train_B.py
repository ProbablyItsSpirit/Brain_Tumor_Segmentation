from __future__ import annotations

import argparse
import copy
import inspect
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    SpatialPadd,
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
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume from (e.g., stage_b_latest.pt)",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="If set, optimizer state from resume checkpoint is ignored",
    )
    parser.add_argument(
        "--max-val-cases",
        type=int,
        default=30,
        help="Max number of validation cases to evaluate each epoch (0 means all)",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        choices=["gli", "mixed"],
        default="gli",
        help="gli: train only on GLI | mixed: train on GLI+PED+MEN",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["dicece", "dicefocal"],
        default="dicece",
        help="Loss function type for class imbalance handling",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="",
        help="Optional comma-separated class weights for 4 classes, e.g. 0.05,0.2,0.25,0.5",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="",
        help="Optional suffix appended to checkpoint_dir for side-by-side experiment runs",
    )
    parser.add_argument(
        "--label-setup",
        type=str,
        choices=["4c", "3c"],
        default="4c",
        help="4c: keep 4 output classes (0..3). 3c: merge labels 3/4 into class 2 (0..2).",
    )
    parser.add_argument(
        "--val-mode",
        type=str,
        choices=["mixed", "gli", "config"],
        default="mixed",
        help="Validation source: mixed(GLI+PED+MEN), gli(GLI only), or config(use splits.val)",
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
        help="Optional epoch override for overfit mode (default 60 when overfit-cases > 0)",
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


def apply_local_path_overrides(cfg: Dict[str, Any], config_dir: Path) -> None:
    data_cfg = cfg["data"]
    repo_root = config_dir.parent.parent

    current_data_root = resolve_path(config_dir, data_cfg["root"])
    if not current_data_root.exists():
        fallback_data_root = repo_root / "BraTS-2024-Complete"
        if fallback_data_root.exists():
            data_cfg["root"] = str(fallback_data_root)
            print(f"[path override] data.root -> {fallback_data_root}")

    current_lists_root = resolve_path(config_dir, data_cfg["patient_lists_dir"])
    if not current_lists_root.exists():
        fallback_lists_root = repo_root / "patient_lists"
        if fallback_lists_root.exists():
            data_cfg["patient_lists_dir"] = str(fallback_lists_root)
            print(f"[path override] data.patient_lists_dir -> {fallback_lists_root}")


def apply_label_setup(cfg: Dict[str, Any], label_setup: str) -> int:
    if label_setup == "3c":
        cfg["data"]["label_mapping"] = {
            0: 0,
            1: 1,
            2: 2,
            3: 2,
            4: 2,
        }
        weights = cfg.get("training", {}).get("loss", {}).get("class_weights")
        if isinstance(weights, list) and len(weights) == 4:
            cfg["training"]["loss"]["class_weights"] = [
                float(weights[0]),
                float(weights[1]),
                float(max(weights[2], weights[3])),
            ]
    else:
        cfg["data"]["label_mapping"] = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 3,
        }
    return int(max(cfg["data"]["label_mapping"].values())) + 1


def read_case_ids(list_file: Path) -> List[str]:
    case_ids: List[str] = []
    with list_file.open("r", encoding="utf-8") as f:
        for line in f:
            case_id = line.strip()
            if case_id and not case_id.startswith("#"):
                case_ids.append(case_id)
    return case_ids


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
    label_mapper = partial(remap_with_mapping, mapping=mapping)
    patch_size = tuple(cfg["patch"]["size"])
    num_samples = int(cfg["patch"].get("num_samples", 1))
    pos = float(cfg["patch"].get("pos", 3))
    neg = float(cfg["patch"].get("neg", 1))

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=label_mapper),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
            # Some mixed-dataset volumes are shallower than patch depth (e.g., 108 < 128).
            # Pad first so random crop ROI is always valid.
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            SqueezeDimd(keys="label", dim=0),
        ]
    )


def build_eval_transforms(cfg: Dict[str, Any]) -> Compose:
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}
    label_mapper = partial(remap_with_mapping, mapping=mapping)
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=label_mapper),
            EnsureTyped(keys="image", dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
        ]
    )


def verify_batch(loader: DataLoader) -> None:
    batch = next(iter(loader))
    image = batch["image"]
    label = batch["label"]

    print(f"image shape: {tuple(image.shape)}")
    print(f"label shape: {tuple(label.shape)}")
    print(f"label unique values: {torch.unique(label).tolist()}")


def build_model(out_channels: int) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
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


def parse_class_weights(raw: str, num_classes: int) -> List[float] | None:
    if not raw.strip():
        return None
    weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(weights) != num_classes:
        raise ValueError(
            f"--class-weights must provide exactly {num_classes} values for classes 0..{num_classes - 1}"
        )
    return weights


def build_loss_function(
    loss_type: str,
    class_weights: List[float] | None,
    num_classes: int,
    cfg: Dict[str, Any],
    device: torch.device,
):
    loss_cfg = cfg.get("training", {}).get("loss", {})

    configured_weights = loss_cfg.get("class_weights")
    if class_weights is None and isinstance(configured_weights, list):
        class_weights = [float(x) for x in configured_weights]

    weight_tensor = None
    if class_weights is not None:
        if len(class_weights) != num_classes:
            raise ValueError(
                f"Class weights must have {num_classes} values for classes 0..{num_classes - 1}"
            )
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    lambda_dice = float(loss_cfg.get("lambda_dice", 0.5))
    lambda_ce = float(loss_cfg.get("lambda_ce", 0.5))
    lambda_focal = float(loss_cfg.get("lambda_focal", 0.5))
    focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))

    if loss_type == "dicefocal":
        focal_kwargs = {
            "to_onehot_y": True,
            "softmax": True,
            "gamma": focal_gamma,
            "lambda_dice": lambda_dice,
            "lambda_focal": lambda_focal,
        }
        if weight_tensor is not None:
            focal_sig = inspect.signature(DiceFocalLoss.__init__)
            if "focal_weight" in focal_sig.parameters:
                focal_kwargs["focal_weight"] = weight_tensor
            elif "weight" in focal_sig.parameters:
                focal_kwargs["weight"] = weight_tensor
        return DiceFocalLoss(**focal_kwargs)

    dicece_kwargs = {
        "to_onehot_y": True,
        "softmax": True,
        "lambda_ce": lambda_ce,
        "lambda_dice": lambda_dice,
    }
    if weight_tensor is not None:
        dicece_sig = inspect.signature(DiceCELoss.__init__)
        if "ce_weight" in dicece_sig.parameters:
            dicece_kwargs["ce_weight"] = weight_tensor
        elif "weight" in dicece_sig.parameters:
            dicece_kwargs["weight"] = weight_tensor

    return DiceCELoss(**dicece_kwargs)


def dice_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> tuple[float | None, bool]:
    pred_c = (pred == class_id).float()
    target_c = (target == class_id).float()
    target_sum = target_c.sum()
    if target_sum.item() == 0:
        return None, False

    denominator = pred_c.sum() + target_sum
    intersection = (pred_c * target_c).sum()
    return float((2.0 * intersection / denominator).item()), True


def evaluate_on_validation(
    model: UNet,
    val_loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, int, int],
    num_classes: int,
) -> Dict[str, Any]:
    model_was_training = model.training
    model.eval()

    case_scores: List[float] = []
    class_scores: Dict[int, List[float]] = {cid: [] for cid in range(1, num_classes)}
    valid_count_per_class: Dict[int, int] = {cid: 0 for cid in range(1, num_classes)}
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if labels.ndim == 5 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )
            preds = torch.argmax(logits, dim=1)

            case_valid_values: List[float] = []
            for cid in range(1, num_classes):
                d, is_valid = dice_for_class(preds[0], labels[0], cid)
                if is_valid and d is not None:
                    class_scores[cid].append(d)
                    valid_count_per_class[cid] += 1
                    case_valid_values.append(d)

            if case_valid_values:
                case_scores.append(float(np.mean(case_valid_values)))

    if model_was_training:
        model.train()

    mean_case = float(np.mean(case_scores)) if case_scores else 0.0
    out: Dict[str, float] = {"mean_dice_no_bg": mean_case}
    for cid in range(1, num_classes):
        out[f"dice_class_{cid}"] = float(np.mean(class_scores[cid])) if class_scores[cid] else 0.0
    out["valid_case_count"] = len(case_scores)
    out["valid_count_per_class"] = {f"class_{cid}": valid_count_per_class[cid] for cid in range(1, num_classes)}
    return out


def run_model_forward_check(loader: DataLoader, loss_fn, out_channels: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(out_channels=out_channels).to(device)

    batch = next(iter(loader))
    images, labels = prepare_batch(batch, device)

    outputs = model(images)
    print(f"model output shape: {tuple(outputs.shape)}")

    loss = loss_fn(outputs, prepare_labels_for_dicece(labels))
    print(f"loss: {loss.item():.6f}")


def save_stage_b_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: UNet,
    optimizer: torch.optim.Optimizer,
    epoch_mean_loss: float,
    val_mean_dice: float | None,
    cfg: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "mean_loss": epoch_mean_loss,
            "val_mean_dice": val_mean_dice,
            "config": cfg,
        },
        checkpoint_path,
    )


def run_stage_b_training(
    loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: Dict[str, Any],
    config_dir: Path,
    train_samples: int,
    test_samples: int,
    val_samples: int,
    loss_type: str,
    class_weights: List[float] | None,
    num_classes: int,
    resume_checkpoint: str = "",
    reset_optimizer: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(out_channels=num_classes).to(device)
    loss_fn = build_loss_function(loss_type, class_weights, num_classes, cfg, device)

    learning_rate = float(cfg["training"]["learning_rate"])
    num_epochs = int(cfg["training"].get("epochs", 10))
    log_every = int(cfg["training"].get("log_every", 20))
    val_interval = int(cfg["training"].get("val_interval", 1))
    patch_size = tuple(cfg["patch"]["size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_dir = resolve_path(config_dir, cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = checkpoint_dir / "stage_b_latest.pt"
    best_ckpt_path = checkpoint_dir / "stage_b_best.pt"
    history_path = checkpoint_dir / "stage_b_metrics_history.json"

    best_mean_loss = float("inf")
    best_val_mean_dice = float("-inf")
    history: List[Dict[str, Any]] = []
    start_epoch = 1

    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            loaded_history = json.load(f)
            if isinstance(loaded_history, list):
                history = loaded_history
                if history:
                    best_mean_loss = float(min(item["mean_loss"] for item in history))
                    val_values = [item.get("val_mean_dice") for item in history if item.get("val_mean_dice") is not None]
                    if val_values:
                        best_val_mean_dice = float(max(val_values))

    if resume_checkpoint:
        resume_path = resolve_path(config_dir, resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

        if (not reset_optimizer) and ("optimizer_state_dict" in state):
            optimizer.load_state_dict(state["optimizer_state_dict"])

        last_epoch = int(state.get("epoch", 0))
        start_epoch = last_epoch + 1
        best_mean_loss = min(best_mean_loss, float(state.get("mean_loss", float("inf"))))
        if state.get("val_mean_dice") is not None:
            best_val_mean_dice = max(best_val_mean_dice, float(state["val_mean_dice"]))
        print(f"Resuming from checkpoint: {resume_path}")
        print(f"Resumed at epoch {last_epoch}; continuing from epoch {start_epoch}")

    if start_epoch > num_epochs:
        print(
            f"Nothing to train: start_epoch ({start_epoch}) is greater than configured epochs ({num_epochs})."
        )
        return

    print(f"Checkpoint directory: {checkpoint_dir}")

    for epoch in range(start_epoch, num_epochs + 1):
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

        should_validate = val_loader is not None and val_samples > 0 and val_interval > 0 and (epoch % val_interval == 0)
        val_metrics = None
        val_mean_dice = None
        if should_validate:
            val_metrics = evaluate_on_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                patch_size=patch_size,
                num_classes=num_classes,
            )
            val_mean_dice = val_metrics["mean_dice_no_bg"]
            print(f"[epoch {epoch}/{num_epochs}] val mean Dice (foreground classes): {val_mean_dice:.6f}")
            class_line = ", ".join(
                [f"C{cid}={val_metrics.get(f'dice_class_{cid}', 0.0):.4f}" for cid in range(1, num_classes)]
            )
            print(f"[epoch {epoch}/{num_epochs}] val per-class Dice: {class_line}")
            print(
                f"[epoch {epoch}/{num_epochs}] valid supports: "
                f"{val_metrics.get('valid_count_per_class', {})}, "
                f"valid_cases={val_metrics.get('valid_case_count', 0)}"
            )

        save_stage_b_checkpoint(
            checkpoint_path=latest_ckpt_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            epoch_mean_loss=epoch_mean_loss,
            val_mean_dice=val_mean_dice,
            cfg=cfg,
        )

        if val_mean_dice is not None:
            is_best = val_mean_dice > best_val_mean_dice
            if is_best:
                best_val_mean_dice = val_mean_dice
            selection_metric = "val_mean_dice"
        else:
            is_best = epoch_mean_loss < best_mean_loss
            selection_metric = "mean_loss"

        if is_best:
            best_mean_loss = epoch_mean_loss
            save_stage_b_checkpoint(
                checkpoint_path=best_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                epoch_mean_loss=epoch_mean_loss,
                val_mean_dice=val_mean_dice,
                cfg=cfg,
            )

        history.append(
            {
                "epoch": epoch,
                "mean_loss": epoch_mean_loss,
                "best_mean_loss_so_far": best_mean_loss,
                "val_mean_dice": val_mean_dice,
                "best_val_mean_dice_so_far": None if best_val_mean_dice == float("-inf") else best_val_mean_dice,
                "val_per_class_dice": None
                if val_metrics is None
                else {f"dice_class_{cid}": val_metrics.get(f"dice_class_{cid}", 0.0) for cid in range(1, num_classes)},
                "valid_case_count": None if val_metrics is None else val_metrics.get("valid_case_count", 0),
                "valid_count_per_class": None if val_metrics is None else val_metrics.get("valid_count_per_class", {}),
                "selection_metric": selection_metric,
                "is_best": is_best,
                "device": str(device),
                "train_samples": train_samples,
                "test_samples": test_samples,
                "val_samples": val_samples,
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
    apply_local_path_overrides(cfg, config_path.parent)

    num_classes = apply_label_setup(cfg, args.label_setup)

    if args.train_mode == "mixed":
        cfg["splits"]["train"] = {"GLI": "train", "PED": "train", "MEN": "train"}

    if args.overfit_cases > 0:
        # Keep overfit experiments short by default unless user overrides.
        cfg["training"]["epochs"] = int(args.overfit_epochs) if args.overfit_epochs > 0 else 60
    elif args.overfit_epochs > 0:
        cfg["training"]["epochs"] = int(args.overfit_epochs)

    if args.checkpoint_suffix:
        base_ckpt_dir = str(cfg["training"]["checkpoint_dir"]).rstrip("/\\")
        cfg["training"]["checkpoint_dir"] = f"{base_ckpt_dir}_{args.checkpoint_suffix}"

    set_determinism(seed=int(cfg.get("seed", 42)))

    cfg_for_loading = copy.deepcopy(cfg)
    if args.val_mode == "mixed":
        cfg_for_loading["splits"]["val"] = {"GLI": "test", "PED": "train", "MEN": "train"}
    elif args.val_mode == "gli":
        cfg_for_loading["splits"]["val"] = {"GLI": "test"}
    elif "val" not in cfg_for_loading["splits"]:
        cfg_for_loading["splits"]["val"] = {"GLI": "test"}

    all_dataset_dicts = build_dataset_dicts(cfg_for_loading, config_path.parent)
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

    if args.val_mode == "mixed":
        val_mapping = {"GLI": "test", "PED": "train", "MEN": "train"}
    elif args.val_mode == "gli":
        val_mapping = {"GLI": "test"}
    elif "val" in cfg["splits"]:
        val_mapping = cfg["splits"]["val"]
    else:
        val_mapping = {"GLI": "test"}

    val_files = select_split_files(
        all_dataset_dicts,
        split_datasets=val_mapping,
        split_name="val",
    )

    if len(val_files) == 0:
        print("[train_B] No usable validation cases in configured val split. Falling back to GLI train.")
        val_files = select_split_files(
            all_dataset_dicts,
            split_datasets={"GLI": "train"},
            split_name="val",
        )

    if args.overfit_cases > 0:
        overfit_n = min(args.overfit_cases, len(train_files))
        train_files = train_files[:overfit_n]
        val_files = train_files.copy()
        print(f"[overfit mode] Using {len(train_files)} train cases and same set for validation.")

    if args.max_val_cases > 0:
        val_files = val_files[: args.max_val_cases]

    print(f"\nTotal train samples: {len(train_files)}")
    print(f"Total test samples: {len(test_files)}")
    print(f"Total val samples (used for checkpoint selection): {len(val_files)}")
    print(f"Training mode: {args.train_mode}")
    print(f"Loss type: {args.loss_type}")
    print(f"Label setup: {args.label_setup} (num_classes={num_classes})")
    print(f"Validation mode: {args.val_mode}")
    print(f"Configured epochs: {cfg['training'].get('epochs')}")

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

    parsed_class_weights = parse_class_weights(args.class_weights, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sanity_loss_fn = build_loss_function(args.loss_type, parsed_class_weights, num_classes, cfg, device)

    print("\nVerifying model forward pass and loss...")
    run_model_forward_check(train_loader, sanity_loss_fn, out_channels=num_classes)

    val_loader = None
    if len(val_files) > 0:
        val_ds = Dataset(data=val_files, transform=build_eval_transforms(cfg))
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    print("\nRunning Stage B: multi-epoch baseline training...")
    run_stage_b_training(
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

    print("Stage B training completed.")


if __name__ == "__main__":
    main()
