from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
)
from monai.utils import set_determinism


MODALITIES = ["t1c", "t2w", "t2f"]


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
    parser = argparse.ArgumentParser(description="Run strict multimodal inference using a trained checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., stage_b_best.pt)",
    )
    parser.add_argument(
        "--case-file",
        type=str,
        default="",
        help="Optional JSON file containing held-out case dicts created by train_strict_multimodal.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baseline_t1ce_multiclass/results/strict_multimodal_inference",
        help="Directory to save inference metrics and optional predictions",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="If set, save per-case prediction volumes (.npy).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on number of cases to run (0 means all)",
    )
    parser.add_argument(
        "--label-setup",
        type=str,
        choices=["4c", "3c"],
        default="4c",
        help="4c: keep 4 output classes (0..3). 3c: merge labels 3/4 into class 2 (0..2).",
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


def load_case_specs(args: argparse.Namespace, checkpoint_path: Path) -> List[Dict[str, Any]]:
    if args.case_file.strip():
        case_file = Path(args.case_file)
    else:
        case_file = checkpoint_path.parent / "strict_val_cases.json"

    if not case_file.exists():
        raise FileNotFoundError(
            f"Case file not found: {case_file}\n"
            "Run train_strict_multimodal.py first so it saves strict_val_cases.json, or pass --case-file explicitly."
        )

    with case_file.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, list) or not loaded:
        raise RuntimeError(f"Case file is empty or invalid: {case_file}")

    case_specs: List[Dict[str, Any]] = []
    for item in loaded:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid case entry in {case_file}: expected dict, got {type(item).__name__}")
        case_specs.append(item)
    return case_specs


class StackModalitiesd:
    def __init__(self, image_keys: Sequence[str]):
        self.image_keys = list(image_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d["image"] = torch.cat([d[key] for key in self.image_keys], dim=0)
        for key in self.image_keys:
            d.pop(key, None)
        return d


def build_transforms(cfg: Dict[str, Any], modalities: Sequence[str]) -> Compose:
    mapping = {int(k): int(v) for k, v in cfg["data"]["label_mapping"].items()}
    image_keys = [f"image_{modality}" for modality in modalities]
    all_image_keys = list(image_keys) + ["label"]

    def label_mapper(lbl: Any):
        return remap_with_mapping(lbl, mapping=mapping)

    return Compose(
        [
            LoadImaged(keys=all_image_keys),
            EnsureChannelFirstd(keys=all_image_keys),
            Orientationd(keys=all_image_keys, axcodes="RAS"),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            Lambdad(keys="label", func=label_mapper),
            EnsureTyped(keys=image_keys, dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.int64),
            StackModalitiesd(image_keys),
        ]
    )


def build_model(out_channels: int, in_channels: int) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def extract_case_id(case_id_field: Any) -> str:
    if isinstance(case_id_field, (list, tuple)):
        return str(case_id_field[0])
    return str(case_id_field)


def dice_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> tuple[float | None, bool]:
    pred_c = (pred == class_id).float()
    target_c = (target == class_id).float()
    target_sum = target_c.sum()
    if target_sum.item() == 0:
        return None, False

    denominator = pred_c.sum() + target_sum
    intersection = (pred_c * target_c).sum()
    return float((2.0 * intersection / denominator).item()), True


def compute_case_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    valid_count_per_class: Dict[str, int] = {}
    class_values: List[float] = []
    for class_id in range(1, num_classes):
        d, is_valid = dice_for_class(pred, target, class_id)
        per_class[f"dice_class_{class_id}"] = None if d is None else d
        valid_count_per_class[f"class_{class_id}"] = 1 if is_valid else 0
        if is_valid and d is not None:
            class_values.append(d)

    mean_dice_no_bg = float(np.mean(class_values)) if class_values else 0.0
    return {
        "mean_dice_no_bg": mean_dice_no_bg,
        "valid_class_count": len(class_values),
        "valid_count_per_class": valid_count_per_class,
        **per_class,
    }


def load_checkpoint(model: UNet, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    repo_root = config_path.parent.parent.parent.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(config_path)
    num_classes = apply_label_setup(cfg, args.label_setup)
    print(f"Label setup: {args.label_setup} (num_classes={num_classes})")
    set_determinism(seed=int(cfg.get("seed", 42)))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    case_specs = load_case_specs(args, checkpoint_path)
    if args.max_cases > 0:
        case_specs = case_specs[: args.max_cases]

    print(f"\nTotal cases for inference: {len(case_specs)}")
    if len(case_specs) == 0:
        raise RuntimeError("No cases found for inference.")

    modalities = MODALITIES
    test_ds = Dataset(data=case_specs, transform=build_transforms(cfg, modalities))
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["dataloader"].get("num_workers", 0)) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(out_channels=num_classes, in_channels=len(modalities)).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_case_metrics: List[Dict[str, Any]] = []
    support_sums: Dict[str, int] = {f"class_{cid}": 0 for cid in range(1, num_classes)}

    with torch.no_grad():
        for batch in test_loader:
            case_id = extract_case_id(batch["case_id"])
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if labels.ndim == 5 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            roi_size = tuple(cfg["patch"]["size"])
            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )
            preds = torch.argmax(logits, dim=1)

            if args.save_predictions:
                pred_np = preds[0].detach().cpu().numpy().astype(np.uint8)
                np.save(pred_dir / f"{case_id}_pred.npy", pred_np)

            metrics = compute_case_metrics(preds[0], labels[0], num_classes=num_classes)
            for key, value in metrics.get("valid_count_per_class", {}).items():
                support_sums[key] = support_sums.get(key, 0) + int(value)
            all_case_metrics.append({"case_id": case_id, **metrics})

    mean_dice_values = [m["mean_dice_no_bg"] for m in all_case_metrics]
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_cases": len(all_case_metrics),
        "mean_dice_no_bg": float(np.mean(mean_dice_values)) if mean_dice_values else 0.0,
        "valid_count_per_class": support_sums,
        "cases": all_case_metrics,
    }

    summary_path = output_dir / "inference_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nInference completed.")
    if args.save_predictions:
        print(f"Predictions saved to: {pred_dir}")
    else:
        print("Predictions were not saved (use --save-predictions to enable).")
    print(f"Metrics saved to: {summary_path}")
    print(f"Mean Dice (classes 1-3): {summary['mean_dice_no_bg']:.6f}")


if __name__ == "__main__":
    main()
