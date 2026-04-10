from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet


def load_train_module():
    this_file = Path(__file__).resolve()
    train_path = this_file.with_name("train_flair_only_gli.py")
    spec = importlib.util.spec_from_file_location("train_flair_only_gli_module", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLAIR-only GLI inference using a trained checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to config file",
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
        help="Optional JSON file created by train_flair_only_gli.py containing evaluation cases",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baseline_t1ce_multiclass/results/flair_only_gli_baseline_eval10",
        help="Directory to save inference metrics and optional predictions",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="If set, save per-case prediction volumes (.npy)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on number of cases to run (0 means all)",
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


def load_case_specs(args: argparse.Namespace, checkpoint_path: Path) -> List[Dict[str, Any]]:
    if args.case_file.strip():
        case_file = Path(args.case_file)
    else:
        case_file = checkpoint_path.parent / "holdout_cases.json"

    if not case_file.exists():
        raise FileNotFoundError(
            f"Case file not found: {case_file}\n"
            "Run train_flair_only_gli.py first so it saves holdout_cases.json, or pass --case-file explicitly."
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


def load_checkpoint(model: UNet, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    repo_root = config_path.parent.parent.parent.parent.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_module = load_train_module()
    tb = train_module.load_train_b_module()
    cfg = load_config(config_path)
    tb.apply_local_path_overrides(cfg, config_path.parent)
    num_classes = tb.apply_label_setup(cfg, "4c")
    print(f"Label setup: 4c (num_classes={num_classes})")

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

    test_ds = Dataset(data=case_specs, transform=train_module.build_inference_transforms(cfg))
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["dataloader"].get("num_workers", 0)) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tb.build_model(out_channels=num_classes).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_case_metrics: List[Dict[str, Any]] = []
    support_sums: Dict[str, int] = {f"class_{cid}": 0 for cid in range(1, num_classes)}

    with torch.no_grad():
        for batch in test_loader:
            case_id = str(batch["case_id"][0])
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
                overlap=0.0,
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
    print(f"Mean Dice (foreground classes): {summary['mean_dice_no_bg']:.6f}")


if __name__ == "__main__":
    main()
