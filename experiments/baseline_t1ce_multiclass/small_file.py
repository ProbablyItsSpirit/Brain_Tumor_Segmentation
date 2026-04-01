from __future__ import annotations

import argparse
import json
import copy
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

from inference import (
    build_dataset_dicts,
    build_inference_transforms,
    build_model,
    compute_case_metrics,
    extract_case_id,
    load_checkpoint,
    load_config,
    resolve_path,
    select_split_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick GLI-only inference on a small number of cases"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (e.g., stage_b_best.pt)",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Number of GLI test cases to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/small_file_gli10",
        help="Directory to save small-run metrics and plot",
    )
    return parser.parse_args()


def save_plot(case_metrics: List[Dict[str, Any]], plot_path: Path) -> None:
    case_ids = [m["case_id"] for m in case_metrics]
    mean_dice = [m["mean_dice_no_bg"] for m in case_metrics]

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(mean_dice)), mean_dice)
    plt.xticks(range(len(case_ids)), case_ids, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Mean Dice (classes 1-3)")
    plt.xlabel("Case ID")
    plt.title("GLI Quick Eval (10 cases)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(config_path)
    set_determinism(seed=int(cfg.get("seed", 42)))

    output_dir = resolve_path(config_path.parent, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Force-load GLI test source split for this quick sanity script,
    # independent of the main experiment's cross-dataset test mapping.
    quick_cfg = copy.deepcopy(cfg)
    quick_cfg["splits"]["test"] = {"GLI": "test"}

    all_dataset_dicts = build_dataset_dicts(quick_cfg, config_path.parent)
    gli_test_files = select_split_files(
        all_dataset_dicts,
        split_datasets={"GLI": "test"},
        split_name="test",
    )

    if len(gli_test_files) == 0:
        print("[small_file] No usable GLI test cases found. Falling back to GLI train cases.")
        quick_cfg_fallback = copy.deepcopy(cfg)
        quick_cfg_fallback["splits"]["test"] = {"GLI": "train"}
        all_dataset_dicts = build_dataset_dicts(quick_cfg_fallback, config_path.parent)
        gli_test_files = select_split_files(
            all_dataset_dicts,
            split_datasets={"GLI": "train"},
            split_name="test",
        )

    num_cases = max(1, int(args.num_cases))
    test_files = gli_test_files[:num_cases]
    if len(test_files) == 0:
        raise RuntimeError("No usable GLI cases found in test or train splits.")

    print(f"Running quick GLI inference on {len(test_files)} case(s).")

    test_ds = Dataset(data=test_files, transform=build_inference_transforms(cfg))
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["dataloader"].get("num_workers", 0)) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_case_metrics: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in test_loader:
            case_id = extract_case_id(batch["case_id"])

            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if labels.ndim == 5 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=tuple(cfg["patch"]["size"]),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )
            preds = torch.argmax(logits, dim=1)

            metrics = compute_case_metrics(preds[0], labels[0], num_classes=4)
            all_case_metrics.append({"case_id": case_id, **metrics})

    mean_dice_values = [m["mean_dice_no_bg"] for m in all_case_metrics]
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_cases": len(all_case_metrics),
        "mean_dice_no_bg": float(np.mean(mean_dice_values)) if mean_dice_values else 0.0,
        "cases": all_case_metrics,
    }

    metrics_path = output_dir / "small_file_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_path = output_dir / "small_file_plot.png"
    save_plot(all_case_metrics, plot_path)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Plot saved to: {plot_path}")
    print(f"Quick-run Mean Dice (classes 1-3): {summary['mean_dice_no_bg']:.6f}")


if __name__ == "__main__":
    main()
