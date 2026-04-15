#!/usr/bin/env python
"""Run the mandatory base-model overfit check on 20 samples and save diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "train_strict_multimodal.py"
INFER_SCRIPT = ROOT / "inference_strict_multimodal.py"
VALIDATE_SCRIPT = ROOT / "validate_10_cases.py"
CONFIG_PATH = ROOT / "config.yaml"


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 80}")
    print(label)
    print(f"{'=' * 80}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def find_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = [
        checkpoint_dir / "stage_b_best.pt",
        checkpoint_dir / "best.pt",
        checkpoint_dir / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    pt_files = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_checkpoint_dir() -> Path:
    import yaml

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (ROOT / checkpoint_dir).resolve()
    return checkpoint_dir


def main() -> None:
    checkpoint_dir = load_checkpoint_dir()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--overfit-cases",
            "20",
            "--overfit-epochs",
            "60",
        ],
        "1) Train on 20 samples for 60 epochs (overfit check)",
    )

    checkpoint = find_checkpoint(checkpoint_dir)
    print(f"Using checkpoint: {checkpoint}")

    inference_out = ROOT / "results" / "overfit_base_inference"
    run(
        [
            sys.executable,
            str(INFER_SCRIPT),
            "--config",
            str(CONFIG_PATH),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(inference_out),
            "--save-predictions",
            "--max-cases",
            "20",
        ],
        "2) Infer on the same 20 samples",
    )

    run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--pred-dir",
            str(inference_out / "predictions"),
            "--num-cases",
            "20",
            "--output-dir",
            str(ROOT / "results" / "overfit_base_visuals"),
            "--modality",
            "t1c",
            "--target",
            "multiclass4",
        ],
        "3) Save visual overlays",
    )

    metrics_file = inference_out / "inference_metrics.json"
    if metrics_file.exists():
        with metrics_file.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        mean_dice = float(metrics.get("mean_dice_no_bg", 0.0))
        print(f"\nMean Dice (classes 1-3): {mean_dice:.4f}")
        if mean_dice < 0.9:
            print("WARNING: Dice is below 0.9. The pipeline still needs debugging.")
        else:
            print("PASS: Training samples overfit successfully.")


if __name__ == "__main__":
    main()