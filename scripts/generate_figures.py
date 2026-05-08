"""
Generate publication-quality figures for brain tumor segmentation paper.

This script reads training history from results/ and generates:
1. Train vs Validation Dice curve
2. Loss curve
3. Bar graph of experimental results
4. WT/TC/ET Dice comparison

Usage:
  python scripts/generate_figures.py --results-dir results/gli_multimodal_mvp_seed42 --output-dir figures/
"""

import json
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Default experimental results (if no history.json, use logged values)
EXPERIMENTAL_RESULTS = {
    "FLAIR Baseline": {"mean": 0.32, "wt": 0.32, "tc": None, "et": None},
    "Early Multi": {"mean": 0.11, "wt": 0.11, "tc": None, "et": None},
    "Strict Multi": {"mean": 0.72, "wt": 0.72, "tc": 0.65, "et": None},
    "Binary GLI": {"mean": 0.78, "wt": 0.78, "tc": None, "et": None},
    "Region ML": {"mean": 0.8373, "wt": 0.8723, "tc": 0.8207, "et": 0.8107},
}


def load_history(results_dir):
    """Load training history from results directory."""
    history_file = Path(results_dir) / "history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)
    else:
        print(f"Warning: history.json not found at {history_file}")
        return None


def plot_train_val_dice(history, output_dir):
    """Plot training vs validation Dice over epochs."""
    if history is None:
        print("Skipping train_val_dice plot (no history).")
        return
    
    epochs = history.get("epochs", list(range(len(history.get("train_dice", [])))))
    train_dice = history.get("train_dice", [])
    val_dice = history.get("val_dice", [])
    
    if not train_dice or not val_dice:
        print("Warning: train_dice or val_dice not found in history.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_dice, "o-", label="Training Dice", linewidth=2, markersize=4)
    ax.plot(epochs, val_dice, "s-", label="Validation Dice", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Dice Score", fontsize=12, fontweight="bold")
    ax.set_title("Training vs Validation Dice Score", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    output_path = Path(output_dir) / "train_val_dice.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_loss_curve(history, output_dir):
    """Plot training vs validation loss over epochs."""
    if history is None:
        print("Skipping loss_curve plot (no history).")
        return
    
    epochs = history.get("epochs", list(range(len(history.get("train_loss", [])))))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    
    if not train_loss or not val_loss:
        print("Warning: train_loss or val_loss not found in history.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_loss, "o-", label="Training Loss", linewidth=2, markersize=4)
    ax.plot(epochs, val_loss, "s-", label="Validation Loss", linewidth=2, markersize=4)
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / "loss_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_experimental_bars(output_dir):
    """Plot bar graph of experimental results (mean Dice across methods)."""
    methods = list(EXPERIMENTAL_RESULTS.keys())
    mean_dices = [EXPERIMENTAL_RESULTS[m]["mean"] for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    bars = ax.bar(methods, mean_dices, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    
    ax.set_ylabel("Mean Dice Score", fontsize=12, fontweight="bold")
    ax.set_title("Experimental Progression: Mean Dice Across Configurations", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    
    output_path = Path(output_dir) / "exp_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_wt_tc_et_bars(output_dir):
    """Plot grouped bar chart for WT/TC/ET Dice comparison."""
    regions = ["WT", "TC", "ET"]
    
    # Use the final Region ML results and a few others
    configs = {
        "FLAIR": [0.32, None, None],
        "Strict Multi": [0.72, 0.65, None],
        "Region ML": [0.8723, 0.8207, 0.8107],
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regions))
    width = 0.25
    
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    for i, (config, values) in enumerate(configs.items()):
        # Replace None with 0 for plotting (or could skip)
        plot_values = [v if v is not None else 0 for v in values]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, plot_values, width, label=config, alpha=0.8, edgecolor="black")
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val is not None:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    
    ax.set_ylabel("Dice Score", fontsize=12, fontweight="bold")
    ax.set_title("WT / TC / ET Dice Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis="y")
    
    output_path = Path(output_dir) / "wt_tc_et.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from training logs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/gli_multimodal_mvp_seed42",
        help="Path to results directory containing history.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for generated figures",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = load_history(args.results_dir)
    
    print("Generating figures...")
    plot_train_val_dice(history, output_dir)
    plot_loss_curve(history, output_dir)
    plot_experimental_bars(output_dir)
    plot_wt_tc_et_bars(output_dir)
    
    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
