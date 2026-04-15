from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
import torch

from config import get_default_config
from dataset import load_gli_splits
from model import build_model
from transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clean GLI-only baseline predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/clean_gli_binary_baseline/visuals")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    return parser.parse_args()


def pick_slice(mask: np.ndarray) -> int:
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    if counts.max() <= 0:
        return mask.shape[0] // 2
    return int(counts.argmax())


def main() -> None:
    args = parse_args()
    cfg = get_default_config()
    cfg.patch_size = tuple(args.patch_size)

    train_cases, val_cases = load_gli_splits(cfg.repo_root, cfg.data_root, cfg.train_list, cfg.val_list)
    cases = train_cases if args.split == "train" else val_cases
    if not cases:
        raise RuntimeError("No cases available for visualization.")
    case = cases[min(args.case_index, len(cases) - 1)]

    ds = Dataset(
        data=[case],
        transform=build_transforms(
            modalities=cfg.modalities,
            patch_size=cfg.patch_size,
            min_fg_ratio=cfg.min_fg_ratio,
            max_tries=cfg.max_sample_tries,
            margin=cfg.tumor_margin,
            training=False,
        ),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_channels=3, out_channels=2).to(device)
    state = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (cfg.repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(loader))
        image = batch["image"].to(device)
        label = batch["label"].cpu().numpy()[0, 0]

        logits = sliding_window_inference(image, tuple(cfg.patch_size), 1, model, overlap=0.25)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()[0]

    mid = pick_slice(label)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    title = case.get("case_id", "case") if isinstance(case, dict) else "case"
    axes[0].imshow(image[0, 0, mid].detach().cpu().numpy(), cmap="gray")
    axes[0].set_title("T1ce")
    axes[1].imshow(label[mid], cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred[mid], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(str(title))
    fig.tight_layout()
    fig.savefig(out_dir / f"{title}_slice{mid}.png")
    plt.close(fig)

    print(f"Saved visualization to: {out_dir}")


if __name__ == "__main__":
    main()
