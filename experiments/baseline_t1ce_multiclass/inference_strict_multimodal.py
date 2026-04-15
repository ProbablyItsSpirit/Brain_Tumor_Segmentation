from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, Orientationd
from monai.utils import set_determinism
try:
    from scipy import ndimage  # type: ignore
except Exception:
    ndimage = None

from dataset_loader import build_case_list, region_channels_from_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean strict multimodal inference")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/baseline_t1ce_multiclass/results/strict_multimodal_clean",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--save-region-prob", action="store_true")
    parser.add_argument("--min-component-size", type=int, default=-1)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RegionLabeld:
    def __init__(self, key: str, et_labels: Sequence[int]):
        self.key = key
        self.et_labels = tuple(int(x) for x in et_labels)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        wt, tc, et = region_channels_from_label(d[self.key], et_labels=self.et_labels)
        d[self.key] = np.stack([wt, tc, et], axis=0).astype(np.float32)
        return d


class StackModalitiesd:
    def __init__(self, image_keys: Sequence[str]):
        self.image_keys = list(image_keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d["image"] = torch.cat([d[k] for k in self.image_keys], dim=0)
        for k in self.image_keys:
            d.pop(k, None)
        return d


def build_transforms(cfg: Dict[str, Any]) -> Compose:
    mods = list(cfg["data"]["modalities"])
    image_keys = [f"image_{m}" for m in mods]
    keys = image_keys + ["label"]

    et_labels = tuple(int(x) for x in cfg["labels"].get("et_labels", [3, 4]))

    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True),
            RegionLabeld(key="label", et_labels=et_labels),
            EnsureTyped(keys=image_keys, dtype=torch.float32),
            EnsureTyped(keys="label", dtype=torch.float32),
            StackModalitiesd(image_keys=image_keys),
        ]
    )


def build_model(cfg: Dict[str, Any]) -> UNet:
    m = cfg["model"]
    return UNet(
        spatial_dims=3,
        in_channels=int(m["in_channels"]),
        out_channels=int(m["out_channels"]),
        channels=tuple(int(x) for x in m["channels"]),
        strides=tuple(int(x) for x in m["strides"]),
        num_res_units=int(m["num_res_units"]),
    )


def region_to_labelmap(wt: np.ndarray, tc: np.ndarray, et: np.ndarray) -> np.ndarray:
    """Convert WT/TC/ET booleans to BraTS-like integer map 0/1/2/4."""
    out = np.zeros(wt.shape, dtype=np.uint8)
    out[wt > 0] = 2
    out[tc > 0] = 1
    out[et > 0] = 4
    return out


def filter_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask

    if ndimage is None:
        return mask

    filtered = np.zeros_like(mask, dtype=np.uint8)
    labeled, n = ndimage.label(mask.astype(np.uint8))
    for i in range(1, n + 1):
        comp = labeled == i
        if int(comp.sum()) >= min_size:
            filtered[comp] = 1
    return filtered


def dice_region(pred: np.ndarray, gt: np.ndarray) -> float:
    eps = 1e-6
    inter = float((pred * gt).sum())
    den = float(pred.sum() + gt.sum())
    return float((2.0 * inter + eps) / (den + eps))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    set_determinism(seed=int(cfg.get("seed", 42)))

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cases = build_case_list(cfg, config_path.parent, split=args.split)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError("No cases found for inference.")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (config_path.parent.parent.parent / output_dir).resolve()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ds = Dataset(data=cases, transform=build_transforms(cfg))
    nw = int(cfg["dataloader"].get("num_workers", 0))
    if sys.platform.startswith("win"):
        nw = 0
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=nw > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    min_comp = args.min_component_size
    if min_comp < 0:
        min_comp = int(cfg["inference"].get("min_component_size", 50))

    rows = []
    thresh = float(cfg["inference"]["threshold"])

    if ndimage is None and min_comp > 0:
        print("WARNING: scipy not available, skipping connected-component post-processing.")

    with torch.no_grad():
        for batch in loader:
            case_id = str(batch["case_id"][0])
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            logits = sliding_window_inference(
                inputs=image,
                roi_size=tuple(int(x) for x in cfg["patch"]["size"]),
                sw_batch_size=1,
                predictor=model,
                overlap=float(cfg["inference"]["overlap"]),
            )
            probs = torch.sigmoid(logits)[0].cpu().numpy()

            wt = (probs[0] > thresh).astype(np.uint8)
            tc = (probs[1] > thresh).astype(np.uint8)
            et = (probs[2] > thresh).astype(np.uint8)

            wt = filter_small_components(wt, min_comp)
            tc = filter_small_components(tc, min_comp)
            et = filter_small_components(et, min_comp)

            # Keep region consistency
            tc = np.logical_and(tc, wt).astype(np.uint8)
            et = np.logical_and(et, tc).astype(np.uint8)

            pred_map = region_to_labelmap(wt, tc, et)
            np.save(pred_dir / f"{case_id}_pred.npy", pred_map)

            if args.save_region_prob:
                np.save(pred_dir / f"{case_id}_regions.npy", np.stack([wt, tc, et], axis=0).astype(np.uint8))

            gt = target[0].cpu().numpy()
            d_wt = dice_region(wt, (gt[0] > 0.5).astype(np.uint8))
            d_tc = dice_region(tc, (gt[1] > 0.5).astype(np.uint8))
            d_et = dice_region(et, (gt[2] > 0.5).astype(np.uint8))
            d_mean = float((d_wt + d_tc + d_et) / 3.0)

            rows.append(
                {
                    "case_id": case_id,
                    "dice_WT": d_wt,
                    "dice_TC": d_tc,
                    "dice_ET": d_et,
                    "dice_mean": d_mean,
                }
            )

    summary = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "num_cases": len(rows),
        "mean_dice_WT": float(np.mean([r["dice_WT"] for r in rows])),
        "mean_dice_TC": float(np.mean([r["dice_TC"] for r in rows])),
        "mean_dice_ET": float(np.mean([r["dice_ET"] for r in rows])),
        "mean_dice": float(np.mean([r["dice_mean"] for r in rows])),
        "cases": rows,
    }

    with (output_dir / "inference_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Inference done on {len(rows)} cases")
    print(f"Predictions saved to: {pred_dir}")
    print(f"Mean Dice WT/TC/ET: {summary['mean_dice_WT']:.4f} / {summary['mean_dice_TC']:.4f} / {summary['mean_dice_ET']:.4f}")
    print(f"Mean Dice overall: {summary['mean_dice']:.4f}")


if __name__ == "__main__":
    main()
