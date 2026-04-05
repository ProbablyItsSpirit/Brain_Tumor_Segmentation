from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any, Dict, List, Union

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
)
from monai.utils import set_determinism

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run inference using a trained Stage B checkpoint")
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
		"--output-dir",
		type=str,
		default="results/inference_stage_b",
		help="Directory to save inference metrics and optional predictions",
	)
	parser.add_argument(
		"--save-predictions",
		action="store_true",
		help="If set, save per-case prediction volumes (.npy). Disabled by default to avoid large disk usage.",
	)
	parser.add_argument(
		"--max-cases",
		type=int,
		default=0,
		help="Optional cap on number of test cases to run (0 means all)",
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


def build_inference_transforms(cfg: Dict[str, Any]) -> Compose:
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


def build_model(out_channels: int) -> UNet:
	return UNet(
		spatial_dims=3,
		in_channels=1,
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

	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	cfg = load_config(config_path)
	num_classes = apply_label_setup(cfg, args.label_setup)
	print(f"Label setup: {args.label_setup} (num_classes={num_classes})")
	set_determinism(seed=int(cfg.get("seed", 42)))

	output_dir = resolve_path(config_path.parent, args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	pred_dir = output_dir / "predictions"
	if args.save_predictions:
		pred_dir.mkdir(parents=True, exist_ok=True)

	all_dataset_dicts = build_dataset_dicts(cfg, config_path.parent)
	test_files = select_split_files(
		all_dataset_dicts,
		split_datasets=cfg["splits"]["test"],
		split_name="test",
	)
	if args.max_cases > 0:
		test_files = test_files[: args.max_cases]

	print(f"\nTotal test samples for inference: {len(test_files)}")
	if len(test_files) == 0:
		raise RuntimeError("No test samples found for inference.")

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
	model = build_model(out_channels=num_classes).to(device)
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
