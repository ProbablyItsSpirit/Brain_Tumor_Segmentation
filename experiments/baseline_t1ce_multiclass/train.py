from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
	"""Load YAML config and fall back to defaults when file is empty."""
	config_path = Path(config_path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	with config_path.open("r", encoding="utf-8") as f:
		loaded = yaml.safe_load(f) or {}

	defaults: dict[str, Any] = {
		"data_root": "/content/BraTS-2024-Complete",
		"datasets": {
			"GLI": {
				"folder": "BraTS-GLI",
				"splits": ["train"],
				"image_keywords": ["t1ce", "t1c"],
				"label_keywords": ["seg"],
			},
			"PED": {
				"folder": "BraTS-PED",
				"splits": ["train"],
				"image_keywords": ["t1ce", "t1c"],
				"label_keywords": ["seg"],
			},
			"MEN": {
				"folder": "BraTS-MEN-RT",
				"splits": ["train"],
				"image_keywords": ["t1ce", "t1c"],
				"label_keywords": ["gtv", "seg"],
			},
		},
		"max_cases_per_dataset": None,
	}

	# Shallow-merge top-level keys, then merge per-dataset keys.
	cfg = {**defaults, **loaded}
	cfg_datasets = defaults["datasets"].copy()
	cfg_datasets.update((loaded.get("datasets") or {}))
	cfg["datasets"] = cfg_datasets
	return cfg


def _find_first_matching_file(
	patient_dir: Path,
	include_keywords: list[str],
	suffix: str = ".nii.gz",
) -> Path | None:
	"""Return first NIfTI file matching any keyword (case-insensitive)."""
	for file_path in sorted(patient_dir.iterdir()):
		if not file_path.is_file():
			continue
		name = file_path.name.lower()
		if name.endswith(suffix) and any(k.lower() in name for k in include_keywords):
			return file_path
	return None


def build_dataset_list(cfg: dict[str, Any]) -> list[dict[str, str]]:
	"""
	Build MONAI-style dataset list entries:
	{"image": <t1ce_path>, "label": <label_path>, "dataset": <GLI|PED|MEN>, "split": <split>, "patient_id": <id>}
	"""
	data_root = Path(cfg["data_root"])
	max_cases = cfg.get("max_cases_per_dataset")

	items: list[dict[str, str]] = []
	for dataset_name, ds_cfg in cfg["datasets"].items():
		ds_root = data_root / ds_cfg["folder"]
		splits = ds_cfg.get("splits", ["train"])
		image_keywords = ds_cfg.get("image_keywords", ["t1ce", "t1c"])
		label_keywords = ds_cfg.get("label_keywords", ["seg"])

		count_for_dataset = 0
		for split in splits:
			split_dir = ds_root / split
			if not split_dir.exists():
				print(f"[WARN] Missing split directory: {split_dir}")
				continue

			patient_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
			for patient_dir in patient_dirs:
				image_file = _find_first_matching_file(patient_dir, image_keywords)
				label_file = _find_first_matching_file(patient_dir, label_keywords)

				if image_file is None or label_file is None:
					continue

				items.append(
					{
						"image": str(image_file),
						"label": str(label_file),
						"dataset": dataset_name,
						"split": split,
						"patient_id": patient_dir.name,
					}
				)
				count_for_dataset += 1

				if max_cases is not None and count_for_dataset >= int(max_cases):
					break

			if max_cases is not None and count_for_dataset >= int(max_cases):
				break

	return items


def main() -> None:
	config_path = Path(__file__).with_name("config.yaml")
	cfg = load_config(config_path)
	dataset_list = build_dataset_list(cfg)

	print(f"Total samples found: {len(dataset_list)}")
	if dataset_list:
		print("Example sample:")
		print(dataset_list[0])


if __name__ == "__main__":
	main()

