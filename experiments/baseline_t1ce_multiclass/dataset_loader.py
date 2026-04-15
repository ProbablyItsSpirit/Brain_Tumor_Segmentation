from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def read_case_ids(list_path: Path) -> List[str]:
    if not list_path.exists():
        return []

    case_ids: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = line.strip()
            if c and not c.startswith("#"):
                case_ids.append(c)
    return case_ids


def find_case_dir(dataset_root: Path, case_id: str) -> Path | None:
    for split in ("train", "val", "train_additional"):
        cand = dataset_root / split / case_id
        if cand.exists():
            return cand
    return None


def region_channels_from_label(label, et_labels: Sequence[int] = (3, 4)):
    """Return WT/TC/ET region channels from raw BraTS label volume."""
    import numpy as np

    et_mask = np.isin(label, list(et_labels))
    tc_mask = np.logical_or(label == 1, et_mask)
    wt_mask = label > 0
    return wt_mask.astype("float32"), tc_mask.astype("float32"), et_mask.astype("float32")


def build_case_list(cfg: Dict, config_dir: Path, split: str) -> List[Dict]:
    data_cfg = cfg["data"]
    root = resolve_path(config_dir, data_cfg["root"])
    patient_lists = resolve_path(config_dir, data_cfg["patient_lists_dir"])

    modalities: List[str] = list(data_cfg["modalities"])
    datasets_cfg = data_cfg["datasets"]

    cases: List[Dict] = []
    for dataset_name in ("GLI", "PED"):
        ds = datasets_cfg[dataset_name]
        dataset_root = root / ds["folder"]

        list_file = ds["list_files"][split]
        ids = read_case_ids(patient_lists / list_file)

        for case_id in ids:
            case_dir = find_case_dir(dataset_root, case_id)
            if case_dir is None:
                continue

            sample: Dict[str, str] = {
                "dataset": dataset_name,
                "case_id": case_id,
                "label": str(case_dir / f"{case_id}{ds['label_suffix']}"),
            }

            missing = False
            for mod in modalities:
                img = case_dir / f"{case_id}-{mod}.nii.gz"
                if not img.exists():
                    missing = True
                    break
                sample[f"image_{mod}"] = str(img)

            if missing or not Path(sample["label"]).exists():
                continue
            cases.append(sample)

    return cases
