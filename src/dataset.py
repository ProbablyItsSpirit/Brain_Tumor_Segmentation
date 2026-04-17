from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def read_case_ids(list_path: Path) -> List[str]:
    if not list_path.exists():
        return []
    ids: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            case_id = line.strip()
            if case_id and not case_id.startswith("#"):
                ids.append(case_id)
    return ids



def find_case_dir(dataset_root: Path, case_id: str) -> Path | None:
    for split in ("train", "val", "train_additional"):
        candidate = dataset_root / split / case_id
        if candidate.exists():
            return candidate
    return None



def build_gli_case_dicts(repo_root: Path, data_root: Path, list_path: Path) -> List[Dict[str, str]]:
    case_ids = read_case_ids(list_path)
    cases: List[Dict[str, str]] = []

    for case_id in case_ids:
        case_dir = find_case_dir(data_root, case_id)
        if case_dir is None:
            continue

        sample = {
            "case_id": case_id,
            "image_t1n": str(case_dir / f"{case_id}-t1n.nii.gz"),
            "image_t1c": str(case_dir / f"{case_id}-t1c.nii.gz"),
            "image_t2w": str(case_dir / f"{case_id}-t2w.nii.gz"),
            "image_t2f": str(case_dir / f"{case_id}-t2f.nii.gz"),
            "label": str(case_dir / f"{case_id}-seg.nii.gz"),
        }

        if all(Path(v).exists() for k, v in sample.items() if k != "case_id"):
            cases.append(sample)

    return cases



def load_gli_splits(repo_root: Path, data_root: Path, train_list: Path, val_list: Path) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train_cases = build_gli_case_dicts(repo_root, data_root, train_list)
    val_cases = build_gli_case_dicts(repo_root, data_root, val_list)
    return train_cases, val_cases


def find_case_dir_preferred(dataset_root: Path, case_id: str, preferred_split: str) -> Path | None:
    preferred = dataset_root / preferred_split / case_id
    if preferred.exists():
        return preferred
    for split in ("train", "val", "train_additional"):
        candidate = dataset_root / split / case_id
        if candidate.exists():
            return candidate
    return None


def build_gli_case_dicts_for_split(data_root: Path, list_path: Path, preferred_split: str) -> List[Dict[str, str]]:
    case_ids = read_case_ids(list_path)
    cases: List[Dict[str, str]] = []

    for case_id in case_ids:
        case_dir = find_case_dir_preferred(data_root, case_id, preferred_split)
        if case_dir is None:
            continue

        sample = {
            "case_id": case_id,
            "image_t1n": str(case_dir / f"{case_id}-t1n.nii.gz"),
            "image_t1c": str(case_dir / f"{case_id}-t1c.nii.gz"),
            "image_t2w": str(case_dir / f"{case_id}-t2w.nii.gz"),
            "image_t2f": str(case_dir / f"{case_id}-t2f.nii.gz"),
            "label": str(case_dir / f"{case_id}-seg.nii.gz"),
        }

        if all(Path(v).exists() for k, v in sample.items() if k != "case_id"):
            cases.append(sample)

    return cases


def load_gli_train_val_test_strict(
    data_root: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train_cases = build_gli_case_dicts_for_split(data_root, train_list, preferred_split="train")
    val_cases = build_gli_case_dicts_for_split(data_root, val_list, preferred_split="val")
    test_cases = build_gli_case_dicts_for_split(data_root, test_list, preferred_split="val")
    return train_cases, val_cases, test_cases
