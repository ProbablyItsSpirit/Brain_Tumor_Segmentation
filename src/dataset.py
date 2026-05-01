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


def build_gli_image_case_dicts(data_root: Path, list_path: Path, preferred_split: str) -> List[Dict[str, str]]:
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
        }

        if all(Path(v).exists() for k, v in sample.items() if k != "case_id"):
            seg_path = case_dir / f"{case_id}-seg.nii.gz"
            if seg_path.exists():
                sample["label"] = str(seg_path)
            cases.append(sample)

    return cases


def load_gli_train_val_test_cases(
    data_root: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train_cases = build_gli_image_case_dicts(data_root, train_list, preferred_split="train")
    val_cases = build_gli_image_case_dicts(data_root, val_list, preferred_split="val")
    test_cases = build_gli_image_case_dicts(data_root, test_list, preferred_split="val")
    return train_cases, val_cases, test_cases


# ============== PED DATASET LOADERS ==============

def build_ped_case_dicts(data_root: Path, list_path: Path, preferred_split: str) -> List[Dict[str, str]]:
    case_ids = read_case_ids(list_path)
    cases: List[Dict[str, str]] = []

    for case_id in case_ids:
        case_dir = find_case_dir_preferred(data_root, case_id, preferred_split)
        if case_dir is None:
            continue

        sample = {
            "case_id": case_id,
            "dataset": "PED",
            "image_t1n": str(case_dir / f"{case_id}-t1n.nii.gz"),
            "image_t1c": str(case_dir / f"{case_id}-t1c.nii.gz"),
            "image_t2w": str(case_dir / f"{case_id}-t2w.nii.gz"),
            "image_t2f": str(case_dir / f"{case_id}-t2f.nii.gz"),
        }

        if all(Path(v).exists() for k, v in sample.items() if k not in ["case_id", "dataset"]):
            seg_path = case_dir / f"{case_id}-seg.nii.gz"
            if seg_path.exists():
                sample["label"] = str(seg_path)
            cases.append(sample)

    return cases


def load_ped_train_val_test_cases(
    data_root: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train_cases = build_ped_case_dicts(data_root, train_list, preferred_split="train")
    val_cases = build_ped_case_dicts(data_root, val_list, preferred_split="val")
    test_cases = build_ped_case_dicts(data_root, test_list, preferred_split="val")
    return train_cases, val_cases, test_cases


# ============== MEN DATASET LOADERS ==============

def build_men_case_dicts(data_root: Path, list_path: Path, preferred_split: str) -> List[Dict[str, str]]:
    """MEN has T1ce only; replicate to other modality slots."""
    case_ids = read_case_ids(list_path)
    cases: List[Dict[str, str]] = []

    for case_id in case_ids:
        case_dir = find_case_dir_preferred(data_root, case_id, preferred_split)
        if case_dir is None:
            continue

        t1ce_path = case_dir / f"{case_id}-t1c.nii.gz"
        if not t1ce_path.exists():
            continue

        sample = {
            "case_id": case_id,
            "dataset": "MEN",
            "image_t1n": str(t1ce_path),  # Replicate T1ce
            "image_t1c": str(t1ce_path),  # Original T1ce
            "image_t2w": str(t1ce_path),  # Replicate T1ce
            "image_t2f": str(t1ce_path),  # Replicate T1ce
        }

        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        if seg_path.exists():
            sample["label"] = str(seg_path)
        cases.append(sample)

    return cases


def load_men_train_val_test_cases(
    data_root: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train_cases = build_men_case_dicts(data_root, train_list, preferred_split="train")
    val_cases = build_men_case_dicts(data_root, val_list, preferred_split="val")
    test_cases = build_men_case_dicts(data_root, test_list, preferred_split="val")
    return train_cases, val_cases, test_cases


# ============== MIXED DATASET LOADERS ==============

def load_mixed_train_val_test_cases(
    repo_root: Path,
    gli_data_root: Path,
    ped_data_root: Path,
    men_data_root: Path,
    gli_train_list: Path,
    gli_val_list: Path,
    gli_test_list: Path,
    ped_train_list: Path,
    ped_val_list: Path,
    ped_test_list: Path,
    men_train_list: Path,
    men_val_list: Path,
    men_test_list: Path,
    val_ratio: float = 0.1,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Load GLI + PED + MEN with balanced train/val splits."""
    
    # Load each dataset
    gli_train, gli_val, gli_test = load_gli_train_val_test_cases(gli_data_root, gli_train_list, gli_val_list, gli_test_list)
    ped_train, ped_val, ped_test = load_ped_train_val_test_cases(ped_data_root, ped_train_list, ped_val_list, ped_test_list)
    men_train, men_val, men_test = load_men_train_val_test_cases(men_data_root, men_train_list, men_val_list, men_test_list)
    
    # Tag each case with dataset source
    for case in gli_train + gli_val + gli_test:
        case["dataset"] = "GLI"
    for case in ped_train + ped_val + ped_test:
        case["dataset"] = "PED"
    for case in men_train + men_val + men_test:
        case["dataset"] = "MEN"
    
    # Combine training cases
    all_train = gli_train + ped_train + men_train
    
    # If val sets are empty, create holdout from training
    all_val = gli_val + ped_val + men_val
    if not all_val:
        import numpy as np
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(all_train))
        val_count = max(1, int(len(all_train) * val_ratio))
        val_idx = set(perm[:val_count].tolist())
        
        split_train = []
        split_val = []
        for i, case in enumerate(all_train):
            if i in val_idx:
                split_val.append(case)
            else:
                split_train.append(case)
        
        all_train = split_train
        all_val = split_val
    
    # Combine test cases
    all_test = gli_test + ped_test + men_test
    
    return all_train, all_val, all_test
