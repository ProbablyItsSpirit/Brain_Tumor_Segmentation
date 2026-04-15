from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BaselineConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    data_root: Path = Path("BraTS-2024-Complete/BraTS-GLI")
    train_list: Path = Path("patient_lists/gli_train.txt")
    val_list: Path = Path("patient_lists/gli_val.txt")
    modalities: List[str] = field(default_factory=lambda: ["t1c", "t2w", "t2f"])
    batch_size: int = 1
    num_workers: int = 0
    patch_size: tuple[int, int, int] = (96, 96, 96)
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    val_interval: int = 1
    overfit_cases: int = 20
    overfit_epochs: int = 60
    min_fg_ratio: float = 0.02
    max_sample_tries: int = 30
    tumor_margin: int = 24
    checkpoint_dir: Path = Path("checkpoints/clean_gli_binary_baseline")
    results_dir: Path = Path("results/clean_gli_binary_baseline")
    seed: int = 42

    def resolve(self) -> "BaselineConfig":
        self.data_root = (self.repo_root / self.data_root).resolve()
        self.train_list = (self.repo_root / self.train_list).resolve()
        self.val_list = (self.repo_root / self.val_list).resolve()
        self.checkpoint_dir = (self.repo_root / self.checkpoint_dir).resolve()
        self.results_dir = (self.repo_root / self.results_dir).resolve()
        return self



def get_default_config() -> BaselineConfig:
    return BaselineConfig().resolve()
