from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandIntensityShiftd,
)


class BinaryLabeld:
    def __init__(self, key: str = "label"):
        self.key = key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        label = d[self.key]
        if torch.is_tensor(label):
            if label.ndim >= 4 and label.shape[0] == 1:
                label = label.squeeze(0)
            label = (label > 0).long()
            d[self.key] = label.unsqueeze(0) if label.ndim == 3 else label
        else:
            arr = np.asarray(label)
            if arr.ndim >= 4 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)
            d[self.key] = (arr > 0).astype(np.int64)[None, ...]
        return d


class ZScoreNormalizeModalitiesd:
    """Z-score normalize each modality independently."""
    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if torch.is_tensor(img):
                img = img.float()
            else:
                img = torch.tensor(img, dtype=torch.float32)
            
            # Compute mean and std for this modality
            mask = img > 0  # Ignore background
            if mask.sum() > 0:
                mean = img[mask].mean()
                std = img[mask].std()
                if std > 1e-6:
                    img = (img - mean) / (std + 1e-6)
            
            d[key] = img
        return d


class StackModalitiesd:
    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        d["image"] = torch.cat([d[k] for k in self.keys], dim=0)
        for k in self.keys:
            d.pop(k, None)
        return d


class TumorCenteredCropd:
    def __init__(self, image_key: str, label_key: str, patch_size: Sequence[int], min_fg_ratio: float, max_tries: int, margin: int):
        self.image_key = image_key
        self.label_key = label_key
        self.patch_size = tuple(int(x) for x in patch_size)
        self.min_fg_ratio = float(min_fg_ratio)
        self.max_tries = int(max_tries)
        self.margin = int(margin)

    @staticmethod
    def _pad_to(t: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
        import torch.nn.functional as F

        _, d, h, w = t.shape
        td, th, tw = target
        pd = max(0, td - d)
        ph = max(0, th - h)
        pw = max(0, tw - w)
        if pd == ph == pw == 0:
            return t
        return F.pad(t, (0, pw, 0, ph, 0, pd), mode="constant", value=0)

    @staticmethod
    def _crop(t: torch.Tensor, z: int, y: int, x: int, patch: tuple[int, int, int]) -> torch.Tensor:
        pd, ph, pw = patch
        return t[:, z : z + pd, y : y + ph, x : x + pw]

    @staticmethod
    def _rand_start(shape: tuple[int, int, int], patch: tuple[int, int, int]):
        d, h, w = shape
        pd, ph, pw = patch
        z = np.random.randint(0, max(d - pd, 0) + 1) if d > pd else 0
        y = np.random.randint(0, max(h - ph, 0) + 1) if h > ph else 0
        x = np.random.randint(0, max(w - pw, 0) + 1) if w > pw else 0
        return int(z), int(y), int(x)

    def _tumor_start(self, label_3d: torch.Tensor):
        fg = torch.nonzero(label_3d > 0, as_tuple=False)
        if fg.numel() == 0:
            return self._rand_start(tuple(label_3d.shape), self.patch_size)

        zmin = max(0, int(fg[:, 0].min()) - self.margin)
        ymin = max(0, int(fg[:, 1].min()) - self.margin)
        xmin = max(0, int(fg[:, 2].min()) - self.margin)
        zmax = min(label_3d.shape[0] - 1, int(fg[:, 0].max()) + self.margin)
        ymax = min(label_3d.shape[1] - 1, int(fg[:, 1].max()) + self.margin)
        xmax = min(label_3d.shape[2] - 1, int(fg[:, 2].max()) + self.margin)

        cz = np.random.randint(zmin, zmax + 1)
        cy = np.random.randint(ymin, ymax + 1)
        cx = np.random.randint(xmin, xmax + 1)

        pd, ph, pw = self.patch_size
        d, h, w = label_3d.shape
        z = max(0, min(cz - pd // 2, d - pd))
        y = max(0, min(cy - ph // 2, h - ph))
        x = max(0, min(cx - pw // 2, w - pw))
        return int(z), int(y), int(x)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        image = self._pad_to(d[self.image_key], self.patch_size)
        label = self._pad_to(d[self.label_key], self.patch_size)

        best_image = image
        best_label = label
        best_ratio = -1.0

        for _ in range(self.max_tries):
            z, y, x = self._tumor_start(label[0])
            ci = self._crop(image, z, y, x, self.patch_size)
            cl = self._crop(label, z, y, x, self.patch_size)
            fg_ratio = float((cl[0] > 0.5).float().mean().item())

            if fg_ratio > best_ratio:
                best_ratio = fg_ratio
                best_image = ci
                best_label = cl

            if fg_ratio >= self.min_fg_ratio:
                d[self.image_key] = ci
                d[self.label_key] = cl
                return d

        d[self.image_key] = best_image
        d[self.label_key] = best_label
        return d



def build_transforms(modalities: Sequence[str], patch_size: Sequence[int], min_fg_ratio: float, max_tries: int, margin: int, training: bool = True) -> Compose:
    image_keys = [f"image_{m}" for m in modalities]
    keys = list(image_keys) + ["label"]

    ops = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ZScoreNormalizeModalitiesd(keys=image_keys),
        BinaryLabeld(key="label"),
        EnsureTyped(keys=image_keys, dtype=torch.float32),
        EnsureTyped(keys="label", dtype=torch.float32),
        StackModalitiesd(keys=image_keys),
    ]

    if training:
        ops.extend([
            TumorCenteredCropd(
                image_key="image",
                label_key="label",
                patch_size=patch_size,
                min_fg_ratio=min_fg_ratio,
                max_tries=max_tries,
                margin=margin,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                translate_range=(10, 10, 5),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
            ),
            RandIntensityShiftd(keys="image", factors=0.1, prob=0.5),
        ])

    return Compose(ops)
