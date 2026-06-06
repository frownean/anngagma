"""
dataset.py — XRF55 multi-modal dataset
Modalities: WiFi (270,1000), RFID (23,148), mmWave (1,17,256,128)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODALITY_META: Dict[str, Dict[str, Any]] = {
    "wifi":   {"folder": "WiFi",   "shape": (270, 1000)},
    "rfid":   {"folder": "RFID",   "shape": (23, 148)},
    "mmwave": {"folder": "mmWave", "shape": (1, 17, 256, 128)},
}
FILE_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)\.npy$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSpec:
    name: str
    subjects: List[int]
    actions: List[int]
    trials: List[int]


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scene_root(data_root: Path, scene: str) -> Path:
    return data_root / scene / scene


def _sample_path(data_root: Path, scene: str, modality: str,
                 sid: int, aid: int, tid: int) -> Path:
    folder = MODALITY_META[modality]["folder"]
    fname = f"{sid:02d}_{aid:02d}_{tid:02d}.npy"   # zero-padded for consistency
    # also try without zero-padding (original dataset uses plain ints)
    root = _scene_root(data_root, scene) / folder
    for name in (fname, f"{sid}_{aid}_{tid}.npy"):
        p = root / name
        if p.exists():
            return p
    return root / fname   # will not exist — caller handles missing


# ---------------------------------------------------------------------------
# Split builder (discovers available actions/trials from disk)
# ---------------------------------------------------------------------------
def build_splits(data_root: Path, scene: str,
                 train_subjects: List[int],
                 val_subjects: List[int],
                 test_subjects: List[int],
                 action_ids: Optional[List[int]],
                 trials: Optional[List[int]]) -> Dict[str, SplitSpec]:
    """Discover valid (action, trial) pairs from the WiFi folder (most complete)."""
    wifi_folder = _scene_root(data_root, scene) / MODALITY_META["wifi"]["folder"]
    actions_found, trials_found = set(), set()
    if wifi_folder.exists():
        for p in wifi_folder.glob("*.npy"):
            m = FILE_PATTERN.match(p.name)
            if m:
                actions_found.add(int(m.group(2)))
                trials_found.add(int(m.group(3)))
    actions = sorted(action_ids or actions_found or range(1, 56))
    trials_list = sorted(trials or trials_found or range(1, 21))
    specs: Dict[str, SplitSpec] = {}
    for name, subjs in (("train", train_subjects),
                        ("val",   val_subjects),
                        ("test",  test_subjects)):
        specs[name] = SplitSpec(name, subjs, actions, trials_list)
    return specs


# ---------------------------------------------------------------------------
# Per-channel global normalization (computed on training set)
# ---------------------------------------------------------------------------
def compute_normalization_stats(
    data_root: Path,
    scene: str,
    train_spec: SplitSpec,
    modalities: Sequence[str],
    eps: float = 1e-6,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute per-channel mean and std over the training split.
    mmWave is log1p-transformed before stats, matching inference preprocessing.
    Stats are computed in float64 for numerical stability.
    """
    stats: Dict[str, Dict[str, List[float]]] = {}
    for modality in modalities:
        shape = MODALITY_META[modality]["shape"]
        channels = shape[0]
        sums   = np.zeros(channels, dtype=np.float64)
        sq_sums = np.zeros(channels, dtype=np.float64)
        count  = 0
        for sid in train_spec.subjects:
            for aid in train_spec.actions:
                for tid in train_spec.trials:
                    p = _sample_path(data_root, scene, modality, sid, aid, tid)
                    if not p.exists():
                        continue
                    arr = np.load(p).astype(np.float64)
                    # match inference preprocessing
                    arr = _preprocess_raw(arr, modality)
                    flat = arr.reshape(channels, -1)  # (C, *)
                    sums   += flat.sum(axis=1)
                    sq_sums += np.square(flat).sum(axis=1)
                    count  += flat.shape[1]
        if count == 0:
            raise RuntimeError(f"No training samples found for modality={modality}")
        mean = sums / count
        var  = np.maximum(sq_sums / count - np.square(mean), 0.0)
        std  = np.maximum(np.sqrt(var), eps)
        stats[modality] = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


def _preprocess_raw(arr: np.ndarray, modality: str) -> np.ndarray:
    """
    Deterministic, modality-specific transforms applied BEFORE normalization.
    mmWave: clamp negatives → log1p (linearises the heavy-tailed power distribution).
    Others: passthrough.
    """
    arr = arr.astype(np.float32, copy=False)
    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if modality == "mmwave":
        np.maximum(arr, 0.0, out=arr)
        np.log1p(arr, out=arr)
    return arr


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class SensorAugmenter:
    """
    Stochastic augmentations for training only.
    Applied AFTER normalization so noise_std is in normalised units.

    Per-modality config keys:
      noise_std          : additive Gaussian noise std
      scale_range        : [lo, hi] multiplicative scale
      time_shift         : max frames to roll along time axis
      time_shift_prob    : probability of applying time shift
      time_mask_prob     : probability of applying time masking
      time_mask_ratio    : fraction of time axis to mask
      channel_dropout_prob  : probability of applying channel dropout
      channel_dropout_ratio : fraction of channels to zero out
    """

    def __init__(self, cfg: Dict[str, Any], training: bool) -> None:
        self.cfg = cfg or {}
        self.training = training

    @staticmethod
    def _uniform_scalar(rng: Tuple[float, float]) -> float:
        lo, hi = min(rng), max(rng)
        return float(torch.empty(1).uniform_(lo, hi).item())

    def __call__(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        mcfg = self.cfg.get(modality, {})
        if not mcfg or not bool(mcfg.get("enabled", True)):
            return x

        x = x.clone()   # never mutate the cached source tensor

        # multiplicative scale
        if "scale_range" in mcfg:
            x = x * self._uniform_scalar(mcfg["scale_range"])

        # additive Gaussian noise
        noise_std = float(mcfg.get("noise_std", 0.0))
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        # time axis: last dim for 1-D modalities, dim-1 for mmWave (B,1,T,H,W)
        time_dim = 1 if (modality == "mmwave" and x.dim() == 4) else -1
        T = x.size(time_dim)

        # time shift (circular)
        shift = int(mcfg.get("time_shift", 0))
        if shift > 0 and torch.rand(1).item() < float(mcfg.get("time_shift_prob", 0.5)):
            amount = int(torch.randint(-shift, shift + 1, (1,)).item())
            if amount:
                x = torch.roll(x, shifts=amount, dims=time_dim)

        # time masking
        mask_prob  = float(mcfg.get("time_mask_prob",  0.0))
        mask_ratio = float(mcfg.get("time_mask_ratio", 0.0))
        if mask_prob > 0 and mask_ratio > 0 and torch.rand(1).item() < mask_prob:
            width = max(1, min(T, int(round(T * mask_ratio))))
            start = int(torch.randint(0, max(1, T - width + 1), (1,)).item())
            idx = [slice(None)] * x.dim()
            idx[time_dim] = slice(start, start + width)
            x[tuple(idx)] = 0.0

        # channel dropout (zero out entire channels)
        drop_prob  = float(mcfg.get("channel_dropout_prob",  0.0))
        drop_ratio = float(mcfg.get("channel_dropout_ratio", 0.0))
        if drop_prob > 0 and drop_ratio > 0 and torch.rand(1).item() < drop_prob:
            C = x.size(0)
            n_drop = max(1, min(C, int(round(C * drop_ratio))))
            x[torch.randperm(C)[:n_drop]] = 0.0

        return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class XRF55Dataset(Dataset):
    """
    Loads XRF55 multi-modal action recognition samples.

    Preprocessing pipeline per sample:
        raw npy → _preprocess_raw (log1p for mmWave)
                → global per-channel z-score normalization
                → samplewise normalization (optional, per modality)
                → augmentation (training only)

    Args:
        data_root       : path to dataset root
        scene           : scene subfolder name, e.g. "Scene1"
        split_spec      : SplitSpec describing this split
        modalities      : list of modality names to load
        norm_stats      : per-modality {"mean": [...], "std": [...]} dicts
        samplewise_norm : dict mapping modality → mode string or False
                          mode "channel" normalises each channel independently;
                          mode "global"  normalises the whole tensor
        samplewise_eps  : epsilon for samplewise normalisation denominator
        augment_cfg     : per-modality augmentation config (see SensorAugmenter)
        strict_missing  : if True, skip samples with any missing modality file
    """

    def __init__(
        self,
        data_root: Path,
        scene: str,
        split_spec: SplitSpec,
        modalities: Sequence[str],
        norm_stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
        samplewise_norm: Optional[Dict[str, Any]] = None,
        samplewise_eps: float = 1e-6,
        augment_cfg: Optional[Dict[str, Any]] = None,
        strict_missing: bool = True,
    ) -> None:
        self.data_root   = Path(data_root)
        self.scene       = scene
        self.split       = split_spec.name
        self.modalities  = [m.lower() for m in modalities]
        self.norm_stats  = norm_stats or {}
        self.sw_norm     = samplewise_norm or {}
        self.sw_eps      = samplewise_eps
        self.augmenter   = SensorAugmenter(augment_cfg or {}, training=(self.split == "train"))
        self.strict_missing = strict_missing

        for m in self.modalities:
            if m not in MODALITY_META:
                raise ValueError(f"Unknown modality: {m}")

        # ── build sample list ──────────────────────────────────────────────
        candidates = [
            (sid, aid, tid)
            for sid in split_spec.subjects
            for aid in split_spec.actions
            for tid in split_spec.trials
        ]
        self.samples, self.missing_records = self._filter(candidates)
        if not self.samples:
            raise RuntimeError(f"No valid samples for split={self.split}")

    # ------------------------------------------------------------------
    def _filter(
        self, candidates: List[Tuple[int, int, int]]
    ) -> Tuple[List[Tuple[int, int, int]], List[Dict]]:
        if not self.strict_missing:
            return list(candidates), []
        valid, missing = [], []
        for sid, aid, tid in candidates:
            absent = [
                m for m in self.modalities
                if not _sample_path(self.data_root, self.scene, m, sid, aid, tid).exists()
            ]
            if absent:
                missing.append({"subject_id": sid, "action_id": aid,
                                 "trial_id": tid, "missing_modalities": absent})
            else:
                valid.append((sid, aid, tid))
        return valid, missing

    # ------------------------------------------------------------------
    def _load(self, modality: str, sid: int, aid: int, tid: int) -> torch.Tensor:
        expected = tuple(MODALITY_META[modality]["shape"])
        path = _sample_path(self.data_root, self.scene, modality, sid, aid, tid)

        if not path.exists():
            if self.strict_missing:
                raise FileNotFoundError(path)
            return torch.zeros(expected, dtype=torch.float32)

        arr = np.load(path)

        # handle mmWave files saved without the leading channel dim
        if modality == "mmwave" and arr.ndim == 3 and tuple(arr.shape) == expected[1:]:
            arr = arr[None]

        if tuple(arr.shape) != expected:
            if self.strict_missing:
                raise ValueError(f"{path}: shape {arr.shape} != expected {expected}")
            return torch.zeros(expected, dtype=torch.float32)

        arr = _preprocess_raw(arr, modality)

        # ── global per-channel z-score ──────────────────────────────────
        if modality in self.norm_stats:
            s = self.norm_stats[modality]
            mean = np.array(s["mean"], dtype=np.float32)
            std  = np.array(s["std"],  dtype=np.float32).clip(1e-6)
            # broadcast over all trailing dims: shape (C, *) → (C, 1, 1, …)
            C = arr.shape[0]
            view = (C,) + (1,) * (arr.ndim - 1)
            arr = (arr - mean.reshape(view)) / std.reshape(view)

        tensor = torch.from_numpy(np.ascontiguousarray(arr))

        # ── optional samplewise normalisation ──────────────────────────
        mode = self.sw_norm.get(modality, False)
        tensor = _samplewise_norm(tensor, mode, self.sw_eps)

        return tensor

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sid, aid, tid = self.samples[index]
        item: Dict[str, torch.Tensor] = {
            "label":      torch.tensor(aid - 1, dtype=torch.long),
            "subject_id": torch.tensor(sid,     dtype=torch.long),
        }
        for m in self.modalities:
            raw = self._load(m, sid, aid, tid)
            item[m] = self.augmenter(m, raw)
        return item


# ---------------------------------------------------------------------------
# Samplewise normalisation helper
# ---------------------------------------------------------------------------
def _samplewise_norm(tensor: torch.Tensor, mode: Any, eps: float) -> torch.Tensor:
    if mode is None or mode is False:
        return tensor
    s = str(mode).lower()
    if s in {"false", "none", "off", "0", ""}:
        return tensor
    if s in {"channel", "per_channel"}:
        # normalise each channel independently over all other dims
        dims = tuple(range(1, tensor.dim()))
        mu  = tensor.mean(dim=dims, keepdim=True)
        sig = tensor.std(dim=dims,  keepdim=True, unbiased=False).clamp_min(eps)
        return (tensor - mu) / sig
    if s in {"global", "sample", "instance"}:
        return (tensor - tensor.mean()) / tensor.std(unbiased=False).clamp_min(eps)
    raise ValueError(f"Unknown samplewise norm mode: {mode!r}")


# ---------------------------------------------------------------------------
# Normalisation stats: load from cache or compute
# ---------------------------------------------------------------------------
def get_or_compute_norm_stats(
    cache_path: Path,
    data_root: Path,
    scene: str,
    train_spec: SplitSpec,
    modalities: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    if cache_path.exists():
        stats = load_json(cache_path)
        if all(m in stats for m in modalities):
            return stats
        # partial cache: compute only what's missing
        existing = dict(stats)
    else:
        existing = {}

    missing_mods = [m for m in modalities if m not in existing]
    if missing_mods:
        print(f"[dataset] Computing normalisation stats for: {missing_mods}")
        new_stats = compute_normalization_stats(data_root, scene, train_spec, missing_mods)
        existing.update(new_stats)
        save_json(cache_path, existing)
    return existing
