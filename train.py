from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import XRF55Dataset, build_splits, get_or_compute_norm_stats, save_json
from model import AGMA, create_model


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        cursor = updates
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    return updates


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(train_cfg: Dict[str, Any]) -> None:
    deterministic = bool(train_cfg.get("deterministic", False))
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    tf32 = bool(train_cfg.get("tf32", True))
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if tf32 else "highest")


def _worker_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _make_amp(device: torch.device, enabled: bool):
    try:
        from torch.amp import GradScaler, autocast

        scaler = GradScaler(device.type, enabled=enabled and device.type == "cuda")

        def ctx():
            return autocast(device_type=device.type, enabled=enabled and device.type == "cuda")

        return scaler, ctx
    except Exception:
        from torch.cuda.amp import GradScaler, autocast

        scaler = GradScaler(enabled=enabled and device.type == "cuda")

        def ctx():
            return autocast(enabled=enabled and device.type == "cuda")

        return scaler, ctx


def build_datasets(
    cfg: Dict[str, Any],
    modalities: Sequence[str],
    out_dir: Path,
) -> Tuple[XRF55Dataset, XRF55Dataset, XRF55Dataset]:
    dc = cfg["data"]
    data_root = Path(dc["data_root"])
    scene = str(dc["scene"])

    def int_list(key: str) -> List[int]:
        return sorted(int(x) for x in dc[key])

    specs = build_splits(
        data_root=data_root,
        scene=scene,
        train_subjects=int_list("train_subjects"),
        val_subjects=int_list("val_subjects"),
        test_subjects=int_list("test_subjects"),
        action_ids=dc.get("action_ids"),
        trials=dc.get("trials"),
    )

    norm_cfg = dc.get("normalization", {}) or {}
    aug_cfg = dc.get("augment", {}) or {}
    samplewise_norm = norm_cfg.get("samplewise_by_modality", {}) or {}
    samplewise_eps = float(norm_cfg.get("samplewise_eps", 1e-6))
    norm_stats = None
    if bool(norm_cfg.get("enabled", True)):
        cache_path = out_dir / norm_cfg.get("cache_name", "normalization_stats.json")
        if bool(norm_cfg.get("shared_cache", True)):
            import hashlib

            payload = json.dumps(
                {
                    "preprocess_version": 2,
                    "data_root": str(data_root.resolve()),
                    "scene": scene,
                    "modalities": list(modalities),
                    "train_subjects": specs["train"].subjects,
                    "actions": specs["train"].actions,
                    "trials": specs["train"].trials,
                    "samplewise_by_modality": samplewise_norm,
                },
                sort_keys=True,
            )
            digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
            cache_root = Path(norm_cfg.get("cache_root", "outputs_cache/normalization"))
            cache_path = cache_root / f"{scene.lower()}_{'_'.join(modalities)}_{digest}.json"
        norm_stats = get_or_compute_norm_stats(cache_path, data_root, scene, specs["train"], modalities)
        save_json(out_dir / "normalization_stats.json", norm_stats)

    common = dict(
        data_root=data_root,
        scene=scene,
        modalities=modalities,
        norm_stats=norm_stats,
        samplewise_norm=samplewise_norm,
        samplewise_eps=samplewise_eps,
        strict_missing=bool(dc.get("strict_missing", True)),
    )
    train_ds = XRF55Dataset(split_spec=specs["train"], augment_cfg=aug_cfg, **common)
    val_ds = XRF55Dataset(split_spec=specs["val"], augment_cfg=None, **common)
    test_ds = XRF55Dataset(split_spec=specs["test"], augment_cfg=None, **common)

    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if ds.missing_records:
            save_json(out_dir / f"missing_{name}.json", ds.missing_records)
            print(f"[data] {name}: skipped {len(ds.missing_records)} missing samples")

    save_json(
        out_dir / "split_info.json",
        {
            name: {"subjects": specs[name].subjects, "n_samples": len(ds)}
            for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds))
        },
    )
    save_json(
        out_dir / "data_audit.json",
        {
            "modalities": list(modalities),
            "normalization_enabled": bool(norm_cfg.get("enabled", True)),
            "samplewise_by_modality": samplewise_norm,
            "strict_missing": bool(dc.get("strict_missing", True)),
            "splits": {
                name: {
                    "subjects": specs[name].subjects,
                    "actions": len(specs[name].actions),
                    "trials": len(specs[name].trials),
                    "n_samples": len(ds),
                    "missing_records": len(ds.missing_records),
                }
                for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds))
            },
        },
    )
    return train_ds, val_ds, test_ds


def make_loader(ds: XRF55Dataset, cfg: Dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    tc = cfg["train"]
    num_workers = int(tc.get("num_workers", 4))
    kwargs: Dict[str, Any] = {
        "batch_size": int(tc.get("batch_size", 8)),
        "num_workers": num_workers,
        "pin_memory": bool(tc.get("pin_memory", True)) and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0 and bool(tc.get("persistent_workers", True)),
        "worker_init_fn": _worker_init,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(tc.get("prefetch_factor", 2))
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(ds, shuffle=shuffle, **kwargs)


def move_batch(batch: Dict[str, torch.Tensor], modalities: Sequence[str], device: torch.device):
    targets = batch["label"].to(device, non_blocking=True)
    inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
    return targets, inputs


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return 100.0 * (logits.argmax(1) == targets).float().mean().item()


def cross_entropy_with_optional_weights(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if sample_weights is None:
        return criterion(logits, targets)
    weight = getattr(criterion, "weight", None)
    if weight is not None:
        weight = weight.to(device=logits.device, dtype=logits.dtype)
    label_smoothing = float(getattr(criterion, "label_smoothing", 0.0))
    per_sample = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    sample_weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype).clamp_min(0.0)
    return (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)


def sample_active_modalities(
    modalities: Sequence[str],
    dropout_cfg: Optional[Dict[str, Any]],
    train: bool,
) -> Optional[List[str]]:
    if not train or not dropout_cfg or not bool(dropout_cfg.get("enabled", False)):
        return None
    mods = [m.lower() for m in modalities]
    if len(mods) <= 1:
        return None

    full_prob = float(dropout_cfg.get("full_batch_prob", 0.5))
    if random.random() < full_prob:
        return mods

    default_p = float(dropout_cfg.get("p", 0.0))
    drop_probs = dropout_cfg.get("drop_probs", {}) or {}
    keep = set(str(m).lower() for m in dropout_cfg.get("always_keep", []) or [])
    active = []
    for modality in mods:
        if modality in keep:
            active.append(modality)
            continue
        p = float(drop_probs.get(modality, default_p))
        if random.random() >= max(0.0, min(1.0, p)):
            active.append(modality)

    min_keep = max(1, min(len(mods), int(dropout_cfg.get("min_keep", 1))))
    if len(active) < min_keep:
        missing = [m for m in mods if m not in active]
        random.shuffle(missing)
        active.extend(missing[: min_keep - len(active)])
    return active


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    modalities: Sequence[str],
    criterion: nn.Module,
    device: torch.device,
    amp_ctx,
    train: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    grad_clip: float = 0.0,
    active_mods: Optional[Sequence[str]] = None,
    modality_dropout_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Dict[str, float]]:
    model.train(train)
    total_loss = total_acc = total_n = 0.0
    meters: Dict[str, List[float]] = {}
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        targets, inputs = move_batch(batch, modalities, device)
        bsz = targets.size(0)
        with torch.set_grad_enabled(train):
            with amp_ctx():
                batch_active_mods = (
                    list(active_mods)
                    if active_mods is not None
                    else sample_active_modalities(modalities, modality_dropout_cfg, train)
                )
                out = model(inputs, targets=targets, active_modalities=batch_active_mods)
                logits = out["logits"]
                cls_loss = criterion(logits, targets)
                loss = cls_loss + sum(out.get("loss_terms", {}).values())
                aux_total = logits.new_zeros(())
                aux_items = []
                aux_weights = out.get("aux_loss_weights", {}) or {}
                aux_sample_weights = out.get("aux_loss_sample_weights", {}) or {}
                for modality, aux_logit in (out.get("aux_logits", {}) or {}).items():
                    aux_ce = cross_entropy_with_optional_weights(
                        aux_logit,
                        targets,
                        criterion,
                        aux_sample_weights.get(modality),
                    )
                    weight_raw = aux_weights.get(modality, 0.0)
                    if torch.is_tensor(weight_raw):
                        aux_weight = weight_raw.to(device=aux_ce.device, dtype=aux_ce.dtype)
                    else:
                        aux_weight = aux_ce.new_tensor(float(weight_raw))
                    aux_loss = aux_ce * aux_weight
                    aux_total = aux_total + aux_loss
                    aux_items.append((modality, aux_ce, aux_loss, aux_weight, aux_logit))
                if train:
                    loss = loss + aux_total
            if train:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.detach().item() * bsz
        total_acc += accuracy(logits.detach(), targets) * bsz
        total_n += bsz
        meters.setdefault("cls_loss", []).append(cls_loss.detach().item())
        if train and active_mods is None and batch_active_mods is not None:
            meters.setdefault("active_modality_count", []).append(float(len(batch_active_mods)))
            for modality in modalities:
                meters.setdefault(f"active_modality_{modality}", []).append(float(modality in batch_active_mods))
        if aux_items:
            meters.setdefault("anchor_aux_total", []).append(aux_total.detach().item())
            for modality, aux_ce, aux_loss, aux_weight, aux_logit in aux_items:
                meters.setdefault(f"anchor_aux_ce_{modality}", []).append(aux_ce.detach().item())
                meters.setdefault(f"anchor_aux_loss_{modality}", []).append(aux_loss.detach().item())
                meters.setdefault(f"anchor_aux_acc_{modality}", []).append(accuracy(aux_logit.detach(), targets))
                meters.setdefault(f"anchor_aux_weight_{modality}", []).append(aux_weight.detach().item())
        for k, v in out.get("loss_terms", {}).items():
            meters.setdefault(k, []).append(v.detach().item())
        for k, v in out.get("loss_stats", {}).items():
            val = v.detach().item()
            if val == val:
                meters.setdefault(k, []).append(val)
    n = max(1, int(total_n))
    return total_loss / n, total_acc / n, {k: float(np.mean(v)) for k, v in meters.items() if v}


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    modalities: Sequence[str],
    num_classes: int,
    device: torch.device,
    amp_ctx,
    active_mods: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    model.eval()
    mat = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for batch in loader:
        targets, inputs = move_batch(batch, modalities, device)
        with amp_ctx():
            out = model(inputs, targets=None, active_modalities=active_mods)
        preds = out["logits"].argmax(1).cpu()
        for y, p in zip(targets.cpu(), preds):
            mat[int(y), int(p)] += 1
    return mat


def metrics_from_cm(mat: torch.Tensor) -> Dict[str, float]:
    mat = mat.float()
    support = mat.sum(1)
    pred_support = mat.sum(0)
    tp = mat.diag()
    precision = tp / pred_support.clamp_min(1.0)
    recall = tp / support.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    valid = support > 0
    return {
        "accuracy": float(tp.sum() / mat.sum().clamp_min(1.0) * 100),
        "f1_macro": float(f1[valid].mean() * 100) if valid.any() else 0.0,
        "balanced_acc": float(recall[valid].mean() * 100) if valid.any() else 0.0,
    }


@torch.no_grad()
def subject_eval(
    model: nn.Module,
    loader: DataLoader,
    modalities: Sequence[str],
    num_classes: int,
    device: torch.device,
    amp_ctx,
) -> Dict[str, Any]:
    model.eval()
    mats: Dict[int, torch.Tensor] = {}
    for batch in loader:
        targets, inputs = move_batch(batch, modalities, device)
        sids = batch["subject_id"].cpu()
        with amp_ctx():
            out = model(inputs, targets=None)
        preds = out["logits"].argmax(1).cpu()
        for sid, y, p in zip(sids, targets.cpu(), preds):
            sid_int = int(sid)
            mats.setdefault(sid_int, torch.zeros(num_classes, num_classes, dtype=torch.long))
            mats[sid_int][int(y), int(p)] += 1
    rows = []
    for sid in sorted(mats):
        rows.append({"subject_id": sid, "n_samples": int(mats[sid].sum()), **metrics_from_cm(mats[sid])})
    summary: Dict[str, float] = {}
    for key in ("accuracy", "f1_macro", "balanced_acc"):
        vals = [row[key] for row in rows]
        summary[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
        summary[f"std_{key}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"per_subject": rows, "summary": summary}


def missing_modality_variants(modalities: List[str]) -> List[Tuple[str, List[str]]]:
    variants = [("full", modalities)]
    variants += [(f"only_{m}", [m]) for m in modalities]
    if len(modalities) > 1:
        variants += [(f"drop_{m}", [x for x in modalities if x != m]) for m in modalities]
    seen, out = set(), []
    for name, active in variants:
        key = tuple(sorted(active))
        if key not in seen:
            seen.add(key)
            out.append((name, active))
    return out


@torch.no_grad()
def anchor_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    modalities: Sequence[str],
    device: torch.device,
    amp_ctx,
    max_batches: int = 32,
) -> Dict[str, Any]:
    if not isinstance(model, AGMA):
        return {}
    model.eval()
    gate_sums: Dict[str, float] = {m: 0.0 for m in modalities}
    n_gate = 0
    offset_vals: List[float] = []
    assign_entropy: Dict[str, List[float]] = {m: [] for m in modalities}
    positive_weight_vals: Dict[str, List[float]] = {m: [] for m in modalities}
    positive_agreement_vals: Dict[str, List[float]] = {m: [] for m in modalities}
    positive_enhanced_vals: Dict[str, List[float]] = {m: [] for m in modalities}
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        _, inputs = move_batch(batch, modalities, device)
        with amp_ctx():
            out = model(inputs, targets=None, return_debug=True)
        dbg = out["debug"]
        active = dbg["active"]
        weights = dbg["modality_weights"]
        for idx, modality in enumerate(active):
            gate_sums[modality] += float(weights[:, :, idx].mean().cpu())
        pos_weights = dbg.get("positive_anchor_weights")
        pos_agreement = dbg.get("positive_anchor_agreement")
        if pos_weights is not None and pos_agreement is not None:
            threshold = float(getattr(model, "positive_threshold", 0.15))
            for idx, modality in enumerate(active):
                positive_weight_vals[modality].append(float(pos_weights[:, :, idx].mean().cpu()))
                positive_agreement_vals[modality].append(float(pos_agreement[:, :, idx].mean().cpu()))
                positive_enhanced_vals[modality].append(
                    float((pos_agreement[:, :, idx] > threshold).float().mean().cpu())
                )
        n_gate += 1
        anchors_cpu = dbg["anchors"].cpu()
        base_cpu = model.anchor_bank.base_anchors.detach().cpu().unsqueeze(0)
        offset_vals.append(float((anchors_cpu - base_cpu).abs().mean()))
        for modality, assign in dbg["token_assignment"].items():
            p = assign.clamp_min(1e-8)
            assign_entropy[modality].append(float((-(p * p.log()).sum(dim=-1)).mean().cpu()))
    return {
        "mean_modality_gate": {m: gate_sums[m] / max(1, n_gate) for m in gate_sums},
        "mean_anchor_offset_abs": float(np.mean(offset_vals)) if offset_vals else 0.0,
        "mean_token_assignment_entropy": {
            m: float(np.mean(vals)) if vals else 0.0 for m, vals in assign_entropy.items()
        },
        "mean_positive_anchor_weight": {
            m: float(np.mean(vals)) if vals else 0.0 for m, vals in positive_weight_vals.items()
        },
        "mean_positive_anchor_agreement": {
            m: float(np.mean(vals)) if vals else 0.0 for m, vals in positive_agreement_vals.items()
        },
        "mean_positive_anchor_enhanced_ratio": {
            m: float(np.mean(vals)) if vals else 0.0 for m, vals in positive_enhanced_vals.items()
        },
    }


def save_ckpt(path: Path, model: nn.Module, opt, sched, scaler, epoch: int, best_val: float, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val,
            "config": cfg,
        },
        path,
    )


ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "no_alignment": {"model": {"alignment_weight": 0.0}},
    "no_anchor_diversity": {"model": {"anchor_diversity_weight": 0.0, "anchor_repulsion_weight": 0.0}},
    "no_assignment": {"model": {"assignment_balance_weight": 0.0, "assignment_entropy_weight": 0.0}},
    "no_dynamic_anchor": {"model": {"dynamic_anchors": False}},
    "no_agreement_gate": {"model": {"agreement_gate_weight": 0.0}},
    "no_reliability_prior": {"model": {"modality_prior_logit_weight": 0.0, "reference_agreement_weight": 0.0}},
    "no_context_prior": {"model": {"context_prior_weight": 0.0}},
    "no_reference_residual": {"model": {"reference_residual_weight": 0.0}},
    "no_anchor_auxiliary": {"model": {"anchor_auxiliary": {"enabled": False}}},
    "no_positive_anchor_mining": {"model": {"positive_anchor_mining": {"enabled": False}}},
    "no_positive_gate_bias": {"model": {"positive_anchor_mining": {"positive_gate_weight": 0.0}}},
    "no_positive_alignment_weighting": {"model": {"positive_anchor_mining": {"positive_alignment_weight": 0.0}}},
    "no_positive_aux_weighting": {"model": {"positive_anchor_mining": {"positive_aux_weight": 0.0}}},
    "class_positive_contrast": {"model": {"contrast_same_class_positives": True}},
    "J4": {"model": {"J": 4}},
    "J12": {"model": {"J": 12}},
    "J16": {"model": {"J": 16}},
}


def train_one(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", cfg)
    save_json(run_dir / "config.json", cfg)

    tc = cfg["train"]
    ec = cfg["experiment"]
    modalities = [m.lower() for m in ec.get("modalities", cfg["data"].get("modalities", []))]
    model_type = str(ec.get("model_type", "agma")).lower()
    num_classes = int(cfg["model"]["num_classes"])
    seed = int(tc.get("seed", 42))

    set_seed(seed)
    configure_runtime(tc)
    device_name = str(tc.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(device_name)

    train_ds, val_ds, test_ds = build_datasets(cfg, modalities, run_dir)
    train_loader = make_loader(train_ds, cfg, shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, cfg, shuffle=False, seed=seed)
    test_loader = make_loader(test_ds, cfg, shuffle=False, seed=seed)

    model = create_model(model_type, modalities, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model_type} | params={n_params:,} | modalities={modalities}")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(tc.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc.get("lr", 2e-4)),
        weight_decay=float(tc.get("weight_decay", 1e-4)),
    )
    epochs = int(tc.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(tc.get("lr", 2e-4)) * 0.01,
    )
    scaler, amp_ctx = _make_amp(device, bool(tc.get("amp", True)))
    grad_clip = float(tc.get("grad_clip_norm", 5.0))

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    patience = int(tc.get("early_stop_patience", 20))
    min_delta = float(tc.get("early_stop_min_delta", 0.0))
    best_val = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, epochs + 1):
        if hasattr(model, "set_epoch_context"):
            model.set_epoch_context(epoch, epochs)
        tr_loss, tr_acc, tr_aux = run_epoch(
            model,
            train_loader,
            modalities,
            criterion,
            device,
            amp_ctx,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            modality_dropout_cfg=tc.get("modality_dropout", {}) or {},
        )
        val_loss, val_acc, val_aux = run_epoch(model, val_loader, modalities, criterion, device, amp_ctx)
        scheduler.step()

        improved = val_acc > best_val + min_delta
        no_improve = 0 if improved else no_improve + 1
        if improved:
            best_val, best_epoch = val_acc, epoch
            save_ckpt(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_val, cfg)
        save_ckpt(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_val, cfg)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **{f"tr_{k}": v for k, v in tr_aux.items()},
            **{f"val_{k}": v for k, v in val_aux.items()},
        }
        append_jsonl(metrics_path, row)
        print(
            f"epoch={epoch:03d} tr_acc={tr_acc:6.2f} val_acc={val_acc:6.2f} "
            f"tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if patience > 0 and no_improve >= patience:
            print(f"[train] Early stop at epoch {epoch}.")
            break

    best_path = run_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"[eval] Loaded best checkpoint: epoch={best_epoch}, val_acc={best_val:.2f}")

    _, test_acc, test_aux = run_epoch(model, test_loader, modalities, criterion, device, amp_ctx)
    mat = compute_confusion_matrix(model, test_loader, modalities, num_classes, device, amp_ctx)
    test_metrics = metrics_from_cm(mat)
    save_json(run_dir / "confusion_matrix_test.json", mat.tolist())
    subj = subject_eval(model, test_loader, modalities, num_classes, device, amp_ctx)
    save_json(run_dir / "subject_eval.json", subj)
    diag = anchor_diagnostics(model, test_loader, modalities, device, amp_ctx)
    if diag:
        save_json(run_dir / "anchor_diagnostics.json", diag)

    append_jsonl(metrics_path, {"epoch": "test", "test_acc": test_acc, **test_metrics, **{f"test_{k}": v for k, v in test_aux.items()}})

    if len(modalities) > 1 and bool(cfg.get("eval", {}).get("missing_modalities", True)):
        records = []
        for name, active in missing_modality_variants(list(modalities)):
            _, acc, _ = run_epoch(model, test_loader, modalities, criterion, device, amp_ctx, active_mods=active)
            cm = compute_confusion_matrix(model, test_loader, modalities, num_classes, device, amp_ctx, active_mods=active)
            met = metrics_from_cm(cm)
            records.append({"variant": name, "active": active, "acc": acc, **met})
            print(f"[missing] {name:18s} acc={acc:.2f} f1={met['f1_macro']:.2f}")
        save_json(run_dir / "missing_modality_eval.json", records)

    result = {
        "model_type": model_type,
        "modalities": modalities,
        "n_params": n_params,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test": {"acc": test_acc, **test_metrics},
        "test_subjects": subj["summary"],
        "anchor_diagnostics": diag,
    }
    save_json(run_dir / "result.json", result)
    print(f"[done] test_acc={test_acc:.2f} f1_macro={test_metrics['f1_macro']:.2f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pure AGMA training")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model_type", default=None, choices=["agma", "concat_fusion", "late_fusion", "single_modal"])
    p.add_argument("--modalities", nargs="+", default=None, choices=["wifi", "rfid", "mmwave"])
    p.add_argument("--exp_name", default=None)
    p.add_argument("--output_root", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--subfolder", default=None)
    p.add_argument("--ablate", default=None, help=f"Choices: {list(ABLATION_CONFIGS)}")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value")
    return p


def apply_args(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    upd: Dict[str, Any] = {"experiment": {}, "data": {}, "train": {}}
    if args.model_type:
        upd["experiment"]["model_type"] = args.model_type
    if args.modalities:
        upd["experiment"]["modalities"] = args.modalities
        upd["data"]["modalities"] = args.modalities
    if args.exp_name:
        upd["experiment"]["exp_name"] = args.exp_name
    if args.output_root:
        upd["output_root"] = args.output_root
    if args.data_root:
        upd["data"]["data_root"] = args.data_root
    for key in ("device", "epochs", "batch_size", "lr", "seed"):
        value = getattr(args, key)
        if value is not None:
            upd["train"][key] = value
    cfg = _deep_update(cfg, upd)
    if args.ablate:
        if args.ablate not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation {args.ablate!r}")
        cfg = _deep_update(cfg, ABLATION_CONFIGS[args.ablate])
    return _deep_update(cfg, _parse_overrides(args.overrides))


def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_args(load_yaml(Path(args.config)), args)
    ec = cfg["experiment"]
    modalities = [m.lower() for m in ec.get("modalities", cfg["data"].get("modalities", []))]
    model_type = str(ec.get("model_type", "agma")).lower()
    exp_name = str(ec.get("exp_name", f"{model_type}_s{cfg['train'].get('seed', 42)}"))
    if args.subfolder:
        subfolder = args.subfolder
    elif args.ablate:
        subfolder = "ablation"
        exp_name = args.ablate
    elif model_type != "agma":
        subfolder = "comparison"
        exp_name = model_type if model_type != "single_modal" else f"single_{modalities[0]}"
    else:
        subfolder = "main"
    run_dir = Path(cfg.get("output_root", "outputs")) / subfolder / exp_name
    result = train_one(cfg, run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
