from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, requested: int) -> int:
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _gn(channels: int, requested: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(_groups(channels, requested), channels)


def _factor_grid(n_tokens: int, native_h: int, native_w: int) -> Tuple[int, int]:
    if n_tokens <= 0 or n_tokens > native_h * native_w:
        raise ValueError(f"Cannot pool {native_h}x{native_w} into {n_tokens} tokens.")
    aspect = native_h / max(1, native_w)
    best: Optional[Tuple[float, int, int]] = None
    out = (1, n_tokens)
    for h in range(1, min(native_h, n_tokens) + 1):
        if n_tokens % h != 0:
            continue
        w = n_tokens // h
        if w > native_w:
            continue
        score = abs(math.log(max(h / max(1, w), 1e-6) / aspect))
        cand = (score, h, w)
        if best is None or cand < best:
            best = cand
            out = (h, w)
    return out


def _mlp_head(d_model: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, num_classes),
    )


class Conv2dGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        groups: int = 8,
    ) -> None:
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False),
            _gn(out_ch, groups),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, groups: int = 8, dropout: float = 0.05) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = _gn(channels, groups)
        self.norm2 = _gn(channels, groups)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.depthwise(x)))
        y = self.drop(self.norm2(self.pointwise(y)))
        return self.act(x + y)


class ChannelAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=-1)
        peak = x.amax(dim=-1)
        gate = torch.sigmoid(self.mlp(avg) + self.mlp(peak)).unsqueeze(-1)
        return x * gate


class WiFiEncoder(nn.Module):
    """Spatial-frequency CNN plus temporal TCN. Output: (B, T, N, D)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"]
        d_model = int(m.get("D", 256))
        width = int((m.get("wifi", {}) or {}).get("width", 64))
        groups = int(m.get("norm_groups", 8))
        self.time_bins = int(m["time_bins_by_modality"]["wifi"])
        self.tokens = int(m["token_budgets"]["wifi"])

        self.sf_cnn = nn.Sequential(
            Conv2dGNAct(1, width, (9, 7), stride=(2, 2), groups=groups),
            Conv2dGNAct(width, width, (5, 5), groups=groups),
            Conv2dGNAct(width, width * 2, (5, 5), stride=(2, 2), groups=groups),
            Conv2dGNAct(width * 2, width * 2, (3, 5), groups=groups),
            Conv2dGNAct(width * 2, d_model, (3, 3), groups=groups),
        )
        self.temporal_tcn = nn.Sequential(
            TCNBlock(d_model, dilation=1, groups=groups),
            TCNBlock(d_model, dilation=2, groups=groups),
            TCNBlock(d_model, dilation=4, groups=groups),
            TCNBlock(d_model, dilation=8, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.sf_cnn(x.unsqueeze(1))
        grid = F.adaptive_avg_pool2d(feat, (self.tokens, self.time_bins))
        grid = grid.permute(0, 3, 2, 1).contiguous()

        temporal = feat.mean(dim=2)
        temporal = self.temporal_tcn(temporal)
        temporal = F.adaptive_avg_pool1d(temporal, self.time_bins)
        temporal = temporal.transpose(1, 2).unsqueeze(2)
        return grid + temporal


class RFIDEncoder(nn.Module):
    """Channel projection, channel attention, and lightweight TCN. Output: (B, T, N, D)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"]
        d_model = int(m.get("D", 256))
        width = int((m.get("rfid", {}) or {}).get("width", 48))
        groups = int(m.get("norm_groups", 8))
        self.time_bins = int(m["time_bins_by_modality"]["rfid"])
        self.tokens = int(m["token_budgets"]["rfid"])

        self.proj = nn.Sequential(
            nn.Conv1d(23, width, kernel_size=1, bias=False),
            _gn(width, groups),
            nn.GELU(),
        )
        self.channel_attention = ChannelAttention1D(width)
        self.tcn = nn.Sequential(
            TCNBlock(width, dilation=1, groups=groups),
            TCNBlock(width, dilation=2, groups=groups),
            TCNBlock(width, dilation=4, groups=groups),
        )
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, d_model, kernel_size=1, bias=False),
            _gn(d_model, groups),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.proj(x)
        feat = self.channel_attention(feat)
        feat = self.out_proj(self.tcn(feat))
        target = self.time_bins * self.tokens
        feat = F.adaptive_avg_pool1d(feat, target)
        b, d, _ = feat.shape
        return feat.view(b, d, self.time_bins, self.tokens).permute(0, 2, 3, 1).contiguous()


class R2Plus1DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, spatial_stride: int = 1) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(1, 3, 3),
                stride=(1, spatial_stride, spatial_stride),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch),
        )
        if in_ch != out_ch or spatial_stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=(1, spatial_stride, spatial_stride), bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.temporal(self.spatial(x)) + self.skip(x))


class MmWaveEncoder(nn.Module):
    """R(2+1)D encoder for log-normalized mmWave heatmaps. Output: (B, T, N, D)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        m = cfg["model"]
        d_model = int(m.get("D", 256))
        width = int((m.get("mmwave", {}) or {}).get("width", 32))
        self.time_bins = int(m["time_bins_by_modality"]["mmwave"])
        self.tokens = int(m["token_budgets"]["mmwave"])

        self.stem = nn.Sequential(
            nn.Conv3d(1, width, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(width),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(width),
            nn.GELU(),
        )
        self.stage1 = R2Plus1DBlock(width, width * 2, spatial_stride=2)
        self.stage2 = R2Plus1DBlock(width * 2, width * 4, spatial_stride=2)
        self.stage3 = R2Plus1DBlock(width * 4, d_model, spatial_stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stage3(self.stage2(self.stage1(self.stem(x))))
        h_bins, w_bins = _factor_grid(self.tokens, feat.size(-2), feat.size(-1))
        feat = F.adaptive_avg_pool3d(feat, (self.time_bins, h_bins, w_bins))
        b, d, t, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 4, 1).reshape(b, t, h * w, d)
        if h * w == self.tokens:
            return feat.contiguous()
        feat = feat.reshape(b * t, h * w, d).transpose(1, 2)
        feat = F.adaptive_avg_pool1d(feat, self.tokens).transpose(1, 2)
        return feat.reshape(b, t, self.tokens, d).contiguous()


def _build_encoder(modality: str, cfg: Dict[str, Any]) -> nn.Module:
    modality = modality.lower()
    if modality == "wifi":
        return WiFiEncoder(cfg)
    if modality == "rfid":
        return RFIDEncoder(cfg)
    if modality == "mmwave":
        return MmWaveEncoder(cfg)
    raise ValueError(f"Unknown modality: {modality}")


class PositionEmbedding4D(nn.Module):
    def __init__(self, d_model: int, time_bins: int, tokens: int) -> None:
        super().__init__()
        self.time_bins = int(time_bins)
        self.tokens = int(tokens)
        self.time_embed = nn.Parameter(torch.zeros(time_bins, d_model))
        self.token_embed = nn.Parameter(torch.zeros(tokens, d_model))
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.token_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,T,N,D), got {tuple(x.shape)}.")
        b, t, n, d = x.shape
        if t != self.time_bins or n != self.tokens:
            raise ValueError(f"Expected T,N=({self.time_bins},{self.tokens}), got ({t},{n}).")
        pos = self.time_embed[:, None, :] + self.token_embed[None, :, :]
        return x + pos.unsqueeze(0)


class SequencePooler(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = max(32, d_model // 4)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1), weights


class SemanticAnchorBank(nn.Module):
    def __init__(self, d_model: int, n_anchors: int, offset_scale: float, dynamic: bool) -> None:
        super().__init__()
        self.n_anchors = int(n_anchors)
        self.offset_scale = float(offset_scale)
        self.dynamic = bool(dynamic)
        self.base_anchors = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)
        if self.dynamic:
            self.offset_net = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, n_anchors * d_model),
            )

    def forward(self, sample_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = sample_context.size(0)
        base = self.base_anchors.unsqueeze(0).expand(b, -1, -1)
        if not self.dynamic:
            return base, base.new_zeros(base.shape)
        raw = self.offset_net(sample_context).view(b, self.n_anchors, -1)
        offset = self.offset_scale * torch.tanh(raw)
        return base + offset, offset


class AnchorSetClassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_anchors: int,
        num_classes: int,
        dropout: float,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.mixer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        else:
            self.mixer = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(n_anchors * d_model),
            nn.Linear(n_anchors * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        mixed = self.mixer(self.norm(anchors))
        return self.head(mixed.reshape(mixed.size(0), -1))


class AGMA(nn.Module):
    """Pure anchor-guided multimodal alignment for action recognition."""

    def __init__(self, modalities: Sequence[str], cfg: Dict[str, Any]) -> None:
        super().__init__()
        if not modalities:
            raise ValueError("AGMA requires at least one modality.")
        self.modalities = [m.lower() for m in modalities]
        m = cfg["model"]
        self.d_model = int(m.get("D", 256))
        self.num_classes = int(m.get("num_classes", 55))
        self.n_anchors = int(m.get("J", 8))
        self.temperature = float(m.get("anchor_temperature", 0.22))
        self.agreement_gate_weight = float(m.get("agreement_gate_weight", 0.5))
        self.diversity_weight = float(m.get("anchor_diversity_weight", 5e-4))
        self.dynamic_diversity_weight = float(m.get("dynamic_anchor_diversity_weight", 0.5))
        self.repulsion_weight = float(m.get("anchor_repulsion_weight", 0.002))
        self.alignment_weight = float(m.get("alignment_weight", 0.02))
        self.alignment_margin = float(m.get("alignment_margin", 0.20))
        self.alignment_warmup = float(m.get("alignment_warmup_ratio", 0.05))
        self.contrastive_alignment_weight = float(m.get("contrastive_alignment_weight", 0.0))
        self.contrastive_temperature = float(m.get("contrastive_temperature", 0.12))
        self.assignment_consistency_weight = float(m.get("assignment_consistency_weight", 0.0))
        self.assignment_balance_weight = float(m.get("assignment_balance_weight", 0.0))
        self.assignment_entropy_weight = float(m.get("assignment_entropy_weight", 0.0015))
        self.gate_entropy_weight = float(m.get("gate_entropy_weight", 0.0))
        self.gate_balance_weight = float(m.get("gate_balance_weight", 0.0))
        self.gate_floor = float(m.get("gate_floor", 0.0))
        self.modality_prior = {
            modality: float((m.get("modality_prior", {}) or {}).get(modality, 1.0))
            for modality in self.modalities
        }
        dropout = float(m.get("dropout", 0.25))

        self.encoders = nn.ModuleDict()
        self.positions = nn.ModuleDict()
        self.modality_embeddings = nn.ParameterDict()
        for modality in self.modalities:
            time_bins = int(m["time_bins_by_modality"][modality])
            tokens = int(m["token_budgets"][modality])
            self.encoders[modality] = _build_encoder(modality, cfg)
            self.positions[modality] = PositionEmbedding4D(self.d_model, time_bins, tokens)
            self.modality_embeddings[modality] = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))
            nn.init.normal_(self.modality_embeddings[modality], std=0.02)

        hidden = max(32, self.d_model // 4)
        self.anchor_bank = SemanticAnchorBank(
            self.d_model,
            self.n_anchors,
            float(m.get("anchor_offset_scale", 0.035)),
            bool(m.get("dynamic_anchors", True)),
        )
        self.anchor_norm = nn.LayerNorm(self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.modality_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.classifier = AnchorSetClassifier(
            d_model=self.d_model,
            n_anchors=self.n_anchors,
            num_classes=self.num_classes,
            dropout=dropout,
            layers=int(m.get("anchor_layers", 1)),
            heads=int(m.get("anchor_heads", 4)),
        )
        aux_cfg = m.get("anchor_auxiliary", {}) or {}
        self.anchor_aux_enabled = bool(aux_cfg.get("enabled", False))
        aux_weights_cfg = aux_cfg.get("weights", {}) or {}
        self.anchor_aux_weights = {
            modality: float(aux_weights_cfg.get(modality, 0.0)) for modality in self.modalities
        }
        self.anchor_aux_classifiers = nn.ModuleDict()
        if self.anchor_aux_enabled:
            aux_layers = int(aux_cfg.get("layers", 0))
            aux_heads = int(aux_cfg.get("heads", m.get("anchor_heads", 4)))
            for modality in self.modalities:
                if self.anchor_aux_weights.get(modality, 0.0) > 0:
                    self.anchor_aux_classifiers[modality] = AnchorSetClassifier(
                        d_model=self.d_model,
                        n_anchors=self.n_anchors,
                        num_classes=self.num_classes,
                        dropout=dropout,
                        layers=aux_layers,
                        heads=aux_heads,
                    )
        self._progress = 1.0

    def set_epoch_context(self, epoch: int, total_epochs: int) -> None:
        if total_epochs <= 1:
            self._progress = 1.0
        else:
            self._progress = (max(1, min(epoch, total_epochs)) - 1) / float(total_epochs - 1)

    def _active_alignment_weight(self) -> float:
        if self.alignment_weight <= 0:
            return 0.0
        if self._progress <= self.alignment_warmup:
            return 0.0
        span = max(1.0 - self.alignment_warmup, 1e-6)
        return self.alignment_weight * min(1.0, (self._progress - self.alignment_warmup) / span)

    def _encode(self, inputs: Dict[str, torch.Tensor], active: Sequence[str]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for modality in active:
            tokens = self.encoders[modality](inputs[modality])
            tokens = self.positions[modality](tokens)
            out[modality] = tokens + self.modality_embeddings[modality]
        return out

    @staticmethod
    def _flatten(tokens: torch.Tensor) -> torch.Tensor:
        b, t, n, d = tokens.shape
        return tokens.reshape(b, t * n, d)

    def _sample_context(self, token_dict: Dict[str, torch.Tensor], active: Sequence[str]) -> torch.Tensor:
        summaries = [self._flatten(token_dict[m]).mean(dim=1) for m in active]
        return torch.stack(summaries, dim=1).mean(dim=1)

    def _anchor_aggregate(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.normalize(self.anchor_norm(anchors), dim=-1)
        k = F.normalize(self.token_norm(tokens), dim=-1)
        logits = torch.bmm(q, k.transpose(1, 2)) / max(self.temperature, 1e-6)
        anchor_to_token = torch.softmax(logits, dim=-1)
        token_to_anchor = torch.softmax(logits.transpose(1, 2), dim=-1)
        aligned = torch.bmm(anchor_to_token, tokens)
        strength = token_to_anchor.mean(dim=1).detach()
        return aligned, anchor_to_token, token_to_anchor, strength

    @staticmethod
    def _diversity_loss(anchors: torch.Tensor) -> torch.Tensor:
        if anchors.size(-2) <= 1:
            return anchors.new_zeros(())
        a = F.normalize(anchors, dim=-1)
        sim = torch.matmul(a, a.transpose(-1, -2))
        mask = ~torch.eye(a.size(-2), device=a.device, dtype=torch.bool)
        if sim.dim() == 2:
            return sim[mask].pow(2).mean()
        mask = mask.unsqueeze(0).expand(sim.size(0), -1, -1)
        return sim.masked_select(mask).pow(2).mean()

    def _assignment_balance_loss(self, assignments: Sequence[torch.Tensor]) -> torch.Tensor:
        if not assignments:
            return self.anchor_bank.base_anchors.new_zeros(())
        losses = []
        for assign in assignments:
            mean_assign = assign.mean(dim=(0, 1)).clamp_min(1e-8)
            uniform = mean_assign.new_full(mean_assign.shape, 1.0 / mean_assign.numel())
            losses.append((mean_assign * (mean_assign / uniform).log()).sum())
        return torch.stack(losses).mean()

    @staticmethod
    def _assignment_entropy(assignments: Sequence[torch.Tensor]) -> torch.Tensor:
        if not assignments:
            raise ValueError("assignments must be non-empty")
        vals = []
        for assign in assignments:
            p = assign.clamp_min(1e-8)
            vals.append(-(p * p.log()).sum(dim=-1).mean())
        return torch.stack(vals).mean()

    def _alignment_loss(
        self,
        stack: torch.Tensor,
        strengths: torch.Tensor,
        modality_weights: torch.Tensor,
    ) -> torch.Tensor:
        if stack.size(2) <= 1:
            return stack.new_zeros(())
        z = F.normalize(stack, dim=-1)
        center_weight = modality_weights.detach().unsqueeze(-1)
        center = F.normalize((center_weight * z).sum(dim=2), dim=-1).detach()
        dist = 1.0 - (z * center.unsqueeze(2)).sum(dim=-1)
        penalty = F.relu(dist - self.alignment_margin)
        weight = (strengths * modality_weights.detach()).clamp_min(1e-4)
        return (penalty * weight).sum() / weight.sum().clamp_min(1e-6)

    def _contrastive_alignment_loss(
        self,
        stack: torch.Tensor,
        targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if stack.size(2) <= 1:
            return stack.new_zeros(())
        bsz, n_anchors, n_mods, dim = stack.shape
        if bsz * n_mods <= 1:
            return stack.new_zeros(())

        features = F.normalize(stack, dim=-1)
        features = features.permute(1, 0, 2, 3).reshape(n_anchors, bsz * n_mods, dim)
        sample_ids = torch.arange(bsz, device=stack.device).repeat_interleave(n_mods)
        modality_ids = torch.arange(n_mods, device=stack.device).repeat(bsz)

        same_sample = sample_ids[:, None].eq(sample_ids[None, :])
        cross_modal = modality_ids[:, None].ne(modality_ids[None, :])
        positives = same_sample & cross_modal
        if targets is not None and targets.numel() == bsz:
            labels = targets.to(device=stack.device).repeat_interleave(n_mods)
            same_label = labels[:, None].eq(labels[None, :])
            positives = positives | (same_label & cross_modal)

        eye = torch.eye(bsz * n_mods, device=stack.device, dtype=torch.bool)
        logits = torch.matmul(features, features.transpose(1, 2)) / max(self.contrastive_temperature, 1e-6)
        logits = logits.masked_fill(eye.unsqueeze(0), -torch.finfo(logits.dtype).max)
        logits = logits - logits.amax(dim=-1, keepdim=True).detach()
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        pos_mask = positives.unsqueeze(0).expand(n_anchors, -1, -1)
        pos_count = pos_mask.sum(dim=-1)
        valid = pos_count > 0
        if not valid.any():
            return stack.new_zeros(())
        pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=-1)
        loss = -pos_log_prob[valid] / pos_count[valid].to(log_prob.dtype)
        return loss.mean()

    @staticmethod
    def _assignment_consistency_loss(strengths: torch.Tensor) -> torch.Tensor:
        if strengths.size(2) <= 1:
            return strengths.new_zeros(())
        profile = strengths.clamp_min(1e-8)
        profile = profile / profile.sum(dim=1, keepdim=True).clamp_min(1e-8)
        center = profile.mean(dim=2, keepdim=True).clamp_min(1e-8)
        return (profile * (profile / center).log()).sum(dim=1).mean()

    def _gate_balance_loss(self, modality_weights: torch.Tensor, active: Sequence[str]) -> torch.Tensor:
        if modality_weights.size(2) <= 1:
            return modality_weights.new_zeros(())
        mean_gate = modality_weights.mean(dim=(0, 1)).clamp_min(1e-8)
        prior = mean_gate.new_tensor([max(self.modality_prior.get(m, 1.0), 1e-8) for m in active])
        target = prior / prior.sum().clamp_min(1e-8)
        return (mean_gate * (mean_gate / target).log()).sum()

    def _fuse_modalities(self, stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if stack.size(2) == 1:
            weights = stack.new_ones(stack.shape[:3])
            return stack[:, :, 0], weights
        gate_logits = self.modality_gate(stack).squeeze(-1)
        z = F.normalize(stack, dim=-1)
        consensus = F.normalize(z.mean(dim=2), dim=-1)
        agreement = (z * consensus.unsqueeze(2)).sum(dim=-1)
        gate_logits = gate_logits + self.agreement_gate_weight * agreement
        weights = torch.softmax(gate_logits, dim=2)
        if self.gate_floor > 0:
            floor = min(self.gate_floor, (1.0 / stack.size(2)) - 1e-6)
            weights = weights * (1.0 - floor * stack.size(2)) + floor
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * stack).sum(dim=2), weights

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        active = [m for m in (active_modalities or self.modalities) if m in self.modalities]
        if not active:
            raise ValueError("No active modalities.")

        token_4d = self._encode(inputs, active)
        context = self._sample_context(token_4d, active)
        anchors, offset = self.anchor_bank(context)

        z_list: List[torch.Tensor] = []
        assign_list: List[torch.Tensor] = []
        strength_list: List[torch.Tensor] = []
        attn_list: List[torch.Tensor] = []
        for modality in active:
            flat = self._flatten(token_4d[modality])
            z, attn, assign, strength = self._anchor_aggregate(flat, anchors)
            z_list.append(z)
            attn_list.append(attn)
            assign_list.append(assign)
            strength_list.append(strength)

        stack = torch.stack(z_list, dim=2)
        strengths = torch.stack(strength_list, dim=2)
        aux_logits: Dict[str, torch.Tensor] = {}
        if self.anchor_aux_enabled:
            for modality, z in zip(active, z_list):
                if modality in self.anchor_aux_classifiers:
                    aux_logits[modality] = self.anchor_aux_classifiers[modality](z)
        fused_anchors, modality_weights = self._fuse_modalities(stack)
        logits = self.classifier(fused_anchors)

        loss_terms: Dict[str, torch.Tensor] = {}
        loss_stats: Dict[str, torch.Tensor] = {}
        if self.diversity_weight > 0:
            static_div = self._diversity_loss(self.anchor_bank.base_anchors)
            dynamic_div = self._diversity_loss(anchors)
            div = static_div + self.dynamic_diversity_weight * dynamic_div
            loss_terms["anchor_diversity"] = self.diversity_weight * div
            loss_stats["anchor_diversity_raw"] = div.detach()
        if self.repulsion_weight > 0:
            rep = self._diversity_loss(fused_anchors)
            loss_terms["anchor_repulsion"] = self.repulsion_weight * rep
            loss_stats["anchor_repulsion_raw"] = rep.detach()
        if self.assignment_balance_weight > 0:
            bal = self._assignment_balance_loss(assign_list)
            loss_terms["assignment_balance"] = self.assignment_balance_weight * bal
            loss_stats["assignment_balance_raw"] = bal.detach()
        if self.assignment_entropy_weight > 0:
            ent = self._assignment_entropy(assign_list)
            loss_terms["assignment_entropy"] = self.assignment_entropy_weight * ent
            loss_stats["assignment_entropy_raw"] = ent.detach()
        align_w = self._active_alignment_weight()
        if align_w > 0 and len(active) > 1:
            align = self._alignment_loss(stack, strengths, modality_weights)
            loss_terms["cross_modal_alignment"] = align_w * align
            loss_stats["cross_modal_alignment_raw"] = align.detach()
        if self.contrastive_alignment_weight > 0 and len(active) > 1:
            con = self._contrastive_alignment_loss(stack, targets)
            loss_terms["contrastive_alignment"] = self.contrastive_alignment_weight * con
            loss_stats["contrastive_alignment_raw"] = con.detach()
        if self.assignment_consistency_weight > 0 and len(active) > 1:
            con_assign = self._assignment_consistency_loss(strengths)
            loss_terms["assignment_consistency"] = self.assignment_consistency_weight * con_assign
            loss_stats["assignment_consistency_raw"] = con_assign.detach()
        if self.gate_balance_weight > 0 and len(active) > 1:
            gate_bal = self._gate_balance_loss(modality_weights, active)
            loss_terms["gate_balance"] = self.gate_balance_weight * gate_bal
            loss_stats["gate_balance_raw"] = gate_bal.detach()
        if self.gate_entropy_weight > 0 and len(active) > 1:
            gate_ent = _entropy(modality_weights.detach())
            loss_terms["gate_entropy"] = self.gate_entropy_weight * _entropy(modality_weights)
            loss_stats["gate_entropy_raw"] = gate_ent

        loss_stats["alignment_weight"] = logits.new_tensor(align_w)
        loss_stats["anchor_offset_abs"] = offset.detach().abs().mean()
        loss_stats["modality_gate_entropy"] = _entropy(modality_weights.detach())
        for idx, modality in enumerate(active):
            loss_stats[f"modality_gate_{modality}"] = modality_weights.detach()[:, :, idx].mean()
            loss_stats[f"assignment_strength_{modality}"] = strengths.detach()[:, :, idx].mean()
        loss_stats["modality_gate_min"] = modality_weights.detach().amin()
        loss_stats["modality_gate_max"] = modality_weights.detach().amax()
        out: Dict[str, Any] = {"logits": logits, "loss_terms": loss_terms, "loss_stats": loss_stats}
        if aux_logits:
            out["aux_logits"] = aux_logits
            out["aux_loss_weights"] = {
                modality: logits.new_tensor(self.anchor_aux_weights.get(modality, 0.0))
                for modality in aux_logits
            }
        if return_debug:
            out["debug"] = {
                "active": active,
                "anchors": anchors.detach(),
                "fused_anchors": fused_anchors.detach(),
                "modality_weights": modality_weights.detach(),
                "anchor_attention": {m: a.detach() for m, a in zip(active, attn_list)},
                "token_assignment": {m: a.detach() for m, a in zip(active, assign_list)},
            }
        return out


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1).mean()


class _PooledEncoder(nn.Module):
    def __init__(self, modality: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.modality = modality.lower()
        m = cfg["model"]
        self.encoder = _build_encoder(self.modality, cfg)
        self.position = PositionEmbedding4D(
            int(m.get("D", 256)),
            int(m["time_bins_by_modality"][self.modality]),
            int(m["token_budgets"][self.modality]),
        )
        self.pool = SequencePooler(int(m.get("D", 256)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.position(self.encoder(x))
        b, t, n, d = tokens.shape
        feat, _ = self.pool(tokens.reshape(b, t * n, d))
        return feat


class SingleModal(nn.Module):
    def __init__(self, modality: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.modality = modality.lower()
        m = cfg["model"]
        self.extractor = _PooledEncoder(self.modality, cfg)
        self.classifier = _mlp_head(int(m.get("D", 256)), int(m.get("num_classes", 55)), float(m.get("dropout", 0.2)))

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        feat = self.extractor(inputs[self.modality])
        return {"logits": self.classifier(feat), "loss_terms": {}, "loss_stats": {}}


class ConcatFusion(nn.Module):
    def __init__(self, modalities: Sequence[str], cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.modalities = [m.lower() for m in modalities]
        m = cfg["model"]
        d_model = int(m.get("D", 256))
        self.extractors = nn.ModuleDict({mod: _PooledEncoder(mod, cfg) for mod in self.modalities})
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * len(self.modalities)),
            nn.Linear(d_model * len(self.modalities), d_model),
            nn.GELU(),
        )
        self.classifier = _mlp_head(d_model, int(m.get("num_classes", 55)), float(m.get("dropout", 0.2)))

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        active = set(active_modalities or self.modalities)
        ref = next(iter(inputs.values()))
        feats = []
        d_model = self.proj[1].out_features
        for modality in self.modalities:
            if modality in active:
                feats.append(self.extractors[modality](inputs[modality]))
            else:
                feats.append(ref.new_zeros(ref.size(0), d_model))
        return {"logits": self.classifier(self.proj(torch.cat(feats, dim=-1))), "loss_terms": {}, "loss_stats": {}}


class LateFusion(nn.Module):
    def __init__(self, modalities: Sequence[str], cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.modalities = [m.lower() for m in modalities]
        m = cfg["model"]
        d_model = int(m.get("D", 256))
        self.extractors = nn.ModuleDict({mod: _PooledEncoder(mod, cfg) for mod in self.modalities})
        self.classifiers = nn.ModuleDict(
            {mod: _mlp_head(d_model, int(m.get("num_classes", 55)), float(m.get("dropout", 0.2))) for mod in self.modalities}
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        active = [m for m in (active_modalities or self.modalities) if m in self.modalities]
        if not active:
            raise ValueError("No active modalities.")
        logits = [self.classifiers[m](self.extractors[m](inputs[m])) for m in active]
        return {"logits": torch.stack(logits, dim=0).mean(dim=0), "loss_terms": {}, "loss_stats": {}}


def create_model(model_type: str, modalities: Sequence[str], cfg: Dict[str, Any]) -> nn.Module:
    model_type = model_type.lower()
    mods = [m.lower() for m in modalities]
    if model_type == "agma":
        return AGMA(mods, cfg)
    if model_type == "single_modal":
        if len(mods) != 1:
            raise ValueError("single_modal requires exactly one modality.")
        return SingleModal(mods[0], cfg)
    if model_type == "concat_fusion":
        return ConcatFusion(mods, cfg)
    if model_type == "late_fusion":
        return LateFusion(mods, cfg)
    raise ValueError("model_type must be one of: agma, single_modality, concat_fusion, late_fusion")
