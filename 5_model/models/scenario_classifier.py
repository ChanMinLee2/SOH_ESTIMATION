"""
scenario_classifier.py — pluggable scenario (latent-class) classifiers.

All classifiers implement:
    classify(probe_x: Tensor, batch: dict) -> Tensor   # (B, n_classes) logits

Plug-in matrix
--------------
Classifier          CE loss  Needs labels  Typical use
MLPProbeClassifier  yes      no            Default (wraps probe_mlp)
RuleClassifier      yes      no            qfrac / vwindow / protocol axes
CentroidClassifier  yes      fit step      cluster axis, few-shot transfer
OracleClassifier    ~0       yes           Upper-bound ablation
NoneClassifier      no (*)   no            Deng et al. baseline replication

(*) Set lambda_scen=0 in SCRLoss, or check classifier.disables_ce.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ScenarioClassifier(ABC):
    """Abstract base for pluggable scenario classifiers."""

    disables_ce: bool = False

    @abstractmethod
    def classify(
        self, probe_x: torch.Tensor, batch: dict
    ) -> torch.Tensor:
        """
        probe_x : (B, N_HI) direction-gated probe features
        batch   : raw batch dict — may contain seg_idx, level, direction, etc.
        Returns : (B, n_classes) logits — fed directly to cross_entropy
        """


# ---------------------------------------------------------------------------
# MLPProbeClassifier
# ---------------------------------------------------------------------------

class MLPProbeClassifier(ScenarioClassifier, nn.Module):
    """
    Drop-in replacement for the inline probe_mlp in SCRModel.

    Architecture mirrors probe_mlp:
        Linear(n_hi, d_hidden) → ReLU → Dropout
        → Linear(d_hidden, d_hidden//2) → ReLU → Linear(d_hidden//2, n_classes)
    """

    def __init__(
        self,
        n_hi: int,
        n_classes: int,
        d_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(n_hi, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def classify(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        return self.mlp(probe_x)


# ---------------------------------------------------------------------------
# CNNProbeClassifier
# ---------------------------------------------------------------------------

class CNNProbeClassifier(ScenarioClassifier, nn.Module):
    """
    원시 V/I 곡선 CNN 임베딩 + HI probe 융합 분류기 (MODEL_SPECS.md §4.2).

    입력:
        probe_x       : (B, n_hi)          direction-gated probe HI
        batch["x_raw"]: (B, RAW_CH, RAW_N) 원시 곡선 (CNN 입력)
        batch["direction"]: (B,)           +1/-1

    구조:
        RawCNN(x_raw) → cnn_emb (d_cnn)
        [cnn_emb || probe_x || direction] → MLP → n_classes logits

    x_raw 가 없거나 전부 0(구 pkl)이면 CNN 경로는 0 임베딩에 수렴 → probe_x 만으로
    동작(MLP 분류기로 자연 degrade). 별도 예외 없이 호환.
    """

    def __init__(
        self,
        n_hi: int,
        n_classes: int,
        d_cnn: int = 64,
        d_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        nn.Module.__init__(self)
        # 지연 import: torch 모델 계층 순환참조 방지
        from models.raw_cnn import RawCNN

        self.cnn = RawCNN(d_out=d_cnn, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_cnn + n_hi + 1, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, n_classes),
        )

    def _fuse(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        x_raw = batch["x_raw"]                          # (B, RAW_CH, RAW_N)
        cnn_emb = self.cnn(x_raw)                        # (B, d_cnn)
        direction = batch["direction"].to(probe_x.dtype).unsqueeze(1)  # (B, 1)
        return torch.cat([cnn_emb, probe_x, direction], dim=1)

    def forward(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        return self.head(self._fuse(probe_x, batch))

    def classify(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        return self.head(self._fuse(probe_x, batch))


# ---------------------------------------------------------------------------
# RuleClassifier
# ---------------------------------------------------------------------------

class RuleClassifier(ScenarioClassifier):
    """
    Deterministic classifier: infers latent class from seg_idx via ScenarioSpec.

    Suitable for qfrac, vwindow, and protocol axes where each segment
    type deterministically maps to one latent class (lo / mid / hi).
    Returns near-one-hot logits (+/- 1e9) — zero CE loss on correct labels.
    """

    def __init__(self, spec) -> None:
        """spec: ScenarioSpec from common.scenario.base"""
        self._spec = spec
        # lookup table: scenario_id → latent_class index  (CPU tensor)
        lut = [spec.scenario_to_dir_class(i)[1] for i in range(spec.n_scenarios)]
        self._lut = torch.tensor(lut, dtype=torch.long)
        self._n_classes = spec.n_classes

    def classify(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        seg_idx = batch["seg_idx"].long()        # (B,)
        device = probe_x.device
        lut = self._lut.to(device)
        cls_ids = lut[seg_idx]                   # (B,)
        B = probe_x.size(0)
        logits = torch.full(
            (B, self._n_classes), -1e9, dtype=probe_x.dtype, device=device
        )
        logits.scatter_(1, cls_ids.unsqueeze(1), 1e9)
        return logits


# ---------------------------------------------------------------------------
# CentroidClassifier
# ---------------------------------------------------------------------------

class CentroidClassifier(ScenarioClassifier, nn.Module):
    """
    Nearest-centroid classifier in probe-feature space.

    Centroids are fitted via fit() (one-shot) or via update() + finalize()
    (online, per-epoch). Assigns the class whose centroid has minimum L2
    distance; negative squared distance serves as logits.

    Typical use: cluster segmentation axis where each cluster has a natural
    centroid but no closed-form rule.
    """

    def __init__(self, n_hi: int, n_classes: int) -> None:
        nn.Module.__init__(self)
        self._n_classes = n_classes
        self._n_hi = n_hi
        self._fitted = False
        self.register_buffer("_sum",   torch.zeros(n_classes, n_hi))
        self.register_buffer("_count", torch.zeros(n_classes))
        self.register_buffer("centroids", torch.zeros(n_classes, n_hi))

    # ------------------------------------------------------------------
    # Fitting API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, probe_x: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate running sums for one batch. Call each training iteration."""
        for c in range(self._n_classes):
            sel = (labels == c)
            if sel.any():
                self._sum[c]   += probe_x[sel].sum(dim=0)
                self._count[c] += sel.sum().float()

    @torch.no_grad()
    def finalize(self) -> None:
        """Compute centroids from accumulated sums. Call at end of epoch."""
        valid = (self._count > 0)
        self.centroids[valid] = (
            self._sum[valid] / self._count[valid].unsqueeze(1)
        )
        self._sum.zero_()
        self._count.zero_()
        self._fitted = True

    @torch.no_grad()
    def fit(self, probe_x: torch.Tensor, labels: torch.Tensor) -> None:
        """One-shot fit from a full dataset (NumPy arrays or Tensors accepted)."""
        if not isinstance(probe_x, torch.Tensor):
            probe_x = torch.tensor(probe_x, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        self._sum.zero_()
        self._count.zero_()
        self.update(probe_x.to(self._sum.device), labels.to(self._sum.device))
        self.finalize()

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------
    def classify(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        if not self._fitted:
            # uniform logits before fitting
            B = probe_x.size(0)
            return torch.zeros(
                B, self._n_classes, dtype=probe_x.dtype, device=probe_x.device
            )
        centroids = self.centroids.to(probe_x.device)        # (C, N_HI)
        diffs = probe_x.unsqueeze(1) - centroids.unsqueeze(0)  # (B, C, N_HI)
        dist2 = (diffs ** 2).sum(dim=-1)                       # (B, C)
        return -dist2                                           # closer → higher logit


# ---------------------------------------------------------------------------
# OracleClassifier
# ---------------------------------------------------------------------------

class OracleClassifier(ScenarioClassifier):
    """
    Returns ground-truth-aligned logits — CE ≈ 0.

    Measures the regression upper bound achievable if scenario classification
    were perfect. Requires batch["level"] at both train and eval time.
    """

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def classify(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        level = batch["level"].long()      # (B,)
        B = probe_x.size(0)
        logits = torch.full(
            (B, self._n_classes), -1e9, dtype=probe_x.dtype, device=probe_x.device
        )
        logits.scatter_(1, level.unsqueeze(1), 1e9)
        return logits


# ---------------------------------------------------------------------------
# NoneClassifier
# ---------------------------------------------------------------------------

class NoneClassifier(ScenarioClassifier):
    """
    Disables scenario classification (Deng et al. baseline).

    Returns all-zero logits. Set lambda_scen=0 in SCRLoss (or check
    classifier.disables_ce) to fully suppress CE from the loss.
    """

    disables_ce: bool = True

    def __init__(self, n_classes: int) -> None:
        self._n_classes = n_classes

    def classify(self, probe_x: torch.Tensor, batch: dict) -> torch.Tensor:
        B = probe_x.size(0)
        return torch.zeros(
            B, self._n_classes, dtype=probe_x.dtype, device=probe_x.device
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "mlp":      MLPProbeClassifier,
    "cnn":      CNNProbeClassifier,
    "rule":     RuleClassifier,
    "centroid": CentroidClassifier,
    "oracle":   OracleClassifier,
    "none":     NoneClassifier,
}


def build_classifier(
    name: str,
    *,
    n_hi: int,
    n_classes: int,
    spec=None,
    d_hidden: int = 64,
    dropout: float = 0.1,
) -> ScenarioClassifier:
    """
    Factory for ScenarioClassifier instances.

    name      : "mlp" | "rule" | "centroid" | "oracle" | "none"
    n_hi      : HI feature dimension (N_HI)
    n_classes : number of latent classes (spec.n_classes)
    spec      : ScenarioSpec — required for "rule"
    d_hidden  : MLP hidden dim (only "mlp")
    dropout   : dropout rate (only "mlp")
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'. Choose from {sorted(_REGISTRY)}")
    if key == "mlp":
        return MLPProbeClassifier(n_hi, n_classes, d_hidden=d_hidden, dropout=dropout)
    if key == "cnn":
        return CNNProbeClassifier(n_hi, n_classes, d_hidden=d_hidden, dropout=dropout)
    if key == "rule":
        if spec is None:
            raise ValueError("spec is required for RuleClassifier")
        return RuleClassifier(spec)
    if key == "centroid":
        return CentroidClassifier(n_hi, n_classes)
    if key == "oracle":
        return OracleClassifier(n_classes)
    # key == "none"
    return NoneClassifier(n_classes)
