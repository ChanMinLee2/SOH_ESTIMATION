"""
cap_heads.py — Capacity Head 팩토리.

Phase 2에서 yaml의 model.regression_model 값에 따라 회귀 헤드를 교체한다.
Phase 1은 항상 MLPHead (build_cap_head 호출 시 model_cfg 미전달).

지원 모델:
  mlp          : 기본 MLP 2-hidden-layer (기존 동작과 완전 동일)
  transformer  : Transformer Encoder — probe/scen/meta 3 semantic 토큰
  i_transformer: Inverted Transformer — 130개 피처 각각이 토큰 (feature-wise attention)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.hi_schema import N_HI

_HEAD_IN = N_HI + N_HI + 1 + 1  # 64+64+1+1 = 130


# ---------------------------------------------------------------------------
# 1. MLP (baseline)
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """기존 cap_head와 동일한 2-hidden-layer MLP."""

    def __init__(self, d_head: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_HEAD_IN, d_head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_head, d_head // 2),
            nn.ReLU(),
            nn.Linear(d_head // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 130)  →  (B,)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Transformer
# ---------------------------------------------------------------------------

class TransformerHead(nn.Module):
    """
    130-dim 입력을 3개의 semantic 토큰으로 분할해 Transformer Encoder에 입력한다.
      Token 0: probe_x  (64-dim)  — 방향별 probe gate 출력
      Token 1: scen_x   (64-dim)  — 시나리오 gate 출력
      Token 2: meta     (2-dim)   — direction + cap_init

    각 토큰을 d_model로 선형 투영 후 TransformerEncoder → mean pool → 스칼라 출력.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.probe_embed = nn.Linear(N_HI, d_model)
        self.scen_embed  = nn.Linear(N_HI, d_model)
        self.meta_embed  = nn.Linear(2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # Pre-LN: 학습 안정성 향상
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.output  = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 130)
        probe_x = x[:, :N_HI]           # (B, 64)
        scen_x  = x[:, N_HI: 2 * N_HI] # (B, 64)
        meta    = x[:, 2 * N_HI:]       # (B, 2)

        t0 = self.probe_embed(probe_x).unsqueeze(1)  # (B, 1, d)
        t1 = self.scen_embed(scen_x).unsqueeze(1)    # (B, 1, d)
        t2 = self.meta_embed(meta).unsqueeze(1)      # (B, 1, d)

        tokens = torch.cat([t0, t1, t2], dim=1)      # (B, 3, d)
        out    = self.encoder(tokens)                 # (B, 3, d)
        pooled = out.mean(dim=1)                      # (B, d)
        return self.output(pooled).squeeze(-1)        # (B,)


# ---------------------------------------------------------------------------
# 3. Inverted Transformer
# ---------------------------------------------------------------------------

class ITransformerHead(nn.Module):
    """
    Inverted Transformer (Liu et al., ICLR 2024 스타일 적용):
    130개 피처 각각을 독립 토큰으로 처리해 feature-wise self-attention을 수행한다.

    각 토큰 = scalar value projection (1→d_model) + 피처 ID embedding (learnable).
    → 어떤 HI 조합이 용량 예측에 중요한지 attention map으로 해석 가능.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_feat     = _HEAD_IN                          # 130
        self.value_proj = nn.Linear(1, d_model)             # scalar → d_model
        self.feat_emb   = nn.Embedding(_HEAD_IN, d_model)   # 피처 ID positional emb

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.output  = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 130)
        vals   = self.value_proj(x.unsqueeze(-1))                          # (B, 130, d)
        pos    = self.feat_emb(torch.arange(self.n_feat, device=x.device)) # (130, d)
        tokens = vals + pos.unsqueeze(0)                                    # (B, 130, d)

        out    = self.encoder(tokens)   # (B, 130, d)
        pooled = out.mean(dim=1)        # (B, d)
        return self.output(pooled).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------

def build_cap_head(model_cfg: dict, d_head: int = 128, dropout: float = 0.1) -> nn.Module:
    """
    model_cfg 의 regression_model 값에 따라 적절한 헤드를 반환한다.
    model_cfg 가 비어 있거나 키가 없으면 MLPHead (Phase 1 기본 동작).

    Args:
        model_cfg : cfg["model"] 딕셔너리 (scr.yaml의 model: 섹션)
        d_head    : MLP hidden dim 또는 Transformer d_model (yaml model.d_head)
        dropout   : dropout rate (yaml model.dropout)
    """
    rtype = model_cfg.get("regression_model", "mlp").lower().replace("-", "_")

    n_heads  = model_cfg.get("tr_n_heads",  4)
    n_layers = model_cfg.get("tr_n_layers", 2)
    d_ff     = model_cfg.get("tr_d_ff",     d_head * 2)

    if rtype == "mlp":
        return MLPHead(d_head=d_head, dropout=dropout)

    if rtype == "transformer":
        return TransformerHead(
            d_model=d_head, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        )

    if rtype in ("i_transformer", "itransformer"):
        return ITransformerHead(
            d_model=d_head, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        )

    raise ValueError(
        f"Unknown regression_model: '{rtype}'. "
        "Choose from: mlp / transformer / i_transformer"
    )
