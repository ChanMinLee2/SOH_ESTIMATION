"""
cap_heads.py — Capacity Head 팩토리.

Phase 2에서 yaml의 model.regression_model 값에 따라 회귀 헤드를 교체한다.
Phase 1은 항상 MLPHead (build_cap_head 호출 시 model_cfg 미전달).

지원 모델:
  mlp            : 기본 MLP 2-hidden-layer (기존 동작과 완전 동일)
  transformer    : Transformer Encoder — probe/scen/meta 3 semantic 토큰
  i_transformer  : Inverted Transformer — 130개 피처 각각이 토큰 (feature-wise attention)
  resnet_tab     : ResNet for tabular (Gorishniy et al., NeurIPS 2021) — skip-connection blocks
  ft_transformer : FT-Transformer + CLS 토큰 + Sparse Attention Mask (Gorishniy et al., NeurIPS 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.hi_schema import N_HI, RAW_CH, RAW_N

_HEAD_IN = N_HI + N_HI + 1 + 1  # 64+64+1+1 = 130
_D_CNN = 64  # RawCNN 출력 차원 — raw_cnn.py의 d_out 기본값과 동일(HI 64D와 정렬)
_HEAD_IN_WITH_CNN = N_HI + N_HI + _D_CNN + 1 + 1  # 64+64+64+1+1 = 194 (REGRESSION_UPGRADE.md §5)
_D_RAW_FLAT = RAW_CH * RAW_N  # 2*48=96 — 방안1(REGRESSION_UPGRADE.md §2 방안1): raw_v‖raw_i flatten
_HEAD_IN_WITH_RAW_FLAT = N_HI + N_HI + _D_RAW_FLAT + 1 + 1  # 64+64+96+1+1 = 226


# ---------------------------------------------------------------------------
# 1. MLP (baseline)
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """
    Configurable MLP head.

    mlp_hidden_dims=None (default) → [d_head, d_head//2] (기존 동작과 동일)
    mlp_hidden_dims=[256,128,64]   → 3-hidden-layer MLP
    dropout은 hidden layers 사이에만 적용 (출력층 직전 제외).
    """

    def __init__(
        self,
        d_head: int = 128,
        dropout: float = 0.1,
        mlp_hidden_dims: list[int] | None = None,
        head_in: int = _HEAD_IN,
    ):
        super().__init__()
        hidden = mlp_hidden_dims if mlp_hidden_dims is not None else [d_head, d_head // 2]
        layers: list[nn.Module] = []
        in_d = head_in
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            if i < len(hidden) - 1:   # hidden layers 사이에만 dropout
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 130)  →  (B,)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. Transformer
# ---------------------------------------------------------------------------

class TransformerHead(nn.Module):
    """
    130-dim(with_raw_cnn=True면 194-dim, with_raw_flat=True면 226-dim) 입력을
    semantic 토큰으로 분할해 Transformer Encoder에 입력한다.
      Token 0: probe_x  (64-dim)  — 방향별 probe gate 출력
      Token 1: scen_x   (64-dim)  — 시나리오 gate 출력
      Token 2: (with_raw_cnn=True) cnn_emb (64-dim) — raw V/|I| CNN 임베딩,
               64개 스칼라로 쪼개지 않고 통째로 한 토큰(REGRESSION_UPGRADE.md §3.2)
               또는 (with_raw_flat=True) raw_flat(96-dim) — raw_v‖raw_i를 압축 없이
               통째로 한 토큰으로 선형 투영(방안1, REGRESSION_UPGRADE.md §2). 96개
               개별 스칼라 토큰화는 어텐션 비용이 급증해(§3.2) 피하고, cnn_emb와
               동일하게 "토큰 1개" 취급으로 통일 — with_raw_cnn/with_raw_flat 동시
               활성은 금지(build_cap_head에서 검증).
      마지막 Token: meta (2-dim) — direction + cap_init

    각 토큰을 d_model로 선형 투영 후 TransformerEncoder → mean pool → 스칼라 출력.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        with_raw_cnn: bool = False,
        with_raw_flat: bool = False,
    ):
        super().__init__()
        assert not (with_raw_cnn and with_raw_flat), \
            "with_raw_cnn과 with_raw_flat을 동시에 켤 수 없습니다."
        self.with_raw_cnn = with_raw_cnn
        self.with_raw_flat = with_raw_flat
        self.probe_embed = nn.Linear(N_HI, d_model)
        self.scen_embed  = nn.Linear(N_HI, d_model)
        self.meta_embed  = nn.Linear(2, d_model)
        self.cnn_embed   = nn.Linear(_D_CNN, d_model) if with_raw_cnn else None
        self.raw_flat_embed = nn.Linear(_D_RAW_FLAT, d_model) if with_raw_flat else None

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
        # x: (B, 130) 또는 (B, 194)/(B, 226) — with_raw_cnn/with_raw_flat 여부에 따라
        probe_x = x[:, :N_HI]                                # (B, 64)
        scen_x  = x[:, N_HI: 2 * N_HI]                       # (B, 64)

        t0 = self.probe_embed(probe_x).unsqueeze(1)          # (B, 1, d)
        t1 = self.scen_embed(scen_x).unsqueeze(1)            # (B, 1, d)
        tokens = [t0, t1]
        offset = 2 * N_HI

        if self.with_raw_cnn:
            cnn_emb = x[:, offset: offset + _D_CNN]           # (B, 64)
            offset += _D_CNN
            tokens.append(self.cnn_embed(cnn_emb).unsqueeze(1))
        elif self.with_raw_flat:
            raw_flat = x[:, offset: offset + _D_RAW_FLAT]     # (B, 96)
            offset += _D_RAW_FLAT
            tokens.append(self.raw_flat_embed(raw_flat).unsqueeze(1))

        meta = x[:, offset:]                                  # (B, 2)
        tokens.append(self.meta_embed(meta).unsqueeze(1))

        tokens = torch.cat(tokens, dim=1)             # (B, 3/4, d)
        out    = self.encoder(tokens)                  # (B, 3/4, d)
        pooled = out.mean(dim=1)                       # (B, d)
        return self.output(pooled).squeeze(-1)         # (B,)


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
        with_raw_cnn: bool = False,
    ):
        super().__init__()
        if with_raw_cnn:
            raise NotImplementedError(
                "i_transformer + with_raw_cnn은 아직 미구현입니다 — cnn_emb(64D)를 개별 "
                "스칼라 토큰 64개로 쪼갤지, 압축된 임베딩 하나로 넣을지 설계가 필요합니다 "
                "(REGRESSION_UPGRADE.md §5 표 참조). 현재는 mlp/transformer/resnet_tab만 "
                "with_raw_cnn을 지원합니다."
            )
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
# 4. ResNet-Tab
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    """Pre-activation residual block: LN → Linear → ReLU → Dropout → Linear → Dropout → +x"""

    def __init__(self, d: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResNetTabHead(nn.Module):
    """
    ResNet for tabular data (Gorishniy et al., NeurIPS 2021).

    Architecture:
      Linear(_HEAD_IN → d) → [ResBlock × n_blocks] → LN → ReLU → Linear(d → 1)

    L0 gate로 이미 피처 선택이 된 sparse 130-dim 입력에서
    skip connection이 gradient 흐름을 안정화하고 depth를 높임.

    yaml 설정:
      regression_model: resnet_tab
      resnet_n_blocks: 4          # 기본값
      resnet_d_hidden_factor: 2.0 # d_hidden = d_head × factor
    """

    def __init__(
        self,
        d: int = 128,
        d_hidden_factor: float = 2.0,
        n_blocks: int = 4,
        dropout: float = 0.1,
        head_in: int = _HEAD_IN,
    ):
        super().__init__()
        d_hidden = int(d * d_hidden_factor)
        self.input_proj = nn.Linear(head_in, d)
        self.blocks = nn.ModuleList(
            [_ResBlock(d, d_hidden, dropout) for _ in range(n_blocks)]
        )
        self.norm   = nn.LayerNorm(d)
        self.output = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 130)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = torch.relu(self.norm(x))
        return self.output(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# 5. FT-Transformer + CLS token + Sparse Attention Mask
# ---------------------------------------------------------------------------

class FTTransformerHead(nn.Module):
    """
    FT-Transformer (Feature Tokenizer + Transformer) with [CLS] aggregation
    and sparse attention masking (Gorishniy et al., NeurIPS 2021).

    iTransformerHead 대비 두 가지 핵심 차이:
      1. 학습 가능한 [CLS] 토큰을 시퀀스 앞에 붙이고, out[:,0]으로 집약
         (mean pool 대신 CLS가 전체 피처 정보를 선택적으로 수집)
      2. src_key_padding_mask: L0 게이트로 0이 된 HI 위치를 어텐션에서 제외
         → 어텐션 budget을 실제 활성 HI에만 집중

    입력 구조 (130-dim):
      x[:, :N_HI*2] = probe_x ∥ scen_x  — L0 게이트 후 0=비활성
      x[:, -2:]     = direction + cap_init — 항상 유효 (마스킹 제외)

    yaml 설정:
      regression_model: ft_transformer
      tr_n_heads  : 4
      tr_n_layers : 2
      tr_d_ff     : 256
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        with_raw_cnn: bool = False,
    ):
        super().__init__()
        if with_raw_cnn:
            raise NotImplementedError(
                "ft_transformer + with_raw_cnn은 아직 미구현입니다 — cnn_emb(64D)를 sparse "
                "attention mask(L0 게이트 0=비활성) 체계에 어떻게 편입할지 설계가 필요합니다 "
                "(cnn_emb는 게이트로 걸러지는 값이 아니라 항상 유효하므로 mask 규칙이 달라짐). "
                "현재는 mlp/transformer/resnet_tab만 with_raw_cnn을 지원합니다."
            )
        self.n_feat     = _HEAD_IN                                   # 130
        self.n_hi       = N_HI * 2                                   # 128 (probe+scen)
        self.value_proj = nn.Linear(1, d_model)                      # scalar → d_model
        self.feat_emb   = nn.Embedding(_HEAD_IN, d_model)            # 피처 ID 임베딩
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))   # 학습 가능한 CLS

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-LN: 학습 안정성
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.output  = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 130)
        B = x.size(0)
        dev = x.device

        # ── 피처 토크나이제이션 ──────────────────────────────────────────
        vals   = self.value_proj(x.unsqueeze(-1))                    # (B, 130, d)
        pos    = self.feat_emb(torch.arange(self.n_feat, device=dev))  # (130, d)
        tokens = vals + pos.unsqueeze(0)                             # (B, 130, d)

        # ── [CLS] 토큰 prepend ───────────────────────────────────────────
        cls    = self.cls_token.expand(B, -1, -1)                    # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)                     # (B, 131, d)

        # ── Sparse attention mask ────────────────────────────────────────
        # probe_x + scen_x (128-dim): 0 = L0 게이트 비활성 → 어텐션 제외
        # direction + cap_init (2-dim): 항상 유효
        # CLS (1-dim): 항상 유효
        hi_pad   = (x[:, :self.n_hi] == 0)                          # (B, 128) bool
        meta_pad = torch.zeros(B, 2, dtype=torch.bool, device=dev)  # (B, 2)  항상 False
        cls_pad  = torch.zeros(B, 1, dtype=torch.bool, device=dev)  # (B, 1)  항상 False
        pad_mask = torch.cat([cls_pad, hi_pad, meta_pad], dim=1)    # (B, 131)

        # ── Encoder + CLS 집약 ───────────────────────────────────────────
        out = self.encoder(tokens, src_key_padding_mask=pad_mask)    # (B, 131, d)
        return self.output(out[:, 0]).squeeze(-1)                    # (B,)


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------

def build_cap_head(model_cfg: dict, d_head: int = 128, dropout: float = 0.1) -> nn.Module:
    """
    model_cfg 의 regression_model 값에 따라 적절한 헤드를 반환한다.
    model_cfg 가 비어 있거나 키가 없으면 MLPHead (Phase 1 기본 동작).

    Args:
        model_cfg : cfg["model"] 딕셔너리 (scr.yaml의 model: 섹션)
        d_head    : MLP hidden dim 또는 Transformer d_model / ResNet block width
        dropout   : dropout rate (yaml model.dropout)
    """
    rtype = model_cfg.get("regression_model", "mlp").lower().replace("-", "_")

    n_heads  = model_cfg.get("tr_n_heads",  4)
    n_layers = model_cfg.get("tr_n_layers", 2)
    d_ff     = model_cfg.get("tr_d_ff",     d_head * 2)

    with_raw_cnn  = bool(model_cfg.get("with_raw_cnn", False))
    with_raw_flat = bool(model_cfg.get("with_raw_flat", False))
    if with_raw_cnn and with_raw_flat:
        raise ValueError("with_raw_cnn과 with_raw_flat을 동시에 켤 수 없습니다 (방안2 vs 방안1).")
    if with_raw_flat and rtype not in ("mlp", "transformer", "resnet_tab"):
        raise NotImplementedError(
            f"with_raw_flat은 아직 mlp/transformer/resnet_tab만 지원합니다 (rtype={rtype}). "
            "i_transformer/ft_transformer는 96개 raw 스칼라의 개별 토큰화 설계가 필요합니다 "
            "(REGRESSION_UPGRADE.md §3.2)."
        )
    head_in = (
        _HEAD_IN_WITH_CNN if with_raw_cnn else
        _HEAD_IN_WITH_RAW_FLAT if with_raw_flat else
        _HEAD_IN
    )

    if rtype == "mlp":
        return MLPHead(d_head=d_head, dropout=dropout,
                       mlp_hidden_dims=model_cfg.get("mlp_hidden_dims"),
                       head_in=head_in)

    if rtype == "transformer":
        return TransformerHead(
            d_model=d_head, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            with_raw_cnn=with_raw_cnn, with_raw_flat=with_raw_flat,
        )

    if rtype in ("i_transformer", "itransformer"):
        return ITransformerHead(
            d_model=d_head, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            with_raw_cnn=with_raw_cnn,
        )

    if rtype == "resnet_tab":
        return ResNetTabHead(
            d=d_head,
            d_hidden_factor=model_cfg.get("resnet_d_hidden_factor", 2.0),
            n_blocks=model_cfg.get("resnet_n_blocks", 4),
            dropout=dropout,
            head_in=head_in,
        )

    if rtype in ("ft_transformer", "fttransformer"):
        return FTTransformerHead(
            d_model=d_head, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            with_raw_cnn=with_raw_cnn,
        )

    raise ValueError(
        f"Unknown regression_model: '{rtype}'. "
        "Choose from: mlp / transformer / i_transformer / resnet_tab / ft_transformer"
    )
