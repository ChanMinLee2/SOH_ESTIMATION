"""
raw_mlp_model.py — HI 추출 없이 원본 리샘플 곡선(raw_v/raw_i)만으로 회귀하는 베이스라인.

목적 (MODEL_DIRECTION.md B-4 계열): "HI 추출 자체가 얼마나 기여하는가"를 재는 baseline.
CNN을 쓰지 않고 x_raw(RAW_CH, RAW_N)를 flatten한 뒤 direction/cap_init과 concat해
평범한 MLP로 cap을 직접 예측한다 — "raw vs HI" 비교를 아키텍처(CNN vs MLP) 차이와
뒤섞지 않기 위해, 기존 MLPHead와 동일한 얕은 2-hidden-layer 구조를 그대로 쓴다.

SCRModel과 동일한 batch/출력 dict 계약을 따라 SCRTrainer/SCREvaluator/test_scr.py를
수정 없이(또는 최소 분기만으로) 재사용할 수 있게 맞췄다:
  - forward(batch) → {"cap_pred", "level_logits", "probe_z", "scen_z"} (뒤 3개는 더미)
  - _fixed_probe=True, _fixed_scen=True, probe_mlp=None
    → SCRLoss._l0_penalty가 즉시 0 반환, CE 항도 자동 비활성화됨(수정 불필요)
  - spec/n_scenarios/n_classes 보유 → SCREvaluator가 그대로 사용 가능
  - get_selected_probe_his()/get_selected_scen_his(): 더미(빈 리스트) — HI 선택 개념이 없음
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from utils.hi_schema import N_HI, RAW_N, RAW_CH, spec_from_qfrac


class RawMLPModel(nn.Module):
    """raw_v/raw_i(flatten) + direction + cap_init → 평범한 MLP로 SOH 직접 회귀."""

    # SCRLoss._l0_penalty / CE 항을 무해화하기 위한 SCRModel 호환 플래그 (게이트 자체가 없음)
    _fixed_probe = True
    _fixed_scen = True
    probe_mlp = None

    def __init__(
        self,
        d_head: int = 128,
        dropout: float = 0.1,
        model_cfg: Optional[dict] = None,
        spec=None,
        **_ignored,   # SCRModel 생성 시그니처와 자리수가 달라도 안전하게 흡수
    ):
        super().__init__()
        if spec is None:
            spec = spec_from_qfrac()
        self.spec = spec
        self.n_scenarios = spec.n_scenarios
        self.n_classes = spec.n_classes

        m_cfg = model_cfg or {}
        hidden = m_cfg.get("mlp_hidden_dims") or [d_head, d_head // 2]

        in_dim = RAW_CH * RAW_N + 2   # flatten(raw_v,raw_i) + direction + cap_init
        layers: list[nn.Module] = []
        d_in = in_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(d_in, h))
            layers.append(nn.ReLU())
            if i < len(hidden) - 1:   # hidden layers 사이에만 dropout (MLPHead와 동일 관례)
                layers.append(nn.Dropout(dropout))
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x_raw = batch["x_raw"]                             # (B, RAW_CH, RAW_N)
        direction = batch["direction"]                     # (B,)
        cap_init = batch["cap_init"]                        # (B,)

        B = x_raw.size(0)
        flat = x_raw.reshape(B, -1)                         # (B, RAW_CH*RAW_N)
        feat = torch.cat([flat, direction.unsqueeze(1), cap_init.unsqueeze(1)], dim=1)
        cap_pred = self.net(feat).squeeze(-1)               # (B,)

        dev = x_raw.device
        return {
            "cap_pred": cap_pred,
            "level_logits": torch.zeros(B, self.n_classes, dtype=x_raw.dtype, device=dev),
            "probe_z": torch.zeros(B, N_HI, dtype=x_raw.dtype, device=dev),
            "scen_z": torch.zeros(B, N_HI, dtype=x_raw.dtype, device=dev),
        }

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(batch)

    @torch.no_grad()
    def get_probe_x(self, x_hi: torch.Tensor, direction: torch.Tensor,
                     seg_idx: torch.Tensor) -> torch.Tensor:
        """SCREvaluator 호환용 — raw 모델은 HI 게이트가 없으므로 x_hi를 그대로 반환."""
        return x_hi

    @torch.no_grad()
    def get_selected_probe_his(self) -> dict[str, list[int]]:
        return {"charge": [], "discharge": []}

    @torch.no_grad()
    def get_selected_scen_his(self) -> dict[int, list[int]]:
        return {s: [] for s in range(self.n_scenarios)}
