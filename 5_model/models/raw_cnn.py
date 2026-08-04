"""
raw_cnn.py — 세그먼트 원시 V/I/t 곡선용 1D CNN 인코더.

입력  : x_raw (B, RAW_CH=3, RAW_N=48)  — 채널 [V, I(signed), t_rel], q_frac 정규화 곡선
출력  : emb   (B, 3) = [h_scen, h_intensity, h_soh]  — 의미가 고정된 3개 스칼라

구조 : Conv stem → ResBlock → 채널확장(공간 다운샘플 없음) → ResBlock → 3개의 독립
       AttentionPool 분기 → 각 분기 전용 Linear(1) → BatchNorm1d(3)

설계 근거(docs/260803_RESULTS.md §10, 원 설계는 docs/MODEL_SPECS.md §4.1):
  - 다운샘플을 1단계(48→24)만 남긴 이유(§10.2): MIT/HUST 둘 다 충전이 2단계
    (fast CC→1C CC→CV)라 전류 계단 전환의 "위치"가 열화에 따라 크게 이동한다
    (HUST cv_qfrac 0.96→0.70). 기존처럼 24→12로 한 번 더 줄이면 이 위치 해상도가
    절반으로 깎인다 — AttentionPool을 길이 24에서 바로 적용해 해상도를 보존한다.
  - 3개의 독립 AttentionPool 분기(§10.3): 공유 풀링 하나에 Linear(64→3)만 얹으면
    세 출력이 "같은 위치 가중치로 본 같은 표현의 선형 투영"에 불과해 분리가
    약하다. 분기마다 별도 score=Linear(64,1)을 둬 각자 다른 위치에 주목할 여지를
    준다(h_intensity는 전류 계단 근처, h_soh는 knee/plateau 근처 등 — 학습으로
    도달, 사전 강제 아님).
  - 출력 BatchNorm1d(3)(§10.4): 기존 64D는 스케일 불일치를 다수 차원이 완충했지만
    3D는 각 차원이 유일한 정보 채널이라 스케일 불일치가 즉시 반영된다.

세 스칼라의 지도 신호(모델 자체가 아니라 학습 스크립트 쪽 책임 — train_classifier.py
참고):
  h_scen      : 세그먼트 내부 플래토 유사도(예 lfp_plateau_frac류) 회귀 타깃(§10.8)
  h_intensity : 세그먼트 내부 전류 레짐 변화 정도(예 stat_i_std류) 회귀 타깃(§10.8)
  h_soh       : 기존 cap_head의 SOH 회귀 손실(및 Phase 1 전용 보조 프로브, §10.9)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.hi_schema import RAW_N, RAW_CH


class _ResBlock1d(nn.Module):
    """Conv-BN-GELU-Conv-BN + skip → GELU (길이·채널 보존)."""

    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=pad)
        self.bn2   = nn.BatchNorm1d(ch)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + x)


class AttentionPool1d(nn.Module):
    """위치별 스코어 softmax 가중합 → (B, C). 단순 평균풀링 대체."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        x_t = x.transpose(1, 2)                # (B, L, C)
        w   = self.score(x_t).softmax(dim=1)   # (B, L, 1)
        return (x_t * w).sum(dim=1)            # (B, C)


class RawCNN(nn.Module):
    """원시 (V, I_signed, t_rel) 곡선 → 3차원 의미고정 임베딩 [h_scen, h_intensity, h_soh].

    Args:
        in_ch    : 입력 채널 (기본 RAW_CH=3)
        c1, c2   : stem / 후반 conv 채널 폭
        dropout  : 각 분기 출력 직전 dropout
    """

    HEAD_NAMES = ("h_scen", "h_intensity", "h_soh")

    def __init__(
        self,
        in_ch: int = RAW_CH,
        c1: int = 32,
        c2: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_out = 3

        # stem: (B, in_ch, 48) → (B, c1, 24)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.res1 = _ResBlock1d(c1)                    # (B, c1, 24)
        # 채널확장(공간 다운샘플 없음, §10.2): (B, c1, 24) → (B, c2, 24)
        self.chan_expand = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=1, stride=1),
            nn.BatchNorm1d(c2),
            nn.GELU(),
        )
        self.res2 = _ResBlock1d(c2)                    # (B, c2, 24)

        # 3개의 독립 AttentionPool 분기 + 전용 projection(§10.3)
        self.pool_scen = AttentionPool1d(c2)
        self.pool_intensity = AttentionPool1d(c2)
        self.pool_soh = AttentionPool1d(c2)
        self.proj_scen = nn.Sequential(nn.Dropout(dropout), nn.Linear(c2, 1))
        self.proj_intensity = nn.Sequential(nn.Dropout(dropout), nn.Linear(c2, 1))
        self.proj_soh = nn.Sequential(nn.Dropout(dropout), nn.Linear(c2, 1))

        # 출력 정규화(§10.4) — 3D는 각 차원이 유일한 정보 채널이라 스케일 불일치가
        # 곧바로 하류(cap_head)에 반영된다. BatchNorm1d(3)로 흡수.
        self.out_norm = nn.BatchNorm1d(3)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """x_raw: (B, in_ch, RAW_N) → (B, 3) = [h_scen, h_intensity, h_soh]."""
        h = self.stem(x_raw)
        h = self.res1(h)
        h = self.chan_expand(h)
        h = self.res2(h)                        # (B, c2, 24)

        h_scen = self.proj_scen(self.pool_scen(h))            # (B, 1)
        h_intensity = self.proj_intensity(self.pool_intensity(h))  # (B, 1)
        h_soh = self.proj_soh(self.pool_soh(h))                # (B, 1)

        emb = torch.cat([h_scen, h_intensity, h_soh], dim=1)   # (B, 3)
        return self.out_norm(emb)


if __name__ == "__main__":
    m = RawCNN()
    x = torch.randn(4, RAW_CH, RAW_N)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"RawCNN: in={tuple(x.shape)} -> out={tuple(y.shape)}  params={n_params:,}")
