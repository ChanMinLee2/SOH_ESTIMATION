"""
raw_cnn.py — 세그먼트 원시 V/I 곡선용 1D CNN 인코더.

입력  : x_raw (B, RAW_CH=2, RAW_N=48)  — 채널 [V, |I|], q_frac 정규화 곡선
출력  : emb   (B, d_out)               — HI 와 융합 가능한 고정 차원 임베딩

구조 : Conv stem → ResBlock ×2 → Attention Pooling → Linear
  - LFP V-Q 곡선의 국부 특징(플래토 진입/이탈 기울기, knee)을 잔차 블록으로 포착
  - Attention Pooling 이 어느 q_frac 위치가 중요한지 학습 (단순 avg-pool 대비 상보적)

MODEL_SPECS.md §4.1 참조. 분류기(CNNProbeClassifier)와 (선택적으로) 회귀 헤드가
동일 인스턴스를 공유할 수 있도록 순수 인코더로만 구현한다.
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
    """원시 (V, |I|) 곡선 → d_out 임베딩.

    Args:
        d_out    : 출력 임베딩 차원 (기본 64 — HI 64D 와 정렬)
        in_ch    : 입력 채널 (기본 RAW_CH=2)
        c1, c2   : stem / 후반 conv 채널 폭
        dropout  : 출력 직전 dropout
    """

    def __init__(
        self,
        d_out: int = 64,
        in_ch: int = RAW_CH,
        c1: int = 32,
        c2: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_out = d_out

        # stem: (B, in_ch, 48) → (B, c1, 24)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.res1 = _ResBlock1d(c1)                    # (B, c1, 24)
        # downsample: (B, c1, 24) → (B, c2, 12)
        self.down = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(c2),
            nn.GELU(),
        )
        self.res2 = _ResBlock1d(c2)                    # (B, c2, 12)
        self.pool = AttentionPool1d(c2)                # (B, c2)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c2, d_out),
            nn.GELU(),
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """x_raw: (B, in_ch, RAW_N) → (B, d_out)."""
        h = self.stem(x_raw)
        h = self.res1(h)
        h = self.down(h)
        h = self.res2(h)
        h = self.pool(h)
        return self.proj(h)


if __name__ == "__main__":
    m = RawCNN(d_out=64)
    x = torch.randn(4, RAW_CH, RAW_N)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"RawCNN: in={tuple(x.shape)} → out={tuple(y.shape)}  params={n_params:,}")
