"""
Hard-Concrete L0 gate (Louizos et al. 2018).

Parameters: beta=2/3, gamma=-0.1, zeta=1.1
Train:  s = sigmoid((log U - log(1-U) + log_alpha) / beta)
        z = clamp(s*(zeta - gamma) + gamma, 0, 1)
Infer:  s = sigmoid(log_alpha)
        active = (z > 0)  ← hard binary mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """
    Learnable Hard-Concrete L0 gate.

    n_features: number of binary gates (one per input HI)
    """

    BETA: float = 2.0 / 3.0
    GAMMA: float = -0.1
    ZETA: float = 1.1

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        # log_alpha initialised near zero (50% open probability at init)
        self.log_alpha = nn.Parameter(torch.zeros(n_features))

    # ------------------------------------------------------------------
    # Probability that gate is non-zero (used in L0 loss)
    # ------------------------------------------------------------------
    def gate_prob(self) -> torch.Tensor:
        """P(z > 0) ≈ sigmoid(log_alpha - beta * log(-gamma/zeta))."""
        offset = self.BETA * torch.log(
            torch.tensor(-self.GAMMA / self.ZETA, device=self.log_alpha.device)
        )
        return torch.sigmoid(self.log_alpha - offset)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (..., n_features)
        Returns:
          masked_x : (..., n_features)  element-wise gated
          z        : (..., n_features)  [0,1] gates
        """
        if self.training:
            z = self._sample_train(x)
        else:
            z = self._hard_infer().expand_as(x)

        return x * z, z

    def _sample_train(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(x[..., :self.n_features]).uniform_().clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid(
            (torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.BETA
        )
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return z

    def _hard_infer(self) -> torch.Tensor:
        s = torch.sigmoid(self.log_alpha)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return (z > 0).float()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def active_indices(self) -> list[int]:
        """Return indices of active gates at inference time."""
        s = torch.sigmoid(self.log_alpha)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return (z > 0).nonzero(as_tuple=False).squeeze(1).tolist()

    @torch.no_grad()
    def active_count(self) -> int:
        return len(self.active_indices())

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}"


class GroupedHardConcreteGate(HardConcreteGate):
    """
    그룹 계층을 추가한 Hard-Concrete L0 게이트.

        log_alpha[i] = log_alpha_group[group_of(i)] + log_alpha_member[i]

    독립 스칼라 하나씩이던 log_alpha를 "그룹을 켤지 말지"(log_alpha_group, 그룹 수만큼)와
    "그룹 안에서 상대적으로 얼마나 더/덜 중요한지"(log_alpha_member, HI 수만큼) 두 단으로
    쪼갠다. 다중공선성이 있는 HI끼리는 모델 출력에 주는 그래디언트가 거의 같아서 원래
    log_alpha가 학습 중 어느 쪽을 밀지 랜덤하게 갈리는데(top-10 랭킹 불안정성의 원인),
    "그룹을 켤지"라는 큰 결정을 그래디언트가 합쳐지는 log_alpha_group 하나로 모으면
    이 흔들림이 줄어든다. forward/gate_prob/active_indices 등은 log_alpha를 그대로
    참조하므로 HardConcreteGate 쪽 구현을 재사용하고, log_alpha만 property로 재정의한다.

    group_ids: 길이 n_features, 값은 0..n_groups-1 (build_synergy_groups.py의
               seg_{s}_groups를 펼친 것). 시너지 그룹 정보가 없는(모든 HI가 자기 혼자인)
               경우 group_ids=range(n_features)를 주면 일반 HardConcreteGate와 동등하다.
    """

    def __init__(self, n_features: int, group_ids: list[int]):
        nn.Module.__init__(self)  # HardConcreteGate.__init__은 건너뜀 — 거기서 만드는 평범한
                                   # log_alpha Parameter가 아래 log_alpha 프로퍼티와 충돌하기 때문
        if len(group_ids) != n_features:
            raise ValueError(
                f"group_ids 길이({len(group_ids)})가 n_features({n_features})와 다릅니다"
            )
        self.n_features = n_features
        self.n_groups = max(group_ids) + 1
        self.register_buffer("group_index", torch.tensor(group_ids, dtype=torch.long))
        self.log_alpha_group  = nn.Parameter(torch.zeros(self.n_groups))
        self.log_alpha_member = nn.Parameter(torch.zeros(n_features))

    @property
    def log_alpha(self) -> torch.Tensor:
        return self.log_alpha_group[self.group_index] + self.log_alpha_member

    @torch.no_grad()
    def group_gate_prob(self) -> torch.Tensor:
        """그룹 레벨만의 게이트 확률(멤버 오프셋 제외) — 그룹 자체가 켜졌는지 보고 싶을 때."""
        offset = self.BETA * torch.log(
            torch.tensor(-self.GAMMA / self.ZETA, device=self.log_alpha_group.device)
        )
        return torch.sigmoid(self.log_alpha_group - offset)

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, n_groups={self.n_groups}"
