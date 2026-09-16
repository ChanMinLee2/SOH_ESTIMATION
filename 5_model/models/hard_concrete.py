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


class ShrinkageHardConcreteGate(nn.Module):
    """
    시나리오 간 부분 풀링(partial pooling) Hard-Concrete 게이트 뱅크.

        log_alpha_s[i] = shared_log_alpha[i] + delta_log_alpha[s, i]

    기존 `nn.ModuleList([HardConcreteGate(width) for s in range(n_scenarios)])`는
    시나리오마다 완전히 독립된 log_alpha를 학습해서, 게이트 하나가 보는 학습
    샘플이 전체의 1/n_scenarios로 쪼개진다(docs/260909_RESULTS.md §6-5(e) —
    이게 시나리오 라벨 축이 해로운 확정된 메커니즘). 이 클래스는 그 대신 전체
    n_scenarios × N 샘플로 학습되는 `shared_log_alpha`(시나리오 공통 성분)와,
    시나리오 s의 것만으로 학습되는 `delta_log_alpha[s]`(시나리오별 편차)로
    분해한다 — `delta_log_alpha`에 별도 L2 벌점(`shrinkage_penalty()`, 학습
    루프에서 `lambda_shrink`를 곱해 총 손실에 더함)을 걸어, 그 시나리오만의
    편차가 실제 데이터로 뒷받침될 때만 0에서 벗어나게 만든다 — "공유할지
    전용으로 할지"를 HI/시나리오 단위로 데이터가 직접 결정하게 하는 것이
    핵심이라, `interaction_json`(v4, HI 전체를 shared/specific으로 이분)보다
    한 단계 더 유연하다.

    forward 인터페이스는 기존 `nn.ModuleList` 기반 `scen_gates`를 그대로
    대체할 수 있도록 맞췄다 — `enumerate(bank)`/`bank[s]`/`len(bank)`가
    각 시나리오의 게이트처럼 동작하는 `_ShrinkageGateView`를 반환하고, 그
    view는 `HardConcreteGate`와 동일한 `gate_prob()`/`active_indices()`/
    `.BETA`(temperature annealing용 인스턴스 오버라이드)를 지원한다 —
    `scr_loss.py`/`phase1_trainer_v2.py`/`train_scr.py`의 게이트 소비 코드
    (L0 벌점, 게이트 포화도, JSON 랭킹 저장, gate_prob 플랏)는 무수정으로
    재사용된다.
    """

    BETA: float = HardConcreteGate.BETA
    GAMMA: float = HardConcreteGate.GAMMA
    ZETA: float = HardConcreteGate.ZETA

    def __init__(self, n_scenarios: int, n_features: int):
        super().__init__()
        self.n_scenarios = n_scenarios
        self.n_features = n_features
        self.shared_log_alpha = nn.Parameter(torch.zeros(n_features))
        self.delta_log_alpha = nn.Parameter(torch.zeros(n_scenarios, n_features))
        self._views = [_ShrinkageGateView(self, s) for s in range(n_scenarios)]

    def log_alpha_for(self, s: int) -> torch.Tensor:
        return self.shared_log_alpha + self.delta_log_alpha[s]

    def shrinkage_penalty(self) -> torch.Tensor:
        """delta_log_alpha의 평균 제곱 크기 — lambda_shrink를 곱해 총 손실에 더하는 용도.
        시나리오 수·폭에 무관하게 "게이트 1개당 평균 편차 크기"로 해석되도록 mean 사용
        (sum이면 lambda_shrink 튜닝이 n_scenarios*n_features 크기에 얽매이게 됨)."""
        return self.delta_log_alpha.pow(2).mean()

    def __iter__(self):
        return iter(self._views)

    def __len__(self) -> int:
        return self.n_scenarios

    def __getitem__(self, s: int) -> "_ShrinkageGateView":
        return self._views[s]

    def extra_repr(self) -> str:
        return f"n_scenarios={self.n_scenarios}, n_features={self.n_features}"


class _ShrinkageGateView:
    """ShrinkageHardConcreteGate의 시나리오 s 슬라이스를 평범한 HardConcreteGate처럼
    호출 가능하게 감싸는 얇은 프록시. 자체 파라미터를 갖지 않고 bank.log_alpha_for(s)만
    읽으므로 nn.Module로 등록하지 않는다(등록하면 shared_log_alpha가 bank 자신과
    각 view에 중복 참조되어 상태사전 키가 불필요하게 불어남 — 파라미터는 bank 하나에만
    귀속시키는 게 깔끔하다). BETA/GAMMA/ZETA를 클래스가 아닌 self로 참조해서, 기존
    HardConcreteGate와 동일하게 인스턴스별 temperature annealing(`gate.BETA = ...`,
    phase1_trainer_v2.py의 Stage2)이 시나리오별로 독립적으로 걸리게 했다."""

    BETA: float = HardConcreteGate.BETA
    GAMMA: float = HardConcreteGate.GAMMA
    ZETA: float = HardConcreteGate.ZETA

    def __init__(self, bank: ShrinkageHardConcreteGate, s: int):
        self.bank = bank
        self.s = s

    @property
    def log_alpha(self) -> torch.Tensor:
        return self.bank.log_alpha_for(self.s)

    @property
    def training(self) -> bool:
        return self.bank.training

    def gate_prob(self) -> torch.Tensor:
        log_alpha = self.log_alpha
        offset = self.BETA * torch.log(
            torch.tensor(-self.GAMMA / self.ZETA, device=log_alpha.device)
        )
        return torch.sigmoid(log_alpha - offset)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            z = self._sample_train(x)
        else:
            z = self._hard_infer().expand_as(x)
        return x * z, z

    def _sample_train(self, x: torch.Tensor) -> torch.Tensor:
        log_alpha = self.log_alpha
        u = torch.zeros_like(x[..., :self.bank.n_features]).uniform_().clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + log_alpha) / self.BETA)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return z

    def _hard_infer(self) -> torch.Tensor:
        s = torch.sigmoid(self.log_alpha)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return (z > 0).float()

    @torch.no_grad()
    def active_indices(self) -> list[int]:
        s = torch.sigmoid(self.log_alpha)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return (z > 0).nonzero(as_tuple=False).squeeze(1).tolist()

    @torch.no_grad()
    def active_count(self) -> int:
        return len(self.active_indices())
