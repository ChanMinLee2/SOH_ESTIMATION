"""
SCR composite loss.

Phase 1 dual-objective:
  L = MSE(cap_pred, cap_target)           ← probe_gate + scen_gates
    + lambda_scen * CE(level_logits, level) ← probe_gate only (via probe_mlp)
    + lambda_l0   * L0_penalty             ← sparsity on all gates

Phase 2 (probe_mlp=None):
  L = MSE + lambda_l0 * L0_penalty

L0_penalty: for each scenario, computes E[cost of active HIs].
  Charging scenarios: use charge_probe_gate + scen_gate[s]
  Discharging scenarios: use discharge_probe_gate + scen_gate[s]
  P(HI_i active in scenario s) = 1 - (1 - p_probe_i)(1 - p_scen_i)
  2026-09-18부터 scen_kernel_gates(커널 HI, Stage B' 단독 — probe 단계 없음)도 동일
  lambda_l0로 패널티에 합산됨 — 예전엔 raw HI만 계산해 커널 게이트에 희소화 압력이
  전혀 없었음(docs/260917_RESULTS.md). 커널별 비용은 model.kernel_hi_costs가 있으면
  그 커널을 만든 멤버 raw HI 카테고리(stat/diff/lfp/morph) 비용의 평균을 쓰고,
  없으면(구 pkl/미지정) 균일 비용 1.0으로 하위호환.
  2026-10-07부터 probe_kernel_gates(분류기용 커널 후보 블록, §scr_model.py)도 같은
  방식으로 패널티에 합산됨 — 회귀뿐 아니라 분류기의 커널 HI 사용에도 희소화 압력.

hi_cost_weighted(2026-09-19, 기본 False로 전환): True면 raw HI는 CATEGORY_COSTS
(stat/diff/lfp/morph), 커널 HI는 멤버 카테고리 비용 평균으로 L0 페널티를 가중한다.
False(기본)면 raw/커널 전부 균일 비용 1.0 — 순수 "활성 게이트 개수"만 페널티가 된다.
raw HI 카테고리 비용이 `4_hi_analysis/hi_profile/hi_timing_cost.json` 실측치로
바뀌면서(stat이 diff/lfp보다 훨씬 비싸게 나옴 — 실측 알고리즘 복잡도 때문이지 "카테고리"
자체의 문제가 아님, 논의 진행 중) 이 항을 잠깐 끄고 검증하기 위해 하위호환 토글로
분리했다 — 예전엔 raw HI 비용 가중치가 스위치 없이 항상 켜져 있었다(프로젝트 최초
커밋부터, git log -S"cost_vec" 확인).

l0_norm_constant (docs/260915_RESULTS.md — L0 정규화 상수 confound 분리):
_l0_penalty가 시나리오별 페널티 합을 model.n_scenarios(라벨ON=6, 라벨OFF=2)로
나누는 게 기본 동작 — 같은 lambda_l0라도 게이트 1개가 느끼는 실효 희소성
압력이 조건마다 다르다(6분의 1 vs 2분의 1). None(기본)이면 기존과 100% 동일
동작(n_scenarios로 나눔); 정수를 주면 n_scenarios 대신 그 고정값으로 나눠서,
라벨ON/OFF 조건 간 이 정규화 강도를 동일하게 맞춘 대조군을 만들 수 있다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import get_hi_cost_vector


class SCRLoss(nn.Module):

    def __init__(
        self,
        lambda_scen: float = 0.0,
        lambda_l0: float = 0.01,
        l0_norm_constant: int | None = None,
        hi_cost_weighted: bool = False,
    ):
        super().__init__()
        self.lambda_scen = lambda_scen  # > 0 → CE 활성 (Phase 1 with probe_mlp)
        self.lambda_l0 = lambda_l0
        self.l0_norm_constant = l0_norm_constant  # None(기본)이면 n_scenarios로 나눔(기존과 동일)
        self.hi_cost_weighted = hi_cost_weighted  # False(기본, 2026-09-19부터) → 균일 비용 1.0

        costs = get_hi_cost_vector("dis_hi")
        self.register_buffer("cost_vec", torch.tensor(costs, dtype=torch.float32))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        cap_pred = outputs["cap_pred"]
        target   = batch["target"]

        mse = F.mse_loss(cap_pred, target)
        l0  = self._l0_penalty(model)

        # CE: probe_mlp 존재 + lambda_scen > 0 일 때만 활성 (Phase 1 dual-objective)
        ce = torch.zeros(1, device=cap_pred.device).squeeze()
        if self.lambda_scen > 0 and getattr(model, "probe_mlp", None) is not None:
            ce = F.cross_entropy(outputs["level_logits"], batch["level"])

        total = mse + self.lambda_scen * ce + self.lambda_l0 * l0
        return {"total": total, "mse": mse, "ce": ce, "l0": l0}

    def _raw_hi_penalty(
        self, model: nn.Module, eff_cost_vec: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """raw HI(probe x scen 곱셈 결합) L0 페널티 — P(z_i != 0) = 1-(1-p_probe_i)(1-p_scen_i).
        probe/scen 둘 다 고정(Phase2 완전 고정)이면 기여 없음(0). shared_gate가
        있으면(v4) specific/shared 폭을 원래 컬럼 순서로 재조립해 cost_vec(N_HI)과
        맞춘다. _l0_penalty에서 분리, 동작 변화 없음."""
        penalty = torch.zeros(1, device=device)
        if model._fixed_probe and model._fixed_scen:
            return penalty

        ones = torch.ones_like(self.cost_vec)
        if not model._fixed_probe:
            p_probe_ch  = model.charge_probe_gate.gate_prob()
            p_probe_dis = model.discharge_probe_gate.gate_prob()
        else:
            p_probe_ch  = ones
            p_probe_dis = ones

        if not model._fixed_scen:
            _charge_ids = frozenset(model.spec.charge_scenario_ids)
            _n_scen     = model.n_scenarios
            shared_gate = getattr(model, "shared_gate", None)  # v4: HI 일부가 scen_gates
                # 대신 시나리오 무관 shared_gate로 라우팅됨 — 있으면 scen_gates[s]는
                # N_HI보다 좁은 폭(specific 몫만)이라, cost_vec(N_HI)과 맞추려면 원래
                # 컬럼 순서로 재조립해야 함. None이면(shared_hi_mask 미지정) 기존과
                # 100% 동일 동작.
            raw_penalty = torch.zeros(1, device=device)
            for s, gate in enumerate(model.scen_gates):
                p_probe  = p_probe_ch if s in _charge_ids else p_probe_dis
                if shared_gate is not None:
                    p_scen = torch.zeros_like(self.cost_vec)
                    p_scen[model._shared_idx] = shared_gate.gate_prob()
                    if len(model._specific_idx) > 0:
                        p_scen[model._specific_idx] = gate.gate_prob()
                else:
                    p_scen = gate.gate_prob()
                p_active = 1.0 - (1.0 - p_probe) * (1.0 - p_scen)
                raw_penalty = raw_penalty + (eff_cost_vec * p_active).sum()
            norm = self.l0_norm_constant if self.l0_norm_constant is not None else _n_scen
            penalty = penalty + raw_penalty / norm
        else:
            # Only probe gates contribute
            penalty = penalty + (
                (eff_cost_vec * p_probe_ch).sum() +
                (eff_cost_vec * p_probe_dis).sum()
            ).unsqueeze(0) / 2
        return penalty

    def _kernel_hi_penalty(self, model: nn.Module, device: torch.device) -> torch.Tensor:
        """커널 융합 HI(scen_kernel_gates) L0 페널티 — probe 단계가 없는 Stage B' 단독
        게이트라 raw처럼 곱셈 결합 없이 게이트 확률 자체가 활성 확률이다.
        hi_cost_weighted=True고 model.kernel_hi_costs가 있으면(kernel.py의 f["cost"] —
        그 커널을 만든 멤버 raw HI들의 카테고리 비용 평균) 그 값으로 가중, 아니면 균일
        비용 1.0. scen_kernel_gates가 없으면 기여 없음(0). raw HI 조기반환(probe/scen
        둘 다 고정인 Phase2 케이스)과 무관하게 항상 계산한다 — 커널 게이트는 고정 마스크
        개념이 아예 없는 Phase1 전용 학습 가능 게이트라서. _l0_penalty에서 분리,
        동작 변화 없음."""
        penalty = torch.zeros(1, device=device)
        kernel_gates = getattr(model, "scen_kernel_gates", None)
        if kernel_gates is None:
            return penalty

        # 2026-09-19: hi_cost_weighted=False(기본)면 kernel_hi_costs가 있어도 무시하고
        # 균일 비용 1.0 — raw HI 쪽과 토글을 맞춘다.
        kernel_costs = getattr(model, "kernel_hi_costs", None) if self.hi_cost_weighted else None
        kernel_penalty = torch.zeros(1, device=device)
        for s, gate in enumerate(kernel_gates):
            p = gate.gate_prob()
            if kernel_costs is not None:
                c = torch.tensor(kernel_costs[s], dtype=p.dtype, device=p.device)
                kernel_penalty = kernel_penalty + (c * p).sum()
            else:
                kernel_penalty = kernel_penalty + p.sum()
        _n_scen = model.n_scenarios
        norm = self.l0_norm_constant if self.l0_norm_constant is not None else _n_scen
        penalty = penalty + kernel_penalty / norm
        return penalty

    def _probe_kernel_hi_penalty(self, model: nn.Module, device: torch.device) -> torch.Tensor:
        """분류기(Stage A)용 커널 후보 블록(model.probe_kernel_gates) L0 페널티
        (2026-10-07 신설) — _kernel_hi_penalty와 완전히 같은 구조(게이트 확률 자체가
        활성 확률, 비용도 같은 model.kernel_hi_costs를 재사용 — 커널 HI 자체의
        계산 비용은 어느 게이트가 읽든 동일하므로)지만 model.scen_kernel_gates
        대신 model.probe_kernel_gates를 본다. probe_kernel_gates가 없으면
        (커널 HI 자체가 없는 run) 기여 없음(0)."""
        penalty = torch.zeros(1, device=device)
        probe_kernel_gates = getattr(model, "probe_kernel_gates", None)
        if probe_kernel_gates is None:
            return penalty

        kernel_costs = getattr(model, "kernel_hi_costs", None) if self.hi_cost_weighted else None
        kernel_penalty = torch.zeros(1, device=device)
        for s, gate in enumerate(probe_kernel_gates):
            p = gate.gate_prob()
            if kernel_costs is not None:
                c = torch.tensor(kernel_costs[s], dtype=p.dtype, device=p.device)
                kernel_penalty = kernel_penalty + (c * p).sum()
            else:
                kernel_penalty = kernel_penalty + p.sum()
        _n_scen = model.n_scenarios
        norm = self.l0_norm_constant if self.l0_norm_constant is not None else _n_scen
        penalty = penalty + kernel_penalty / norm
        return penalty

    def _l0_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        L0 penalty summed across all 6 scenarios.
        Charging scenarios use charge_probe; discharging use discharge_probe.

        2026-09-18(v4 로직 수정): 커널 HI 게이트(model.scen_kernel_gates)도 이제 같은
        lambda_l0로 패널티를 받는다 — 이전엔 raw HI(probe+scen)만 계산하고 커널 게이트는
        전혀 계산에 안 들어가서, 커널 쪽엔 희소화 압력이 원천적으로 없었다(실측:
        hi_selection_matrix.png에서 커널 HI가 시나리오당 16~24개씩 거의 다 gate_prob≥0.9로
        선택됨 — docs/260917_RESULTS.md).
        2026-10-04: 단일 책임 원칙에 따라 raw HI 몫(_raw_hi_penalty)과 커널 HI
        몫(_kernel_hi_penalty)을 분리했다(동작 변화 없는 순수 구조 정리) — 각 함수
        docstring에 계산 디테일 참고.
        2026-10-07: 분류기(Stage A)용 커널 후보 블록(_probe_kernel_hi_penalty)도
        추가 — 회귀와 동일하게 분류기의 커널 HI 사용에도 희소화 압력을 건다.
        """
        device = self.cost_vec.device
        eff_cost_vec = self.cost_vec if self.hi_cost_weighted else torch.ones_like(self.cost_vec)

        penalty = self._raw_hi_penalty(model, eff_cost_vec, device)
        penalty = penalty + self._kernel_hi_penalty(model, device)
        penalty = penalty + self._probe_kernel_hi_penalty(model, device)
        return penalty.squeeze()
