"""
SCR composite loss.

Phase 1 dual-objective:
  L = MSE(cap_pred, cap_target)           ← probe_gate + scen_gates
    + lambda_scen * CE(level_logits, level) ← probe_gate only (via probe_mlp)
    + lambda_l0   * L0_penalty             ← sparsity on all gates
    + lambda_shrink * shrinkage_penalty    ← scen_gates가 ShrinkageHardConcreteGate일 때만

Phase 2 (probe_mlp=None):
  L = MSE + lambda_l0 * L0_penalty

L0_penalty: for each scenario, computes E[cost of active HIs].
  Charging scenarios: use charge_probe_gate + scen_gate[s]
  Discharging scenarios: use discharge_probe_gate + scen_gate[s]
  P(HI_i active in scenario s) = 1 - (1 - p_probe_i)(1 - p_scen_i)

shrinkage_penalty (docs/260909_RESULTS.md §6-5(e) 대응책 ①): model.scen_gates가
ShrinkageHardConcreteGate 뱅크(scr_model.py의 shrinkage_gate=True)일 때만 존재하는
mean(delta_log_alpha^2) — 시나리오별 편차를 0쪽으로 눌러 shared_log_alpha가
전체 시나리오 데이터로 학습되도록 유도한다. 그 외(기존 nn.ModuleList 게이트)에는
이 항이 아예 0으로 꺼진다 — 기존 run과 100% 동일 동작.

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
        lambda_shrink: float = 0.0,
        l0_norm_constant: int | None = None,
    ):
        super().__init__()
        self.lambda_scen = lambda_scen  # > 0 → CE 활성 (Phase 1 with probe_mlp)
        self.lambda_l0 = lambda_l0
        self.lambda_shrink = lambda_shrink  # > 0 → scen_gates가 ShrinkageHardConcreteGate일 때만 의미 있음
        self.l0_norm_constant = l0_norm_constant  # None(기본)이면 n_scenarios로 나눔(기존과 동일)

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

        # shrinkage: model.scen_gates가 ShrinkageHardConcreteGate일 때만 실제 항이 붙음
        shrink = torch.zeros(1, device=cap_pred.device).squeeze()
        _shrink_fn = getattr(model.scen_gates, "shrinkage_penalty", None)
        if self.lambda_shrink > 0 and _shrink_fn is not None:
            shrink = _shrink_fn()

        total = (mse + self.lambda_scen * ce + self.lambda_l0 * l0
                 + self.lambda_shrink * shrink)
        return {"total": total, "mse": mse, "ce": ce, "l0": l0, "shrink": shrink}

    def _l0_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        L0 penalty summed across all 6 scenarios.
        Charging scenarios use charge_probe; discharging use discharge_probe.
        P(z_i != 0) = 1 - (1-p_probe_i)(1-p_scen_i)
        """
        device = self.cost_vec.device
        penalty = torch.zeros(1, device=device)

        if model._fixed_probe and model._fixed_scen:
            return penalty.squeeze()

        ones = torch.ones_like(self.cost_vec)

        # Probe gate probs per direction
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
                penalty  = penalty + (self.cost_vec * p_active).sum()
            norm = self.l0_norm_constant if self.l0_norm_constant is not None else _n_scen
            penalty = penalty / norm
        else:
            # Only probe gates contribute
            penalty = (
                (self.cost_vec * p_probe_ch).sum() +
                (self.cost_vec * p_probe_dis).sum()
            ).unsqueeze(0) / 2

        return penalty.squeeze()
