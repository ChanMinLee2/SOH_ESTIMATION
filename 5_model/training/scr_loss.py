"""
SCR composite loss.

L = MSE(cap_pred, cap_target)
  + lambda_scen * CE(level_logits, level)
  + lambda_l0   * L0_penalty

L0_penalty: for each of 6 scenarios, computes E[cost of active HIs].
  Charging scenarios (0-2): use charge_probe_gate + scen_gate[s]
  Discharging scenarios (3-5): use discharge_probe_gate + scen_gate[s]
  P(HI_i active in scenario s) = 1 - (1 - p_probe_i)(1 - p_scen_i)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import get_hi_cost_vector, N_SEGS

_CHARGE_SEGS    = frozenset({0, 1, 2})
_DISCHARGE_SEGS = frozenset({3, 4, 5})


class SCRLoss(nn.Module):

    def __init__(
        self,
        lambda_scen: float = 0.5,
        lambda_l0: float = 0.01,
    ):
        super().__init__()
        self.lambda_scen = lambda_scen
        self.lambda_l0 = lambda_l0

        costs = get_hi_cost_vector("dis_hi")
        self.register_buffer("cost_vec", torch.tensor(costs, dtype=torch.float32))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        cap_pred     = outputs["cap_pred"]
        level_logits = outputs["level_logits"]
        target       = batch["target"]
        level        = batch["level"]

        mse = F.mse_loss(cap_pred, target)
        ce  = F.cross_entropy(level_logits, level)
        l0  = self._l0_penalty(model)

        total = mse + self.lambda_scen * ce + self.lambda_l0 * l0

        return {"total": total, "mse": mse, "ce": ce, "l0": l0}

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
            for s, gate in enumerate(model.scen_gates):
                p_probe = p_probe_ch if s in _CHARGE_SEGS else p_probe_dis
                p_scen  = gate.gate_prob()
                p_active = 1.0 - (1.0 - p_probe) * (1.0 - p_scen)
                penalty  = penalty + (self.cost_vec * p_active).sum()
            penalty = penalty / N_SEGS
        else:
            # Only probe gates contribute
            penalty = (
                (self.cost_vec * p_probe_ch).sum() +
                (self.cost_vec * p_probe_dis).sum()
            ).unsqueeze(0) / 2

        return penalty.squeeze()
