"""
SCR composite loss.

L = MSE(cap_pred, cap_target)
  + lambda_scen * CE(level_logits, level)
  + lambda_l0   * sum_i cost_i * P(gate_i active)

P(gate_i active) = 1 - (1 - p_probe_i)(1 - p_B_i)
  where p_probe_i and p_B_i are gate_prob() from HardConcreteGate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import get_hi_cost_vector, N_SEGS


class SCRLoss(nn.Module):

    def __init__(
        self,
        lambda_scen: float = 0.5,
        lambda_l0: float = 0.01,
    ):
        super().__init__()
        self.lambda_scen = lambda_scen
        self.lambda_l0 = lambda_l0

        # cost vector (same segment used — costs are identical across segments)
        costs = get_hi_cost_vector("dis_hi")  # shape (N_HI,)
        self.register_buffer("cost_vec", torch.tensor(costs, dtype=torch.float32))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys: total, mse, ce, l0
        """
        cap_pred = outputs["cap_pred"]          # (B,)
        level_logits = outputs["level_logits"]  # (B, 3)
        target = batch["target"]                # (B,)
        level = batch["level"]                  # (B,) int

        mse = F.mse_loss(cap_pred, target)
        ce = F.cross_entropy(level_logits, level)

        l0 = self._l0_penalty(model)

        total = mse + self.lambda_scen * ce + self.lambda_l0 * l0

        return {"total": total, "mse": mse, "ce": ce, "l0": l0}

    def _l0_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Aggregate L0 regularisation cost over all active gates.
        P(z_i != 0) = 1 - (1 - p_probe_i)(1 - p_scen_i)
        """
        device = self.cost_vec.device
        penalty = torch.zeros(1, device=device)

        if model._fixed_probe and model._fixed_scen:
            return penalty.squeeze()

        # Probe gate probabilities
        if not model._fixed_probe:
            p_probe = model.probe_gate.gate_prob()  # (N_HI,)
        else:
            p_probe = torch.ones(self.cost_vec.shape, device=device)

        # Per-segment gate probabilities (average across segments as penalty)
        if not model._fixed_scen:
            # sum over all scenario gates weighted by cost
            for gate in model.scen_gates:
                p_scen = gate.gate_prob()  # (N_HI,)
                p_active = 1.0 - (1.0 - p_probe) * (1.0 - p_scen)
                penalty = penalty + (self.cost_vec * p_active).sum()
            penalty = penalty / N_SEGS  # normalise by scenario count
        else:
            # Only probe contributes to L0
            penalty = (self.cost_vec * p_probe).sum().unsqueeze(0)

        return penalty.squeeze()
