"""
SCR (Scenario-Conditioned Routing) model.

Stage A — direction-aware probe gate (HI subset selection, regression only)
  1. Direction-aware HardConcreteGate:
       charge_probe_gate    (N_HI) — charging segments
       discharge_probe_gate (N_HI) — discharging segments
     Each gate independently selects m HIs optimal for regression.
  Note: probe_mlp (level classification) removed. CE 손실 비활성.

Stage B — scenario-conditioned regression
  2. Per-scenario HardConcreteGate (n_scenarios × N_HI) selects k additional HIs
  3. Capacity head: [probe_x || scen_x || direction || cap_init] → capacity_Ah

At inference JSON masks may replace the L0 gates (fixed binary vectors).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import N_HI, spec_from_qfrac
from models.cap_heads import build_cap_head


class SCRModel(nn.Module):
    """
    Args:
        d_probe          : hidden dim for Stage A MLP
        d_head           : hidden dim for capacity head
        dropout          : dropout rate
        charge_probe_mask   : fixed bool (N_HI,) for charging probe (Phase 2 / test)
        discharge_probe_mask: fixed bool (N_HI,) for discharging probe (Phase 2 / test)
        scen_masks       : fixed bool (N_SEGS, N_HI) for scenario gates (Phase 2 / test)
    """

    def __init__(
        self,
        d_probe: int = 64,
        d_head: int = 128,
        dropout: float = 0.1,
        charge_probe_mask: Optional[torch.Tensor] = None,    # (N_HI,) bool
        discharge_probe_mask: Optional[torch.Tensor] = None, # (N_HI,) bool
        scen_masks: Optional[torch.Tensor] = None,           # (n_scenarios, N_HI) bool
        model_cfg: Optional[dict] = None,  # Phase 2 전용: regression_model 선택
        spec=None,   # ScenarioSpec | None  (None → qfrac default)
    ):
        super().__init__()
        self.d_probe = d_probe
        self.d_head = d_head

        # Resolve spec (stored as plain Python attr; save separately as JSON)
        if spec is None:
            spec = spec_from_qfrac()
        self.spec = spec
        self.n_scenarios = spec.n_scenarios
        self.n_classes   = spec.n_classes

        from models.hard_concrete import HardConcreteGate

        # ----------------------------------------------------------------
        # Stage A — direction-aware probe gates
        # ----------------------------------------------------------------
        fixed_probe = (charge_probe_mask is not None and
                       discharge_probe_mask is not None)
        self._fixed_probe = fixed_probe

        if not fixed_probe:
            self.charge_probe_gate    = HardConcreteGate(N_HI)
            self.discharge_probe_gate = HardConcreteGate(N_HI)
        else:
            self.charge_probe_gate    = None
            self.discharge_probe_gate = None
            self.register_buffer("_charge_probe_mask_buf",    charge_probe_mask.float())
            self.register_buffer("_discharge_probe_mask_buf", discharge_probe_mask.float())

        # ----------------------------------------------------------------
        # Stage B — per-scenario gates (n_scenarios × N_HI)
        # ----------------------------------------------------------------
        if scen_masks is None:
            self.scen_gates = nn.ModuleList(
                [HardConcreteGate(N_HI) for _ in range(self.n_scenarios)]
            )
            self._fixed_scen = False
        else:
            self.register_buffer("_scen_masks_buf", scen_masks.float())
            self.scen_gates = None
            self._fixed_scen = True

        # ----------------------------------------------------------------
        # Capacity head
        # input: probe_x (N_HI) || scen_x (N_HI) || direction (1) || cap_init (1)
        # = m active probe HIs + k active scen HIs + 2 scalars
        # Phase 1: 항상 MLP (model_cfg=None)
        # Phase 2: model_cfg["regression_model"] 에 따라 MLP/Transformer/iTransformer
        # ----------------------------------------------------------------
        self.cap_head = build_cap_head(model_cfg or {}, d_head=d_head, dropout=dropout)

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------
    def _apply_probe_gate(
        self, x: torch.Tensor, direction: torch.Tensor, seg_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Routes each sample to its direction-specific probe gate.
        x         : (B, N_HI)
        direction : (B,)  +1.0=charge, -1.0=discharge
        seg_idx   : (B,)  0-5
        Returns (probe_x, probe_z): both (B, N_HI)
        """
        B = x.size(0)
        probe_x = torch.zeros_like(x)
        probe_z = torch.zeros_like(x)

        ch_sel  = (direction > 0)   # charging samples
        dis_sel = (direction <= 0)  # discharging samples

        if self._fixed_probe:
            if ch_sel.any():
                m = self._charge_probe_mask_buf          # (N_HI,)
                n = int(ch_sel.sum().item())
                probe_x[ch_sel] = x[ch_sel] * m
                probe_z[ch_sel] = m.unsqueeze(0).expand(n, -1)
            if dis_sel.any():
                m = self._discharge_probe_mask_buf
                n = int(dis_sel.sum().item())
                probe_x[dis_sel] = x[dis_sel] * m
                probe_z[dis_sel] = m.unsqueeze(0).expand(n, -1)
        else:
            if ch_sel.any():
                mx, zz = self.charge_probe_gate(x[ch_sel])
                probe_x[ch_sel] = mx
                probe_z[ch_sel] = zz
            if dis_sel.any():
                mx, zz = self.discharge_probe_gate(x[dis_sel])
                probe_x[dis_sel] = mx
                probe_z[dis_sel] = zz

        return probe_x, probe_z

    def _apply_scen_gate(
        self, x: torch.Tensor, seg_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each sample, apply its scenario-specific gate.
        x: (B, N_HI), seg_idx: (B,) int in [0, N_SEGS)
        Returns (masked_x, z): both (B, N_HI)
        """
        B = x.size(0)
        masked = torch.zeros_like(x)
        z_out  = torch.zeros_like(x)

        if self._fixed_scen:
            for s in range(self.n_scenarios):
                sel = (seg_idx == s)
                if sel.any():
                    m = self._scen_masks_buf[s]
                    n_sel = int(sel.sum().item())
                    masked[sel] = x[sel] * m
                    z_out[sel]  = m.unsqueeze(0).expand(n_sel, -1)
        else:
            for s, gate in enumerate(self.scen_gates):
                sel = (seg_idx == s)
                if sel.any():
                    mx, zz = gate(x[sel])
                    masked[sel] = mx
                    z_out[sel]  = zz

        return masked, z_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        batch keys: x_hi (B,N_HI), nan_mask (B,N_HI), direction (B,),
                    seg_idx (B,), cap_init (B,)

        Returns:
          cap_pred     : (B,)      normalised capacity prediction
          level_logits : (B,n_cls) zeros — CE 비활성, 로그 호환용
          probe_z      : (B,N_HI)
          scen_z       : (B,N_HI)
        """
        x         = batch["x_hi"]                           # (B, N_HI)
        nan_mask  = batch["nan_mask"]                       # (B, N_HI)
        direction = batch["direction"]                      # (B,)
        seg_idx   = batch["seg_idx"]                        # (B,)

        x = x * nan_mask  # NaN positions → 0

        # Stage A: direction-aware probe gate (regression HI selection only)
        probe_x, probe_z = self._apply_probe_gate(x, direction, seg_idx)

        # Stage B: scenario-conditioned gate
        scen_x, scen_z = self._apply_scen_gate(x, seg_idx) # (B, N_HI)

        # Capacity head: probe_x + scen_x + direction + cap_init
        feat = torch.cat(
            [probe_x, scen_x,
             direction.unsqueeze(1),
             batch["cap_init"].unsqueeze(1)],
            dim=1,
        )                                                   # (B, 2*N_HI+2)
        cap_pred = self.cap_head(feat)                      # (B,)

        # CE 비활성 — zeros 반환 (evaluator 로그 호환)
        level_logits = torch.zeros(
            x.size(0), self.n_classes, dtype=x.dtype, device=x.device
        )

        return {
            "cap_pred":     cap_pred,
            "level_logits": level_logits,
            "probe_z":      probe_z,
            "scen_z":       scen_z,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(batch)

    @torch.no_grad()
    def get_selected_probe_his(self) -> dict[str, list[int]]:
        """Returns {"charge": [active_hi_indices], "discharge": [...]}."""
        if self._fixed_probe:
            return {
                "charge":    self._charge_probe_mask_buf.nonzero(as_tuple=False).squeeze(1).tolist(),
                "discharge": self._discharge_probe_mask_buf.nonzero(as_tuple=False).squeeze(1).tolist(),
            }
        return {
            "charge":    self.charge_probe_gate.active_indices(),
            "discharge": self.discharge_probe_gate.active_indices(),
        }

    @torch.no_grad()
    def get_selected_scen_his(self) -> dict[int, list[int]]:
        """Returns {seg_idx: [active_hi_indices]}."""
        if self._fixed_scen:
            return {
                s: self._scen_masks_buf[s].nonzero(as_tuple=False).squeeze(1).tolist()
                for s in range(self.n_scenarios)
            }
        return {s: gate.active_indices() for s, gate in enumerate(self.scen_gates)}
