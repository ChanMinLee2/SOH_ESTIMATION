"""
SCR (Scenario-Conditioned Routing) model.

Stage A — probe selection + scenario classifier
  1. Direction-aware HardConcreteGate:
       charge_probe_gate  (N_HI)  — charging segments (chg_lo/mid/hi)
       discharge_probe_gate (N_HI) — discharging segments (dis_hi/mid/lo)
     Each gate independently learns its optimal HI subset.
  2. Shared MLP: probe_x → level logits (3-way classification)

Stage B — scenario-conditioned regression
  3. Per-scenario HardConcreteGate (6 x N_HI) selects k additional HIs
  4. Capacity head: [probe_x || scen_x || direction] → capacity_Ah (normalised)
     Input = m (probe, already computed) + k (scen) + 1 (direction)

At inference JSON masks may replace the L0 gates (fixed binary vectors).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import N_HI, N_SEGS, N_LEVELS

# Segment indices by direction (based on SCEN_MAP ordering)
_CHARGE_SEGS    = frozenset({0, 1, 2})   # chg_lo, chg_mid, chg_hi
_DISCHARGE_SEGS = frozenset({3, 4, 5})   # dis_hi, dis_mid, dis_lo


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
        scen_masks: Optional[torch.Tensor] = None,           # (N_SEGS, N_HI) bool
    ):
        super().__init__()
        self.d_probe = d_probe
        self.d_head = d_head

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

        # Shared MLP: probe_x (N_HI, direction-masked) → level logits
        self.probe_mlp = nn.Sequential(
            nn.Linear(N_HI, d_probe),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_probe, d_probe // 2),
            nn.ReLU(),
            nn.Linear(d_probe // 2, N_LEVELS),
        )

        # ----------------------------------------------------------------
        # Stage B — per-scenario gates (6 scenarios × N_HI)
        # ----------------------------------------------------------------
        if scen_masks is None:
            self.scen_gates = nn.ModuleList(
                [HardConcreteGate(N_HI) for _ in range(N_SEGS)]
            )
            self._fixed_scen = False
        else:
            self.register_buffer("_scen_masks_buf", scen_masks.float())
            self.scen_gates = None
            self._fixed_scen = True

        # ----------------------------------------------------------------
        # Capacity head
        # input: probe_x (N_HI) || scen_x (N_HI) || direction (1)
        #        || dataset_id (1) || cap_init (1)
        # = m active probe HIs + k active scen HIs + 3 scalars
        # ----------------------------------------------------------------
        head_in = N_HI + N_HI + 1 + 1 + 1  # 64+64+1+1+1 = 131
        self.cap_head = nn.Sequential(
            nn.Linear(head_in, d_head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_head, d_head // 2),
            nn.ReLU(),
            nn.Linear(d_head // 2, 1),
        )

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
            for s in range(N_SEGS):
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
                    seg_idx (B,), level (B,)

        Returns:
          cap_pred     : (B,)    normalised capacity prediction
          level_logits : (B,3)   scenario classification logits
          probe_z      : (B,N_HI)
          scen_z       : (B,N_HI)
        """
        x         = batch["x_hi"]                           # (B, N_HI)
        nan_mask  = batch["nan_mask"]                       # (B, N_HI)
        direction = batch["direction"]                      # (B,)
        seg_idx   = batch["seg_idx"]                        # (B,)

        x = x * nan_mask  # NaN positions → 0

        # Stage A: direction-aware probe gate
        probe_x, probe_z = self._apply_probe_gate(x, direction, seg_idx)
        level_logits = self.probe_mlp(probe_x)              # (B, 3)

        # Stage B: scenario-conditioned gate
        scen_x, scen_z = self._apply_scen_gate(x, seg_idx) # (B, N_HI)

        # Capacity head: probe_x + scen_x + direction + dataset_id + cap_init
        feat = torch.cat(
            [probe_x, scen_x,
             direction.unsqueeze(1),
             batch["dataset_id"].unsqueeze(1),
             batch["cap_init"].unsqueeze(1)],
            dim=1,
        )                                                   # (B, 2*N_HI+3 = 131)
        cap_pred = self.cap_head(feat).squeeze(1)           # (B,)

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
                for s in range(N_SEGS)
            }
        return {s: gate.active_indices() for s, gate in enumerate(self.scen_gates)}
