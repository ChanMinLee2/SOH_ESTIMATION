"""
SCR (Scenario-Conditioned Routing) model.

Stage A — probe selection + scenario classifier
  1. HardConcreteGate (size=N_HI) selects m probe HIs
  2. MLP: probe_x → level logits (3-way classification)

Stage B — scenario-conditioned regression
  3. Per-scenario HardConcreteGate (6 x N_HI) selects k additional HIs
     conditioned on the classified level
  4. Capacity head: [probe_x || B_x || direction] → capacity_Ah (normalised)

At inference JSON masks may replace the L0 gates (fixed binary vectors).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hi_schema import N_HI, N_SEGS, N_LEVELS


class SCRModel(nn.Module):
    """
    Args:
        d_probe   : hidden dim for Stage A MLP
        d_head    : hidden dim for capacity head
        dropout   : dropout rate
        probe_mask: fixed boolean tensor (N_HI,) if probe HIs are loaded from JSON
        scen_masks: fixed boolean tensor (N_SEGS, N_HI) if scenario HIs are loaded from JSON
    """

    def __init__(
        self,
        d_probe: int = 64,
        d_head: int = 128,
        dropout: float = 0.1,
        probe_mask: Optional[torch.Tensor] = None,   # (N_HI,) bool
        scen_masks: Optional[torch.Tensor] = None,   # (N_SEGS, N_HI) bool
    ):
        super().__init__()
        self.d_probe = d_probe
        self.d_head = d_head

        # ----------------------------------------------------------------
        # Stage A
        # ----------------------------------------------------------------
        from models.hard_concrete import HardConcreteGate

        if probe_mask is None:
            self.probe_gate: nn.Module = HardConcreteGate(N_HI)
            self._fixed_probe = False
        else:
            # Register as buffer (not optimised, always active)
            self.register_buffer("_probe_mask_buf", probe_mask.float())
            self.probe_gate = None
            self._fixed_probe = True

        # MLP: probe → level logits
        self.probe_mlp = nn.Sequential(
            nn.Linear(N_HI, d_probe),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_probe, d_probe // 2),
            nn.ReLU(),
            nn.Linear(d_probe // 2, N_LEVELS),
        )

        # ----------------------------------------------------------------
        # Stage B — one gate per scenario (6 scenarios × N_HI)
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
        # ----------------------------------------------------------------
        # input: probe_x (N_HI) || scen_x (N_HI) || direction (1)
        head_in = N_HI + N_HI + 1
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
    def _apply_probe_gate(self, x: torch.Tensor):
        """Returns (masked_x, z). x: (B, N_HI)."""
        if self._fixed_probe:
            mask = self._probe_mask_buf  # (N_HI,)
            return x * mask, mask.unsqueeze(0).expand_as(x)
        return self.probe_gate(x)

    def _apply_scen_gate(self, x: torch.Tensor, seg_idx: torch.Tensor):
        """
        For each sample, apply its scenario-specific gate.
        x: (B, N_HI), seg_idx: (B,) int in [0, N_SEGS)
        Returns (masked_x, z): both (B, N_HI)
        """
        B = x.size(0)
        masked = torch.zeros_like(x)
        z_out = torch.zeros_like(x)

        if self._fixed_scen:
            for s in range(N_SEGS):
                sel = (seg_idx == s)
                if sel.any():
                    m = self._scen_masks_buf[s]  # (N_HI,)
                    n_sel = int(sel.sum().item())
                    masked[sel] = x[sel] * m
                    z_out[sel] = m.unsqueeze(0).expand(n_sel, -1)
        else:
            for s, gate in enumerate(self.scen_gates):
                sel = (seg_idx == s)
                if sel.any():
                    mx, zz = gate(x[sel])
                    masked[sel] = mx
                    z_out[sel] = zz

        return masked, z_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        batch keys: x_hi (B,N_HI), nan_mask (B,N_HI), direction (B,),
                    seg_idx (B,), level (B,)

        Returns dict with:
          cap_pred    : (B,)   normalised capacity prediction
          level_logits: (B,3)  scenario classification logits
          probe_z     : (B,N_HI)
          scen_z      : (B,N_HI)
        """
        x = batch["x_hi"]                        # (B, N_HI)
        nan_mask = batch["nan_mask"]              # (B, N_HI)
        direction = batch["direction"].unsqueeze(1)  # (B, 1)
        seg_idx = batch["seg_idx"]                # (B,)

        # NaN positions must not contribute — zero them before gating
        x = x * nan_mask

        # Stage A
        probe_x, probe_z = self._apply_probe_gate(x)             # (B, N_HI)
        level_logits = self.probe_mlp(probe_x)                    # (B, 3)

        # Stage B
        scen_x, scen_z = self._apply_scen_gate(x, seg_idx)       # (B, N_HI)

        # Capacity head
        feat = torch.cat([probe_x, scen_x, direction], dim=1)    # (B, 2*N_HI+1)
        cap_pred = self.cap_head(feat).squeeze(1)                 # (B,)

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
        """Eval mode forward — hard binary gates."""
        self.eval()
        return self.forward(batch)

    @torch.no_grad()
    def get_selected_probe_his(self) -> list[int]:
        if self._fixed_probe:
            return self._probe_mask_buf.nonzero(as_tuple=False).squeeze(1).tolist()
        return self.probe_gate.active_indices()

    @torch.no_grad()
    def get_selected_scen_his(self) -> dict[int, list[int]]:
        """Returns {seg_idx: [active_hi_indices]}."""
        if self._fixed_scen:
            result = {}
            for s in range(N_SEGS):
                result[s] = self._scen_masks_buf[s].nonzero(as_tuple=False).squeeze(1).tolist()
            return result
        return {s: gate.active_indices() for s, gate in enumerate(self.scen_gates)}
