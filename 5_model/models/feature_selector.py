"""FeatureSelector: applies routing gates to segment-category feature groups.

Handles NaN masking (features absent due to chg_gap_seg or computation failure)
independently of routing gates. A feature is zeroed if it is either (a) NaN
in the original data or (b) its group was not selected by the router.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple


class FeatureSelector(nn.Module):
    """Applies per-group routing gates to feature tensors.

    This module is stateless — it only performs element-wise multiplication.

    For each group k:
        x_out_k = x_k * mask_k * gate_k

    where:
        x_k     : (B, n_k)  normalized feature values (NaN → 0 already)
        mask_k  : (B, n_k)  validity mask (1=original value, 0=was NaN)
        gate_k  : (B, 1)    routing gate for group k

    During training, gate_k is a continuous value in (0, 1), enabling
    gradient flow for end-to-end optimization.
    During inference, gate_k ∈ {0, 1} for actual computational savings.
    """

    def forward(
        self,
        group_features: List[torch.Tensor],
        nan_masks: List[torch.Tensor],
        gates: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply gates and NaN masks to each feature group.

        Args:
            group_features: list of (B, n_k) tensors, one per group.
            nan_masks:      list of (B, n_k) float tensors (1=valid, 0=NaN).
            gates:          (B, n_groups) routing weights in [0, 1].

        Returns:
            List of (B, n_k) gated, masked feature tensors.
        """
        gated: List[torch.Tensor] = []
        for k, (x_k, mask_k) in enumerate(zip(group_features, nan_masks)):
            g_k = gates[:, k : k + 1]          # (B, 1)
            x_out = x_k * mask_k * g_k         # (B, n_k)
            gated.append(x_out)
        return gated

    def compute_effective_sparsity(
        self,
        gates: torch.Tensor,
        group_costs: List[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sparsity statistics for monitoring.

        Returns:
            mean_active:    mean number of active groups per sample.
            weighted_cost:  mean computational cost per sample (gate × cost).
        """
        costs = torch.tensor(group_costs, device=gates.device, dtype=gates.dtype)
        mean_active = (gates > 0.5).float().sum(dim=-1).mean()
        weighted_cost = (gates * costs).mean()
        return mean_active, weighted_cost
