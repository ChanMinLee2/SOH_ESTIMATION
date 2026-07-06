"""DFRLoss: MSE reconstruction + cost-weighted sparsity regularization.

Total loss = MSE(cap_pred, cap_true) + λ_sparse * sparsity_loss

Sparsity loss penalizes the use of expensive HI groups, encouraging the
router to select fewer, cheaper features while maintaining accuracy.

If cost_weighted=True, sparsity_loss = mean(gates * costs), which penalizes
selecting high-cost groups (morph > lfp > diff > stat) more heavily.
If cost_weighted=False, sparsity_loss = mean(gates) (uniform penalty).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DFRLoss(nn.Module):
    """Dynamic Feature Routing loss.

    Args:
        lambda_sparse: Weight for the sparsity regularization term.
        group_costs:   List of per-group computational costs aligned with group_keys.
        cost_weighted: If True, weight sparsity penalty by group cost.
    """

    def __init__(
        self,
        lambda_sparse: float,
        group_costs: List[float],
        cost_weighted: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.cost_weighted = cost_weighted
        self.register_buffer(
            "group_costs",
            torch.tensor(group_costs, dtype=torch.float32),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss.

        Args:
            pred:   (B,) predicted capacity_Ah.
            target: (B,) true capacity_Ah.
            gates:  (B, n_groups) routing gate values in [0, 1].

        Returns:
            total_loss: scalar tensor.
            breakdown:  dict with 'mse', 'sparse', 'total' for logging.
        """
        mse = F.mse_loss(pred, target)

        if self.cost_weighted:
            costs = self.group_costs.to(gates.device)   # (n_groups,)
            # Normalize costs to [0, 1] range for stable weighting
            costs_norm = costs / costs.max()
            sparse = (gates * costs_norm).mean()
        else:
            sparse = gates.mean()

        total = mse + self.lambda_sparse * sparse

        return total, {
            "mse": mse.item(),
            "sparse": sparse.item(),
            "total": total.item(),
        }
