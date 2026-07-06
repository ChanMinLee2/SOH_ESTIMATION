"""FeatureRouter: maps latent z to per-group routing gates.

Uses Gumbel-Sigmoid relaxation during training for differentiable
binary routing decisions. Hard (threshold) gates are used at inference.

Each gate g_k ∈ (0, 1) represents "should HI group k be computed?"
Sparsity is encouraged via L1 regularization on the gates in the loss.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FeatureRouter(nn.Module):
    """Maps latent encoding z to per-group routing gates.

    Architecture: z → MLP → logits → Gumbel-Sigmoid (train) / Sigmoid (eval)

    The Gumbel-Sigmoid (also known as concrete Bernoulli) provides a
    differentiable relaxation of discrete binary routing decisions.
    Temperature annealing gradually sharpens soft gates toward binary.

    Args:
        d_enc:       Dimension of latent encoding z.
        n_groups:    Number of HI groups (24 = 6 segments × 4 categories).
        hidden_dim:  Router MLP hidden dimension.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        d_enc: int,
        n_groups: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_groups = n_groups

        self.net = nn.Sequential(
            nn.Linear(d_enc, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_groups),
        )

    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """Compute routing gates.

        Args:
            z:           (B, d_enc) latent encoding.
            temperature: Gumbel temperature (training only). Lower = more binary.
            hard:        If True, apply hard threshold at 0.5 (inference mode).

        Returns:
            gates: (B, n_groups) in (0, 1) during training,
                   {0, 1} float if hard=True.
        """
        logits = self.net(z)  # (B, n_groups)

        if self.training:
            # Gumbel-Sigmoid: add Gumbel noise for stochastic binary relaxation
            # Equivalent to sampling from a concrete/BinConcrete distribution
            u = torch.zeros_like(logits).uniform_().clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            gates = torch.sigmoid((logits + gumbel) / temperature)
        else:
            gates = torch.sigmoid(logits)
            if hard:
                gates = (gates > 0.5).float()

        return gates  # (B, n_groups)

    def get_gate_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic gate probabilities (no Gumbel noise, no hard threshold).

        Useful for analysis and visualization.
        """
        with torch.no_grad():
            return torch.sigmoid(self.net(z))
