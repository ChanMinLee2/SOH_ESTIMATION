"""
Hard-Concrete L0 gate (Louizos et al. 2018).

Parameters: beta=2/3, gamma=-0.1, zeta=1.1
Train:  s = sigmoid((log U - log(1-U) + log_alpha) / beta)
        z = clamp(s*(zeta - gamma) + gamma, 0, 1)
Infer:  s = sigmoid(log_alpha)
        active = (z > 0)  ← hard binary mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """
    Learnable Hard-Concrete L0 gate.

    n_features: number of binary gates (one per input HI)
    """

    BETA: float = 2.0 / 3.0
    GAMMA: float = -0.1
    ZETA: float = 1.1

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        # log_alpha initialised near zero (50% open probability at init)
        self.log_alpha = nn.Parameter(torch.zeros(n_features))

    # ------------------------------------------------------------------
    # Probability that gate is non-zero (used in L0 loss)
    # ------------------------------------------------------------------
    def gate_prob(self) -> torch.Tensor:
        """P(z > 0) ≈ sigmoid(log_alpha - beta * log(-gamma/zeta))."""
        offset = self.BETA * torch.log(
            torch.tensor(-self.GAMMA / self.ZETA, device=self.log_alpha.device)
        )
        return torch.sigmoid(self.log_alpha - offset)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (..., n_features)
        Returns:
          masked_x : (..., n_features)  element-wise gated
          z        : (..., n_features)  [0,1] gates
        """
        if self.training:
            z = self._sample_train(x)
        else:
            z = self._hard_infer().expand_as(x)

        return x * z, z

    def _sample_train(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.zeros_like(x[..., :self.n_features]).uniform_().clamp(1e-8, 1 - 1e-8)
        s = torch.sigmoid(
            (torch.log(u) - torch.log(1.0 - u) + self.log_alpha) / self.BETA
        )
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return z

    def _hard_infer(self) -> torch.Tensor:
        s = torch.sigmoid(self.log_alpha)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return (z > 0).float()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def active_indices(self) -> list[int]:
        """Return indices of active gates at inference time."""
        s = torch.sigmoid(self.log_alpha)
        z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)
        return (z > 0).nonzero(as_tuple=False).squeeze(1).tolist()

    @torch.no_grad()
    def active_count(self) -> int:
        return len(self.active_indices())

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}"
