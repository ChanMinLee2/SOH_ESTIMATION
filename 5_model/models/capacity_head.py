"""CapacityHead: regression head mapping fused embedding to capacity_Ah."""

from __future__ import annotations
import torch
import torch.nn as nn


class CapacityHead(nn.Module):
    """Maps fused representation to scalar battery capacity.

    Architecture: MLP with residual projection + scalar output.

    Input:  h_fused ∈ R^{B × d_fus}
    Output: cap     ∈ R^{B}  (capacity in Ah, unbounded positive)
    """

    def __init__(self, d_fus: int, d_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_fus, d_head),
            nn.LayerNorm(d_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_head, 1),
        )

    def forward(self, h_fused: torch.Tensor) -> torch.Tensor:
        """h_fused: (B, d_fus) → cap: (B,)"""
        return self.net(h_fused).squeeze(-1)
