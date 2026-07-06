"""InitialEncoder: encodes always-available Global HIs into latent z."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class _MLPBlock(nn.Module):
    """Linear → LayerNorm → GELU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InitialEncoder(nn.Module):
    """Maps global HIs (always computed) to a latent representation z.

    Architecture: MLP with layer normalization.
    Input:  x_global ∈ R^{B × n_global}
    Output: z        ∈ R^{B × d_enc}
    """

    def __init__(
        self,
        n_global: int,
        d_enc: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = d_enc * 2
        layers: List[nn.Module] = []

        in_dim = n_global
        for i in range(n_layers - 1):
            layers.append(_MLPBlock(in_dim, hidden_dim, dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, d_enc))
        layers.append(nn.LayerNorm(d_enc))

        self.net = nn.Sequential(*layers)
        self.d_enc = d_enc

    def forward(self, x_global: torch.Tensor) -> torch.Tensor:
        """x_global: (B, n_global) → z: (B, d_enc)"""
        return self.net(x_global)


class CategoryEncoder(nn.Module):
    """Shared encoder for one HI category (stat / diff / lfp / morph).

    The same encoder weights are applied to all segments of the same category,
    reducing parameter count and encouraging generalizable representations.

    Input:  x ∈ R^{B × n_features}  (features of one segment-category group)
    Output: h ∈ R^{B × d_grp}
    """

    def __init__(self, n_features: int, d_grp: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_grp * 2),
            nn.LayerNorm(d_grp * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_grp * 2, d_grp),
            nn.LayerNorm(d_grp),
        )
        self.d_grp = d_grp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features) → h: (B, d_grp)"""
        return self.net(x)
