"""FeatureFusion: fuses global encoding z with gated group embeddings."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class FeatureFusion(nn.Module):
    """Fuses global encoding z with per-group embeddings h_k.

    Architecture: concat([z, h_0, h_1, ..., h_{K-1}]) → MLP → h_fused

    The concatenation approach handles variable group activity naturally:
    inactive groups contribute zero vectors (gate=0 → h_k=0), so the
    input dimensionality is fixed regardless of routing decisions.

    Args:
        d_enc:    Dimension of global encoding z.
        n_groups: Number of HI groups.
        d_grp:    Per-group embedding dimension.
        d_fus:    Output (fusion) dimension.
        n_layers: Number of MLP hidden layers.
        dropout:  Dropout rate.
    """

    def __init__(
        self,
        d_enc: int,
        n_groups: int,
        d_grp: int,
        d_fus: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = d_enc + n_groups * d_grp

        layers: List[nn.Module] = []
        cur_dim = in_dim
        for i in range(n_layers - 1):
            layers += [
                nn.Linear(cur_dim, d_fus),
                nn.LayerNorm(d_fus),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            cur_dim = d_fus
        layers += [
            nn.Linear(cur_dim, d_fus),
            nn.LayerNorm(d_fus),
        ]
        self.net = nn.Sequential(*layers)
        self.d_fus = d_fus

    def forward(
        self,
        z: torch.Tensor,
        h_groups: List[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse global encoding and group embeddings.

        Args:
            z:        (B, d_enc) global latent encoding.
            h_groups: list of K (B, d_grp) tensors, one per group.

        Returns:
            h_fused: (B, d_fus)
        """
        h_cat = torch.cat(h_groups, dim=-1)     # (B, K * d_grp)
        x = torch.cat([z, h_cat], dim=-1)       # (B, d_enc + K * d_grp)
        return self.net(x)                       # (B, d_fus)
