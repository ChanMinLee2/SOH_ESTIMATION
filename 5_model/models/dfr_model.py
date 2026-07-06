"""DFRModel: Dynamic Feature Routing model for battery capacity estimation.

Full forward pipeline:
  1. InitialEncoder:    x_global            → z
  2. FeatureRouter:     z                   → gates (B, n_groups)
  3. FeatureSelector:   group_features, gates → gated_features
  4. CategoryEncoders:  gated_features[k]   → h_groups[k]  (shared per category)
  5. FeatureFusion:     z, h_groups         → h_fused
  6. CapacityHead:      h_fused             → capacity_Ah
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from models.encoder import InitialEncoder, CategoryEncoder
from models.router import FeatureRouter
from models.feature_selector import FeatureSelector
from models.feature_fusion import FeatureFusion
from models.capacity_head import CapacityHead
from utils.hi_groups import HIGroupInfo


class DFRModel(nn.Module):
    """Dynamic Feature Routing model.

    Args:
        hi_info: HIGroupInfo built from the data columns.
        cfg:     Nested config dict (model / router sub-sections).
    """

    def __init__(self, hi_info: HIGroupInfo, cfg: Dict) -> None:
        super().__init__()

        mcfg = cfg["model"]
        rcfg = cfg["router"]

        d_enc = mcfg["d_enc"]
        d_grp = mcfg["d_grp"]
        d_fus = mcfg["d_fus"]
        d_head = mcfg["d_head"]
        dropout = mcfg["dropout"]

        self.hi_info = hi_info
        self.group_keys = hi_info.group_keys  # list of (seg, cat) tuples

        # 1. InitialEncoder
        self.encoder = InitialEncoder(
            n_global=hi_info.n_global,
            d_enc=d_enc,
            n_layers=mcfg["n_layers_enc"],
            dropout=dropout,
        )

        # 2. FeatureRouter
        self.router = FeatureRouter(
            d_enc=d_enc,
            n_groups=hi_info.n_groups,
            hidden_dim=rcfg["hidden_dim"],
            dropout=dropout,
        )

        # 3. FeatureSelector (stateless)
        self.selector = FeatureSelector()

        # 4. Category-shared GroupEncoders (one encoder per category, shared across segs)
        self.group_encoders = nn.ModuleDict({
            cat: CategoryEncoder(
                n_features=hi_info.cat_sizes[cat],
                d_grp=d_grp,
                dropout=dropout,
            )
            for cat in hi_info.cat_sizes
        })

        # 5. FeatureFusion
        self.fusion = FeatureFusion(
            d_enc=d_enc,
            n_groups=hi_info.n_groups,
            d_grp=d_grp,
            d_fus=d_fus,
            n_layers=mcfg["n_layers_fus"],
            dropout=dropout,
        )

        # 6. CapacityHead
        self.head = CapacityHead(d_fus=d_fus, d_head=d_head, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        batch: Dict[str, object],
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full DFR forward pass.

        Args:
            batch:       Dict from DataLoader/collate_fn containing:
                           x_global      (B, n_global)
                           group_features list of K (B, n_k)
                           nan_masks      list of K (B, n_k)
            temperature: Router Gumbel temperature (training).
            hard:        Whether to use hard (binary) gates (inference).

        Returns:
            cap:   (B,)        predicted capacity_Ah
            gates: (B, K)      routing gate values
        """
        x_global: torch.Tensor = batch["x_global"]
        group_features: List[torch.Tensor] = batch["group_features"]
        nan_masks: List[torch.Tensor] = batch["nan_masks"]

        # Step 1: Initial encoding
        z = self.encoder(x_global)                     # (B, d_enc)

        # Step 2: Routing
        gates = self.router(z, temperature, hard)      # (B, n_groups)

        # Step 3: Feature selection (NaN zeroing + gate multiplication)
        gated = self.selector(group_features, nan_masks, gates)

        # Step 4: Category-shared group encoding
        h_groups: List[torch.Tensor] = []
        for k, (seg, cat) in enumerate(self.group_keys):
            h_k = self.group_encoders[cat](gated[k])   # (B, d_grp)
            h_groups.append(h_k)

        # Step 5: Fusion
        h_fused = self.fusion(z, h_groups)             # (B, d_fus)

        # Step 6: Capacity estimation
        cap = self.head(h_fused)                       # (B,)

        return cap, gates

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, object],
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-mode forward with hard gates.

        Moves batch to device, runs hard routing, returns (predictions, gates)
        on CPU.
        """
        self.eval()
        batch_dev = {
            "x_global": batch["x_global"].to(device),
            "group_features": [gf.to(device) for gf in batch["group_features"]],
            "nan_masks": [m.to(device) for m in batch["nan_masks"]],
        }
        cap, gates = self.forward(batch_dev, hard=True)
        return cap.cpu(), gates.cpu()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def group_names(self) -> List[str]:
        return [self.hi_info.group_name(s, c) for s, c in self.group_keys]
