"""HI feature group definitions for Dynamic Feature Routing.

Groups are defined as (segment × category) pairs.
Global HIs are always computed; segment-category groups are dynamically routed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pathlib
import pickle
import pandas as pd
from . import compat  # noqa: F401  — numpy 1.x/2.x pkl compatibility

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

META_COLS = ["dataset", "cell_id", "cycle", "capacity_Ah"]

GLOBAL_HIS: List[str] = [
    "q_dis", "energy_dis", "v_mean_dis", "r_dc_est", "q_plateau_frac",
    "ica_peak1_v", "ica_peak1_h", "ica_peak1_area", "ica_peak1_asym",
    "dva_valley_q", "dva_valley_depth", "ce", "cv_q_frac", "cv_time_frac",
    "chg_ica_peak1_h",
]

SEGMENTS: List[str] = ["dis_hi", "dis_mid", "dis_lo", "chg_lo", "chg_mid", "chg_hi"]
CATEGORIES: List[str] = ["stat", "diff", "lfp", "morph"]

# Relative computational cost per category (used for cost-weighted sparsity)
CATEGORY_COSTS: Dict[str, float] = {
    "stat": 1.0,   # simple statistics
    "diff": 1.5,   # SG smoothing + numerical differentiation
    "lfp": 2.0,    # LFP-specific analysis, plateau detection
    "morph": 3.0,  # DTW / Fréchet distance, most expensive
}

# ---------------------------------------------------------------------------
# Group info dataclass
# ---------------------------------------------------------------------------

@dataclass
class HIGroupInfo:
    """Holds all metadata about HI feature groups."""

    global_his: List[str]
    # group_keys: ordered list of (segment, category) tuples
    group_keys: List[Tuple[str, str]]
    # groups: group_name → sorted list of column names
    groups: Dict[str, List[str]]
    # category sizes: category → number of features per segment
    cat_sizes: Dict[str, int]
    # per-group costs aligned with group_keys
    group_costs: List[float]

    @property
    def n_global(self) -> int:
        return len(self.global_his)

    @property
    def n_groups(self) -> int:
        return len(self.group_keys)

    def group_name(self, seg: str, cat: str) -> str:
        return f"{seg}_{cat}"

    def group_index(self, seg: str, cat: str) -> int:
        return self.group_keys.index((seg, cat))

    def group_sizes(self) -> List[int]:
        return [len(self.groups[self.group_name(s, c)]) for s, c in self.group_keys]


# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------

def _discover_groups(columns: List[str]) -> Dict[str, List[str]]:
    """Auto-detect segment-category groups from DataFrame columns."""
    groups: Dict[str, List[str]] = {}
    for seg in SEGMENTS:
        for cat in CATEGORIES:
            prefix = f"{cat}_"
            suffix = f"_{seg}"
            cols = sorted([c for c in columns if c.startswith(prefix) and c.endswith(suffix)])
            if cols:
                groups[f"{seg}_{cat}"] = cols
    return groups


def build_hi_group_info(columns: List[str]) -> HIGroupInfo:
    """Build HIGroupInfo from a list of DataFrame columns."""
    raw_groups = _discover_groups(columns)

    group_keys: List[Tuple[str, str]] = []
    groups: Dict[str, List[str]] = {}
    group_costs: List[float] = []

    for seg in SEGMENTS:
        for cat in CATEGORIES:
            name = f"{seg}_{cat}"
            if name in raw_groups:
                group_keys.append((seg, cat))
                groups[name] = raw_groups[name]
                group_costs.append(CATEGORY_COSTS[cat])

    # Infer category sizes (number of features per segment per category)
    cat_sizes: Dict[str, int] = {}
    for seg in SEGMENTS:
        for cat in CATEGORIES:
            name = f"{seg}_{cat}"
            if name in groups:
                cat_sizes[cat] = max(cat_sizes.get(cat, 0), len(groups[name]))

    # Verify global HIs exist
    missing = [h for h in GLOBAL_HIS if h not in columns]
    if missing:
        raise ValueError(f"Global HIs missing from data: {missing}")

    return HIGroupInfo(
        global_his=GLOBAL_HIS,
        group_keys=group_keys,
        groups=groups,
        cat_sizes=cat_sizes,
        group_costs=group_costs,
    )


def build_hi_group_info_from_data(data_dir: str) -> HIGroupInfo:
    """Build HIGroupInfo by scanning a single pkl file in data_dir."""
    root = pathlib.Path(data_dir)
    for dataset in ["MIT", "HUST"]:
        ds_dir = root / dataset
        if ds_dir.exists():
            pkls = list(ds_dir.glob("*.pkl"))
            if pkls:
                with open(pkls[0], "rb") as f:
                    df: pd.DataFrame = pickle.load(f)
                return build_hi_group_info(list(df.columns))
    raise FileNotFoundError(f"No pkl files found under {data_dir}")
