"""Shared style module for the paper's Figure 1-8 drafts (Journal of Power
Sources camera-ready look). Import from here so every figure shares fonts,
palette, and panel-label placement -- keep this the single place to retune
the look across the whole set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_directories import DATA_4_HI_ROOT  # noqa: E402,F401

# ---------------------------------------------------------------- palette --

INK = "#141414"
SUBINK = "#5A5A5A"
FAINT = "#9C9C9C"
GRID = "#E7E4DC"

# SOC-zone triad (shared meaning across all figures that touch scenarios)
Z_LEFT = "#3D6E8C"    # bucket [0, n1]      -> chg_lo / dis_hi
Z_MID = "#D9A544"     # bucket [.5-n1/2,.5+n1/2] -> chg_mid / dis_mid
Z_RIGHT = "#B14A3C"   # bucket [1-n1, 1]    -> chg_hi / dis_lo
ZONE_COLOR = {"left": Z_LEFT, "mid": Z_MID, "right": Z_RIGHT}

# chemistry duo (LFP vs NCM), used wherever cross-chemistry contrast appears
LFP_COLOR = "#152A45"   # navy
NCM_COLOR = "#9C3B26"   # rust
LCO_COLOR = "#3E6B4A"   # forest green (CALCE, used sparingly)

# model / regression-head triad
MLP_COLOR = "#3D6E8C"
TRANSFORMER_COLOR = "#B14A3C"
RESNET_COLOR = "#6B4E8C"

# routing-mode triad (oracle/hard/soft), used across classification+regression figs
ORACLE_COLOR = "#3D6E8C"
HARD_COLOR = "#D9A544"
SOFT_COLOR = "#B14A3C"

N1 = 0.35
N2 = 0.20
ZONE_BOUNDS = {
    "left": (0.00, N1),
    "mid": (0.5 - N1 / 2, 0.5 + N1 / 2),
    "right": (1.0 - N1, 1.0),
}
OVERLAPS = [(ZONE_BOUNDS["mid"][0], ZONE_BOUNDS["left"][1]),
            (ZONE_BOUNDS["right"][0], ZONE_BOUNDS["mid"][1])]

SCEN_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
SCEN_POS_COLOR = [Z_LEFT, Z_MID, Z_RIGHT, Z_LEFT, Z_MID, Z_RIGHT]


def setup_rcparams(base_size: float = 8.3):
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Arial" if "Arial" in available else "sans-serif",
        "font.size": base_size,
        "text.color": INK,
        "axes.edgecolor": SUBINK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": SUBINK,
        "ytick.color": SUBINK,
        "xtick.labelsize": base_size - 0.7,
        "ytick.labelsize": base_size - 0.7,
        "axes.labelsize": base_size + 0.1,
        "axes.titlesize": base_size + 0.3,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.default": "regular",
        "axes.grid": False,
    })


def label_panel(fig, ax, text, dx=-0.028, dy=0.006, fontsize=10.5):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + dy, text, fontsize=fontsize, fontweight="bold",
              color=INK, va="bottom", ha="left")


def strip_top_right(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)
