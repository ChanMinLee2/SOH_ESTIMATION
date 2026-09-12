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
#
# 2026-09-11: desaturated pass -- referenced against typical dQ/dV & capacity-
# fade figures in the battery-degradation literature (Dahn-group DVA/ICA
# papers, Nature Energy / Joule cycling studies), which favor dark, low-chroma
# hues (steel-blue, muted ochre, brick, slate) over saturated primaries so
# multi-line overlays stay legible in greyscale print and don't compete with
# data. Every hue below was pulled down in saturation/lightness from the prior
# pass; hue identity (which color means what) is unchanged.

INK = "#1A1A1A"
SUBINK = "#5C5C5C"
FAINT = "#9B9B9B"
GRID = "#E7E4DC"

# SOC-zone triad (shared meaning across all figures that touch scenarios)
Z_LEFT = "#3A6178"    # bucket [0, n1]      -> chg_lo / dis_hi
Z_MID = "#A9863F"     # bucket [.5-n1/2,.5+n1/2] -> chg_mid / dis_mid
Z_RIGHT = "#96473A"   # bucket [1-n1, 1]    -> chg_hi / dis_lo
ZONE_COLOR = {"left": Z_LEFT, "mid": Z_MID, "right": Z_RIGHT}

# chemistry duo (LFP vs NCM), used wherever cross-chemistry contrast appears
LFP_COLOR = "#182E45"   # navy
NCM_COLOR = "#8A3F2E"   # rust
LCO_COLOR = "#3C5C42"   # forest green (CALCE, used sparingly)

# per-dataset triad for MIT/HUST/TJU overlays (fig1): MIT & HUST share the
# LFP hue family (same chemistry, different manufacturer -- dark vs. lighter
# tint of the same blue) so the encoding itself shows "two LFP cells, one
# NCM cell" at a glance; TJU (NCM) gets the distinct rust hue.
MIT_COLOR = "#182E45"    # LFP, dark navy (== LFP_COLOR)
HUST_COLOR = "#5B7C93"   # LFP, lighter slate-blue tint of the same family
TJU_COLOR = "#8A3F2E"    # NCM, rust (== NCM_COLOR)
DATASET_COLOR = {"MIT": MIT_COLOR, "HUST": HUST_COLOR, "TJU": TJU_COLOR}

# model / regression-head triad
MLP_COLOR = "#3A6178"
TRANSFORMER_COLOR = "#96473A"
RESNET_COLOR = "#644E70"

# routing-mode triad (oracle/hard/soft), used across classification+regression figs
ORACLE_COLOR = "#3A6178"
HARD_COLOR = "#A9863F"
SOFT_COLOR = "#96473A"

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
