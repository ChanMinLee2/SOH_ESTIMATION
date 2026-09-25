"""Architecture diagram -- SCR (Scenario-Conditioned Routing) model, i.e. the
"Scenario-Conditioned Subset Learning" mechanism this paper's title refers to.

2026-09-16, 3rd revision (full visual redesign per explicit request -- "too
plain, boxes with text, redo at a professional design level"): rebuilt the
whole rendering system rather than reusing the flat box-and-arrow look.
There is no python library that gives a hand-crafted conference-figure look
"for free" (the reference image this was modeled on is manual vector-art
polish, likely PowerPoint/Illustrator) -- so this invests directly in
matplotlib craft instead of a different toolchain:
  - drop shadows (matplotlib.patheffects.SimplePatchShadow) on every card
  - pseudo-3D stacked-layer cards (offset duplicate patches) wherever the
    real model has a multi-channel/multi-scenario block (raw HI, both gate
    boxes, the swappable head, the output)
  - colored "chip" stage headers (bold white-on-color) instead of plain
    italic captions
  - a simple drawn battery icon for the input card
  - two-row composition (data/Stage-A processing on top, the paper's actual
    contribution -- Stage B gating + head + output -- on the bottom, given
    more room) instead of one long single-row strip

All real-data insets from the 2nd revision are kept as-is (nothing here is
decorative filler):
  - input segment + 6 scenario-lane sparklines: actual V-t curves, HUST
    cell 1-7, cycle 2, one per seg_name (canonical q_frac_ref seg pkl)
  - raw-HI / kernel-HI gate cards: real gate_prob crops from the canonical
    v4 run's gates/regression_HIs.json + regression_kernel_HIs.json (same
    source Fig9 uses)
  - output card: the real SOH-vs-cycle trajectory (HUST 1-7, same cell/run
    as Fig6/Fig7)

Shows the real forward-pass mechanism (model_lib/models/scr_model.py):
  1. Raw HI extraction (64, 4 categories) -- direction-agnostic
  2. Direction-aware Stage-A probe gate (L0 HardConcrete) -- dual gradient
     (MSE regression + CE classification, Phase-1 only)
  3. Scenario router (probe_mlp) -- produces the 6-way routing decision,
     evaluated 3 ways at inference (oracle / hard / soft)
  4. Kernel fusion (Nystroem+Ridge on synergy groups) -- parallel feature
     branch, gated per scenario same as raw HI
  5. Scenario-conditioned subset gates -- the paper's central mechanism:
     raw HI uses a hybrid shared/specific gate, kernel HI is fully
     per-scenario (no shared option, flagged as a caption note)
  6. Swappable capacity (regression) head -> SOH prediction

Run: python docs/figures/fig_arch_scr_model.py
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from _style import (INK, SUBINK, FAINT, Z_LEFT, Z_MID, Z_RIGHT,
                     MLP_COLOR, TRANSFORMER_COLOR, RESNET_COLOR,
                     ORACLE_COLOR, HARD_COLOR, SOFT_COLOR, setup_rcparams, PROJECT_ROOT)
from data_directories import DATA_4_HI_ROOT
from fig3_hi_design_rationale import base_names_and_categories, CAT_COLOR
from fig10_v4_example import RUN_DIR as P1V4_RUN_DIR, load_cell, _true_line, CHG_LEVELS, DIS_LEVELS

setup_rcparams()

SCEN_COLORS = [Z_LEFT, Z_MID, Z_RIGHT, Z_LEFT, Z_MID, Z_RIGHT]
SCEN_LABELS = ["chg lo", "chg mid", "chg hi", "dis hi", "dis mid", "dis lo"]
SCEN_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]

# -- palette: soft tinted stage grounds + a dedicated card system on top --
STAGE_DATA_BG = "#EEF2F1"
STAGE_A_BG = "#EAF0EF"
STAGE_B_BG = "#F7F1E6"
HEAD_BG = "#F2EEF5"
CARD_FILL = "#FFFFFF"
CARD_EDGE = "#D8D4C8"
SHADOW_RGB = (0.15, 0.15, 0.16)

SEG_PKL = (DATA_4_HI_ROOT / "q_frac_ref" /
           "n1-35%_n2-20%_N-2_minpts5_lag-1_noise-3%_ou-200_calib-100_offA-5mA" /
           "seg" / "HUST" / "1-7.pkl")
GATE_JSON = P1V4_RUN_DIR / "gates" / "regression_HIs.json"
KERNEL_GATE_JSON = P1V4_RUN_DIR / "gates" / "regression_kernel_HIs.json"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_navy", ["#F4F1E8", "#182E45"])


# ---------------------------------------------------------------------------
# Real-data loaders (identical sources to Fig6/Fig9)
# ---------------------------------------------------------------------------

def load_scenario_curves(cycle=2):
    with open(SEG_PKL, "rb") as f:
        d = pickle.load(f)
    df = d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
    sub = df[df.cycle == cycle]
    out = {}
    for sn in SCEN_NAMES:
        row = sub[sub.seg_name == sn].iloc[0]
        v = np.asarray(row["raw_v"], dtype=float)
        t = np.asarray(row["raw_t"], dtype=float)
        t = (t - t.min()) / (t.max() - t.min() + 1e-12)
        out[sn] = (t, v)
    return out


def load_output_curve(cell="1-7"):
    df = load_cell(P1V4_RUN_DIR, cell)
    cycles = np.sort(df.cycle.unique())
    chg_true = _true_line(df[df.seg_name.isin([s for s, _ in CHG_LEVELS])], cycles)
    dis_true = _true_line(df[df.seg_name.isin([s for s, _ in DIS_LEVELS])], cycles)
    true_ah = 0.5 * (chg_true + dis_true)
    soh = true_ah / true_ah[0]
    return cycles.astype(float), soh


def load_raw_gate_crop(n_rows=9):
    d = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    order, cats = base_names_and_categories()
    mat = np.zeros((len(order), 6))
    for s, sname in enumerate(SCEN_NAMES):
        idx_to_prob = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_probs"]))
        idx_to_name = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_names"]))
        name_to_idx = {v.replace(f"_{sname}", ""): k for k, v in idx_to_name.items()}
        for r, hi in enumerate(order):
            mat[r, s] = idx_to_prob[name_to_idx[hi]]
    variation = mat.max(axis=1) - mat.min(axis=1)
    top = sorted(np.argsort(-variation)[:n_rows].tolist())
    return mat[top]


def load_kernel_gate_crop(n_rows=8):
    d = json.loads(KERNEL_GATE_JSON.read_text(encoding="utf-8"))
    base_names = sorted(d["seg_0_names"])
    mat = np.zeros((len(base_names), 6))
    for s, sname in enumerate(SCEN_NAMES):
        name_to_prob = dict(zip(d[f"seg_{s}_names"], d[f"seg_{s}_probs"]))
        for r, kname in enumerate(base_names):
            mat[r, s] = name_to_prob[kname]
    order = np.argsort(-mat.mean(axis=1))[:n_rows]
    return mat[order]


# ---------------------------------------------------------------------------
# Design system: cards with shadow + pseudo-3D stacking, chips, icons
# ---------------------------------------------------------------------------

SHADOW = [pe.SimplePatchShadow(offset=(0.045, -0.045), shadow_rgbFace=SHADOW_RGB, alpha=0.22), pe.Normal()]
SHADOW_SOFT = [pe.SimplePatchShadow(offset=(0.03, -0.03), shadow_rgbFace=SHADOW_RGB, alpha=0.15), pe.Normal()]


def card(ax, xy, w, h, fc=CARD_FILL, ec=CARD_EDGE, lw=1.1, depth=0, depth_color=None,
         rounding=0.09, zorder=3, shadow=True):
    """A single card, optionally with `depth` fainter offset copies behind it
    (pseudo-3D stack -- used for every "many channels at once" block: raw
    HI, both gate boxes, the swappable head, the output)."""
    x, y = xy
    depth_color = depth_color or ec
    off = 0.10
    for d in range(depth, 0, -1):
        back = FancyBboxPatch((x + d * off, y + d * off), w, h,
                               boxstyle=f"round,pad=0.012,rounding_size={rounding}",
                               linewidth=0.9, edgecolor=depth_color, facecolor="#FBFAF6",
                               zorder=zorder - 0.1 * d)
        if shadow:
            back.set_path_effects(SHADOW_SOFT)
        ax.add_patch(back)
    front = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.012,rounding_size={rounding}",
                            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
    if shadow:
        front.set_path_effects(SHADOW)
    ax.add_patch(front)
    return front


def card_text(ax, xy, w, h, title=None, body=None, title_size=7.4, body_size=6.7,
              title_color=INK, body_color=SUBINK, pad_top=0.20):
    x, y = xy
    if title:
        ax.text(x + w / 2, y + h - pad_top, title, ha="center", va="top",
                fontsize=title_size, color=title_color, fontweight="bold", zorder=6, linespacing=1.3)
    if body:
        ax.text(x + w / 2, y + h / 2 - (0.16 if title else 0), body, ha="center",
                va="center" if not title else "top",
                fontsize=body_size, color=body_color, zorder=6, linespacing=1.35)


def chip(ax, xy, w, h, text, color, text_color="white", fontsize=8.6, zorder=6):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.055",
                        linewidth=0, facecolor=color, zorder=zorder)
    p.set_path_effects(SHADOW_SOFT)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=zorder + 1)
    return p


def arrow(ax, p0, p1, color=INK, lw=1.3, ls="-", connectionstyle="arc3,rad=0.0",
          mutation_scale=11, zorder=2, alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mutation_scale,
                         linewidth=lw, edgecolor=color, facecolor=color, ls=ls, alpha=alpha,
                         connectionstyle=connectionstyle, zorder=zorder,
                         shrinkA=0, shrinkB=0, capstyle="round")
    ax.add_patch(a)
    return a


def curve_inset(ax, rect, t, v, color=SUBINK, lw=1.1, fill=False, zorder=5):
    """t and v are normalised to [0,1] internally regardless of their native
    scale (t may be an already-[0,1] fraction, as for the scenario curves,
    or raw cycle numbers, as for the output curve -- both must work)."""
    x0, y0, w, h = rect
    t = np.asarray(t, dtype=float)
    tn = (t - t.min()) / (t.max() - t.min() + 1e-9)
    vn = (v - v.min()) / (v.max() - v.min() + 1e-9)
    xs, ys = x0 + tn * w, y0 + vn * h
    if fill:
        ax.fill_between(xs, y0, ys, color=color, alpha=0.16, zorder=zorder - 1, linewidth=0)
    ax.plot(xs, ys, color=color, lw=lw, zorder=zorder, solid_capstyle="round")


def heatmap_inset(ax, rect, mat, cmap=SEQ_CMAP, zorder=5):
    x0, y0, w, h = rect
    ax.imshow(mat, extent=(x0, x0 + w, y0, y0 + h), origin="upper", aspect="auto",
              cmap=cmap, vmin=0.0, vmax=1.0, zorder=zorder)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor=SUBINK, lw=0.5, zorder=zorder + 1))


def battery_icon(ax, cx, cy, w=0.55, h=0.34, color=INK, zorder=6):
    """A small drawn battery glyph -- body + terminal nub + one internal
    charge bar -- for the raw input card, echoing the reference figure's
    battery icon without importing an image asset."""
    body = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                           linewidth=1.3, edgecolor=color, facecolor="none", zorder=zorder)
    ax.add_patch(body)
    nub_w, nub_h = w * 0.10, h * 0.42
    ax.add_patch(plt.Rectangle((cx + w / 2, cy - nub_h / 2), nub_w, nub_h,
                                facecolor=color, edgecolor="none", zorder=zorder))
    bar_w = w * 0.16
    for i in range(3):
        bx = cx - w / 2 + w * 0.14 + i * (bar_w + w * 0.06)
        ax.add_patch(plt.Rectangle((bx, cy - h * 0.30), bar_w, h * 0.60,
                                    facecolor=color, edgecolor="none", alpha=0.85, zorder=zorder))


def fusion_glyph(ax, x0, y0, colors, zorder=5):
    """Several thin member lines converging into one thick fused line --
    kernel-fusion concept icon."""
    for k, dy in enumerate([0.42, 0.30, 0.18, 0.06]):
        ax.plot([x0, x0 + 0.55], [y0 + dy, y0 + 0.24], color=colors[k], lw=1.1, alpha=0.9, zorder=zorder)
    ax.plot([x0 + 0.55, x0 + 1.45], [y0 + 0.24, y0 + 0.24], color=INK, lw=2.0, zorder=zorder)


def concat_node(ax, xy, r=0.32):
    c = Circle(xy, r, fc="white", ec=INK, lw=1.4, zorder=6)
    c.set_path_effects(SHADOW)
    ax.add_patch(c)
    ax.text(*xy, "$\\oplus$", ha="center", va="center", fontsize=13, zorder=7)


# ---------------------------------------------------------------------------
def build_figure():
    fig, ax = plt.subplots(figsize=(17.2, 11.4))
    ax.set_xlim(0, 19.6)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    scen_curves = load_scenario_curves()
    raw_crop = load_raw_gate_crop(n_rows=9)
    ker_crop = load_kernel_gate_crop(n_rows=8)
    cycles, soh = load_output_curve()

    # ================================================================
    # ROW 1 (top) -- data processing + Stage A
    # ================================================================
    ROW1_BOT = 6.15
    ax.add_patch(FancyBboxPatch((0.15, ROW1_BOT), 5.05, 11.35 - ROW1_BOT,
                                 boxstyle="round,pad=0.02,rounding_size=0.16", fc=STAGE_DATA_BG, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((5.40, ROW1_BOT), 4.35, 11.35 - ROW1_BOT,
                                 boxstyle="round,pad=0.02,rounding_size=0.16", fc=STAGE_A_BG, ec="none", zorder=0))
    chip(ax, (0.40, 10.85), 3.55, 0.42, "STAGE 1 — Data processing", "#4C7A6E", fontsize=8.6)
    chip(ax, (5.65, 10.85), 3.85, 0.42, "STAGE A — Direction-aware probe gate", "#5C7A52", fontsize=8.6)

    # -- input segment card (battery icon + real V-t curve) --
    card(ax, (0.45, 8.55), 2.05, 1.75)
    battery_icon(ax, 1.475, 10.02, w=0.62, h=0.30, color=INK)
    ax.text(1.475, 9.72, "Partial charge/\ndischarge segment", ha="center", va="top",
            fontsize=6.7, color=INK, linespacing=1.25)
    t0, v0 = scen_curves["chg_mid"]
    curve_inset(ax, (0.62, 8.68, 1.72, 0.62), t0, v0, color=Z_MID, lw=1.3, fill=True)
    ax.text(1.475, 8.60, "real V(t)", ha="center", fontsize=5.4, color=SUBINK, style="italic")

    # -- 64 raw HI, pseudo-3D stacked card --
    card(ax, (2.95, 8.75), 1.85, 1.35, depth=2, depth_color="#CFE0DA")
    card_text(ax, (2.95, 8.75), 1.85, 1.35, title="64 raw HI", body=None, title_size=7.6, pad_top=0.22)
    cat_y = 9.62
    for cat, n, lbl in [("stat", 18, "stat"), ("diff", 20, "diff"), ("lfp", 20, "lfp"), ("morph", 6, "morph")]:
        ax.add_patch(plt.Rectangle((3.14, cat_y - 0.03), 0.15, 0.15, facecolor=CAT_COLOR[cat], edgecolor="none", zorder=6))
        ax.text(3.36, cat_y + 0.045, f"{lbl} ({n})", fontsize=5.6, color=SUBINK, va="center", zorder=6)
        cat_y -= 0.225
    arrow(ax, (2.52, 9.42), (2.95, 9.42))

    # -- Stage A: direction-aware probe gate (stacked charge/discharge) --
    card(ax, (5.85, 9.10), 3.55, 1.45, depth=1, depth_color="#D9E3DB")
    card_text(ax, (5.85, 9.10), 3.55, 1.45, title="Direction-aware probe gate (L0)",
              body="charge_probe_gate /\ndischarge_probe_gate", title_size=7.2, body_size=6.7)
    arrow(ax, (5.00, 9.42), (5.85, 9.65))

    chip(ax, (6.35, 10.58), 2.55, 0.24, "CE loss (Phase 1 only)", "#8B8B86", fontsize=6.2)
    arrow(ax, (7.62, 10.55), (7.62, 10.58), color=SUBINK, ls=(0, (3, 2)), mutation_scale=8)

    # -- scenario router --
    card(ax, (5.85, 6.55), 3.55, 1.55, depth=1, depth_color="#D9E3DB")
    card_text(ax, (5.85, 6.55), 3.55, 1.55, title="Scenario router",
              body="probe_mlp classifier\n6-way SOC window", title_size=7.2, body_size=6.7)
    arrow(ax, (7.62, 9.10), (7.62, 8.10))

    pill_x = 6.00
    for lbl, c in zip(["oracle", "hard", "soft"], [ORACLE_COLOR, HARD_COLOR, SOFT_COLOR]):
        chip(ax, (pill_x, ROW1_BOT + 0.10), 0.95, 0.36, lbl, c, fontsize=6.4)
        pill_x += 1.05
    ax.text(7.62, ROW1_BOT + 0.55, "3 inference-time routing modes", fontsize=5.6, color=SUBINK,
            ha="center", style="italic")

    # ================================================================
    # ROW 2 (bottom) -- Stage B + head + output
    # ================================================================
    ROW2_TOP = 5.85
    ax.add_patch(FancyBboxPatch((0.15, 0.20), 2.65, ROW2_TOP - 0.20,
                                 boxstyle="round,pad=0.02,rounding_size=0.16", fc=STAGE_DATA_BG, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((3.05, 0.20), 8.55, ROW2_TOP - 0.20,
                                 boxstyle="round,pad=0.02,rounding_size=0.16", fc=STAGE_B_BG, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((11.85, 0.20), 7.50, ROW2_TOP - 0.20,
                                 boxstyle="round,pad=0.02,rounding_size=0.16", fc=HEAD_BG, ec="none", zorder=0))
    chip(ax, (0.40, ROW2_TOP - 0.55), 2.15, 0.42, "Kernel fusion", "#8A5A2E", fontsize=7.6)
    chip(ax, (3.30, ROW2_TOP - 0.55), 6.05, 0.42, "STAGE B — scenario-conditioned subset gates", "#B08A3E", fontsize=8.6)
    chip(ax, (12.10, ROW2_TOP - 0.55), 3.30, 0.42, "Swappable head + output", "#7A5C88", fontsize=8.0)

    # vertical connector: raw HI (row1) -> kernel fusion (row2)
    arrow(ax, (3.30, 8.75), (1.72, ROW2_TOP - 0.05), connectionstyle="arc3,rad=0.25", color=SUBINK, lw=1.1)
    ax.text(2.55, 7.3, "raw HI\n(shared input)", fontsize=5.4, color=SUBINK, ha="left", linespacing=1.2)

    # -- kernel fusion card --
    card(ax, (0.40, 0.55), 2.15, 1.65, depth=1, depth_color="#EAD9BE")
    fusion_glyph(ax, 0.62, 1.55, [CAT_COLOR["stat"], CAT_COLOR["diff"], CAT_COLOR["lfp"], CAT_COLOR["morph"]])
    ax.text(2.00, 1.79, "1 kernel HI", fontsize=5.0, color=SUBINK, ha="center", style="italic")
    ax.text(1.475, 1.20, "synergy groups ->\nNyström + ridge\n+7-12 HI / scenario",
            ha="center", va="top", fontsize=6.0, color=SUBINK, linespacing=1.3)

    # -- scenario fan-out, real per-scenario V-t sparkline in each lane --
    fan_x0, fan_top, fan_w, fan_h = 3.30, 5.15, 1.05, 4.65
    gap = 0.12
    lane_h = (fan_h - gap * 5) / 6
    for i, (sn, lbl, c) in enumerate(zip(SCEN_NAMES, SCEN_LABELS, SCEN_COLORS)):
        y = fan_top - (i + 1) * lane_h - i * gap
        p = FancyBboxPatch((fan_x0, y), fan_w, lane_h, boxstyle="round,pad=0.008,rounding_size=0.045",
                            linewidth=0, facecolor=c, zorder=3)
        p.set_path_effects(SHADOW_SOFT)
        ax.add_patch(p)
        ax.text(fan_x0 + 0.26, y + lane_h / 2, lbl.replace(" ", "\n"), ha="center", va="center",
                fontsize=5.4, color="white", fontweight="bold", zorder=5, linespacing=0.95)
        t, v = scen_curves[sn]
        curve_inset(ax, (fan_x0 + 0.48, y + 0.05, fan_w - 0.58, lane_h - 0.10), t, v,
                     color="white", lw=1.0, fill=False, zorder=5)
    ax.text(fan_x0 + fan_w / 2, fan_top + 0.10, "real V(t)\nper scenario", fontsize=5.0, color=SUBINK,
            ha="center", style="italic", linespacing=1.05)
    arrow(ax, (5.85, 7.10), (fan_x0 + fan_w / 2, fan_top + 0.35), connectionstyle="arc3,rad=0.28", mutation_scale=11)
    ax.text(4.55, 6.55, "route", fontsize=5.6, color=SUBINK, ha="center", style="italic")

    # -- Stage B: raw-HI + kernel-HI subset gates, real gate_prob crops --
    card(ax, (4.55, 3.05), 3.75, 2.30, depth=2, depth_color="#E9DFC4")
    card_text(ax, (4.55, 3.05), 3.75, 2.30, title="Raw-HI subset gate (hybrid, L0)",
              body=None, title_size=7.2, pad_top=0.22)
    ax.text(4.75, 4.98, "shared_gate (25) + scen_gates[s] (39)", fontsize=5.8, color=SUBINK, ha="left", va="top")
    heatmap_inset(ax, (4.75, 3.22, 3.35, 1.45), raw_crop)
    for j, sc in enumerate(SCEN_COLORS):
        ax.add_patch(plt.Rectangle((4.75 + j * (3.35 / 6), 3.14), 3.35 / 6, 0.06, facecolor=sc, edgecolor="none", zorder=6))
    ax.text(6.425, 3.07, "real gate probability -- 9 most scenario-sensitive HI", fontsize=4.9, color=SUBINK,
            ha="center", style="italic")

    card(ax, (4.55, 0.55), 3.75, 2.10, depth=2, depth_color="#E9DFC4")
    card_text(ax, (4.55, 0.55), 3.75, 2.10, title="Kernel-HI subset gate (L0)", body=None, title_size=7.2, pad_top=0.22)
    ax.text(4.75, 2.28, "scen_kernel_gates[s], fully per-scenario", fontsize=5.8, color=SUBINK, ha="left", va="top")
    heatmap_inset(ax, (4.75, 0.72, 3.35, 1.25), ker_crop)
    for j, sc in enumerate(SCEN_COLORS):
        ax.add_patch(plt.Rectangle((4.75 + j * (3.35 / 6), 0.64), 3.35 / 6, 0.06, facecolor=sc, edgecolor="none", zorder=6))
    ax.text(6.425, 0.475, "real gate probability -- top-8 kernel HI by mean prob", fontsize=4.9, color=SUBINK,
            ha="center", style="italic")

    arrow(ax, (4.35, 4.0), (4.55, 4.0), connectionstyle="arc3,rad=0.0")
    arrow(ax, (4.35, 1.55), (4.55, 1.55), connectionstyle="arc3,rad=0.0")

    # -- concat node --
    concat_node(ax, (10.45, 3.20))
    arrow(ax, (8.30, 4.55), (10.15, 3.35), connectionstyle="arc3,rad=-0.18")
    arrow(ax, (8.30, 1.55), (10.15, 3.05), connectionstyle="arc3,rad=0.20")
    ax.text(10.45, 4.40, "probe_x $\\oplus$ scen_x $\\oplus$\nkernel_x $\\oplus$ direction\n$\\oplus$ cap_init",
            ha="center", fontsize=5.6, color=SUBINK)

    # -- swappable capacity head, stacked cards --
    head_colors = [RESNET_COLOR, TRANSFORMER_COLOR, MLP_COLOR]
    head_labels = ["ResNet-tab.", "Transformer", "MLP (default)"]
    for i, (c, lbl) in enumerate(zip(head_colors, head_labels)):
        off = (2 - i) * 0.20
        p = FancyBboxPatch((12.25 + off, 3.55 - off), 1.55, 0.95, boxstyle="round,pad=0.012,rounding_size=0.08",
                            linewidth=1.5, edgecolor=c, facecolor="white", zorder=4 + i)
        p.set_path_effects(SHADOW_SOFT)
        ax.add_patch(p)
        ax.text(12.25 + off + 0.775, 3.55 - off + 0.475, lbl, ha="center", va="center", fontsize=6.3, color=INK, zorder=7)
    ax.text(13.32, 2.55, "swappable\ncapacity head", ha="center", fontsize=5.6, color=SUBINK, style="italic")
    arrow(ax, (10.77, 3.30), (12.25, 3.75), connectionstyle="arc3,rad=-0.10")

    # -- output card, real SOH-vs-cycle curve --
    card(ax, (12.05, 0.55), 2.55, 2.10, fc=INK, ec=INK, depth=2, depth_color="#3A3A38", rounding=0.10)
    ax.text(13.325, 2.35, "Predicted SOH ($\\hat{c}$)", ha="center", va="top", fontsize=7.4,
            color="white", fontweight="bold", zorder=7)
    curve_inset(ax, (12.25, 0.72, 2.15, 1.15), cycles, soh, color="#EADFC8", lw=1.4, fill=True)
    ax.text(13.325, 0.60, "real SOH vs. cycle, HUST 1-7", fontsize=5.2, color="#C9C2AE", ha="center", style="italic")
    arrow(ax, (13.32, 3.30), (13.32, 2.70), connectionstyle="arc3,rad=0.0")

    # ================================================================
    # Bottom equation + legend
    # ================================================================
    ax.text(9.8, -0.02,
            r"$x_{hi}(64) \rightarrow probe_x,\ scen_x,\ kernel_x$"
            r"$\ \Rightarrow\ \oplus\,(dir,\,cap_{init})\ \Rightarrow\ cap\_head \rightarrow SOH$",
            ha="center", va="bottom", fontsize=7.8, color=INK)

    leg_elems = [
        Line2D([0], [0], color=INK, lw=1.4, marker=">", markersize=4.5, label="feature / data flow"),
        Line2D([0], [0], color=SUBINK, lw=1.3, ls=(0, (3, 2)), label="auxiliary gradient (Phase 1 only)"),
        Line2D([0], [0], color=Z_MID, lw=1.6, label="real V(t) / SOH(cycle) curve"),
        plt.Rectangle((0, 0), 1, 1, fc=SEQ_CMAP(0.85), label="high gate probability (selected)"),
        plt.Rectangle((0, 0), 1, 1, fc=SEQ_CMAP(0.05), label="low gate probability (pruned)"),
    ]
    ax.legend(handles=leg_elems, loc="upper left", bbox_to_anchor=(10.05, 11.30),
              bbox_transform=ax.transData, fontsize=6.8, frameon=False, ncol=1,
              handlelength=2.1, labelspacing=0.7, borderaxespad=0.0)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.02)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig6_arch_scr_model.png"
    out_pdf = "docs/figures/fig6_arch_scr_model.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
