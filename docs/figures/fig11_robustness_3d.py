"""Figure 11 -- robustness: segment length (n2%) x regression head.

STATUS: placeholder / style draft only, per main.tex's own note ("Status:
blocked, no figure yet" -- existing n2-range runs mix multiple segment
lengths within one training run as augmentation, so per-value performance
can't be isolated; iTransformer has no trained run at all). The three
existing MLP/transformer/ResNet-tabular runs that DO exist for a model
sweep (0907_*_all4_*) additionally pool all 4 chemistries (MIT+HUST+TJU+
CALCE), not the canonical MIT+HUST-only setting this figure is supposed to
isolate -- using them here would silently conflate "regression head" with
"+2 more chemistries", the same confound problem already avoided when
scoping Fig6 to MIT+HUST only. So no panel has a clean real dataset yet.

What's real here: the "True" capacity trajectory (HUST cell 1-7, same cell
and run as Fig6) and, for the model sweep only, the *relative ranking/
magnitude* of accuracy across MLP/transformer/ResNet-tabular from the all4
comparison in docs/260915_RESULTS.md (transformer > resnet_tab > mlp) --
used only to calibrate how far apart the synthetic curves sit, not as
literal predicted values. Everything else is a clearly-labeled SYNTHETIC
placeholder: a fixed noise realization laid over the true trajectory, its
amplitude scaled by a per-condition "badness" factor, meant only to prove
out the plotting style before the real sweep runs exist. Do not cite numeric
values read off these plots.

Layout (2026-09-16, 2nd revision): the natural instinct was one 3D panel per
scenario (8 total, 2 sweeps x 4 scenarios) but that fragments into 8 small
plots whose axis labels/shape become hard to read individually -- 3D reads
well as "one big picture", not as a small-multiples grid. Split responsibility
instead: the two 3D panels (a)/(b) keep the aggregate (all-scenario) true
trajectory -- the "what does the sweep do to the overall aging curve" story
-- and two heatmaps (c)/(d) carry the scenario-level detail (row=all 6
scenarios, column=swept condition, value=a placeholder error summary), the
same scenario x something visual language already used in Fig1/5/9/10.

Panel (b) (regression head) is a CATEGORICAL axis (no ordering between
MLP/Transformer/ResNet-tab/iTransformer) -- unlike (a)'s n2%, which is a
genuine continuous quantity, there is nothing meaningful "between" two model
names, so panel (b) draws no interpolated surface fill at all (that would
visually claim a continuous blend across categories that doesn't exist) --
only the discrete per-model True/Predicted trajectory line pairs, at their
own fixed y-position.

Style: x=Cycle, y=swept condition, z=SOH (panel a only; panel b has no
z-surface, see above); light-gray pane background, no grid at all, corner
("far-vertex") view so no axis is edge-on; panel (a)'s surface is linearly
interpolated along both x (cycle) and y (n2%, continuous) so the sweep reads
as one continuous sheet.

Run: python docs/figures/fig7_robustness_3d.py   (PNG only, no PDF --
draft not yet placed in the paper; do not treat as a submittable result).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d

from _style import (INK, SUBINK, setup_rcparams, label_panel,
                     MLP_COLOR, TRANSFORMER_COLOR, RESNET_COLOR, ITRANSFORMER_COLOR)
from fig10_v4_example import RUN_DIR, load_cell, _true_line, CHG_LEVELS, DIS_LEVELS

setup_rcparams()

PANE_GRAY = (0.90, 0.90, 0.90, 1.0)
RNG = np.random.default_rng(7)

ALL_SCENARIOS = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]

# -- n2% sweep -- placeholder "badness" (higher = further from true) --
# shorter segments assumed noisier/less informative, a plausible but
# UNVERIFIED prior, not a measured result.
N2_VALUES = [5, 10, 15, 20]
N2_BADNESS = [1.00, 0.62, 0.38, 0.28]
# sequential ramp (n2% is an ordered quantity, unlike the categorical model
# names below) -- light to dark steel-blue, worst (5%) to best (20%)
N2_COLOR = ["#A8BFCE", "#6E92A8", "#3A6178", "#182E45"]

# -- regression head sweep -- badness calibrated from the REAL relative
# ranking in docs/260915_RESULTS.md (all4 pooling, not this canonical
# setting -- see module docstring): transformer best, resnet_tab close
# second, mlp worst. iTransformer has never been run; placed between
# transformer and resnet_tab as a neutral placeholder, not a prediction.
MODEL_NAMES = ["MLP", "Transformer", "ResNet-tab", "iTransformer"]
MODEL_COLOR = [MLP_COLOR, TRANSFORMER_COLOR, RESNET_COLOR, ITRANSFORMER_COLOR]
MODEL_BADNESS = [1.00, 0.30, 0.42, 0.36]

N_Y_FINE = 80     # cycle resolution of the rendered surface
N_X_FINE = 60     # interpolated resolution along the swept axis (panel a only)

ERR_CMAP = LinearSegmentedColormap.from_list("seq_rust", ["#F4F1E8", "#8A3F2E"])


def true_and_cycles(cell="1-7", seg_name=None):
    """seg_name=None -> aggregate (chg+dis averaged, panels a/b); a specific
    scenario name -> that scenario's own true line (heatmap panels c/d)."""
    df = load_cell(RUN_DIR, cell)
    cycles = np.sort(df.cycle.unique())
    if seg_name is None:
        chg_true = _true_line(df[df.seg_name.isin([s for s, _ in CHG_LEVELS])], cycles)
        dis_true = _true_line(df[df.seg_name.isin([s for s, _ in DIS_LEVELS])], cycles)
        true_ah = 0.5 * (chg_true + dis_true)
    else:
        true_ah = _true_line(df[df.seg_name == seg_name], cycles)
    soh = true_ah / true_ah[0]
    return cycles.astype(float), soh


def synthetic_predicted_surface(cycles, soh_true, badness_by_x, rng=None):
    """(len(badness_by_x), len(cycles)) placeholder predicted-SOH surface:
    true trajectory + a fixed smooth noise pattern scaled by each
    x-condition's badness factor, plus a light SOH-dependent term so error
    visibly grows at low SOH (the one part of the shape below that mirrors a
    real, already-established pattern from Fig5/Fig6)."""
    rng = rng or RNG
    n = len(cycles)
    base_noise = np.sin(np.linspace(0, 6.5, n) + rng.uniform(0, 3.0)) * 0.5 + \
        rng.normal(0, 0.35, n).cumsum() / n * 3.0
    base_noise -= base_noise.mean()
    soh_penalty = (1.0 - soh_true) ** 1.3
    surf = np.empty((len(badness_by_x), n))
    for i, b in enumerate(badness_by_x):
        amp = 0.075 * b
        surf[i] = soh_true + amp * base_noise * (0.4 + 2.2 * soh_penalty)
    return surf


def interpolate_surface(x_known, surf_known, x_fine):
    f = interp1d(x_known, surf_known, axis=0, kind="linear")
    return f(x_fine)


def style_3d_axes(ax, xlabel="Cycle", ylabel="", zlabel="SOH"):
    """Light-gray pane fill, no grid mesh at all -- instead just the 3 axis
    reference lines themselves (one per dimension, meeting at the front-
    bottom-left corner), the minimal "3-axis-grid" look requested in place
    of either a full grid mesh or no grid reference at all."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE_GRAY)
        axis.pane.set_edgecolor(PANE_GRAY)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    axis_lw, axis_color = 0.9, SUBINK
    ax.plot([x0, x1], [y0, y0], [z0, z0], color=axis_color, lw=axis_lw, zorder=1)
    ax.plot([x0, x0], [y0, y1], [z0, z0], color=axis_color, lw=axis_lw, zorder=1)
    ax.plot([x0, x0], [y0, y0], [z0, z1], color=axis_color, lw=axis_lw, zorder=1)
    # re-pin limits -- plotting the axis lines can otherwise nudge matplotlib's
    # 3D autoscale margin outward (e.g. Cycle axis drifting to show -250).
    ax.set_xlim3d(x0, x1); ax.set_ylim3d(y0, y1); ax.set_zlim3d(z0, z1)
    ax.set_xlabel(xlabel, fontsize=7.2, labelpad=5)
    ax.set_ylabel(ylabel, fontsize=7.2, labelpad=4)
    ax.set_zlabel(zlabel, fontsize=7.2, labelpad=1)
    ax.tick_params(axis="both", labelsize=6.0, pad=0, colors=SUBINK)
    ax.tick_params(axis="z", labelsize=6.0, pad=0, colors=SUBINK)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1.15, 1.6, 0.85))


def panel_n2(ax, cycles, soh_true):
    """Panel (a): n2% is a genuine continuous quantity -- full interpolated
    surface, both x (cycle) and y (n2%)."""
    surf_known = synthetic_predicted_surface(cycles, soh_true, N2_BADNESS)
    y_fine = np.linspace(min(N2_VALUES), max(N2_VALUES), N_X_FINE)
    surf_fine = interpolate_surface(np.array(N2_VALUES, dtype=float), surf_known, y_fine)
    x_fine = np.linspace(cycles.min(), cycles.max(), N_Y_FINE)
    surf_fine = interp1d(cycles, surf_fine, axis=1, kind="linear")(x_fine)

    Y, X = np.meshgrid(y_fine, x_fine, indexing="ij")
    soh_true_fine = interp1d(cycles, soh_true, kind="linear")(x_fine)
    true_surf = np.tile(soh_true_fine, (len(y_fine), 1))

    ax.plot_surface(X, Y, true_surf, color=INK, alpha=0.25, linewidth=0,
                     antialiased=True, shade=False, zorder=3)
    ax.plot_surface(X, Y, surf_fine, color=TRANSFORMER_COLOR, alpha=0.40,
                     linewidth=0, antialiased=True, shade=True, zorder=4)
    for yv, c in zip(N2_VALUES, N2_COLOR):
        ax.plot(cycles, np.full_like(cycles, yv, dtype=float), surf_known[N2_VALUES.index(yv)],
                 color=c, lw=1.3, zorder=7)
        ax.plot([cycles.min()], [yv], [soh_true.min() - 0.005], marker="o", ms=3.2, color=c, zorder=8)

    ax.set_yticks(N2_VALUES)
    ax.set_yticklabels([f"{v}%" for v in N2_VALUES], fontsize=6.2)
    ax.set_zlim(soh_true.min() - 0.01, 1.005)
    style_3d_axes(ax, ylabel="n2 (%)")
    ax.set_title("Segment length (n2%) sweep  [SYNTHETIC]", loc="left", fontsize=8.0, pad=10)


def panel_model(ax, cycles, soh_true):
    """Panel (b): regression head is CATEGORICAL -- no ordering, so no
    interpolated surface fill (that would falsely imply a continuous blend
    between model architectures). Only discrete True/Predicted trajectory
    line pairs, one per model, at fixed y-positions."""
    surf_known = synthetic_predicted_surface(cycles, soh_true, MODEL_BADNESS)
    y_positions = [0, 1, 2, 3]
    for yv, c in zip(y_positions, MODEL_COLOR):
        ax.plot(cycles, np.full_like(cycles, yv, dtype=float), soh_true,
                 color=INK, alpha=0.35, lw=1.0, zorder=6)
        ax.plot(cycles, np.full_like(cycles, yv, dtype=float), surf_known[yv],
                 color=c, lw=1.4, zorder=7)
        ax.plot([cycles.min()], [yv], [soh_true.min() - 0.005], marker="o", ms=3.2, color=c, zorder=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(MODEL_NAMES, fontsize=6.0)
    ax.set_zlim(soh_true.min() - 0.01, 1.005)
    style_3d_axes(ax, ylabel="Regression head")
    ax.set_title("Regression head sweep  [SYNTHETIC, no surface -- categorical axis]",
                  loc="left", fontsize=8.0, pad=10)


def scenario_error_matrix(x_values, badness_list, seed_offset):
    """(6 scenarios, len(x_values)) mean relative error (%) over the last
    20% of cycles -- a single placeholder summary number per (scenario,
    condition) cell, feeding the heatmap panels."""
    mat = np.zeros((len(ALL_SCENARIOS), len(x_values)))
    for i, sname in enumerate(ALL_SCENARIOS):
        cycles, soh_true = true_and_cycles(seg_name=sname)
        rng = np.random.default_rng(seed_offset + i)
        surf = synthetic_predicted_surface(cycles, soh_true, badness_list, rng=rng)
        tail = cycles >= np.percentile(cycles, 80)
        rel_err = np.abs(surf[:, tail] - soh_true[tail]) / soh_true[tail] * 100
        mat[i] = rel_err.mean(axis=1)
    return mat


def draw_heatmap(ax, mat, col_labels, title):
    im = ax.imshow(mat, aspect="auto", cmap=ERR_CMAP, vmin=0.0, vmax=mat.max() * 1.05)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7.4)
    ax.set_yticks(range(len(ALL_SCENARIOS)))
    ax.set_yticklabels(ALL_SCENARIOS, fontsize=7.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.axhline(2.5, color="white", lw=1.4)
    for i in range(len(ALL_SCENARIOS)):
        for j in range(len(col_labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.2,
                     color="white" if v > mat.max() * 0.6 else INK)
    ax.set_title(title, loc="left", fontsize=8.0, pad=8)
    return im


def build_figure():
    cycles, soh_true = true_and_cycles()

    fig = plt.figure(figsize=(11.5, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.32, hspace=0.42,
                           top=0.90, bottom=0.06, left=0.045, right=0.96)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[1, 0], projection="3d")
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_n2(ax_a, cycles, soh_true)
    panel_model(ax_b, cycles, soh_true)

    mat_n2 = scenario_error_matrix(N2_VALUES, N2_BADNESS, seed_offset=100)
    mat_model = scenario_error_matrix([0, 1, 2, 3], MODEL_BADNESS, seed_offset=200)
    im_c = draw_heatmap(ax_c, mat_n2, [f"{v}%" for v in N2_VALUES],
                          "Per-scenario error, last-20% cycles [SYNTHETIC]\n(n2% sweep)")
    im_d = draw_heatmap(ax_d, mat_model, MODEL_NAMES,
                          "Per-scenario error, last-20% cycles [SYNTHETIC]\n(regression head sweep)")
    for im, ax in ((im_c, ax_c), (im_d, ax_d)):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("mean |rel. error| (%)", fontsize=6.8, color=SUBINK)
        cbar.ax.tick_params(labelsize=6.0, length=2)
        cbar.outline.set_visible(False)

    handles = [plt.Line2D([0], [0], color=INK, lw=4, alpha=0.5, label="True"),
               plt.Line2D([0], [0], color=TRANSFORMER_COLOR, lw=4, alpha=0.7, label="Predicted (placeholder)")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.05, 0.965),
                      ncol=2, frameon=False, fontsize=7.4, handlelength=1.5)
    for t in leg.get_texts():
        t.set_color(SUBINK)

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)"), (ax_d, "(d)")):
        label_panel(fig, ax, letter, dx=-0.025, dy=0.012)

    fig.suptitle("Fig. 7 draft -- cell HUST 1-7 -- SYNTHETIC placeholder data, "
                  "pending isolated n2 runs + iTransformer training", fontsize=8.8, color="#B33", y=0.995)
    return fig


if __name__ == "__main__":
    print("[fig7] WARNING: all 'Predicted' surfaces/heatmap values are synthetic "
          "placeholders (see module docstring) -- style/layout draft only, not a result.")
    fig = build_figure()
    out_png = "docs/figures/fig11_robustness_3d.png"
    fig.savefig(out_png, dpi=400)
    print(f"saved: {out_png}")
