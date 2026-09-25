"""Figure 12 -- Ablation vs. the final v4 model (H23-style contribution check).

Four conditions, all oracle test-set, MIT+HUST pooled, zone tiling:
  no scen       -- assign="none" (no scenario label), kernel ON      (noscen_kernel_zonetile)
  v4 (w/o group)-- scenario label ON, kernel ON, but interaction_json=""
                   so raw HI never uses the shared/specific hybrid split   (label_kernel_noshare)
  v4 (w/o kernel)-- scenario label ON, kernel OFF                          (rawonly)
                   NOTE: this run also lacks the hybrid group split (interaction_json=null),
                   so it is not a pure "minus kernel only" cell -- it removes kernel AND
                   group simultaneously. No clean "v4 minus only kernel" run exists yet
                   (would need interaction_json set + kernel off, not yet trained).
  v4            -- scenario label ON, kernel ON, interaction_json set (hybrid group)  (canonical)

(a) per-cell RMSE distribution (boxplot + jitter)
(b) per-cell MAPE distribution (boxplot + jitter)

Real data: predictions/test_predictions.csv (oracle) from each run listed above.

Run: python docs/figures/fig8_ablation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _style import INK, SUBINK, GRID, setup_rcparams, label_panel, PROJECT_ROOT

setup_rcparams()

RUNS_DIR = PROJECT_ROOT / "legacy_results/experiments/phase1_lab/results/p1v2_runs"

CONDITIONS = [
    ("no scen", RUNS_DIR / "0911_0541_p1v2_p1v4_noscen_kernel_zonetile_seed42", "#3A6178"),
    ("v4 (w/o group)", RUNS_DIR / "0911_0026_p1v2_p1v4_label_kernel_noshare_seed42", "#A9863F"),
    ("v4 (w/o kernel)*", RUNS_DIR / "0909_1318_p1v2_p1v4_rawonly_seed42_seed42", "#96473A"),
    ("v4", RUNS_DIR / "0904_1708_p1v2_p1v4_minpts5_calib100_offA5mA_seed42", "#182E45"),
]


def per_cell_metrics(run_dir):
    df = pd.read_csv(run_dir / "predictions" / "test_predictions.csv")
    rows = []
    for cell, g in df.groupby("cell_id"):
        err = g.cap_pred_Ah - g.cap_true_Ah
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mape = float(np.mean(np.abs(err) / g.cap_true_Ah) * 100)
        rows.append((cell, rmse, mape))
    return pd.DataFrame(rows, columns=["cell_id", "rmse", "mape"])


def panel_box(ax, data_by_cond, metric_key, ylabel, title, fmt="{:.4f}"):
    labels = [c[0] for c in CONDITIONS]
    colors = [c[2] for c in CONDITIONS]
    vals = [data_by_cond[label][metric_key].values for label in labels]

    bp = ax.boxplot(vals, positions=range(len(labels)), widths=0.5, patch_artist=True,
                     showfliers=True, flierprops=dict(marker="o", ms=2.5, alpha=0.35,
                                                       markerfacecolor=SUBINK, markeredgewidth=0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor(INK)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(1.6)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set_color(SUBINK)

    rng = np.random.default_rng(11)
    for i, v in enumerate(vals):
        jitter = rng.normal(0, 0.045, size=len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=6, color=colors[i], alpha=0.45,
                   linewidths=0, zorder=1)
        med = np.median(v)
        ax.text(i, med, f"  {fmt.format(med)}", fontsize=6.2, color=SUBINK,
                va="bottom", ha="center")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.2, rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=6, fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)


def build_figure():
    data_by_cond = {label: per_cell_metrics(d) for label, d, _ in CONDITIONS}

    fig = plt.figure(figsize=(7.6, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.36, top=0.87, bottom=0.22, left=0.10, right=0.975)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_box(ax_a, data_by_cond, "rmse", "Per-cell RMSE (Ah)", "Per-cell RMSE vs. v4")
    panel_box(ax_b, data_by_cond, "mape", "Per-cell MAPE (%)", "Per-cell MAPE vs. v4",
              fmt="{:.2f}")

    fig.text(0.5, 0.015, "*rawonly also lacks the raw-HI shared/specific group split "
             "(interaction_json unset), so it removes kernel AND group at once, not kernel alone.",
             ha="center", va="bottom", fontsize=6.2, color=SUBINK, style="italic")

    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        label_panel(fig, ax, letter, dx=-0.06, dy=0.014)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig12_ablation.png"
    out_pdf = "docs/figures/fig12_ablation.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}\nsaved: {out_pdf}")
