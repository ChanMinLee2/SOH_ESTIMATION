"""Figure 9 -- cross-chemistry gate
selection similarity (Kendall's tau), raw HI and kernel HI.

The abstract cites this as evidence for the interpretability claim ("the gate
genuinely selects different subsets per chemistry") and Discussion Sec.
"Industrial problems addressed" references it directly, but no figure
existed anywhere in the repo -- computed fresh here from the same three
independently-trained single-chemistry runs Fig5 already uses (no new
training).

2026-09-18: switched BOTH panels from Jaccard(top-K set overlap) to
Kendall's tau(full ranking). Jaccard needed an arbitrary top-K cutoff
(scen_k_count=25) to turn each chemistry's continuous gate_prob into a
binary "selected/not" set -- these single-chemistry runs train on far less
data than the pooled MIT+HUST canonical run and their gates saturate much
less sharply, so a fixed prob>=0.9 cut collapses to near-empty, degenerate
sets (confirmed empirically -- e.g. TJU selects 0-1/64 raw HI at that
threshold), and top-25 was itself an arbitrary stand-in. Kendall's tau
avoids the cutoff entirely (uses the FULL ranking) and comes with a
p-value for free -- strictly more information, no arbitrary parameter,
so there is no remaining reason to keep the Jaccard panels once tau is
computed for both raw and kernel comparisons.

Panel (a) raw HI: gates/regression_HIs.json gives, per scenario, a
gate_prob-ranked list over the same 64-name raw-HI catalogue (confirmed
identical across MIT/HUST/TJU) -- tau computed over the full 64-length
gate_prob vector per chemistry pair per scenario.

Panel (b) kernel HI: gates/regression_kernel_HIs.json's own names
("kernel_chg_lo_g4") are NOT comparable across chemistries -- the group
index g4 comes from an independent partial-correlation clustering run per
chemistry, so "g4" means a different feature in MIT vs. TJU (confirmed: 81
/85/75 kernel features per chemistry, no shared indexing). What IS
comparable is each kernel's member_names (the raw HI it was built from, in
the same common 64-name vocabulary) -- kernel_group_features_*.pkl, same
schema plot_hi_selection_matrix.py already reads. So panel (b) gives each
raw HI in the common 64-name vocabulary a continuous "kernel-footprint
weight" per chemistry: the gate_prob-weighted sum, over ALL of that
chemistry's kernels (full ranking, no top-K cutoff), of the kernels that
include it as a member -- "how strongly does this chemistry's kernel gate
lean on this raw signal, however it got bundled into a kernel". Tau is then
computed over this 64-length weight vector per chemistry pair per scenario,
exactly parallel to panel (a). A raw HI untouched by any kernel for a given
chemistry/scenario legitimately gets weight 0 (not missing data).

Cells with p>=0.05 are marked (not significant at this n=64).

Run: python docs/figures/fig9_jaccard_cross_chemistry.py   (PNG only, no
PDF -- draft not yet placed in the paper).
"""

from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
from scipy.stats import kendalltau
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from _style import INK, SUBINK, setup_rcparams, label_panel, PROJECT_ROOT
from fig3_hi_design_rationale import base_names_and_categories

setup_rcparams()

RUNS_DIR = PROJECT_ROOT / "5_model/experiments/phase1_lab/results/p1v2_runs"
RUNS = {
    "MIT":  RUNS_DIR / "0908_0100_p1v2_p1v4_mit_only_seed42",
    "HUST": RUNS_DIR / "0908_0358_p1v2_p1v4_hust_only_seed42",
    "TJU":  RUNS_DIR / "0910_1533_p1v2_p1v4_tju_only_seed42",
}
KERNEL_PKL = {
    "MIT":  PROJECT_ROOT / "5_model/experiments/phase1_lab/results/kernel_group_features_mit_only.pkl",
    "HUST": PROJECT_ROOT / "5_model/experiments/phase1_lab/results/kernel_group_features_hust_only.pkl",
    "TJU":  PROJECT_ROOT / "5_model/experiments/phase1_lab/results/kernel_group_features_tju_only.pkl",
}
PAIRS = [("MIT", "HUST"), ("MIT", "TJU"), ("HUST", "TJU")]
SCEN_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
RAW_HI_NAMES, _ = base_names_and_categories()   # 64-name catalogue, identical across chemistries

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_rust", ["#F4F1E8", "#8A3F2E"])
DIVERGE_CMAP = LinearSegmentedColormap.from_list("div_rust_blue", ["#3D6E8C", "#F4F1E8", "#8A3F2E"])


def full_rank_probs(run_dir):
    """{scenario: {base raw-HI name: gate_prob}} -- full 64-length ranking, for Kendall's tau."""
    d = json.loads((run_dir / "gates" / "regression_HIs.json").read_text(encoding="utf-8"))
    out = {}
    for s, sname in enumerate(SCEN_NAMES):
        names, probs = d[f"seg_{s}_names"], d[f"seg_{s}_probs"]
        base = [n.replace(f"_{sname}", "") for n in names]
        out[sname] = dict(zip(base, probs))
    return out


def kernel_footprint_weights(run_dir, kernel_pkl):
    """{scenario: {base raw-HI name: weight}} -- gate_prob-weighted sum, over ALL
    of this chemistry's kernels (full ranking, no top-K cutoff), of the kernels
    that include that raw HI as a member. Raw HI never touched by any kernel for
    this scenario legitimately get weight 0.0 (not missing)."""
    d = json.loads((run_dir / "gates" / "regression_kernel_HIs.json").read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(kernel_pkl, "rb") as f:
            artifact = pickle.load(f)
    members = {feat["name"]: feat["member_names"] for feat in artifact["features"]}

    out = {}
    for s, sname in enumerate(SCEN_NAMES):
        names, probs = d[f"seg_{s}_names"], d[f"seg_{s}_probs"]
        weight = {n: 0.0 for n in RAW_HI_NAMES}
        for kname, prob in zip(names, probs):
            # members는 feat["name"](이미 시나리오 접미사 포함, 예: "kernel_chg_lo_g0")로
            # 키가 잡혀 있다 -- kname도 동일 형식이라 접미사를 벗기면 안 됨(예전
            # kernel_footprint_sets()의 base_kname 스트립은 조회를 항상 실패시키는 버그였음,
            # 이번에 nan/전부-0 결과로 발견해 수정).
            for m in members.get(kname, []):
                base_m = m.replace(f"_{sname}", "")
                weight[base_m] = weight.get(base_m, 0.0) + prob
        out[sname] = weight
    return out


def tau_matrix(scores_by_chem):
    """scores_by_chem[chem][scenario] = {base raw-HI name: score} over the fixed
    RAW_HI_NAMES catalogue (missing keys treated as 0.0) -- works identically for
    panel (a)'s gate_prob and panel (b)'s kernel-footprint weight."""
    tau_mat = np.zeros((len(SCEN_NAMES), len(PAIRS)))
    sig_mat = np.zeros((len(SCEN_NAMES), len(PAIRS)), dtype=bool)
    for j, (c1, c2) in enumerate(PAIRS):
        for i, sname in enumerate(SCEN_NAMES):
            p1, p2 = scores_by_chem[c1][sname], scores_by_chem[c2][sname]
            v1 = [p1.get(k, 0.0) for k in RAW_HI_NAMES]
            v2 = [p2.get(k, 0.0) for k in RAW_HI_NAMES]
            tau, pval = kendalltau(v1, v2)
            tau_mat[i, j] = tau
            sig_mat[i, j] = pval < 0.05
    return tau_mat, sig_mat


def draw_panel(ax, mat, title, cmap=None, sig=None, sig_note=None):
    # tau가 음수를 포함하면(예: 커널 footprint 패널) 0을 축으로 하는 발산형 컬러맵으로,
    # 전부 양수면(예: raw HI 패널, 관측 범위 0.11~0.56) 기존처럼 0-시작 순차형으로 그린다.
    has_negative = mat.min() < 0
    if has_negative:
        cmap = cmap or DIVERGE_CMAP
        vlim = max(abs(mat.min()), abs(mat.max())) * 1.05
        vmin, vmax = -vlim, vlim
    else:
        cmap = cmap or SEQ_CMAP
        vmin, vmax = 0.0, mat.max() * 1.05
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels([f"{a}–{b}" for a, b in PAIRS], fontsize=8.2)
    ax.set_yticks(range(len(SCEN_NAMES)))
    ax.set_yticklabels(SCEN_NAMES, fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.axhline(2.5, color="white", lw=1.6)
    for i in range(len(SCEN_NAMES)):
        for j in range(len(PAIRS)):
            v = mat[i, j]
            label = f"{v:.2f}"
            if sig is not None and not sig[i, j]:
                label += "$^{ns}$"
            frac = abs(v - vmin) / max(vmax - vmin, 1e-9)
            ax.text(j, i, label, ha="center", va="center", fontsize=7.2,
                     color="white" if frac > 0.55 else INK)
    ax.set_title(title, loc="left", fontsize=8.6, pad=10)
    if sig_note:
        ax.text(0.0, -0.16, sig_note, transform=ax.transAxes, fontsize=6.0,
                 color=SUBINK, style="italic")
    return im


def build_figure():
    probs_raw = {chem: full_rank_probs(d) for chem, d in RUNS.items()}
    probs_kernel = {chem: kernel_footprint_weights(d, KERNEL_PKL[chem]) for chem, d in RUNS.items()}
    mat_raw_tau, sig_raw = tau_matrix(probs_raw)
    mat_kernel_tau, sig_kernel = tau_matrix(probs_kernel)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.09, right=0.94, wspace=0.55)

    im_a = draw_panel(ax_a, mat_raw_tau, "Raw HI -- Kendall's $\\tau$\n(full 64-HI ranking)",
                        sig=sig_raw, sig_note="$^{ns}$: not significant at p<0.05")
    im_b = draw_panel(ax_b, mat_kernel_tau,
                        "Kernel HI -- Kendall's $\\tau$, via raw-HI\nmembership-weighted footprint (full ranking)",
                        sig=sig_kernel, sig_note="$^{ns}$: not significant at p<0.05")

    for im, ax in ((im_a, ax_a), (im_b, ax_b)):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Kendall's $\\tau$", fontsize=7.2, color=SUBINK)
        cbar.ax.tick_params(labelsize=6.2, length=2)
        cbar.outline.set_visible(False)

    fig.suptitle("Cross-chemistry gate selection similarity, per scenario", fontsize=9.2, y=0.97)
    label_panel(fig, ax_a, "(a)", dx=-0.065, dy=0.012)
    label_panel(fig, ax_b, "(b)", dx=-0.065, dy=0.012)

    print(f"[fig9] raw Kendall tau range: {mat_raw_tau.min():.3f}-{mat_raw_tau.max():.3f} "
          f"({int(sig_raw.sum())}/{sig_raw.size} significant at p<0.05)")
    print(f"[fig9] kernel-footprint Kendall tau range: {mat_kernel_tau.min():.3f}-{mat_kernel_tau.max():.3f} "
          f"({int(sig_kernel.sum())}/{sig_kernel.size} significant at p<0.05)")
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig9_jaccard_cross_chemistry.png"
    fig.savefig(out_png, dpi=600)
    print(f"saved: {out_png}")
