"""Figure 9 -- cross-chemistry gate
selection Jaccard similarity, raw HI and kernel HI.

The abstract cites this as evidence for the interpretability claim ("the gate
genuinely selects different subsets per chemistry", Jaccard 0.12-0.43) and
Discussion Sec. "Industrial problems addressed" references it directly, but
no figure existed anywhere in the repo -- computed fresh here from the same
three independently-trained single-chemistry runs Fig5 already uses (no new
training).

Panel (a) raw HI: gates/regression_HIs.json gives, per scenario, a
gate_prob-ranked list over the same 64-name raw-HI catalogue (confirmed
identical across MIT/HUST/TJU) -- top scen_k_count(=25, config.yaml
regression.scen_k_count, confirmed identical across all three runs) per
scenario as each chemistry's "selected set". A fixed-size top-k, not a
probability threshold: these single-chemistry runs train on far less data
than the pooled MIT+HUST canonical run and their gates saturate much less
sharply, so a fixed prob>=0.9 cut collapses to near-empty, degenerate sets
(confirmed empirically -- e.g. TJU selects 0-1/64 raw HI at that threshold).

Panel (b) kernel HI: gates/regression_kernel_HIs.json's own names
("kernel_chg_lo_g4") are NOT comparable across chemistries -- the group
index g4 comes from an independent partial-correlation clustering run per
chemistry, so "g4" means a different feature in MIT vs. TJU (confirmed: 81
/85/75 kernel features per chemistry, no shared indexing). What IS
comparable is each kernel's member_names (the raw HI it was built from, in
the same common 64-name vocabulary) -- kernel_group_features_*.pkl, same
schema plot_hi_selection_matrix.py already reads. So panel (b) compares each
chemistry's top-25 selected kernels' *raw-HI membership footprint* (the
union of member raw HI, scenario-suffix stripped) rather than kernel
identity -- "do the two chemistries' kernel gates end up drawing on the same
underlying raw signal, even via different kernel groupings".

Jaccard = |A∩B| / |A∪B| per pair, per scenario, for both panels.

Panel (c) Kendall's tau (raw HI): Jaccard only looks at whether a HI clears
an arbitrary top-25 cutoff (binary in/out); tau uses the FULL gate_prob
ranking over all 64 HI, so it doesn't depend on that cutoff and comes with a
p-value for free. Computed and checked before adding (not assumed useful) --
it tells a consistent but non-redundant story: MIT-HUST tau=0.41-0.56 (all
p<0.001), cross-chemistry pairs tau=0.11-0.25 (mostly p<0.05, a few
borderline ~0.07-0.19) -- confirms the Jaccard finding via an independent
method, and adds a significance readout Jaccard alone doesn't have. Cells
with p>=0.05 are marked (not significant at this n=64).

Run: python docs/figures/fig10_jaccard_cross_chemistry.py   (PNG only, no
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
TOP_K = 25   # regression.scen_k_count, confirmed identical across all three runs' config.yaml

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_rust", ["#F4F1E8", "#8A3F2E"])


def raw_selected_sets(run_dir):
    """{scenario: set(base raw-HI name)} -- top TOP_K by gate_prob.
    seg_s_names/seg_s_probs are already gate_prob-descending (verified), so
    the first TOP_K entries of seg_s_names are exactly the top-k selection."""
    d = json.loads((run_dir / "gates" / "regression_HIs.json").read_text(encoding="utf-8"))
    out = {}
    for s, sname in enumerate(SCEN_NAMES):
        top_names = d[f"seg_{s}_names"][:TOP_K]
        out[sname] = {n.replace(f"_{sname}", "") for n in top_names}
    return out


def kernel_footprint_sets(run_dir, kernel_pkl):
    """{scenario: set(base raw-HI name)} -- union of member_names (scenario
    suffix stripped) across the top TOP_K selected kernels, i.e. each
    chemistry's kernel-gate "raw signal footprint" in the common vocabulary."""
    d = json.loads((run_dir / "gates" / "regression_kernel_HIs.json").read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(kernel_pkl, "rb") as f:
            artifact = pickle.load(f)
    members = {feat["name"]: feat["member_names"] for feat in artifact["features"]}

    out = {}
    for s, sname in enumerate(SCEN_NAMES):
        top_kernels = d[f"seg_{s}_names"][:TOP_K]
        footprint = set()
        for kname in top_kernels:
            base_kname = kname.replace(f"_{sname}", "")
            for m in members.get(base_kname, []):
                footprint.add(m.replace(f"_{sname}", ""))
        out[sname] = footprint
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pair_matrix(sets_by_chem):
    mat = np.zeros((len(SCEN_NAMES), len(PAIRS)))
    for j, (c1, c2) in enumerate(PAIRS):
        for i, sname in enumerate(SCEN_NAMES):
            mat[i, j] = jaccard(sets_by_chem[c1][sname], sets_by_chem[c2][sname])
    return mat


def full_rank_probs(run_dir):
    """{scenario: {base raw-HI name: gate_prob}} -- full 64-length ranking,
    not just the top-K, for Kendall's tau."""
    d = json.loads((run_dir / "gates" / "regression_HIs.json").read_text(encoding="utf-8"))
    out = {}
    for s, sname in enumerate(SCEN_NAMES):
        names, probs = d[f"seg_{s}_names"], d[f"seg_{s}_probs"]
        base = [n.replace(f"_{sname}", "") for n in names]
        out[sname] = dict(zip(base, probs))
    return out


def tau_matrix(probs_by_chem):
    tau_mat = np.zeros((len(SCEN_NAMES), len(PAIRS)))
    sig_mat = np.zeros((len(SCEN_NAMES), len(PAIRS)), dtype=bool)
    for j, (c1, c2) in enumerate(PAIRS):
        for i, sname in enumerate(SCEN_NAMES):
            p1, p2 = probs_by_chem[c1][sname], probs_by_chem[c2][sname]
            keys = sorted(p1.keys())
            tau, pval = kendalltau([p1[k] for k in keys], [p2[k] for k in keys])
            tau_mat[i, j] = tau
            sig_mat[i, j] = pval < 0.05
    return tau_mat, sig_mat


def draw_panel(ax, mat, title, cmap=None, sig=None, sig_note=None):
    cmap = cmap or SEQ_CMAP
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=mat.max() * 1.05)
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
            ax.text(j, i, label, ha="center", va="center", fontsize=7.2,
                     color="white" if v > mat.max() * 0.55 else INK)
    ax.set_title(title, loc="left", fontsize=8.6, pad=10)
    if sig_note:
        ax.text(0.0, -0.16, sig_note, transform=ax.transAxes, fontsize=6.0,
                 color=SUBINK, style="italic")
    return im


def build_figure():
    raw_sets = {chem: raw_selected_sets(d) for chem, d in RUNS.items()}
    kernel_sets = {chem: kernel_footprint_sets(d, KERNEL_PKL[chem]) for chem, d in RUNS.items()}
    probs_sets = {chem: full_rank_probs(d) for chem, d in RUNS.items()}
    mat_raw = pair_matrix(raw_sets)
    mat_kernel = pair_matrix(kernel_sets)
    mat_tau, sig_tau = tau_matrix(probs_sets)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.4, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.07, right=0.96, wspace=0.62)

    im_a = draw_panel(ax_a, mat_raw, f"Raw HI -- Jaccard\n(top-{TOP_K} by gate prob)")
    im_b = draw_panel(ax_b, mat_tau, "Raw HI -- Kendall's $\\tau$\n(full 64-HI ranking)",
                        sig=sig_tau, sig_note="$^{ns}$: not significant at p<0.05")
    im_c = draw_panel(ax_c, mat_kernel,
                        f"Kernel HI -- Jaccard, via raw-HI\nmembership footprint (top-{TOP_K} kernels)")

    for im, ax, lbl in ((im_a, ax_a, "Jaccard similarity"), (im_b, ax_b, "Kendall's $\\tau$"),
                        (im_c, ax_c, "Jaccard similarity")):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(lbl, fontsize=7.2, color=SUBINK)
        cbar.ax.tick_params(labelsize=6.2, length=2)
        cbar.outline.set_visible(False)

    fig.suptitle("Cross-chemistry gate selection overlap, per scenario", fontsize=9.2, y=0.97)
    label_panel(fig, ax_a, "(a)", dx=-0.055, dy=0.012)
    label_panel(fig, ax_b, "(b)", dx=-0.055, dy=0.012)
    label_panel(fig, ax_c, "(c)", dx=-0.055, dy=0.012)

    print(f"[fig10] raw Jaccard range: {mat_raw.min():.3f}-{mat_raw.max():.3f}")
    print(f"[fig10] raw Kendall tau range: {mat_tau.min():.3f}-{mat_tau.max():.3f} "
          f"({int(sig_tau.sum())}/{sig_tau.size} significant at p<0.05)")
    print(f"[fig10] kernel-footprint Jaccard range: {mat_kernel.min():.3f}-{mat_kernel.max():.3f}")
    return fig


if __name__ == "__main__":
    fig = build_figure()
    out_png = "docs/figures/fig9_jaccard_cross_chemistry.png"
    fig.savefig(out_png, dpi=600)
    print(f"saved: {out_png}")
