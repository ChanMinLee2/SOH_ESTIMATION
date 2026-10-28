"""Learned-representation embedding figure (draft, candidate new figure for §3).

What the trained model "sees", on the LFP (MIT+HUST) test split under HARD
routing (classifier argmax, the deployment condition):
  (a) routing-classifier latent (probe_mlp penultimate layer), coloured by the
      ground-truth scenario; misrouted segments ringed in black
  (b) regressor latent (cap_head penultimate layer), coloured by scenario
  (c) the same regressor latent, coloured by true SOH

Pipeline (two environments, see docs/figures/_extract_embedding.py):
  1. project env  : _extract_embedding.py -> _cache/embed_lfp.npz  (forward hooks)
  2. figure env   : this script -> UMAP (cached to _cache/umap_lfp_fig.npz) -> PNG/PDF

UMAP uses large n_neighbors (200) so global structure (how scenarios relate,
where low-SOH segments sit) is preserved rather than over-fragmenting each
scenario into per-cell islands. Embedding axes carry no units -- only relative
position is meaningful -- so axes are drawn as a small corner arrow pair, the
convention in Nature-family latent-space figures (e.g. iMOE, Nat. Commun. 2026).

NCM is not shown: the only TJU checkpoint is a pre-refactor run whose kernel
gate layout no longer matches SCRModel (load_state_dict shape mismatch), so a
TJU panel needs TJU retrained on the current pipeline first.

Run (figure env): python docs/figures/fig_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cmocean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from _style import INK, SUBINK, Z_LEFT, Z_MID, Z_RIGHT, setup_rcparams, label_panel  # noqa: E402

setup_rcparams(base_size=7.6)
CACHE = HERE / "_cache"


def _tint(hex_color: str, f: float) -> tuple:
    c = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(c + (1 - c) * f)


# position triad (lo/mid/hi) shared with every scenario figure; discharge = lighter tint
SCEN_COLOR = [Z_LEFT, Z_MID, Z_RIGHT, _tint(Z_LEFT, 0.42), _tint(Z_MID, 0.42), _tint(Z_RIGHT, 0.42)]
SCEN_LABEL = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]


def _umap(d) -> dict:
    path = CACHE / "umap_lfp_fig.npz"
    if path.exists():
        return dict(np.load(path))
    import umap
    out = {}
    for key in ("clf_h", "reg_h"):
        X = d[key].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        out[key] = umap.UMAP(n_components=2, n_neighbors=200, min_dist=0.6,
                             random_state=7).fit_transform(X).astype(np.float32)
    np.savez_compressed(path, **out)
    return out


def _frame(ax, e):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    pad = 0.04 * (e.max(0) - e.min(0))
    ax.set_xlim(e[:, 0].min() - pad[0], e[:, 0].max() + pad[0])
    ax.set_ylim(e[:, 1].min() - pad[1], e[:, 1].max() + pad[1])
    ax.set_aspect("equal", adjustable="datalim")
    # corner arrow pair, Nature-style unitless embedding axes
    x0, y0 = 0.0, 0.0
    kw = dict(xycoords="axes fraction", textcoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", lw=0.7, color=SUBINK, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(x0 + 0.13, y0), xytext=(x0, y0), **kw)
    ax.annotate("", xy=(x0, y0 + 0.13), xytext=(x0, y0), **kw)
    ax.text(x0 + 0.015, y0 - 0.035, "UMAP 1", transform=ax.transAxes, fontsize=5.8, color=SUBINK, va="top")
    ax.text(x0 - 0.03, y0 + 0.015, "UMAP 2", transform=ax.transAxes, fontsize=5.8, color=SUBINK,
            rotation=90, ha="right")


def _scatter_scen(ax, e, scen, order_seed=3):
    rng = np.random.default_rng(order_seed)
    o = rng.permutation(len(e))  # random draw order so no scenario systematically hides another
    ax.scatter(e[o, 0], e[o, 1], s=0.9, c=[SCEN_COLOR[i] for i in scen[o]],
               linewidths=0, alpha=0.85, rasterized=True)


def build_figure():
    d = np.load(CACHE / "embed_lfp.npz", allow_pickle=True)
    u = _umap(d)
    scen = d["scen_true"].astype(int)
    soh = d["soh_true"]
    wrong = d["scen_pred"] != d["scen_true"]

    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, left=0.03, right=0.90, bottom=0.17, top=0.88, wspace=0.12)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) classifier latent
    e = u["clf_h"]
    _scatter_scen(ax_a, e, scen)
    if wrong.any():
        ax_a.scatter(e[wrong, 0], e[wrong, 1], s=9, facecolors="none", edgecolors=INK, linewidths=0.6, zorder=5)
    _frame(ax_a, e)
    ax_a.set_title("(a)  Routing classifier latent", loc="left", fontsize=7.8, pad=4)
    ax_a.text(0.99, 0.02, f"misrouted {int(wrong.sum())}/{len(wrong):,}\n(hard-routing acc. {float(d['acc'])*100:.2f}%)",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5.6, color=SUBINK)

    # (b) regressor latent by scenario
    e = u["reg_h"]
    _scatter_scen(ax_b, e, scen)
    _frame(ax_b, e)
    ax_b.set_title("(b)  Regressor latent, by scenario", loc="left", fontsize=7.8, pad=4)

    # (c) regressor latent by SOH -- low SOH drawn last so the degraded core stays visible
    o = np.argsort(-soh)
    sc = ax_c.scatter(e[o, 0], e[o, 1], s=0.9, c=soh[o], cmap=cmocean.cm.deep_r,
                      vmin=0.72, vmax=1.0, linewidths=0, alpha=0.9, rasterized=True)
    _frame(ax_c, e)
    ax_c.set_title("(c)  Regressor latent, by SOH", loc="left", fontsize=7.8, pad=4)
    cax = fig.add_axes([0.915, 0.27, 0.010, 0.50])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("True SOH", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_visible(False)

    handles = [Line2D([0], [0], marker="o", ls="", ms=4.2, mfc=SCEN_COLOR[i], mec="none", label=SCEN_LABEL[i])
               for i in range(6)]
    handles.append(Line2D([0], [0], marker="o", ls="", ms=4.6, mfc="none", mec=INK, mew=0.7, label="misrouted"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.44, 0.0),
                     frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=1.1)
    for t in leg.get_texts():
        t.set_color(INK)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    for ext, kw in (("png", dict(dpi=600)), ("pdf", {})):
        p = HERE / f"fig_embedding.{ext}"
        fig.savefig(p, **kw)
        print("saved:", p)
