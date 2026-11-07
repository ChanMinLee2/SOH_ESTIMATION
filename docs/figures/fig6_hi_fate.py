"""Figure 6 — HI fate across the three dataset models (2026-11-04).

The figure follows every raw HI through the scenario-wise pipeline of Section 2.4 and the
trained gates of Section 2.5 for the MIT, HUST and TJU models, so that one picture shows
(i) that each scenario ends with its own subset and (ii) that the subsets differ between
the two LFP datasets and the NCM dataset.

(a) Fate matrix. Rows are raw HIs (grouped by category), columns are 3 datasets x 6 scenarios.
    A thin strip left of each dataset block marks the stage-1 test (scenario-specific vs shared).
    Cell: stage-5 combined redundancy removal (x), otherwise the effective open probability of
    the raw HI in that scenario, i.e. the probabilistic union of the direction-wise gate and the
    scenario gate (Section 2.5). A dot marks P >= tau_sel = 0.9 (= HI_sel^(s)). A small gray
    corner tick marks HIs collapsed as attached in stage 2 (they stay raw inputs, but do not seed
    or join synergy groups).
(b) Size of HI_sel^(s) per column by category, with selected kernel HIs stacked on top.
(c) Jaccard similarity between the 18 selected raw-HI subsets.

Data: <run>/gates/{regression_HIs, classification_HIs, regression_kernel_HIs}.json,
      <run>/synergy_groups_*.json, <run>/*_combined_redundancy.json,
      <run>/hi_scenario_interaction_*.json
Run:  python docs/figures/fig6_hi_fate.py
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs"
OUT = Path(__file__).resolve().parent / "fig6_hi_fate"

RUNS = [("MIT", "1031_0117"), ("HUST", "1031_0431"), ("TJU", "1031_0938")]
SCEN = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]
SCEN_LABEL = ["c-lo", "c-mid", "c-hi", "d-hi", "d-mid", "d-lo"]
TAU = 0.9
CATS = ["stat", "diff", "lfp", "morph"]
CAT_LABEL = {"stat": "stat", "diff": "diff", "lfp": "LFP", "morph": "morph"}
CAT_COLOR = {"stat": "#1f77b4", "diff": "#ff7f0e", "lfp": "#2ca02c", "morph": "#9467bd"}
KERNEL_COLOR = "#555555"
SPEC_COLOR, SHARED_COLOR = "#c0392b", "#d9d9d9"
DATASET_COLOR = {"MIT": "#3a6ea5", "HUST": "#5b9bd5", "TJU": "#c0504d"}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.6, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9, "legend.fontsize": 6.9, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.3, "ytick.major.size": 2.3,
    "pdf.fonttype": 42,
})


def _base(name: str) -> str:
    return re.sub(r"_(chg|dis)_(lo|mid|hi)$", "", name)


def _cat(name: str) -> str:
    return name.split("_")[0]


def _load(run_id: str) -> dict:
    d = Path(glob.glob(str(RUNS_DIR / f"{run_id}_*"))[0])
    rd = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    reg = rd(d / "gates/regression_HIs.json")
    cls = rd(d / "gates/classification_HIs.json")
    ker = rd(d / "gates/regression_kernel_HIs.json")
    syn = rd(glob.glob(str(d / "synergy_groups_*.json"))[0])
    red = rd(glob.glob(str(d / "*_combined_redundancy.json"))[0])["by_scenario"]
    inter = rd(glob.glob(str(d / "hi_scenario_interaction_*.json"))[0])["per_hi"]

    p_dir = {"chg": {_base(n): p for n, p in zip(cls["charge_names"], cls["charge_probs"])},
             "dis": {_base(n): p for n, p in zip(cls["discharge_names"], cls["discharge_probs"])}}
    out = {"p": {}, "p_scen": {}, "p_dir": p_dir, "removed": {}, "attached": {}, "ker_sel": {},
           "ker_tot": {}, "specific": {h: bool(v["significant"]) for h, v in inter.items()}}
    for k in range(6):
        s = reg[f"seg_{k}_seg_name"]
        p_scen = {_base(n): p for n, p in zip(reg[f"seg_{k}_names"], reg[f"seg_{k}_probs"])}
        pd_ = p_dir[s[:3]]
        out["p_scen"][s] = p_scen
        out["p"][s] = {h: 1 - (1 - pd_.get(h, 0.0)) * (1 - p_scen[h]) for h in p_scen}
        out["removed"][s] = {_base(n) for n in red[s]["removed_raw_names"]}
        assert syn[f"seg_{k}_seg_name"] == s
        out["attached"][s] = {_base(n) for g in syn[f"seg_{k}_group_attached_names"] for n in g}
        kp = ker[f"seg_{k}_probs"]
        assert ker[f"seg_{k}_seg_name"] == s
        out["ker_sel"][s] = int(sum(q >= TAU for q in kp))
        out["ker_tot"][s] = len(kp)
    out["sel"] = {s: {h for h, q in out["p"][s].items() if q >= TAU and h not in out["removed"][s]}
                  for s in SCEN}
    return out


def build():
    data = {name: _load(rid) for name, rid in RUNS}
    all_his = list(next(iter(data.values()))["p"]["chg_lo"].keys())
    ever = [h for h in all_his if any(h in data[n]["sel"][s] for n, _ in RUNS for s in SCEN)]
    rows = sorted(ever, key=lambda h: (CATS.index(_cat(h)),
                                       -sum(h in data[n]["sel"][s] for n, _ in RUNS for s in SCEN), h))
    cols = [(n, s) for n, _ in RUNS for s in SCEN]

    # ---------------------------------------------------------------- layout
    W, H = 174 / 25.4, 140 / 25.4   # 2026-11-05: height 168 -> 140 mm, panel (a) about 80%
    fig = plt.figure(figsize=(W, H))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.06],
                  left=0.15, right=0.93, top=0.925, bottom=0.135, wspace=0.34, hspace=0.55)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------------- (a) fate matrix
    cmap = LinearSegmentedColormap.from_list("gate", ["#f7fbff", "#c6dbef", "#6baed6", "#08519c"])
    cmap_dir = LinearSegmentedColormap.from_list("dgate", ["#fff8f0", "#fdd0a2", "#fd8d3c", "#a63603"])
    # one direction-gate column in front of the three scenario columns of each direction
    SLOT = {"chg": 0, "chg_lo": 1, "chg_mid": 2, "chg_hi": 3,
            "dis": 4, "dis_hi": 5, "dis_mid": 6, "dis_lo": 7}
    SLOT_LABEL = ["chg gate", "c-lo", "c-mid", "c-hi", "dis gate", "d-hi", "d-mid", "d-lo"]
    NSLOT = len(SLOT_LABEL)
    strip_w, gap = 0.55, 0.45
    x0 = {}
    x = 0.0
    for n, _ in RUNS:
        x0[n] = x
        x += strip_w + NSLOT + gap
    nrow = len(rows)
    for i, h in enumerate(rows):
        y = i
        for n, _ in RUNS:
            dd = data[n]
            ax_a.add_patch(Rectangle((x0[n], y), strip_w, 1,
                                     color=SPEC_COLOR if dd["specific"].get(h) else SHARED_COLOR, lw=0))
            for d_ in ("chg", "dis"):
                cx = x0[n] + strip_w + SLOT[d_]
                ax_a.add_patch(Rectangle((cx, y), 1, 1, facecolor=cmap_dir(dd["p_dir"][d_].get(h, 0.0)),
                                         edgecolor="none"))
            for s in SCEN:
                cx = x0[n] + strip_w + SLOT[s]
                if h in dd["removed"][s]:
                    ax_a.add_patch(Rectangle((cx, y), 1, 1, facecolor="white", edgecolor="none"))
                    ax_a.plot([cx + .25, cx + .75], [y + .25, y + .75], color="#999999", lw=0.6)
                    ax_a.plot([cx + .25, cx + .75], [y + .75, y + .25], color="#999999", lw=0.6)
                else:
                    q = dd["p_scen"][s][h]
                    ax_a.add_patch(Rectangle((cx, y), 1, 1, facecolor=cmap(q), edgecolor="none"))
                    if dd["p"][s][h] >= TAU:
                        if q >= TAU:
                            ax_a.plot(cx + .5, y + .5, "o", ms=1.9, color="white", mec="#08306b", mew=0.4)
                        else:
                            ax_a.plot(cx + .5, y + .5, "o", ms=2.1, color="#fd8d3c", mec="#a63603", mew=0.4)
                if h in dd["attached"][s]:
                    ax_a.add_patch(plt.Polygon([[cx + 1, y], [cx + 0.62, y], [cx + 1, y + 0.38]],
                                               color="#7f7f7f", lw=0))
    # grid lines
    for n, _ in RUNS:
        for j in range(NSLOT + 1):
            ax_a.plot([x0[n] + strip_w + j] * 2, [0, nrow], color="white", lw=0.35)
        ax_a.add_patch(Rectangle((x0[n], 0), strip_w + NSLOT, nrow, fill=False, ec="#333333", lw=0.7))
        ax_a.plot([x0[n] + strip_w + 4] * 2, [0, nrow], color="#333333", lw=0.6)
        for j in (1, 5):
            ax_a.plot([x0[n] + strip_w + j] * 2, [0, nrow], color="#333333", lw=0.3, ls=(0, (2, 1)))
        ax_a.text(x0[n] + strip_w + NSLOT / 2, -0.4, n + (" (NCM)" if n == "TJU" else " (LFP)"),
                  ha="center", va="bottom", fontsize=8, fontweight="bold", color=DATASET_COLOR[n])
        for j, lab in enumerate(SLOT_LABEL):
            is_dir = j in (0, 4)
            ax_a.text(x0[n] + strip_w + j + .5, nrow + 0.35, lab, rotation=90, ha="center", va="top",
                      fontsize=6.4, color="#a63603" if is_dir else "black",
                      fontweight="bold" if is_dir else "normal")
    for i in range(nrow + 1):
        for n, _ in RUNS:
            ax_a.plot([x0[n] + strip_w, x0[n] + strip_w + NSLOT], [i, i], color="white", lw=0.35)
    # category side strip + row labels
    xl = -0.25
    for i, h in enumerate(rows):
        c = _cat(h)
        ax_a.add_patch(Rectangle((xl - 0.55, i), 0.5, 1, color=CAT_COLOR[c], lw=0, clip_on=False))
        ax_a.text(xl - 0.75, i + 0.5, h, ha="right", va="center", fontsize=6.4, color=CAT_COLOR[c])
    # category separators
    prev = None
    for i, h in enumerate(rows):
        c = _cat(h)
        if prev is not None and c != prev:
            for n, _ in RUNS:
                ax_a.plot([x0[n], x0[n] + strip_w + NSLOT], [i, i], color="#333333", lw=0.6)
        prev = c
    ax_a.set_xlim(xl - 0.6, x - gap + 0.05)
    ax_a.set_ylim(nrow + 3.3, -2.0)
    ax_a.axis("off")
    ax_a.set_title(f"(a) Fate of each raw HI ({nrow} of 63 selected at least once)", loc="left",
                   fontweight="bold", x=-0.30, pad=4)
    # legend for (a)
    handles = [Patch(color=SPEC_COLOR, label="Scenario-specific (stage 1)"),
               Patch(color=SHARED_COLOR, label="Shared (stage 1)"),
               Line2D([], [], marker="x", ls="", color="#999999", ms=3.5, mew=0.7,
                      label="Removed in stage 5"),
               Patch(color="#7f7f7f", label="Attached in stage 2"),
               Line2D([], [], marker="o", ls="", color="white", mec="#08306b", mew=0.5, ms=3,
                      label="Selected by scenario gate"),
               Line2D([], [], marker="o", ls="", color="#fd8d3c", mec="#a63603", mew=0.5, ms=3,
                      label="Selected by direction gate only")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.005), ncol=2,
               frameon=False, handlelength=1.0, columnspacing=0.8, handletextpad=0.4, fontsize=6.4)
    for xc, cm_, ttl in ((0.50, cmap, "Scenario gate $P(z_j \\neq 0)$"),
                         (0.70, cmap_dir, "Direction gate $P(z_j \\neq 0)$")):
        cax = fig.add_axes([xc, 0.04, 0.12, 0.011])
        cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=cm_, norm=mpl.colors.Normalize(0, 1)), cax=cax,
                          orientation="horizontal")
        cb.set_ticks([0, 0.5, 0.9])
        cb.ax.tick_params(labelsize=6.4, length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)
        cax.set_title(ttl, fontsize=6.4, pad=2)

    # ---------------------------------------------------------------- (b) composition
    xb = []
    pos = 0.0
    for n, _ in RUNS:
        for j in range(6):
            xb.append(pos + j)
        pos += 6.8
    xb = np.array(xb)
    bottoms = np.zeros(len(cols))
    for c in CATS:
        vals = np.array([sum(_cat(h) == c for h in data[n]["sel"][s]) for n, s in cols])
        ax_b.bar(xb, vals, bottom=bottoms, width=0.82, color=CAT_COLOR[c], label=CAT_LABEL[c], lw=0)
        bottoms += vals
    kv = np.array([data[n]["ker_sel"][s] for n, s in cols])
    ax_b.bar(xb, kv, bottom=bottoms, width=0.82, color=KERNEL_COLOR, label="kernel", lw=0)
    for xx, b, k, (n, s) in zip(xb, bottoms, kv, cols):
        ax_b.text(xx, b + k + 0.3, f"{int(b + k)}", ha="center", va="bottom", fontsize=6.4)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(SCEN_LABEL * 3, rotation=90, fontsize=6.4)
    for n, _ in RUNS:
        i0 = [i for i, (nn, _) in enumerate(cols) if nn == n]
        ax_b.text(xb[i0].mean(), 1.0, n, transform=ax_b.get_xaxis_transform(), ha="center",
                  va="bottom", fontsize=7.5, fontweight="bold", color=DATASET_COLOR[n])
    ax_b.set_ylabel(r"HIs in $\mathrm{HI}_\mathrm{sel}^{(s)}$")
    ax_b.set_ylim(0, max(bottoms + kv) * 1.22)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=0.9,
                columnspacing=0.6, handletextpad=0.3, fontsize=6.4)
    ax_b.set_title(r"(b) Size of $\mathrm{HI}_\mathrm{sel}^{(s)}$ by category", loc="left",
                   fontweight="bold", pad=10)

    # ---------------------------------------------------------------- (c) Jaccard
    sets = [data[n]["sel"][s] for n, s in cols]
    J = np.array([[len(a & b) / len(a | b) if (a | b) else np.nan for b in sets] for a in sets])
    im = ax_c.imshow(J, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    for k in (6, 12):
        ax_c.axhline(k - .5, color="white", lw=1.2)
        ax_c.axvline(k - .5, color="white", lw=1.2)
    ax_c.set_xticks(range(18))
    ax_c.set_xticklabels(SCEN_LABEL * 3, rotation=90, fontsize=6.4)
    ax_c.set_yticks(range(18))
    ax_c.set_yticklabels(SCEN_LABEL * 3, fontsize=6.4)
    for k, (n, _) in enumerate(RUNS):
        ax_c.text(6 * k + 2.5, -0.9, n, ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                  color=DATASET_COLOR[n])
        ax_c.text(-3.6, 6 * k + 2.5, n, ha="right", va="center", fontsize=7.5, fontweight="bold",
                  color=DATASET_COLOR[n], rotation=90)
    ax_c.tick_params(length=1.2, pad=1)
    cb2 = fig.colorbar(im, ax=ax_c, fraction=0.045, pad=0.03)
    cb2.ax.tick_params(labelsize=6.4, length=1.5)
    cb2.set_label("Jaccard similarity", fontsize=6.7)
    cb2.outline.set_linewidth(0.4)
    ax_c.set_title("(c) Overlap of the raw subsets", loc="left", fontweight="bold", pad=16,
                   x=-0.30)

    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    return data, rows, cols, J


def report(data, rows, cols, J):
    print("rows shown", len(rows))
    for n, _ in RUNS:
        dd = data[n]
        ever = set().union(*dd["sel"].values())
        allsix = set.intersection(*[dd["sel"][s] for s in SCEN])
        sizes = [len(dd["sel"][s]) for s in SCEN]
        lfp = {s: sum(_cat(h) == "lfp" for h in dd["sel"][s]) for s in SCEN}
        catd = {d_: {c: sum(sum(_cat(h) == c for h in dd["sel"][s]) for s in SCEN if s.startswith(d_))
                     for c in CATS} for d_ in ("chg", "dis")}
        print(f"{n}: ever {len(ever)}, all six {len(allsix)} {sorted(allsix)}, per scen {sizes}")
        print(f"   LFP-specific per scen {lfp}; category sums by direction {catd}")
        print(f"   kernel sel {dd['ker_sel']} of {dd['ker_tot']}; specific(stage1) {sum(dd['specific'].values())}")
        und = {s: sum(0.1 < q < 0.9 for h, q in dd['p'][s].items() if h not in dd['removed'][s]) for s in SCEN}
        print(f"   undecided union probs per scen {und}")
    idx = {n: [i for i, (nn, _) in enumerate(cols) if nn == n] for n, _ in RUNS}

    def mean_block(a, b):
        v = [J[i, j] for i in a for j in b if i != j]
        return np.nanmean(v)
    for n, _ in RUNS:
        print(f"Jaccard within {n}: {mean_block(idx[n], idx[n]):.3f}")
    print(f"Jaccard MIT-HUST: {mean_block(idx['MIT'], idx['HUST']):.3f}")
    print(f"Jaccard LFP-TJU: {mean_block(idx['MIT'] + idx['HUST'], idx['TJU']):.3f}")
    # same scenario across datasets
    same = lambda a, b: np.mean([J[idx[a][k], idx[b][k]] for k in range(6)])
    print(f"same-scenario MIT-HUST {same('MIT','HUST'):.3f}, MIT-TJU {same('MIT','TJU'):.3f}, HUST-TJU {same('HUST','TJU'):.3f}")
    # within dataset same direction vs cross direction
    for n, _ in RUNS:
        ch, di = idx[n][:3], idx[n][3:]
        print(f"   {n}: same-direction {np.nanmean([mean_block(ch, ch), mean_block(di, di)]):.3f}, cross-direction {mean_block(ch, di):.3f}")


if __name__ == "__main__":
    report(*build())
    print("saved", OUT.with_suffix(".png"))
