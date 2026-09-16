"""
K 선택 진단 -- axis_comparison.py가 저장한 hi_window_features.csv를 재사용해서
(재추출 없음, 빠름) K=2..15 범위에서 inertia(elbow)/gap statistic/silhouette를
꺾은선으로 그린다. K_RANGE=(2,6)이 매번 상한에 붙던 문제(원래 axis_comparison.py)
의 원인을 눈으로 확인하기 위함.

Run: python 4_hi_analysis/k_selection_diagnostics.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "4_hi_analysis" / "outputs" / "axis_comparison"
FEAT_PATH = OUT_DIR / "hi_window_features.csv"

K_MIN, K_MAX = 2, 12
SIL_SAMPLE = 3000  # silhouette는 O(n^2)라 샘플링


def gap_statistic_curve(X: np.ndarray, k_min: int, k_max: int, n_ref: int = 5):
    """cluster.py::_gap_statistic과 동일 계산이지만, best_k 하나가 아니라
    전체 (gaps, stds) 배열을 반환한다 -- 곡선을 보기 위함."""
    ks = list(range(k_min, k_max + 1))
    gaps, stds = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        wk = float(km.inertia_)
        ref_wks = []
        rng = np.random.default_rng(0)
        for _ in range(n_ref):
            Xref = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_r = KMeans(n_clusters=k, n_init=3, random_state=0).fit(Xref)
            ref_wks.append(np.log(km_r.inertia_ + 1e-10))
        log_wk_ref = float(np.mean(ref_wks))
        sdk = float(np.std(ref_wks)) * np.sqrt(1 + 1 / n_ref)
        gaps.append(log_wk_ref - np.log(wk + 1e-10))
        stds.append(sdk)
    return ks, gaps, stds


def pick_1se(ks, gaps, stds):
    for idx in range(len(ks) - 1):
        if gaps[idx] >= gaps[idx + 1] - stds[idx + 1]:
            return ks[idx]
    return ks[int(np.argmax(gaps))]


def main():
    t0 = time.time()
    feat_df = pd.read_csv(FEAT_PATH)
    hi_cols = [c for c in feat_df.columns
               if c not in ("dataset", "cell_id", "cycle", "direction", "win_center_qfrac")]
    print(f"[k_diag] loaded {len(feat_df)} rows, {len(hi_cols)} HI cols")

    groups = list(feat_df.groupby(["dataset", "direction"]))
    fig_inertia, ax_inertia = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fig_gap, ax_gap = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    fig_sil, ax_sil = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    pos = {("MIT", "charge"): (0, 0), ("MIT", "discharge"): (0, 1),
           ("HUST", "charge"): (1, 0), ("HUST", "discharge"): (1, 1)}

    summary_rows = []
    for (dataset, direction), g in groups:
        X = g[hi_cols].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = StandardScaler().fit_transform(X)
        rng = np.random.default_rng(0)
        sil_idx = rng.choice(len(Xs), size=min(SIL_SAMPLE, len(Xs)), replace=False)

        ks = list(range(K_MIN, K_MAX + 1))
        inertias, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(Xs[sil_idx], km.labels_[sil_idx]))

        gap_ks, gaps, stds = gap_statistic_curve(Xs, K_MIN, K_MAX)
        best_k_1se = pick_1se(gap_ks, gaps, stds)
        best_k_sil = ks[int(np.argmax(sils))]

        r, c = pos[(dataset, direction)]
        title = f"{dataset} / {direction} (n={len(g)})"

        ax_inertia[r, c].plot(ks, inertias, marker="o", ms=3)
        ax_inertia[r, c].set_title(title, fontsize=9)
        ax_inertia[r, c].set_ylabel("inertia (WCSS)")

        ax_gap[r, c].errorbar(gap_ks, gaps, yerr=stds, marker="o", ms=3, capsize=2)
        ax_gap[r, c].axvline(best_k_1se, color="red", ls="--", lw=1,
                              label=f"1-SE pick K={best_k_1se}")
        ax_gap[r, c].set_title(title, fontsize=9)
        ax_gap[r, c].set_ylabel("gap statistic")
        ax_gap[r, c].legend(fontsize=7)

        ax_sil[r, c].plot(ks, sils, marker="o", ms=3, color="#96473A")
        ax_sil[r, c].axvline(best_k_sil, color="red", ls="--", lw=1,
                              label=f"max silhouette K={best_k_sil}")
        ax_sil[r, c].set_title(title, fontsize=9)
        ax_sil[r, c].set_ylabel("silhouette")
        ax_sil[r, c].legend(fontsize=7)

        summary_rows.append(dict(dataset=dataset, direction=direction,
                                  best_k_gap_1se=best_k_1se, best_k_silhouette=best_k_sil))
        print(f"[k_diag] {dataset}/{direction}: 1-SE K={best_k_1se}, "
              f"silhouette-max K={best_k_sil}, {time.time()-t0:.0f}s")

    for fig, axes, name in ((fig_inertia, ax_inertia, "inertia_elbow"),
                             (fig_gap, ax_gap, "gap_statistic"),
                             (fig_sil, ax_sil, "silhouette")):
        for ax in axes[-1, :]:
            ax.set_xlabel("K")
        fig.suptitle(f"K selection diagnostic -- {name}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = OUT_DIR / f"k_diagnostic_{name}.png"
        fig.savefig(out, dpi=200)
        print(f"[k_diag] saved: {out}")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "k_diagnostic_summary.csv", index=False)
    print(f"[k_diag] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
