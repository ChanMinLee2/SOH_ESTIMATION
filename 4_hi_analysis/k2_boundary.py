"""
K=2 고정 클러스터링으로 경계(q_frac 전이 지점)를 뽑고, q_frac_ref(0.35/0.65)와
vqslope 플래토 진입점(대표 셀 기준)과 직접 비교.

hi_window_features.csv 재사용(재추출 없음).
Run: python 4_hi_analysis/k2_boundary.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.scenario.vqslope import VQSlopeSegmenter  # noqa: E402

OUT_DIR = PROJECT_ROOT / "4_hi_analysis" / "outputs" / "axis_comparison"
FEAT_PATH = OUT_DIR / "hi_window_features.csv"
CLEAN_ROOT = None  # set below via axis_comparison for cell loading

K_FIXED = 2
QFRAC_REF = [0.35, 0.65]
REP_CELL = {"MIT": "b1c0", "HUST": "1-1"}


def find_boundaries_fixed_k(g: pd.DataFrame, hi_cols: list[str], k: int):
    X = np.nan_to_num(g[hi_cols].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    labels = km.labels_

    centers_raw = g["win_center_qfrac"].to_numpy()
    uniq_centers = np.array(sorted(set(np.round(centers_raw, 4))))
    bin_majority = np.full(len(uniq_centers), -1)
    for bi, cval in enumerate(uniq_centers):
        sel = labels[np.isclose(centers_raw, cval, atol=1e-4)]
        if len(sel) > 0:
            bin_majority[bi] = np.bincount(sel).argmax()

    boundaries = []
    for bi in range(1, len(uniq_centers)):
        if bin_majority[bi] != bin_majority[bi - 1] and bin_majority[bi] >= 0 and bin_majority[bi - 1] >= 0:
            boundaries.append(float((uniq_centers[bi] + uniq_centers[bi - 1]) / 2))
    return boundaries, uniq_centers.tolist(), bin_majority.tolist()


def vqslope_entry_exit(dataset: str):
    """대표 셀의 대표 사이클에서 vqslope 플래토 진입/이탈 q_frac (charge/discharge 각각)."""
    import axis_comparison as ac
    cell_id = REP_CELL[dataset]
    f = ac.CLEAN_ROOT / dataset / f"{cell_id}.pkl"
    _, df = ac.load_cell(f)
    cycles = sorted(df["cycle"].unique())
    mid_cyc = cycles[len(cycles) // 2]
    c = df[df["cycle"] == mid_cyc]

    vqs = VQSlopeSegmenter(mode="dva", n_samples=1)
    out = {}
    for direction in ("charge", "discharge"):
        arr = ac.direction_arrays(c, direction)
        if arr is None:
            continue
        v, i_abs, dt, t = arr
        q = np.cumsum(i_abs * dt) / 3600.0
        q_tot = float(q[-1])
        rng = vqs._plateau_q_range(v, i_abs, dt, q, q_tot)
        if rng:
            out[direction] = (rng[0] / q_tot, rng[1] / q_tot)
    return out


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import axis_comparison as ac  # noqa

    feat_df = pd.read_csv(FEAT_PATH)
    hi_cols = [c for c in feat_df.columns
               if c not in ("dataset", "cell_id", "cycle", "direction", "win_center_qfrac")]

    print(f"{'dataset':6} {'dir':10} {'K2 boundary(qfrac)':22} "
          f"{'q_frac_ref':14} {'vqslope entry/exit':20}")
    rows = []
    for dataset in ("MIT", "HUST"):
        vqs_landmarks = vqslope_entry_exit(dataset)
        for direction in ("charge", "discharge"):
            g = feat_df[(feat_df.dataset == dataset) & (feat_df.direction == direction)]
            boundaries, centers, maj = find_boundaries_fixed_k(g, hi_cols, K_FIXED)
            vqs_ee = vqs_landmarks.get(direction)
            print(f"{dataset:6} {direction:10} {str([round(b,3) for b in boundaries]):22} "
                  f"{str(QFRAC_REF):14} "
                  f"{str([round(x,3) for x in vqs_ee]) if vqs_ee else '-':20}")
            rows.append(dict(dataset=dataset, direction=direction,
                              k2_boundaries=boundaries,
                              qfrac_ref=QFRAC_REF,
                              vqslope_entry=vqs_ee[0] if vqs_ee else None,
                              vqslope_exit=vqs_ee[1] if vqs_ee else None))

    pd.DataFrame(rows).to_csv(OUT_DIR / "k2_boundary_comparison.csv", index=False)
    print(f"\n[k2_boundary] saved: {OUT_DIR / 'k2_boundary_comparison.csv'}")


if __name__ == "__main__":
    main()
