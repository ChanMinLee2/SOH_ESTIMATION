"""
축 비교 — q_frac_ref(고정 n1=0.35) vs vqslope(DVA 랜드마크) vs vwindow(문헌 고정
전압) vs raw-HI 클러스터링(데이터 기반), MIT/HUST(LFP) 전용.

docs/DRAFT.md 재설계 논의 "원칙4 — 조건화 축 자체를 데이터 기반으로 재검증" 의
1단계. q_frac_ref는 q_tot(≈SOH)의 함수라 순환성 문제가 있으므로(vqslope.py
docstring 참고), 클러스터링 입력 세그먼트는 q_abs(BOL 기준 절대용량)로 자른다.

측정 항목:
  1. vqslope/vwindow 세그먼트 탈락률(attempted vs yielded) — 기존 Segmenter
     내장 카운터(vqslope) 또는 직접 집계(vwindow, 카운터 없음)
  2. MIT/HUST 각각, 충/방전 각각 raw HI(58개, morph 제외) 슬라이딩 윈도우
     (q_abs 10%폭/8%stride=2%겹침) → K-means(Gap statistic, cluster.py 재사용)
     → 클러스터 경계(q_frac 전이 지점)
  3. 대표 셀 1개씩(MIT b1c0, HUST 1-1) 대표 사이클에 대해 4개 방식의 경계를
     V-Q 곡선 위에 겹쳐 그린 비교 figure

출력: 4_hi_analysis/outputs/axis_comparison/
  - dropout_summary.csv       (vqslope/vwindow 탈락률)
  - cluster_boundaries.csv    (데이터셋×방향별 클러스터 전이 q_frac)
  - hi_window_features.csv    (원자료, 재사용 가능하게 저장)
  - comparison_MIT.png / comparison_HUST.png

Run: python 4_hi_analysis/axis_comparison.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
sys.path.insert(0, str(PROJECT_ROOT / "docs" / "figures"))

from data_directories import DATA_4_HI_ROOT              # noqa: E402
from common.scenario.vqslope import VQSlopeSegmenter      # noqa: E402
from common.scenario.vwindow import VWindowSegmenter, _LFP_CHG_EDGES_3, _LFP_DIS_EDGES_3  # noqa: E402
from common.scenario.cluster import _gap_statistic        # noqa: E402
from hi_compute import compute_his                         # noqa: E402

CLEAN_ROOT = DATA_4_HI_ROOT / "clean"
OUT_DIR = PROJECT_ROOT / "4_hi_analysis" / "outputs" / "axis_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CYCLES_PER_CELL = 12     # 런타임 제어 (HI 슬라이딩 윈도우가 비쌈)
WIN_WIDTH = 0.10             # q_abs 기준 10%폭
WIN_STRIDE = 0.08            # 8% stride = 2% 겹침
MIN_CURRENT_A = 0.02
MIN_PTS_WIN = 8
K_RANGE = (2, 6)

# q_frac_ref 기준 (n1=0.35) — docs/figures/_style.py ZONE_BOUNDS와 동일
QFRAC_REF_BOUNDS = {"left": (0.0, 0.35), "mid": (0.325, 0.675), "right": (0.65, 1.0)}

REP_CELL = {"MIT": "b1c0", "HUST": "1-1"}


# ---------------------------------------------------------------------------
# 원시 사이클 로딩 / 방향 분리
# ---------------------------------------------------------------------------

def load_cell(pkl_path: Path):
    d = pd.read_pickle(pkl_path)
    return d["meta"]["cell_id"], d["cycles"]


def direction_arrays(c: pd.DataFrame, direction: str):
    """direction: 'charge' | 'discharge'. Returns (v,i_abs,dt,t) sorted by time, or None."""
    sel = c.current_A > MIN_CURRENT_A if direction == "charge" else c.current_A < -MIN_CURRENT_A
    sub = c[sel].sort_values("time_s")
    if len(sub) < MIN_PTS_WIN:
        return None
    v = sub["voltage_V"].to_numpy()
    i_abs = np.abs(sub["current_A"].to_numpy())
    t = sub["time_s"].to_numpy()
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 1.0))
    dt = np.clip(dt, 1e-6, None)
    return v, i_abs, dt, t


# ---------------------------------------------------------------------------
# 1. vqslope / vwindow 탈락률 -- 실제 집계 루프는 main()에 있음. vqslope의
#    n_attempted/n_yielded는 scenario_name(chg_lo 등) 단위 dict라, 방향별로
#    합산하려면 접두어(chg/dis)로 묶어야 한다 -- 그 합산 헬퍼만 여기 둔다.
# ---------------------------------------------------------------------------

def sum_vqslope_counters(vqs: VQSlopeSegmenter, prefix: str) -> tuple[int, int]:
    att = sum(v for k, v in vqs.n_attempted.items() if k.startswith(prefix))
    yld = sum(v for k, v in vqs.n_yielded.items() if k.startswith(prefix))
    return att, yld


# ---------------------------------------------------------------------------
# 2. raw HI 슬라이딩 윈도우 (q_abs 기준) -> 클러스터링
# ---------------------------------------------------------------------------

def sliding_windows_qabs(q_tot_cap_ref: float, width=WIN_WIDTH, stride=WIN_STRIDE):
    """[0, cap_ref] 위 10%폭/8%stride 윈도우 경계 리스트 [(lo_frac,hi_frac), ...]."""
    starts = np.arange(0.0, 1.0 - width + 1e-9, stride)
    return [(float(s), float(s + width)) for s in starts]


def extract_window_features(dataset: str, files: list[Path]):
    """모든 셀/사이클/방향에 대해 슬라이딩 윈도우 raw HI 벡터를 뽑는다."""
    rows = []
    for f in files:
        cell_id, df = load_cell(f)
        cycles = sorted(df["cycle"].unique())
        idx = np.linspace(0, len(cycles) - 1, min(MAX_CYCLES_PER_CELL, len(cycles))).astype(int)
        cap_ref = {}
        for cyc in sorted(set(cycles[k] for k in idx)):
            c = df[df["cycle"] == cyc]
            for direction in ("charge", "discharge"):
                arr = direction_arrays(c, direction)
                if arr is None:
                    continue
                v, i_abs, dt, t = arr
                q = np.cumsum(i_abs * dt) / 3600.0
                q_tot = float(q[-1])
                if q_tot < 0.05:
                    continue
                if direction not in cap_ref:
                    cap_ref[direction] = q_tot  # 이 셀에서 처음 만난(=가장 이른) 유효 사이클 기준 BOL
                ref = cap_ref[direction]

                for lo_fr, hi_fr in sliding_windows_qabs(ref):
                    lo_q, hi_q = lo_fr * ref, hi_fr * ref
                    m = (q >= lo_q) & (q < hi_q)
                    if m.sum() < MIN_PTS_WIN:
                        continue
                    his = compute_his(v[m], i_abs[m], t[m])
                    row = dict(dataset=dataset, cell_id=cell_id, cycle=int(cyc),
                               direction=direction, win_center_qfrac=(lo_fr + hi_fr) / 2)
                    row.update(his)
                    rows.append(row)
    return pd.DataFrame(rows)


def cluster_and_find_boundaries(feat_df: pd.DataFrame, hi_cols: list[str]):
    """dataset x direction 별로 K-means, 클러스터 경계(q_frac 전이 지점) 도출."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    results = {}
    for (dataset, direction), g in feat_df.groupby(["dataset", "direction"]):
        X = g[hi_cols].to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if len(g) < K_RANGE[0] * 5:
            continue
        Xs = StandardScaler().fit_transform(X)
        k = _gap_statistic(Xs, K_RANGE)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        labels = km.labels_

        # 실제 슬라이딩 윈도우 중심 위치(WIN_WIDTH/WIN_STRIDE로 결정되는 이산 격자)를
        # 그대로 "bin" 그리드로 쓴다 -- 임의의 고정 n_bins를 쓰면 실제 윈도우 해상도보다
        # 촘촘해져서 빈 bin이 절반씩 생기고 인접비교가 전부 스킵되는 버그가 났었다.
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
        results[(dataset, direction)] = dict(K=k, boundaries=boundaries,
                                              bin_majority=bin_majority.tolist(),
                                              bin_centers=uniq_centers.tolist())
        print(f"[axis_comparison] cluster {dataset}/{direction}: K={k}, "
              f"n={len(g)}, boundaries(qfrac)={[round(b,3) for b in boundaries]}")
    return results


# ---------------------------------------------------------------------------
# 3. 시각화
# ---------------------------------------------------------------------------

def _example_vq(dataset: str, cell_id: str):
    f = CLEAN_ROOT / dataset / f"{cell_id}.pkl"
    _, df = load_cell(f)
    cycles = sorted(df["cycle"].unique())
    mid_cyc = cycles[len(cycles) // 2]
    c = df[df["cycle"] == mid_cyc]
    return c, mid_cyc


def _shade(ax, bounds_list, colors, labels):
    for (lo, hi), c, lbl in zip(bounds_list, colors, labels):
        ax.axvspan(lo, hi, color=c, alpha=0.18, label=lbl)


def plot_comparison(dataset: str, cluster_results: dict):
    cell_id = REP_CELL[dataset]
    c, cyc = _example_vq(dataset, cell_id)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey="row")
    palette = ["#3A6178", "#A9863F", "#96473A"]

    for row, direction in enumerate(("charge", "discharge")):
        arr = direction_arrays(c, direction)
        if arr is None:
            continue
        v, i_abs, dt, t = arr
        q = np.cumsum(i_abs * dt) / 3600.0
        q_tot = float(q[-1])
        qf = q / q_tot

        # (a) q_frac_ref
        ax = axes[row, 0]
        ax.plot(qf, v, color="black", lw=1.0)
        _shade(ax, list(QFRAC_REF_BOUNDS.values()), palette, list(QFRAC_REF_BOUNDS.keys()))
        ax.set_title(f"{direction} -- q_frac_ref (n1=0.35)", fontsize=9)

        # (b) vqslope
        ax = axes[row, 1]
        ax.plot(qf, v, color="black", lw=1.0)
        vqs = VQSlopeSegmenter(mode="dva", n_samples=1)
        rng = vqs._plateau_q_range(v, i_abs, dt, q, q_tot)
        if rng:
            qe, qx = rng[0] / q_tot, rng[1] / q_tot
            _shade(ax, [(0, qe), (qe, qx), (qx, 1.0)], palette, ["head", "plateau", "tail"])
        ax.set_title(f"{direction} -- vqslope (DVA landmark)", fontsize=9)

        # (c) vwindow
        ax = axes[row, 2]
        ax.plot(qf, v, color="black", lw=1.0)
        edges = _LFP_CHG_EDGES_3 if direction == "charge" else _LFP_DIS_EDGES_3
        for k in range(3):
            v_lo, v_hi = edges[k], edges[k + 1]
            m = (v >= v_lo) & (v <= v_hi)
            if m.sum() > 0:
                ax.axvspan(qf[m].min(), qf[m].max(), color=palette[k], alpha=0.18)
        ax.set_title(f"{direction} -- vwindow (fixed V, LFP lit.)", fontsize=9)

        # (d) cluster
        ax = axes[row, 3]
        ax.plot(qf, v, color="black", lw=1.0)
        res = cluster_results.get((dataset, direction))
        if res:
            centers = res["bin_centers"]; maj = res["bin_majority"]
            cur = None; seg_start = 0.0
            ci = 0
            for bi, (bc, lab) in enumerate(zip(centers, maj)):
                if lab != cur:
                    if cur is not None and cur >= 0:
                        ax.axvspan(seg_start, bc, color=palette[ci % len(palette)], alpha=0.18)
                        ci += 1
                    seg_start = bc
                    cur = lab
            ax.axvspan(seg_start, 1.0, color=palette[ci % len(palette)], alpha=0.18)
        ax.set_title(f"{direction} -- raw-HI cluster (K={res['K'] if res else '?'})", fontsize=9)

        for ax in axes[row]:
            ax.set_xlabel("q_frac"); ax.set_xlim(0, 1)
        axes[row, 0].set_ylabel("V")

    fig.suptitle(f"{dataset} cell={cell_id} cycle={cyc} -- axis boundary comparison", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / f"comparison_{dataset}.png"
    fig.savefig(out, dpi=200)
    print(f"[axis_comparison] saved: {out}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    dropout_rows = []
    all_feats = []

    for dataset in ("MIT", "HUST"):
        files = sorted((CLEAN_ROOT / dataset).glob("*.pkl"))
        print(f"[axis_comparison] {dataset}: {len(files)} cells -- dropout pass")

        # -- vqslope/vwindow dropout (전용 인스턴스라 카운터 접두어 방식으로 재집계) --
        vqs = VQSlopeSegmenter(mode="dva", n_samples=1)
        vwin = VWindowSegmenter.from_lfp()
        vwin_att = {"charge": 0, "discharge": 0}
        vwin_yld = {"charge": 0, "discharge": 0}

        for fi, f in enumerate(files):
            cell_id, df = load_cell(f)
            cycles = sorted(df["cycle"].unique())
            idx = np.linspace(0, len(cycles) - 1, min(MAX_CYCLES_PER_CELL, len(cycles))).astype(int)
            for cyc in sorted(set(cycles[k] for k in idx)):
                c = df[df["cycle"] == cyc]
                dis = direction_arrays(c, "discharge")
                chg = direction_arrays(c, "charge")
                dv, di, ddt = (dis[0], dis[1], dis[2]) if dis else (np.array([]), np.array([]), np.array([]))
                dq = np.cumsum(di * ddt) / 3600.0 if dis else np.array([])
                if chg:
                    cv, ci, cdt = chg[0], chg[1], chg[2]
                    cq = np.cumsum(ci * cdt) / 3600.0
                else:
                    cv = ci = cdt = cq = None

                list(vqs.iter_segments(cell_id, cyc, dv, di, ddt, dq, cv, ci, cdt, cq))

                if dis is not None:
                    vwin_att["discharge"] += vwin.n_windows
                    vwin_yld["discharge"] += len(list(vwin.iter_segments(cell_id, cyc, dv, di, ddt, dq)))
                if chg is not None:
                    vwin_att["charge"] += vwin.n_windows + 1
                    # 방전 인자에 빈 배열을 줘야 charge만 yield됨 -- 실수로 dv/di/ddt/dq
                    # (실제 방전 배열)을 같이 넘기면 discharge 레코드까지 "charge" 탈락률에
                    # 섞여 들어가 n_yielded > n_attempted(음수 dropout_rate)가 되는 버그였음.
                    _empty = np.array([])
                    vwin_yld["charge"] += len(list(vwin.iter_segments(
                        cell_id, cyc, _empty, _empty, _empty, _empty, cv, ci, cdt, cq)))

            if (fi + 1) % 40 == 0:
                print(f"[axis_comparison] {dataset} dropout {fi+1}/{len(files)}, {time.time()-t0:.0f}s")

        for dkey in ("charge", "discharge"):
            att, yld = sum_vqslope_counters(vqs, "chg" if dkey == "charge" else "dis")
            dropout_rows.append(dict(dataset=dataset, method="vqslope", direction=dkey,
                                      n_attempted=att, n_yielded=yld,
                                      dropout_rate=(1 - yld / att) if att > 0 else np.nan,
                                      plateau_fail=vqs.n_plateau_fail.get(dkey, 0)))
            dropout_rows.append(dict(dataset=dataset, method="vwindow", direction=dkey,
                                      n_attempted=vwin_att[dkey], n_yielded=vwin_yld[dkey],
                                      dropout_rate=(1 - vwin_yld[dkey] / vwin_att[dkey])
                                      if vwin_att[dkey] > 0 else np.nan, plateau_fail=None))

        print(f"[axis_comparison] {dataset}: raw-HI sliding-window feature extraction")
        feat = extract_window_features(dataset, files)
        all_feats.append(feat)
        print(f"[axis_comparison] {dataset}: {len(feat)} window-HI rows, {time.time()-t0:.0f}s elapsed")

    dropout_df = pd.DataFrame(dropout_rows)
    dropout_path = OUT_DIR / "dropout_summary.csv"
    dropout_df.to_csv(dropout_path, index=False)
    print(dropout_df.to_string(index=False))
    print(f"[axis_comparison] saved: {dropout_path}")

    feat_df = pd.concat(all_feats, ignore_index=True)
    feat_path = OUT_DIR / "hi_window_features.csv"
    feat_df.to_csv(feat_path, index=False)
    print(f"[axis_comparison] saved: {feat_path} ({len(feat_df)} rows)")

    hi_cols = [c for c in feat_df.columns
               if c not in ("dataset", "cell_id", "cycle", "direction", "win_center_qfrac")]
    cluster_results = cluster_and_find_boundaries(feat_df, hi_cols)

    bnd_rows = []
    for (dataset, direction), res in cluster_results.items():
        for b in res["boundaries"]:
            bnd_rows.append(dict(dataset=dataset, direction=direction, K=res["K"], boundary_qfrac=b))
    bnd_df = pd.DataFrame(bnd_rows)
    bnd_path = OUT_DIR / "cluster_boundaries.csv"
    bnd_df.to_csv(bnd_path, index=False)
    print(bnd_df.to_string(index=False))
    print(f"[axis_comparison] saved: {bnd_path}")

    for dataset in ("MIT", "HUST"):
        plot_comparison(dataset, cluster_results)

    print(f"[axis_comparison] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
