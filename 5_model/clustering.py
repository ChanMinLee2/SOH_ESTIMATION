"""
clustering.py  —  충전/방전 그룹별 HI 클러스터링 + 레벨(Lo/Mid/Hi) 분류 분석

핵심 질문: "3개 SOH 레벨(Lo/Mid/Hi)을 구분하는 데 HI가 몇 개 필요한가?"

분석 단위 (2개):
  - charging    : scen ∈ {1, 2, 3}   → Lo=0 / Mid=1 / Hi=2
  - discharging : scen ∈ {-1, -2, -3} → Lo=0 / Mid=1 / Hi=2
  충/방전은 입력 단계에서 이미 알고 있으므로 분류 대상이 아님.

HI 선택 기준 — mRMR (Minimum Redundancy Maximum Relevance):
  - Relevance  : MI(HI_i ; y_level)  — target과의 상호정보량
  - Redundancy : mean |Pearson corr(HI_i, HI_selected)|  — 선택된 HI와의 중복
  - Score = relevance_norm - redundancy  → 매 step 가장 높은 후보 선택
  단순 MI 랭킹은 상관된 HI를 중복 선택하는 문제가 있어 mRMR로 대체.

출력 플롯 (그룹별 2×3):
  [0,0] PCA scatter — best_k 군집
  [0,1] PCA scatter — k=3 군집
  [0,2] PCA scatter — 실제 레벨 색상
  [1,0] Silhouette + KNN Purity curve (k=2..k_max)
  [1,1] sil_by_m (K-Means k=3) + acc_by_m (LogReg 5-fold) dual-axis
  [1,2] mRMR top-20 MI 값 bar chart

Usage:
  python 5_model/clustering.py
  python 5_model/clustering.py --k-max 9 --mrmr-m 20 --knn 10
  python 5_model/clustering.py --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

# joblib 병렬 스레드 × OpenBLAS 내부 스레드 중첩으로 최대 스레드(50) 초과 방지
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))

SEG_DIRS = [
    PROJECT_ROOT / "_4_data_hi" / "seg" / "MIT",
    PROJECT_ROOT / "_4_data_hi" / "seg" / "HUST",
]
META_COLS   = {"cell_id", "cycle", "segment_id", "capacity_Ah", "scen",
               "stat_q_abs", "stat_energy_seg"}

SIL_SAMPLE  = 10_000   # silhouette 계산용 서브샘플
KNN_SAMPLE  = 10_000   # KNN purity 계산용 서브샘플
PLOT_SAMPLE = 20_000   # scatter plot용 서브샘플
CLF_SAMPLE  = 50_000   # LogReg CV 서브샘플 (전체가 너무 크면 느림)
MRMR_SAMPLE = 50_000   # mRMR 상관계수 계산용 서브샘플

PLOT_DIR    = PROJECT_ROOT / "docs" / "clustering_plots"
LEVEL_NAMES = ["Lo", "Mid", "Hi"]
LEVEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# 충전/방전 그룹 정의 (scen 코드)
GROUPS: dict[str, list[int]] = {
    "charging":    [1, 2, 3],
    "discharging": [-1, -2, -3],
}


# ---------------------------------------------------------------------------
# mRMR 랭킹
# ---------------------------------------------------------------------------

def mrmr_rank(X_sc: np.ndarray, y_lvl: np.ndarray,
              max_m: int, sample: int = MRMR_SAMPLE) -> tuple[list[int], np.ndarray]:
    """
    mRMR (Minimum Redundancy Maximum Relevance) feature ranking.

    Relevance  = MI(X_i ; y_level)          — target과의 정보량
    Redundancy = mean |corr(X_i, X_sel)|    — 이미 선택된 feature와의 선형 중복
    Score      = rel_norm(X_i) - redundancy

    Returns
    -------
    rank   : 선택 순서의 feature 인덱스 리스트 (길이 max_m)
    rel_mi : 전체 feature의 raw MI 값 (길이 n_features), 시각화용
    """
    n = len(X_sc)
    rng = np.random.RandomState(42)
    if n > sample:
        idx = rng.choice(n, sample, replace=False)
        Xs = X_sc[idx]; ys = y_lvl[idx]
    else:
        Xs = X_sc; ys = y_lvl

    # relevance: MI with target
    rel_mi = mutual_info_classif(Xs, ys, random_state=42)          # (n_feat,)
    rel_norm = rel_mi / (rel_mi.max() + 1e-12)                     # normalize to [0,1]

    selected: list[int] = []
    remaining = list(range(X_sc.shape[1]))

    for _ in range(min(max_m, X_sc.shape[1])):
        if not selected:
            best = remaining[int(np.argmax([rel_norm[i] for i in remaining]))]
        else:
            # corr matrix: (n_remaining + n_selected) × (n_remaining + n_selected)
            # 행렬 크기가 커지면 느리므로 필요한 블록만 계산
            X_rem = Xs[:, remaining]          # (n, n_rem)
            X_sel = Xs[:, selected]           # (n, n_sel)
            # 분모로 std 0인 경우 처리
            std_rem = X_rem.std(axis=0)
            std_sel = X_sel.std(axis=0)
            std_rem = np.where(std_rem < 1e-12, 1.0, std_rem)
            std_sel = np.where(std_sel < 1e-12, 1.0, std_sel)
            Xr_n = (X_rem - X_rem.mean(axis=0)) / std_rem          # normalized
            Xs_n = (X_sel - X_sel.mean(axis=0)) / std_sel
            corr = (Xr_n.T @ Xs_n) / len(Xs)                       # (n_rem, n_sel)
            red = np.abs(corr).mean(axis=1)                         # (n_rem,)
            rel_arr = np.array([rel_norm[i] for i in remaining])
            scores = rel_arr - red
            best = remaining[int(np.argmax(scores))]

        selected.append(best)
        remaining.remove(best)

    return selected, rel_mi


# ---------------------------------------------------------------------------
# 병렬 작업 단위
# ---------------------------------------------------------------------------

def _knn_purity(X_sc: np.ndarray, labels: np.ndarray, k_nn: int) -> float:
    n = len(X_sc)
    rng = np.random.RandomState(42)
    if n > KNN_SAMPLE:
        idx = rng.choice(n, KNN_SAMPLE, replace=False)
        X_sub, L_sub = X_sc[idx], labels[idx]
    else:
        X_sub, L_sub = X_sc, labels
    nbrs = NearestNeighbors(n_neighbors=k_nn + 1, algorithm="auto", n_jobs=1)
    nbrs.fit(X_sub)
    indices = nbrs.kneighbors(X_sub, return_distance=False)[:, 1:]
    purity = float((L_sub[indices] == L_sub[:, None]).mean())
    return round(purity, 4)


def _kmeans_job(k: int, X_sc: np.ndarray, y_lvl: np.ndarray,
                n_init: int, k_nn: int = 10) -> dict:
    km = MiniBatchKMeans(n_clusters=k, n_init=n_init, batch_size=10_000,
                         random_state=42, max_iter=300)
    labels = km.fit_predict(X_sc)
    n = len(X_sc)
    sil     = silhouette_score(X_sc, labels, sample_size=min(SIL_SAMPLE, n), random_state=42)
    ari     = adjusted_rand_score(y_lvl, labels)
    nmi     = normalized_mutual_info_score(y_lvl, labels)
    knn_pur = _knn_purity(X_sc, labels, k_nn)
    return {
        "k": k, "labels": labels,
        "inertia":    float(km.inertia_),
        "sil":        round(float(sil), 4),
        "ari":        round(float(ari), 4),
        "nmi":        round(float(nmi), 4),
        "knn_purity": knn_pur,
    }


def _sil_m_job(m: int, rank: list[int], X_sc: np.ndarray, n_init: int) -> float:
    X_sub = X_sc[:, rank[:m]]
    km = MiniBatchKMeans(n_clusters=3, n_init=n_init, batch_size=10_000, random_state=42)
    labels = km.fit_predict(X_sub)
    s = silhouette_score(X_sub, labels, sample_size=min(SIL_SAMPLE, len(X_sub)), random_state=42)
    return round(float(s), 4)


def _acc_m_job(m: int, rank: list[int], X_sc: np.ndarray,
               y_lvl: np.ndarray) -> float:
    """LogReg 5-fold CV accuracy on subsampled data."""
    rng = np.random.RandomState(42)
    n = len(X_sc)
    if n > CLF_SAMPLE:
        idx = rng.choice(n, CLF_SAMPLE, replace=False)
        X_sub = X_sc[idx][:, rank[:m]]
        y_sub = y_lvl[idx]
    else:
        X_sub = X_sc[:, rank[:m]]
        y_sub = y_lvl
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    acc = cross_val_score(clf, X_sub, y_sub, cv=5, scoring="accuracy", n_jobs=1).mean()
    return round(float(acc), 4)


# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------

_CLUSTER_CMAP = plt.cm.get_cmap("tab10")


def _scatter(ax: plt.Axes, X_2d: np.ndarray, labels: np.ndarray,
             title: str, colors: list[str], class_names: list[str]) -> None:
    """단일 데이터셋(MIT 또는 HUST) scatter."""
    for c, (color, name) in enumerate(zip(colors, class_names)):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, s=4, alpha=0.3,
                   label=f"{name} ({mask.sum():,})", rasterized=True)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)
    ax.legend(fontsize=6, markerscale=3, loc="best")
    ax.tick_params(labelsize=6)


def plot_group(
    group_name: str,
    X_sc: np.ndarray,
    y_lvl: np.ndarray,
    y_ds: np.ndarray,
    hi_cols: list[str],
    rank: list[int],
    rel_mi: np.ndarray,
    k_range: list[int],
    labels_per_k: dict[int, np.ndarray],
    best_k_sil: int,
    sil_list: list[float],
    ari_list: list[float],
    nmi_list: list[float],
    knn_pur_list: list[float],
    sil_by_m: list[float],
    acc_by_m: list[float],
    plateau_m: int,
    plot_dir: Path,
    k_nn: int,
) -> None:
    """
    3×3 레이아웃:
      행 0: best_k 군집  — MIT | HUST | Sil+KNN curve
      행 1: k=3 군집     — MIT | HUST | sil_by_m + acc_by_m
      행 2: 실제 레벨    — MIT | HUST | mRMR bar chart
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    n = len(X_sc)

    # PCA + 서브샘플 (시각화 전용)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, min(PLOT_SAMPLE, n), replace=False)
    X_sub   = X_sc[idx]
    y_sub   = y_lvl[idx]
    ds_sub  = y_ds[idx]
    pca     = PCA(n_components=2, random_state=42)
    X_2d    = pca.fit_transform(X_sub)
    vr      = pca.explained_variance_ratio_

    labels_best = labels_per_k[best_k_sil][idx]
    labels_k3   = labels_per_k[3][idx]

    # MIT / HUST 서브샘플 마스크
    m_mit  = ds_sub == "MIT"
    m_hust = ds_sub == "HUST"
    mit_n  = int((y_ds == "MIT").sum())
    hust_n = int((y_ds == "HUST").sum())

    bk_colors = [_CLUSTER_CMAP(i / best_k_sil) for i in range(best_k_sil)]
    k3_colors = [_CLUSTER_CMAP(i / 3)           for i in range(3)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(
        f"Clustering — {group_name.capitalize()}  "
        f"(N={n:,}  |  MIT={mit_n:,}  HUST={hust_n:,})  "
        f"PC1+PC2={vr.sum()*100:.1f}%  k_nn={k_nn}",
        fontsize=11, y=1.01
    )

    # ── 행 0: best_k 군집 ─────────────────────────────────────────────
    _scatter(axes[0, 0], X_2d[m_mit],  labels_best[m_mit],
             f"Best-k={best_k_sil}  MIT",  bk_colors, [f"C{i}" for i in range(best_k_sil)])
    _scatter(axes[0, 1], X_2d[m_hust], labels_best[m_hust],
             f"Best-k={best_k_sil}  HUST", bk_colors, [f"C{i}" for i in range(best_k_sil)])

    ax_sil = axes[0, 2]
    ax_knn = ax_sil.twinx()
    ax_sil.plot(k_range, sil_list,     "b-o",  ms=4, label="Silhouette")
    ax_knn.plot(k_range, knn_pur_list, "r--s", ms=4, label="KNN Purity")
    ax_sil.axvline(x=best_k_sil, color="b", ls=":", lw=1)
    ax_sil.set_xlabel("k", fontsize=8)
    ax_sil.set_ylabel("Silhouette", color="b", fontsize=8)
    ax_knn.set_ylabel("KNN Purity",  color="r", fontsize=8)
    ax_sil.set_title(f"Cluster Quality vs k  (best_k={best_k_sil})", fontsize=9)
    ax_sil.tick_params(labelsize=7); ax_knn.tick_params(labelsize=7)
    lines_q = ax_sil.get_lines() + ax_knn.get_lines()
    ax_sil.legend(lines_q, [l.get_label() for l in lines_q], fontsize=7, loc="lower left")

    # ── 행 1: k=3 군집 (Lo/Mid/Hi 대응) ──────────────────────────────
    _scatter(axes[1, 0], X_2d[m_mit],  labels_k3[m_mit],
             "K-Means k=3  MIT   [Lo/Mid/Hi 대응]",  k3_colors, [f"C{i}" for i in range(3)])
    _scatter(axes[1, 1], X_2d[m_hust], labels_k3[m_hust],
             "K-Means k=3  HUST  [Lo/Mid/Hi 대응]", k3_colors, [f"C{i}" for i in range(3)])

    ax_m = axes[1, 2]
    ax_a = ax_m.twinx()
    ms = list(range(1, len(sil_by_m) + 1))
    ax_m.plot(ms, sil_by_m, "b-o",  ms=4, label="Sil (k=3)")
    ax_a.plot(ms, acc_by_m, "g--^", ms=4, label="Acc (LogReg 5-CV)")
    ax_m.axvline(x=plateau_m, color="gray", ls=":", lw=1.2, label=f"plateau m={plateau_m}")
    ax_m.set_xlabel("m  (mRMR top-m HIs)", fontsize=8)
    ax_m.set_ylabel("Silhouette", color="b", fontsize=8)
    ax_a.set_ylabel("Accuracy",   color="g", fontsize=8)
    ax_a.set_ylim(0, 1.05)
    ax_m.set_title(f"Quality vs #HIs  (plateau_m={plateau_m})", fontsize=9)
    ax_m.tick_params(labelsize=7); ax_a.tick_params(labelsize=7)
    lines_m = ax_m.get_lines() + ax_a.get_lines()
    ax_m.legend(lines_m, [l.get_label() for l in lines_m], fontsize=7, loc="lower right")

    # ── 행 2: 실제 레벨 ───────────────────────────────────────────────
    _scatter(axes[2, 0], X_2d[m_mit],  y_sub[m_mit],
             "True Level  MIT   (Lo / Mid / Hi)",  LEVEL_COLORS, LEVEL_NAMES)
    _scatter(axes[2, 1], X_2d[m_hust], y_sub[m_hust],
             "True Level  HUST  (Lo / Mid / Hi)", LEVEL_COLORS, LEVEL_NAMES)

    # mRMR bar chart
    ax_bar = axes[2, 2]
    top_n     = min(20, len(rank))
    top_names = [hi_cols[i]       for i in rank[:top_n]]
    top_mi    = [float(rel_mi[i]) for i in rank[:top_n]]
    y_pos     = list(range(top_n))
    ax_bar.barh(y_pos, top_mi, color="steelblue", alpha=0.75)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([nm[:30] for nm in top_names], fontsize=6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("MI (relevance, not mRMR score)", fontsize=7)
    ax_bar.set_title(
        f"Top-{top_n} HIs by mRMR\n"
        "(순서=mRMR 선택 순, 막대=MI 크기)", fontsize=8
    )
    ax_bar.tick_params(labelsize=6)
    if plateau_m <= top_n:
        ax_bar.axhline(y=plateau_m - 0.5, color="red", ls="--", lw=1,
                       label=f"plateau m={plateau_m}")
        ax_bar.legend(fontsize=7)

    plt.tight_layout()
    out_path = plot_dir / f"{group_name}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  플롯 저장: {out_path}")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _plateau_m(sil_by_m: list[float], thr: float = 0.005) -> int:
    """sil 증분이 thr 미만이 되는 첫 번째 m."""
    for i in range(len(sil_by_m) - 1):
        if sil_by_m[i + 1] - sil_by_m[i] < thr:
            return i + 1
    return len(sil_by_m)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="충전/방전 그룹별 HI 클러스터링 + mRMR 기반 레벨 분류 분석"
    )
    p.add_argument("--k-max",   type=int, default=9,
                   help="K-Means 최대 k (default 9)")
    p.add_argument("--n-init",  type=int, default=5,
                   help="MiniBatchKMeans n_init (default 5)")
    p.add_argument("--knn",     type=int, default=10,
                   help="KNN Purity 이웃 수 (default 10)")
    p.add_argument("--mrmr-m",  type=int, default=20,
                   help="mRMR로 선택할 최대 HI 수 (default 20)")
    p.add_argument("--no-plot", action="store_true",
                   help="플롯 생성 생략")
    p.add_argument("--n-jobs",  type=int, default=-1,
                   help="joblib 병렬 수 (default -1 = all)")
    p.add_argument("--out",     default="docs/_clustering_results.json",
                   help="결과 JSON 경로")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    t0      = time.time()
    k_range = list(range(2, args.k_max + 1))

    # ── 데이터 로드 ──────────────────────────────────────────────────────
    print("데이터 로드 중...")
    frames = []
    for d in SEG_DIRS:
        ds_name = d.name   # "MIT" or "HUST"
        for p in sorted(d.glob("*.pkl")):
            with open(p, "rb") as f:
                df_tmp = pickle.load(f)
            df_tmp["_dataset"] = ds_name
            frames.append(df_tmp)
    all_data = pd.concat(frames, ignore_index=True)
    hi_cols  = [c for c in all_data.columns if c not in META_COLS | {"_dataset"}]
    print(f"  전체 {len(all_data):,}행  HI {len(hi_cols)}개  로드: {time.time()-t0:.1f}s\n")

    results = {}

    for group_name, scen_codes in GROUPS.items():
        t_g = time.time()

        # 해당 그룹 필터링
        df    = all_data[all_data["scen"].isin(scen_codes)]
        y_lvl = (np.abs(df["scen"].values) - 1).astype(int)   # 0=Lo, 1=Mid, 2=Hi
        y_ds  = df["_dataset"].values                          # "MIT" or "HUST"
        X_raw = df[hi_cols].fillna(0).values

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_raw)
        n      = len(X_sc)
        lvl_dist = np.bincount(y_lvl).tolist()

        print(f"[{group_name}]  scen={scen_codes}  N={n:,}  레벨분포={dict(zip(LEVEL_NAMES, lvl_dist))}")

        # 1. K-Means k=2..k_max (병렬)
        print(f"  K-Means k={k_range} 병렬 실행...")
        km_raw = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(_kmeans_job)(k, X_sc, y_lvl, args.n_init, args.knn)
            for k in k_range
        )
        labels_per_k = {r["k"]: r["labels"] for r in km_raw}
        sil_list     = [r["sil"]        for r in km_raw]
        ari_list     = [r["ari"]        for r in km_raw]
        nmi_list     = [r["nmi"]        for r in km_raw]
        knn_pur_list = [r["knn_purity"] for r in km_raw]
        inertia_list = [r["inertia"]    for r in km_raw]
        best_k_sil   = k_range[int(np.argmax(sil_list))]
        k3_idx       = k_range.index(3)

        print(f"  best_k={best_k_sil}  sil@best={max(sil_list):.4f}  "
              f"k=3 → sil={sil_list[k3_idx]:.4f}  ari={ari_list[k3_idx]:.4f}  nmi={nmi_list[k3_idx]:.4f}")

        # 2. mRMR 랭킹
        print(f"  mRMR 랭킹 계산 중 (max_m={args.mrmr_m}, sample={MRMR_SAMPLE:,})...")
        rank, rel_mi = mrmr_rank(X_sc, y_lvl, max_m=args.mrmr_m)
        print(f"  mRMR top-5: {[hi_cols[i] for i in rank[:5]]}")

        # 3. sil_by_m: mRMR top-m HI로 K-Means k=3 silhouette (병렬)
        print(f"  sil_by_m (m=1..{args.mrmr_m}) 계산 중...")
        sil_by_m = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(_sil_m_job)(m, rank, X_sc, args.n_init)
            for m in range(1, args.mrmr_m + 1)
        )

        # 4. acc_by_m: LogReg 5-fold CV accuracy (병렬)
        print(f"  LogReg 5-fold CV accuracy (m=1..{args.mrmr_m}, subsampled={CLF_SAMPLE:,})...")
        acc_by_m = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(_acc_m_job)(m, rank, X_sc, y_lvl)
            for m in range(1, args.mrmr_m + 1)
        )

        # 5. plateau_m
        pm = _plateau_m(sil_by_m)
        print(f"  plateau_m={pm}  acc@m=1={acc_by_m[0]:.4f}  acc@m={pm}={acc_by_m[pm-1]:.4f}  "
              f"acc@m={args.mrmr_m}={acc_by_m[-1]:.4f}")
        print(f"  [{group_name}] 완료: {time.time()-t_g:.1f}s\n")

        results[group_name] = {
            "scen_codes":       scen_codes,
            "n_samples":        n,
            "level_dist":       dict(zip(LEVEL_NAMES, lvl_dist)),
            "best_k_sil":       best_k_sil,
            "sil_curve":        dict(zip(k_range, sil_list)),
            "ari_curve":        dict(zip(k_range, ari_list)),
            "nmi_curve":        dict(zip(k_range, nmi_list)),
            "knn_pur_curve":    dict(zip(k_range, knn_pur_list)),
            "inertia_curve":    dict(zip(k_range, inertia_list)),
            "k3_sil":           round(sil_list[k3_idx], 4),
            "k3_ari":           round(ari_list[k3_idx], 4),
            "k3_nmi":           round(nmi_list[k3_idx], 4),
            "mrmr_rank":        rank,
            "mrmr_top_names":   [hi_cols[i] for i in rank],
            "mrmr_mi_values":   [round(float(rel_mi[i]), 5) for i in rank],
            "sil_by_m":         sil_by_m,
            "acc_by_m":         acc_by_m,
            "plateau_m":        pm,
        }

        if not args.no_plot:
            plot_group(
                group_name=group_name,
                X_sc=X_sc, y_lvl=y_lvl, y_ds=y_ds, hi_cols=hi_cols,
                rank=rank, rel_mi=rel_mi,
                k_range=k_range, labels_per_k=labels_per_k,
                best_k_sil=best_k_sil,
                sil_list=sil_list, ari_list=ari_list,
                nmi_list=nmi_list, knn_pur_list=knn_pur_list,
                sil_by_m=sil_by_m, acc_by_m=acc_by_m,
                plateau_m=pm,
                plot_dir=PLOT_DIR, k_nn=args.knn,
            )

    # ── 결과 저장 ─────────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 JSON 저장: {out_path}")

    # ── 요약 테이블 ───────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'그룹':15s} {'N':>9s} {'best_k':>7s} {'k3-Sil':>8s} "
          f"{'k3-ARI':>8s} {'k3-NMI':>8s} {'plateau_m':>10s} {'acc@plt_m':>10s}")
    print("-" * 75)
    for g, r in results.items():
        pm   = r["plateau_m"]
        acc  = r["acc_by_m"][pm - 1]
        print(f"{g:15s} {r['n_samples']:>9,} {r['best_k_sil']:>7d} "
              f"{r['k3_sil']:>8.4f} {r['k3_ari']:>8.4f} {r['k3_nmi']:>8.4f} "
              f"{pm:>10d} {acc:>10.4f}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run(parse_args())
