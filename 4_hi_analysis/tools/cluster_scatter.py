"""
raw-HI 피처 공간을 PCA 2D로 투영해서 실제 분포를 산점도로 확인.
hi_window_features.csv 재사용(재추출 없음, 빠름).

두 가지 색칠로 같은 산점도를 나란히 그린다:
  1행: K=2 클러스터 라벨로 색칠 -- "실제로 두 덩어리로 분리되는가"
  2행: win_center_qfrac(연속값)으로 색칠 -- "그 분리가 SOC 위치와 일치하는가"
       (일치하면 클러스터=SOC 위치 구조, 안 일치하면 클러스터가 다른 걸(셀/열화 등)
       잡고 있다는 뜻)

Run: python 4_hi_analysis/tools/cluster_scatter.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "4_hi_analysis" / "outputs" / "axis_comparison"
FEAT_PATH = OUT_DIR / "hi_window_features.csv"

K_FIXED = 2


def main():
    feat_df = pd.read_csv(FEAT_PATH)
    hi_cols = [c for c in feat_df.columns
               if c not in ("dataset", "cell_id", "cycle", "direction", "win_center_qfrac")]

    groups = [("MIT", "charge"), ("MIT", "discharge"), ("HUST", "charge"), ("HUST", "discharge")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for col, (dataset, direction) in enumerate(groups):
        g = feat_df[(feat_df.dataset == dataset) & (feat_df.direction == direction)]
        X = np.nan_to_num(g[hi_cols].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        Xs = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_

        km = KMeans(n_clusters=K_FIXED, n_init=10, random_state=42).fit(Xs)
        labels = km.labels_

        ax = axes[0, col]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="coolwarm", s=4, alpha=0.35, linewidths=0)
        ax.set_title(f"{dataset}/{direction}\nK={K_FIXED} cluster label", fontsize=9)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")

        ax2 = axes[1, col]
        sc2 = ax2.scatter(Z[:, 0], Z[:, 1], c=g["win_center_qfrac"], cmap="viridis",
                           s=4, alpha=0.35, linewidths=0)
        ax2.set_title(f"{dataset}/{direction}\ncolored by win_center_qfrac", fontsize=9)
        ax2.set_xlabel(f"PC1 ({evr[0]*100:.0f}%)")
        ax2.set_ylabel(f"PC2 ({evr[1]*100:.0f}%)")
        fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

        print(f"[cluster_scatter] {dataset}/{direction}: n={len(g)}, "
              f"PC1+PC2 explain {evr.sum()*100:.1f}% variance, "
              f"K=2 split sizes={np.bincount(labels).tolist()}")

    fig.suptitle("Raw-HI feature space (PCA 2D) -- cluster label vs. SOC position", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "cluster_scatter_pca.png"
    fig.savefig(out, dpi=200)
    print(f"[cluster_scatter] saved: {out}")


if __name__ == "__main__":
    main()
