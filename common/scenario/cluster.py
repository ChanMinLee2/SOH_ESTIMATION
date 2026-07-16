"""
common/scenario/cluster.py — 축 4: 미세분할 + K-means 클러스터 (Ke 2026 재현).

방향별 q_frac 균등 n_fine분할 → 미세 세그먼트별 Δq 통계벡터(mean/std/skew of dQ)
→ K-means. K는 Gap statistic + 1-SE rule로 k_range 내 자동 선정.

fit():
  train split 셀의 통계벡터를 수집 → K-means → centroids.npz 저장.
  반드시 train split 셀만 사용 (split_manifest.json 활용).

분류기: centroid (기본) — 최근접 센트로이드 배정, 결정론.
  mlp_probe로 바꾸면 "학습형 라우팅 vs 고정 클러스터" head-to-head (E5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import ScenarioSpec, SegmentRecord, Segmenter

_DEFAULT_N_FINE = 10
_DEFAULT_K_RANGE = (2, 8)


def _fine_seg_features(
    v: np.ndarray,
    i: np.ndarray,
    dt: np.ndarray,
    q: np.ndarray,
    n_fine: int,
) -> np.ndarray | None:
    """q_frac n_fine분할 → 각 미세세그먼트의 (dQ_mean, dQ_std, dQ_skew) → (n_fine, 3)."""
    from scipy.stats import skew as sp_skew
    q_tot = float(q[-1]) if len(q) > 0 else 0.0
    if q_tot < 0.05:
        return None
    feats = []
    for k in range(n_fine):
        lo = (k / n_fine) * q_tot
        hi = ((k + 1) / n_fine) * q_tot
        m = (q >= lo) & (q < hi)
        if m.sum() < 3:
            feats.append([0.0, 0.0, 0.0])
            continue
        dq = np.diff(q[m], prepend=q[m][0])
        feats.append([float(np.mean(dq)), float(np.std(dq)),
                      float(sp_skew(dq)) if len(dq) > 2 else 0.0])
    return np.array(feats, dtype=np.float32).flatten()  # (n_fine*3,)


def _gap_statistic(X: np.ndarray, k_range: tuple[int, int], n_ref: int = 10) -> int:
    """Gap statistic + 1-SE rule로 최적 K 선정."""
    from sklearn.cluster import KMeans
    ks = list(range(k_range[0], k_range[1] + 1))
    gaps, stds = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
        wk = float(km.inertia_)
        # 참조 분포 (균등 분포 박스 내 랜덤)
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
    # 1-SE rule: 가장 작은 k s.t. Gap(k) >= Gap(k+1) - std(k+1)
    best_k = ks[0]
    for idx in range(len(ks) - 1):
        if gaps[idx] >= gaps[idx + 1] - stds[idx + 1]:
            best_k = ks[idx]
            break
    else:
        best_k = ks[np.argmax(gaps)]
    return best_k


class ClusterSegmenter(Segmenter):
    """미세분할 K-means 클러스터 세그멘터."""

    name = "cluster"

    def __init__(
        self,
        n_fine: int = _DEFAULT_N_FINE,
        k_range: tuple[int, int] = _DEFAULT_K_RANGE,
        k_select: str = "gap_1se",   # "gap_1se" | int(고정)
        split_direction: bool = False,
        min_pts: int = 10,
    ):
        self.n_fine = n_fine
        self.k_range = k_range
        self.k_select = k_select
        self.split_direction = split_direction
        self.min_pts = min_pts
        self._centroids: dict[str, np.ndarray] | None = None   # {"dis": (K,D), "chg": (K,D)}
        self._K: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # fit: train cells에서 K-means 학습
    # ------------------------------------------------------------------
    def fit(self, train_cells) -> None:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        dirs = ["dis", "chg"] if self.split_direction else ["all"]
        feat_buf: dict[str, list[np.ndarray]] = {d: [] for d in dirs}

        for cell in train_cells:
            for key in ("dis_v", "chg_v"):
                direction_tag = "dis" if key == "dis_v" else "chg"
                tag = direction_tag if self.split_direction else "all"
                if key not in cell:
                    continue
                v = cell[key]; i = cell[key.replace("_v", "_i")]
                dt = cell[key.replace("_v", "_dt")]; q = cell[key.replace("_v", "_q")]
                feat = _fine_seg_features(v, i, dt, q, self.n_fine)
                if feat is not None:
                    feat_buf[tag].append(feat)

        self._centroids = {}
        self._K = {}

        for tag, feats in feat_buf.items():
            if len(feats) < self.k_range[0]:
                self._K[tag] = self.k_range[0]
            else:
                X = np.stack(feats)
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                if self.k_select == "gap_1se":
                    k = _gap_statistic(Xs, self.k_range)
                elif isinstance(self.k_select, int):
                    k = int(self.k_select)
                else:
                    k = self.k_range[0]
                km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
                self._centroids[tag] = km.cluster_centers_
                self._K[tag] = k
                print(f"[cluster] {tag}: K={k}, n_cells_feats={len(feats)}")

    def save_artifacts(self, out_dir: Path) -> None:
        super().save_artifacts(out_dir)
        if self._centroids:
            np.savez(Path(out_dir) / "centroids.npz", **self._centroids)
            k_info = self._K or {}
            (Path(out_dir) / "cluster_k.json").write_text(
                json.dumps(k_info, indent=2))

    def _total_k(self) -> int:
        if self._K is None:
            return self.k_range[0]
        return max(self._K.values())

    def get_spec(self) -> ScenarioSpec:
        K = self._total_k()
        if self.split_direction:
            n_scen = 2 * K
            names = [f"chg_c{k}" for k in range(K)] + [f"dis_c{k}" for k in range(K)]
            routing = [list(range(K)), list(range(K, 2 * K))]
        else:
            n_scen = K
            names = [f"c{k}" for k in range(K)]
            routing = [list(range(K)), list(range(K))]

        return ScenarioSpec(
            axis="cluster",
            n_scenarios=n_scen,
            scenario_names=names,
            n_classes=K,
            class_names=[f"c{k}" for k in range(K)],
            routing=routing,
            classifier_default="centroid",
            params={
                "n_fine":          self.n_fine,
                "k_range":         list(self.k_range),
                "k_select":        self.k_select if isinstance(self.k_select, str) else int(self.k_select),
                "split_direction": self.split_direction,
                "K":               self._K or {},
            },
        )

    def _assign_cluster(
        self, feat: np.ndarray, tag: str
    ) -> int:
        if self._centroids is None or tag not in self._centroids:
            return 0
        dists = np.linalg.norm(self._centroids[tag] - feat, axis=1)
        return int(np.argmin(dists))

    def _process_direction(
        self,
        v, i, dt, q,
        direction: int,
        cell_id: str,
        cycle: int,
        seg_local_start: int,
    ) -> tuple[list[SegmentRecord], int]:
        tag = ("dis" if direction == -1 else "chg") if self.split_direction else "all"
        dir_idx = 0 if direction == 1 else 1
        spec = self.get_spec()
        records = []
        seg_local = seg_local_start
        q_tot = float(q[-1]) if len(q) > 0 else 0.0
        if q_tot < 0.05:
            return records, seg_local

        for k in range(self.n_fine):
            lo = (k / self.n_fine) * q_tot
            hi = ((k + 1) / self.n_fine) * q_tot
            m = (q >= lo) & (q < hi)
            if m.sum() < self.min_pts:
                continue
            feat = _fine_seg_features(v[m], i[m], dt[m], q[m], 1)
            latent_class = self._assign_cluster(feat, tag) if feat is not None else 0
            scenario_id = spec.routing[dir_idx][latent_class]
            records.append(SegmentRecord(
                cell_id=cell_id, cycle=cycle, seg_local_id=seg_local,
                scenario_id=scenario_id,
                latent_class=latent_class,
                direction=direction,
                v=v[m], i=i[m], dt=dt[m], q=q[m],
                meta={"fine_k": k, "cluster": latent_class},
            ))
            seg_local += 1
        return records, seg_local

    def iter_segments(
        self,
        cell_id, cycle,
        dis_v, dis_i, dis_dt, dis_q,
        chg_v=None, chg_i=None, chg_dt=None, chg_q=None,
    ) -> Iterator[SegmentRecord]:
        seg_local = 0

        if len(dis_v) >= self.min_pts:
            recs, seg_local = self._process_direction(
                dis_v, dis_i, dis_dt, dis_q, -1, cell_id, cycle, seg_local)
            yield from recs

        if chg_v is not None and len(chg_v) >= self.min_pts:
            recs, seg_local = self._process_direction(
                chg_v, chg_i, chg_dt, chg_q, +1, cell_id, cycle, seg_local)
            yield from recs
