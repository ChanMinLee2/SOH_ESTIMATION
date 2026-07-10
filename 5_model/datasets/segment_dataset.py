"""
Segment-level dataset for SCR.

Wide pkl format (one row = one cycle, HI columns per segment):
  _4_data_hi/{MIT,HUST}/*.pkl

Each cycle is reshaped to N_SEGS rows.  The cycle-level capacity_Ah is used
as the SOH target for every segment in that cycle.

Native seg format (future):
  _4_data_hi/seg/{MIT,HUST}/*.pkl
  one row = one segment, columns: cell_id, cycle, seg_id, scen, capacity_Ah, hi_0..hi_64
  If this directory exists, it is loaded directly without reshape.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.hi_schema import (
    STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS,
    SEGMENTS, SCEN_MAP, SEG_LEVEL, SEG_DIRECTION,
    get_hi_cols_for_seg, get_hi_cost_vector, N_HI, N_SEGS,
)


# ---------------------------------------------------------------------------
# Cell-level train / val / test split
# ---------------------------------------------------------------------------

def split_cells(
    cell_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Reproducible cell-level split → (train, val, test) lists."""
    rng = np.random.default_rng(seed)
    ids = sorted(cell_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[n_train: n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_wide_pkl(pkl_path: Path) -> pd.DataFrame:
    """Load one wide-format cell pkl (one row = one cycle)."""
    try:
        from utils.compat import install_numpy2_shim
        install_numpy2_shim()
    except ImportError:
        pass
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame(data)
    raise ValueError(f"Unexpected pkl type: {type(data)}")


def _wide_to_segments(df: pd.DataFrame, cell_id: str) -> pd.DataFrame:
    """
    Reshape one cell's wide DataFrame (one row = one cycle) into
    segment-level DataFrame (one row = one segment).

    Output columns:
      cell_id, cycle, seg_name, seg_idx, scen, direction, level,
      capacity_Ah, hi_00 .. hi_64
    """
    rows = []
    for _, cycle_row in df.iterrows():
        cycle_num = int(cycle_row.get("cycle", cycle_row.name))
        cap_ah = float(cycle_row["capacity_Ah"])

        for seg in SEGMENTS:
            hi_cols = get_hi_cols_for_seg(seg)
            # Some segments may not exist in older data — silently skip
            if not all(c in cycle_row.index for c in hi_cols):
                continue

            hi_vals = cycle_row[hi_cols].values.astype(np.float32)
            scen_code, seg_idx = SCEN_MAP[seg]
            level = SEG_LEVEL[seg]
            direction = SEG_DIRECTION[seg]

            rows.append({
                "cell_id":     cell_id,
                "cycle":       cycle_num,
                "seg_name":    seg,
                "seg_idx":     seg_idx,
                "scen":        scen_code,
                "direction":   direction,
                "level":       level,
                "capacity_Ah": cap_ah,
                **{f"hi_{i:02d}": v for i, v in enumerate(hi_vals)},
            })

    return pd.DataFrame(rows)


def load_dataset_wide(
    data_dir: Path,
    datasets: Sequence[str],
    min_cycles: int = 10,
) -> pd.DataFrame:
    """
    Load wide pkl files for all datasets and return a combined segment-level DataFrame.

    data_dir: e.g. Path("_4_data_hi")
    datasets: e.g. ["MIT", "HUST"]
    """
    all_segs: list[pd.DataFrame] = []
    for ds in datasets:
        ds_dir = data_dir / ds
        if not ds_dir.exists():
            print(f"[dataset] WARNING: {ds_dir} not found, skipping")
            continue
        for pkl_path in sorted(ds_dir.glob("*.pkl")):
            cell_id = pkl_path.stem
            try:
                df = _load_wide_pkl(pkl_path)
            except Exception as e:
                print(f"[dataset] ERROR loading {pkl_path}: {e}")
                continue
            if len(df) < min_cycles:
                continue
            seg_df = _wide_to_segments(df, cell_id)
            if len(seg_df) == 0:
                continue
            seg_df["dataset"] = ds
            all_segs.append(seg_df)

    if not all_segs:
        raise RuntimeError("No data loaded — check data_dir and datasets config.")
    combined = pd.concat(all_segs, ignore_index=True)
    return combined


def _get_native_hi_cols() -> list[str]:
    """64 HI column names in native seg format (no seg suffix, stat_q_abs/stat_energy_seg excluded)."""
    _STAT_EXCLUDE = {"q_abs", "energy_seg"}
    cols: list[str] = []
    for key in STAT_KEYS:
        if key in _STAT_EXCLUDE:
            continue
        cols.append(f"stat_{key}")
    for key in DIFF_KEYS:
        cols.append(f"diff_{key}")
    for key in LFP_KEYS:
        cols.append(f"lfp_{key}")
    for key in MORPH_KEYS:
        cols.append(f"morph_{key}")
    return cols


_NATIVE_HI_COLS: list[str] = _get_native_hi_cols()
_SCEN_TO_SEG: dict[int, str] = {code: seg for seg, (code, _) in SCEN_MAP.items()}


def load_dataset_native_seg(
    seg_data_dir: Path,
    datasets: Sequence[str],
    wide_data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load native segment-format pkls (one row = one segment).

    capacity_Ah in native seg = stat_q_abs_{seg} (partial Ah per segment),
    NOT the total cycle capacity. The correct SOH target is loaded from the
    corresponding wide pkl in wide_data_dir if provided.

    HI columns are mapped to hi_00..hi_64 in the same order as _wide_to_segments.
    """
    try:
        from utils.compat import install_numpy2_shim
        install_numpy2_shim()
    except ImportError:
        pass
    import pickle

    all_dfs: list[pd.DataFrame] = []
    for ds in datasets:
        ds_seg_dir = seg_data_dir / ds
        if not ds_seg_dir.exists():
            continue
        wide_ds_dir = (wide_data_dir / ds) if wide_data_dir is not None else None

        for pkl_path in sorted(ds_seg_dir.glob("*.pkl")):
            cell_id = pkl_path.stem
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

            # Replace per-segment capacity_Ah with cycle-level total capacity
            cycle_cap: dict[int, float] | None = None
            if wide_ds_dir is not None:
                wide_pkl = wide_ds_dir / f"{cell_id}.pkl"
                if wide_pkl.exists():
                    with open(wide_pkl, "rb") as fw:
                        wd = pickle.load(fw)
                    wide_df = wd if isinstance(wd, pd.DataFrame) else pd.DataFrame(wd)
                    if "capacity_Ah" in wide_df.columns and "cycle" in wide_df.columns:
                        cycle_cap = wide_df.set_index("cycle")["capacity_Ah"].to_dict()

            if cycle_cap is not None:
                df["capacity_Ah"] = df["cycle"].map(cycle_cap)
            else:
                print(f"[dataset] WARNING: no wide pkl for {ds}/{cell_id}, using segment capacity_Ah")

            # Add metadata columns from scen/segment_id
            df["seg_name"]  = df["scen"].map(_SCEN_TO_SEG)
            df["seg_idx"]   = df["segment_id"].astype(np.int64)
            df["direction"] = df["scen"].apply(lambda x: 1.0 if x > 0 else -1.0).astype(np.float32)
            df["level"]     = (df["scen"].abs() - 1).astype(np.int64)

            # Map native HI cols → hi_00..hi_64 (exclude stat_q_abs)
            available = [c for c in _NATIVE_HI_COLS if c in df.columns]
            rename_map = {old: f"hi_{i:02d}" for i, old in enumerate(available)}
            df = df.rename(columns=rename_map)
            for i in range(len(available), N_HI):
                df[f"hi_{i:02d}"] = np.nan

            keep = (["cell_id", "cycle", "seg_name", "seg_idx", "scen",
                     "direction", "level", "capacity_Ah"]
                    + [f"hi_{i:02d}" for i in range(N_HI)])
            df = df[keep].dropna(subset=["capacity_Ah"])
            df["dataset"] = ds
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Normalizer (fitted on train split)
# ---------------------------------------------------------------------------

class SegmentNormalizer:
    """Z-score normalization per HI feature; NaN → 0.0 after z-score."""

    def __init__(self):
        self.mean_: np.ndarray | None = None  # (N_HI,)
        self.std_: np.ndarray | None = None   # (N_HI,)
        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0

    @property
    def hi_cols(self) -> list[str]:
        return [f"hi_{i:02d}" for i in range(N_HI)]

    def fit(self, df: pd.DataFrame) -> "SegmentNormalizer":
        x = df[self.hi_cols].values.astype(np.float64)
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # constant columns → no-op

        cap = df["capacity_Ah"].values.astype(np.float64)
        self.target_mean_ = float(np.nanmean(cap))
        self.target_std_ = float(np.nanstd(cap))
        if self.target_std_ < 1e-8:
            self.target_std_ = 1.0
        return self

    def transform_x(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.hi_cols].values.astype(np.float64)
        nan_mask = np.isnan(x)
        x = (x - self.mean_) / self.std_
        x[nan_mask] = 0.0
        return x.astype(np.float32), (~nan_mask).astype(np.float32)

    def transform_target(self, cap: np.ndarray) -> np.ndarray:
        return ((cap - self.target_mean_) / self.target_std_).astype(np.float32)

    def inverse_target(self, cap_norm: np.ndarray) -> np.ndarray:
        return cap_norm * self.target_std_ + self.target_mean_


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

HI_COLS = [f"hi_{i:02d}" for i in range(N_HI)]


class SegmentDataset(Dataset):
    """
    One sample = one (cell, cycle, segment) triple.

    Tensors:
      x_hi       : (N_HI,)   normalised HI features [float32]
      nan_mask   : (N_HI,)   1.0 = valid, 0.0 = was NaN [float32]
      direction  : scalar     +1.0 or -1.0 [float32]
      level      : scalar     0/1/2 int64  (ground truth for classifier loss)
      seg_idx    : scalar     0-5 int64
      target     : scalar     normalised capacity_Ah [float32]
      dataset_id : scalar     데이터셋 인덱스 (datasets 리스트 순서, 0-based) [float32]
      cap_init   : scalar     정규화된 초기/정격 용량 [float32]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        normalizer: SegmentNormalizer,
        fit_normalizer: bool = False,
        data_cfg: dict | None = None,
    ):
        if fit_normalizer:
            normalizer.fit(df)

        x, mask = normalizer.transform_x(df)
        cap_norm = normalizer.transform_target(df["capacity_Ah"].values)

        self.x_hi = torch.from_numpy(x)
        self.nan_mask = torch.from_numpy(mask)
        self.direction = torch.tensor(df["direction"].values, dtype=torch.float32)
        self.level = torch.tensor(df["level"].values, dtype=torch.long)
        self.seg_idx = torch.tensor(df["seg_idx"].values, dtype=torch.long)
        self.target = torch.tensor(cap_norm, dtype=torch.float32)

        # ----------------------------------------------------------------
        # 메타 스칼라: dataset_id + cap_init
        # ----------------------------------------------------------------
        cfg = data_cfg or {}
        datasets_list = cfg.get("datasets", sorted(df["dataset"].unique().tolist()))

        # dataset_id: 데이터셋 순서 인덱스를 [0, 1] 범위로 정규화
        n_ds = max(len(datasets_list) - 1, 1)
        ds_to_id = {ds: float(i) / n_ds for i, ds in enumerate(datasets_list)}
        self.dataset_id = torch.tensor(
            df["dataset"].map(ds_to_id).fillna(0.0).values.astype(np.float32),
            dtype=torch.float32,
        )

        # cap_init: 초기/정격 용량 (target 정규화 기준 동일 적용)
        if cfg.get("use_initial_capacity", False):
            # 셀별 최소 사이클의 capacity_Ah → 첫 사이클 실측 용량
            first_cap = (
                df.sort_values("cycle")
                .groupby("cell_id")["capacity_Ah"]
                .first()
            )
            cap_init_raw = df["cell_id"].map(first_cap).values.astype(np.float32)
        else:
            nominal = cfg.get("nominal_capacities", {})
            cap_init_raw = df["dataset"].map(nominal).fillna(1.0).values.astype(np.float32)

        cap_init_norm = normalizer.transform_target(cap_init_raw)
        self.cap_init = torch.tensor(cap_init_norm, dtype=torch.float32)

        # metadata (not returned by __getitem__, but useful for evaluation)
        self.cell_ids = df["cell_id"].values.tolist()
        self.cycles = df["cycle"].values.tolist()
        self.seg_names = df["seg_name"].values.tolist()
        self.capacity_raw = df["capacity_Ah"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_hi":      self.x_hi[idx],
            "nan_mask":  self.nan_mask[idx],
            "direction": self.direction[idx],
            "level":     self.level[idx],
            "seg_idx":   self.seg_idx[idx],
            "target":    self.target[idx],
            "cap_init":  self.cap_init[idx],
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ---------------------------------------------------------------------------
# Top-level builder used by train_scr.py
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict) -> tuple[SegmentDataset, SegmentDataset, SegmentDataset, SegmentNormalizer]:
    """
    Load data → split → build train/val/test SegmentDatasets.

    is_cross_dataset_evaluate=false (default):
        셀 단위 random split. datasets 전체에서 train/val/test 분리.
    is_cross_dataset_evaluate=true:
        datasets[0]을 train/val 소스, datasets[1]을 test 소스로 사용.
        normalizer는 datasets[0] train 셀 기준으로만 fit.
    """
    data_cfg      = cfg["data"]
    root          = PROJECT_ROOT
    seg_dir       = root / data_cfg["seg_data_dir"]
    wide_dir      = root / data_cfg["data_dir"]
    datasets_list = data_cfg["datasets"]
    is_cross      = data_cfg.get("is_cross_dataset_evaluate", False)

    # ------------------------------------------------------------------
    # Data loading (공통)
    # ------------------------------------------------------------------
    native_df = pd.DataFrame()
    if seg_dir.exists():
        native_df = load_dataset_native_seg(seg_dir, datasets_list, wide_dir)

    if len(native_df) > 0:
        df = native_df
    else:
        df = load_dataset_wide(
            wide_dir, datasets_list,
            min_cycles=data_cfg.get("min_cycles_per_cell", 10),
        )

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    train_ratio = data_cfg.get("train_ratio", 0.6)
    val_ratio   = data_cfg.get("val_ratio",   0.2)
    seed        = data_cfg.get("split_seed",  42)

    if is_cross:
        if len(datasets_list) < 2:
            raise ValueError(
                "is_cross_dataset_evaluate=true requires at least 2 datasets in config"
            )
        train_src = datasets_list[0]
        test_src  = datasets_list[1]

        df_tv   = df[df["dataset"] == train_src].reset_index(drop=True)
        df_test = df[df["dataset"] == test_src].reset_index(drop=True)

        if len(df_tv) == 0:
            raise RuntimeError(f"[dataset] train source '{train_src}' has no data")
        if len(df_test) == 0:
            raise RuntimeError(f"[dataset] test source '{test_src}' has no data")

        # train/val 비율을 train_src 내에서 재정규화 (test_ratio 부분 제외)
        tv_total    = train_ratio + val_ratio
        adj_train   = train_ratio / tv_total
        adj_val     = val_ratio   / tv_total

        tv_cells = df_tv["cell_id"].unique().tolist()
        train_cells, val_cells, _ = split_cells(
            tv_cells,
            train_ratio=adj_train,
            val_ratio=adj_val,
            seed=seed,
        )
        test_cells = df_test["cell_id"].unique().tolist()

        train_df = df_tv[df_tv["cell_id"].isin(train_cells)].reset_index(drop=True)
        val_df   = df_tv[df_tv["cell_id"].isin(val_cells)].reset_index(drop=True)
        test_df  = df_test

        print(f"[dataset] cross-dataset: train/val={train_src} | test={test_src}")
    else:
        cell_ids = df["cell_id"].unique().tolist()
        train_cells, val_cells, test_cells = split_cells(
            cell_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        train_df = df[df["cell_id"].isin(train_cells)].reset_index(drop=True)
        val_df   = df[df["cell_id"].isin(val_cells)].reset_index(drop=True)
        test_df  = df[df["cell_id"].isin(test_cells)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build datasets (normalizer는 train 기준 fit)
    # ------------------------------------------------------------------
    norm     = SegmentNormalizer()
    train_ds = SegmentDataset(train_df, norm, fit_normalizer=True,  data_cfg=data_cfg)
    val_ds   = SegmentDataset(val_df,   norm, fit_normalizer=False, data_cfg=data_cfg)
    test_ds  = SegmentDataset(test_df,  norm, fit_normalizer=False, data_cfg=data_cfg)

    print(f"[dataset] cells  train={len(train_cells)} val={len(val_cells)} test={len(test_cells)}")
    print(f"[dataset] segs   train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    return train_ds, val_ds, test_ds, norm
