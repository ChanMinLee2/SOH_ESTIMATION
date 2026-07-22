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
    get_hi_cols_for_seg, get_hi_cost_vector, N_HI, spec_from_qfrac,
    RAW_N, RAW_CH,
)

_proj_root = Path(__file__).resolve().parent.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))
from common.scenario.base import ScenarioSpec

_DEFAULT_SPEC: ScenarioSpec = spec_from_qfrac()


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


def _wide_to_segments(
    df: pd.DataFrame,
    cell_id: str,
    spec: ScenarioSpec | None = None,
) -> pd.DataFrame:
    """
    Reshape one cell's wide DataFrame (one row = one cycle) into
    segment-level DataFrame (one row = one segment).

    Output columns:
      cell_id, cycle, seg_name, seg_idx, scen, direction, level,
      capacity_Ah, hi_00 .. hi_64
    """
    _spec = spec or _DEFAULT_SPEC
    rows = []
    for _, cycle_row in df.iterrows():
        cycle_num = int(cycle_row.get("cycle", cycle_row.name))
        cap_ah = float(cycle_row["capacity_Ah"])

        for seg_idx, seg in enumerate(_spec.scenario_names):
            hi_cols = get_hi_cols_for_seg(seg)
            if not all(c in cycle_row.index for c in hi_cols):
                continue

            hi_vals = cycle_row[hi_cols].values.astype(np.float32)
            dir_idx, latent_class = _spec.scenario_to_dir_class(seg_idx)
            direction_val = 1 if dir_idx == 0 else -1
            level = latent_class

            rows.append({
                "cell_id":     cell_id,
                "cycle":       cycle_num,
                "seg_name":    seg,
                "seg_idx":     seg_idx,
                "scen":        seg_idx,   # legacy field; == scenario_id
                "direction":   direction_val,
                "level":       level,
                "capacity_Ah": cap_ah,
                **{f"hi_{i:02d}": v for i, v in enumerate(hi_vals)},
            })

    return pd.DataFrame(rows)


def load_dataset_wide(
    data_dir: Path,
    datasets: Sequence[str],
    min_cycles: int = 10,
    spec: ScenarioSpec | None = None,
) -> pd.DataFrame:
    """
    Load wide pkl files for all datasets and return a combined segment-level DataFrame.

    data_dir: e.g. Path("_4_data_hi")
    datasets: e.g. ["MIT", "HUST"]
    spec: ScenarioSpec to use for segment routing (default: qfrac)
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
            seg_df = _wide_to_segments(df, cell_id, spec=spec)
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


def load_dataset_native_seg(
    seg_data_dir: Path,
    datasets: Sequence[str],
    wide_data_dir: Path | None = None,
    spec: ScenarioSpec | None = None,
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

            # Add metadata columns from segment_id via spec
            _spec = spec or _DEFAULT_SPEC
            _seg_names  = _spec.scenario_names
            _id_to_name = {i: n for i, n in enumerate(_seg_names)}
            _id_to_dir  = {i: (1 if _spec.scenario_to_dir_class(i)[0] == 0 else -1)
                           for i in range(_spec.n_scenarios)}
            _id_to_lvl  = {i: _spec.scenario_to_dir_class(i)[1]
                           for i in range(_spec.n_scenarios)}
            df["seg_idx"]   = df["segment_id"].astype(np.int64)
            df["seg_name"]  = df["seg_idx"].map(_id_to_name)
            df["direction"] = df["seg_idx"].map(_id_to_dir).astype(np.float32)
            df["level"]     = df["seg_idx"].map(_id_to_lvl).astype(np.int64)

            # Map native HI cols → hi_00..hi_64 (exclude stat_q_abs)
            available = [c for c in _NATIVE_HI_COLS if c in df.columns]
            rename_map = {old: f"hi_{i:02d}" for i, old in enumerate(available)}
            df = df.rename(columns=rename_map)
            for i in range(len(available), N_HI):
                df[f"hi_{i:02d}"] = np.nan

            keep = (["cell_id", "cycle", "seg_name", "seg_idx", "scen",
                     "direction", "level", "capacity_Ah"]
                    + [f"hi_{i:02d}" for i in range(N_HI)])
            # 원시 곡선 컬럼(raw_v/raw_i)이 있으면 함께 보존 (CNN 입력용)
            raw_cols = [c for c in ("raw_v", "raw_i") if c in df.columns]
            df = df[keep + raw_cols].dropna(subset=["capacity_Ah"])
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
        self.cap_init_mean_: float = 0.0   # cap_init z-score 전용 (SOH target과 무관)
        self.cap_init_std_: float = 1.0

    @property
    def hi_cols(self) -> list[str]:
        return [f"hi_{i:02d}" for i in range(N_HI)]

    def fit(self, df: pd.DataFrame) -> "SegmentNormalizer":
        import warnings
        x = df[self.hi_cols].values.astype(np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nanmean(x, axis=0)
            self.std_ = np.nanstd(x, axis=0)
        all_nan = np.isnan(self.mean_)  # 학습 데이터에서 전부 NaN인 컬럼
        self.mean_[all_nan] = 0.0       # no-op 처리 (x - 0)
        self.std_[all_nan] = 1.0        # no-op 처리 (/ 1)
        self.std_[self.std_ < 1e-8] = 1.0  # constant columns → no-op

        # cap_init_mean_/cap_init_std_: cap_init z-scoring 전용 (SOH target은 정규화 불필요)
        cap = df["capacity_Ah"].values.astype(np.float64)
        self.cap_init_mean_ = float(np.nanmean(cap))
        self.cap_init_std_ = float(np.nanstd(cap))
        if self.cap_init_std_ < 1e-8:
            self.cap_init_std_ = 1.0
        return self

    def transform_x(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = df[self.hi_cols].values.astype(np.float64)
        nan_mask = np.isnan(x)
        x = (x - self.mean_) / self.std_
        x[nan_mask] = 0.0
        return x.astype(np.float32), (~nan_mask).astype(np.float32)

    def transform_cap_init(self, cap: np.ndarray) -> np.ndarray:
        """cap_init Ah → z-scored (모델 conditioning 입력용)."""
        return ((cap - self.cap_init_mean_) / self.cap_init_std_).astype(np.float32)

    def inverse_cap_init(self, cap_norm: np.ndarray) -> np.ndarray:
        return cap_norm * self.cap_init_std_ + self.cap_init_mean_


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

HI_COLS = [f"hi_{i:02d}" for i in range(N_HI)]


def _stack_raw_col(series, n: int) -> np.ndarray:
    """object 컬럼(list/ndarray per row) → (n, RAW_N) float32.
    누락(NaN/스칼라)·길이불일치는 zero-pad/truncate 로 방어."""
    out = np.zeros((n, RAW_N), np.float32)
    for idx, val in enumerate(series.values):
        if isinstance(val, (list, np.ndarray)):
            arr = np.asarray(val, dtype=np.float32).ravel()
            m = min(len(arr), RAW_N)
            out[idx, :m] = arr[:m]
        # else: NaN/스칼라 → zero (원시 곡선 없는 세그먼트)
    return out


def _build_raw_tensor(df: pd.DataFrame) -> torch.Tensor:
    """DataFrame → x_raw 텐서 (N, RAW_CH, RAW_N).
    raw_v/raw_i 컬럼이 없으면(구 pkl / wide 포맷) zero 텐서로 fallback."""
    n = len(df)
    if "raw_v" in df.columns and "raw_i" in df.columns:
        rv = _stack_raw_col(df["raw_v"], n)   # (n, RAW_N)
        ri = _stack_raw_col(df["raw_i"], n)   # (n, RAW_N)
    else:
        rv = np.zeros((n, RAW_N), np.float32)
        ri = np.zeros((n, RAW_N), np.float32)
    raw = np.stack([rv, ri], axis=1)          # (n, RAW_CH=2, RAW_N)
    return torch.from_numpy(raw)


class SegmentDataset(Dataset):
    """
    One sample = one (cell, cycle, segment) triple.

    Tensors:
      x_hi       : (N_HI,)   normalised HI features [float32]
      nan_mask   : (N_HI,)   1.0 = valid, 0.0 = was NaN [float32]
      direction  : scalar     +1.0 or -1.0 [float32]
      level      : scalar     0/1/2 int64  (ground truth for classifier loss)
      seg_idx    : scalar     0-5 int64
      target     : scalar     SOH ratio = capacity_Ah / cap_init_Ah ∈ (0, 1] [float32]
      dataset_id : scalar     데이터셋 인덱스 (datasets 리스트 순서, 0-based) [float32]
      cap_init   : scalar     z-scored 초기/정격 용량 Ah (모델 conditioning용) [float32]

    Attributes (not in __getitem__):
      cap_init_raw  : (N,) float32 numpy — 초기/정격 용량 Ah (평가 시 SOH→Ah 변환용)
      capacity_raw  : (N,) float32 numpy — 실측 capacity_Ah
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
        cap_raw = df["capacity_Ah"].values.astype(np.float32)

        self.x_hi = torch.from_numpy(x)
        self.nan_mask = torch.from_numpy(mask)
        self.x_raw = _build_raw_tensor(df)   # (N, RAW_CH, RAW_N) — CNN 입력 (구 pkl → zeros)
        self.direction = torch.tensor(df["direction"].values, dtype=torch.float32)
        self.level = torch.tensor(df["level"].values, dtype=torch.long)
        self.seg_idx = torch.tensor(df["seg_idx"].values, dtype=torch.long)

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

        # cap_init: 초기/정격 용량 raw Ah
        if cfg.get("use_initial_capacity", False):
            first_cap = (
                df.sort_values("cycle")
                .groupby("cell_id")["capacity_Ah"]
                .first()
            )
            cap_init_raw = df["cell_id"].map(first_cap).values.astype(np.float32)
        else:
            nominal = cfg.get("nominal_capacities", {})
            cap_init_raw = df["dataset"].map(nominal).fillna(1.0).values.astype(np.float32)

        # target: SOH ratio (dataset-agnostic, 정규화 불필요)
        soh = cap_raw / np.where(cap_init_raw > 0, cap_init_raw, 1.0)
        self.target = torch.tensor(soh, dtype=torch.float32)

        # cap_init 모델 입력: z-scored Ah → 셀 크기 conditioning (cross-dataset 시 구분 정보)
        cap_init_norm = normalizer.transform_cap_init(cap_init_raw)
        self.cap_init = torch.tensor(cap_init_norm, dtype=torch.float32)

        # metadata (not returned by __getitem__, but useful for evaluation)
        self.cap_init_raw = cap_init_raw                          # SOH→Ah 변환용
        self.cell_ids = df["cell_id"].values.tolist()
        self.cycles = df["cycle"].values.tolist()
        self.seg_names = df["seg_name"].values.tolist()
        self.capacity_raw = cap_raw

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_hi":      self.x_hi[idx],
            "x_raw":     self.x_raw[idx],
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


def filter_dataset_by_cells(ds: "SegmentDataset", cell_ids: list[str]) -> "SegmentDataset":
    """SegmentDataset에서 지정된 cell_id 행만 추출해 새 Dataset 반환."""
    cell_set = set(cell_ids)
    indices  = [i for i, c in enumerate(ds.cell_ids) if c in cell_set]
    return _subset_dataset(ds, indices)


def _subset_dataset(ds: "SegmentDataset", indices: list[int]) -> "SegmentDataset":
    new_ds = object.__new__(SegmentDataset)
    new_ds.x_hi        = ds.x_hi[indices]
    new_ds.x_raw        = ds.x_raw[indices]
    new_ds.nan_mask     = ds.nan_mask[indices]
    new_ds.direction    = ds.direction[indices]
    new_ds.level        = ds.level[indices]
    new_ds.seg_idx      = ds.seg_idx[indices]
    new_ds.target       = ds.target[indices]
    new_ds.cap_init     = ds.cap_init[indices]
    new_ds.dataset_id   = ds.dataset_id[indices]
    new_ds.cap_init_raw = ds.cap_init_raw[indices]
    new_ds.cell_ids     = [ds.cell_ids[i] for i in indices]
    new_ds.cycles       = [ds.cycles[i] for i in indices]
    new_ds.seg_names    = [ds.seg_names[i] for i in indices]
    new_ds.capacity_raw = ds.capacity_raw[indices]
    return new_ds


# ---------------------------------------------------------------------------
# Top-level builder used by train_scr.py
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: dict,
    spec: ScenarioSpec | None = None,
) -> tuple[SegmentDataset, SegmentDataset, SegmentDataset, SegmentNormalizer]:
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
        native_df = load_dataset_native_seg(seg_dir, datasets_list, wide_dir, spec=spec)

    if len(native_df) > 0:
        df = native_df
    else:
        df = load_dataset_wide(
            wide_dir, datasets_list,
            min_cycles=data_cfg.get("min_cycles_per_cell", 10),
            spec=spec,
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


def build_random_seg_dataset(
    seg_data_dir: Path,
    datasets: list[str],
    normalizer: SegmentNormalizer,
    data_cfg: dict,
    spec: ScenarioSpec | None = None,
) -> "SegmentDataset":
    """
    랜덤 세그먼트(test_rs 등) PKL을 로드해 SegmentDataset 반환.

    normalizer: 반드시 학습 체크포인트에서 복원한 것을 전달 (refit 없음).
    spec: 랜덤 세그먼트 데이터의 ScenarioSpec (test_rs → n_scenarios=2).
          None 이면 load_dataset_native_seg 기본값 사용.
    """
    root = PROJECT_ROOT
    seg_dir  = root / seg_data_dir
    wide_dir = seg_dir.parent / "cycle"   # test_rs/cycle/{MIT,HUST}/*.pkl

    df = load_dataset_native_seg(seg_dir, datasets, wide_dir, spec=spec)
    if len(df) == 0:
        raise RuntimeError(f"[build_random_seg_dataset] 데이터 없음: {seg_dir}")

    print(f"[dataset] random_seg: {len(df):,} 세그먼트 로드 ({seg_data_dir})")
    ds = SegmentDataset(df, normalizer, fit_normalizer=False, data_cfg=data_cfg)
    return ds
