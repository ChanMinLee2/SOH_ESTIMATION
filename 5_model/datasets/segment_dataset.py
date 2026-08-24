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
    EXCLUDE_STAT_LEAK,
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

def _get_native_hi_cols() -> list[str]:
    """66 HI column names in native seg format (no seg suffix).

    2026-08-07: stat_q_abs/stat_energy_seg 포함(5_model/utils/hi_schema.py와 동일
    변경 — 이 함수가 그 파일의 _STAT_EXCLUDE 로직을 별도로 복제해서 갖고 있었음).
    2026-08-08: EXCLUDE_STAT_LEAK도 hi_schema.py와 동일하게 반영(SOH_EXCLUDE_STAT_LEAK=1).
    """
    _STAT_EXCLUDE: set[str] = {"q_abs", "energy_seg"} if EXCLUDE_STAT_LEAK else set()
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

    HI columns are mapped to hi_00..hi_64 via _get_native_hi_cols().
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

            # h_scen/h_intensity 보조손실 타깃 (docs/260803_RESULTS.md §10.8) — 이미
            # x_hi(hi_XX)에 포함되는 기존 HI를 그대로 재사용하므로 rename 전에 원본
            # 이름으로 복사해 보존한다(둘 다 세그먼트 내부 상대량이라 q_tot 불필요).
            if "lfp_plateau_frac" in df.columns:
                df["aux_scen_target"] = df["lfp_plateau_frac"]
            if "stat_i_std" in df.columns:
                df["aux_intensity_target"] = df["stat_i_std"]

            # Map native HI cols → hi_00..hi_64 (exclude stat_q_abs)
            available = [c for c in _NATIVE_HI_COLS if c in df.columns]
            rename_map = {old: f"hi_{i:02d}" for i, old in enumerate(available)}
            df = df.rename(columns=rename_map)
            for i in range(len(available), N_HI):
                df[f"hi_{i:02d}"] = np.nan

            _hi_cols = [f"hi_{i:02d}" for i in range(N_HI)]
            keep = (["cell_id", "cycle", "seg_name", "seg_idx", "scen",
                     "direction", "level", "capacity_Ah"]
                    + _hi_cols)
            # 원시 곡선 컬럼(raw_v/raw_i/raw_t)이 있으면 함께 보존 (CNN 입력용)
            raw_cols = [c for c in ("raw_v", "raw_i", "raw_t") if c in df.columns]
            aux_cols = [c for c in ("aux_scen_target", "aux_intensity_target") if c in df.columns]
            df = df[keep + raw_cols + aux_cols].dropna(subset=["capacity_Ah"])
            # HI 66개가 전부 NaN인 세그먼트 제외 — hi_correlation.py가 계산 자체를
            # 못한 경우(예: 충전 데이터 부족으로 q_tc < cap*0.6, _chg_incomplete)로,
            # SegmentNormalizer.fit()이 이미 nanmean/nanstd로 이런 행을 정규화 통계
            # 계산에서 건너뛰므로(segment_dataset.py:308-309) 제거해도 부작용이 없다.
            # "입력=0벡터, 타깃=실제 SOH"라는 학습 불가능한 잡음 쌍을 원천 배제한다.
            df = df.dropna(subset=_hi_cols, how="all")
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
    raw_v/raw_i/raw_t 컬럼이 없으면(구 pkl / wide 포맷) zero 텐서로 fallback.
    raw_t 컬럼만 없는 구 pkl(RAW_CH=2 시절 캐시)은 raw_t 채널만 0으로 채운다
    (docs/260803_RESULTS.md §10.10 — 하위호환)."""
    n = len(df)
    if "raw_v" in df.columns and "raw_i" in df.columns:
        rv = _stack_raw_col(df["raw_v"], n)   # (n, RAW_N)
        ri = _stack_raw_col(df["raw_i"], n)   # (n, RAW_N)
    else:
        rv = np.zeros((n, RAW_N), np.float32)
        ri = np.zeros((n, RAW_N), np.float32)
    if "raw_t" in df.columns:
        rt = _stack_raw_col(df["raw_t"], n)   # (n, RAW_N)
    else:
        rt = np.zeros((n, RAW_N), np.float32)
    raw = np.stack([rv, ri, rt], axis=1)      # (n, RAW_CH=3, RAW_N)
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

        # h_scen/h_intensity 보조손실 타깃 (docs/260803_RESULTS.md §10.8, Phase 1
        # CNN 학습 전용 — 구 pkl에 컬럼이 없으면 NaN → 학습 시 마스킹 처리)
        if "aux_scen_target" in df.columns:
            self.aux_scen_target = torch.tensor(
                df["aux_scen_target"].values.astype(np.float32), dtype=torch.float32)
        else:
            self.aux_scen_target = torch.full((len(df),), float("nan"), dtype=torch.float32)
        if "aux_intensity_target" in df.columns:
            self.aux_intensity_target = torch.tensor(
                df["aux_intensity_target"].values.astype(np.float32), dtype=torch.float32)
        else:
            self.aux_intensity_target = torch.full((len(df),), float("nan"), dtype=torch.float32)

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
            "aux_scen_target":      self.aux_scen_target[idx],
            "aux_intensity_target": self.aux_intensity_target[idx],
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class FastTensorLoader:
    """사전 구축된 (N,·) 텐서를 슬라이싱해 배치 dict를 생성하는 빠른 이터레이터.

    SegmentDataset은 모든 필드를 이미 (N,·) 텐서로 메모리에 올려두므로, 표준
    DataLoader의 per-sample ``__getitem__`` + ``collate_fn``(키마다 2048회 ``torch.stack``)
    오버헤드가 GPU를 굶긴다. 이 로더는 인덱스 슬라이싱 한 번으로 배치를 만들어
    그 오버헤드를 제거한다 (대규모 세그먼트 학습에서 에폭 시간 대폭 단축).

    SCRModel 회귀 forward(Phase 1/2)는 ``x_raw``를 사용하지 않으므로 기본 제외한다.
    CNN 분류기 학습 등 ``x_raw``가 필요하면 ``include_raw=True``. ``h_scen``/
    ``h_intensity`` 보조손실 타깃(docs/260803_RESULTS.md §10.8)이 필요하면
    ``include_aux=True`` (Phase 1 CNN 학습 전용).

    트레이너의 ``_to_device``가 배치를 GPU로 옮기므로 텐서는 CPU에 유지한다
    (배치당 1회 연속 전송 → per-sample stack보다 훨씬 빠르고 GPU 메모리 상주 없음).
    """

    _MODEL_KEYS = ["x_hi", "nan_mask", "direction", "level",
                   "seg_idx", "target", "cap_init"]

    def __init__(
        self,
        ds: "SegmentDataset",
        batch_size: int,
        shuffle: bool = False,
        include_raw: bool = False,
        include_aux: bool = False,
        drop_last: bool = False,
    ) -> None:
        keys = list(self._MODEL_KEYS)
        if include_raw:
            keys.insert(1, "x_raw")
        if include_aux:
            keys += ["aux_scen_target", "aux_intensity_target"]
        if hasattr(ds, "x_kernel"):
            # build_kernel_group_features.py 커널 융합 HI 블록(phase1_trainer_v2.py가
            # 학습 직전 ds.x_kernel로 붙여둠) — 있으면 자동으로 배치에 포함, 없으면
            # 기존과 완전히 동일 동작.
            keys.append("x_kernel")
        self.keys = keys
        self.tensors = {k: getattr(ds, k) for k in keys}
        self.n = len(ds)
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.bs
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        idx = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            sel = idx[i:i + self.bs]
            if self.drop_last and sel.numel() < self.bs:
                break
            yield {k: self.tensors[k][sel] for k in self.keys}


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
    new_ds.aux_scen_target      = ds.aux_scen_target[indices]
    new_ds.aux_intensity_target = ds.aux_intensity_target[indices]
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
    # Data loading (공통) — native seg pkl 전용.
    #
    # 2026-08-18(af9be9c) 시나리오 마지막 세그먼트만 살아남던 덮어쓰기 버그를 고치면서
    # 세그먼트별 HI(diff_dqdv_area_chg_lo 등)가 wide(사이클 단위) pkl에는 더 이상 저장되지
    # 않고 native seg pkl에만 저장되도록 바뀌었다. 예전엔 native seg pkl이 없으면 wide
    # pkl에서 세그먼트별 컬럼을 읽어 재구성하는 폴백(load_dataset_wide/_wide_to_segments)이
    # 있었지만, 그 컬럼 자체가 더 이상 wide pkl에 없으므로 그 폴백은 항상 빈 데이터만
    # 반환하다 아래와 무관한 "No data loaded" 에러로 죽었다 — 실제 원인(Step4 미실행)을
    # 전혀 알려주지 못해서 폴백 자체를 제거하고 아래처럼 조기에 명확한 에러를 낸다.
    # ------------------------------------------------------------------
    if not seg_dir.exists():
        raise RuntimeError(
            f"[dataset] native seg pkl 디렉터리가 없습니다: {seg_dir}\n"
            f"  이 axis/config 조합은 아직 Step4(네이티브 세그먼트 추출, 4_hi_analysis/"
            f"hi_correlation.py 또는 run_pipeline.py 4)를 실행하지 않은 것으로 보입니다.\n"
            f"  wide(사이클 단위) pkl에는 세그먼트별 HI가 더 이상 저장되지 않으므로 "
            f"(2026-08-18 이후) wide pkl로부터 재구성해 계속 진행할 수 없습니다 — "
            f"data_dir/datasets 설정 문제가 아니라 Step4 추출 여부를 먼저 확인하세요."
        )
    df = load_dataset_native_seg(seg_dir, datasets_list, wide_dir, spec=spec)
    if len(df) == 0:
        raise RuntimeError(
            f"[dataset] {seg_dir} 에 pkl 파일은 있지만 유효한 세그먼트 데이터가 0건입니다 — "
            f"datasets={datasets_list} 설정이 실제 저장된 데이터셋 폴더명(대소문자 포함)과 "
            f"일치하는지 확인하세요."
        )

    # 방향 필터 ("charge"|"discharge"|None) — direction: +1.0=충전, -1.0=방전
    direction_filter = data_cfg.get("direction_filter")
    if direction_filter and "direction" in df.columns:
        _dir_val = 1.0 if direction_filter == "charge" else -1.0
        _n_before = len(df)
        df = df[df["direction"] == _dir_val].reset_index(drop=True)
        print(f"[dataset] direction_filter={direction_filter!r} 적용: {_n_before:,} → {len(df):,} rows")

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
        forced_test = data_cfg.get("forced_test_cells")
        if forced_test:
            # 실험9(§8-5, docs/260810_RESULTS.md): datasets 구성이 다른 run들(풀링/MIT-only/
            # HUST-only) 간에 test 셀을 고정해 짝지어진(paired) 비교가 가능하도록 한다.
            # split_cells()는 그 run의 cell_ids 리스트 "전체"를 shuffle하므로, datasets가
            # 다르면 같은 seed를 써도 다른 셀이 뽑힌다(리스트 길이·내용이 달라지기 때문) —
            # forced_test_cells를 먼저 떼어내 이 문제를 우회한다. 이 필드가 없으면(기본값)
            # 아래 분기는 전혀 실행되지 않아 기존 동작과 100% 동일하다.
            forced_test_set = set(forced_test) & set(cell_ids)
            remaining = [c for c in cell_ids if c not in forced_test_set]
            tv_total  = train_ratio + val_ratio
            adj_train = train_ratio / tv_total
            adj_val   = val_ratio   / tv_total
            train_cells, val_cells, _ = split_cells(
                remaining,
                train_ratio=adj_train,
                val_ratio=adj_val,
                seed=seed,
            )
            test_cells = sorted(forced_test_set)
            print(f"[dataset] forced_test_cells 지정: {len(test_cells)}개 고정 test "
                  f"(전체 지정 {len(forced_test)}개 중 이 run에 존재하는 것만) — "
                  f"나머지 {len(remaining)}개를 train/val로 재분할")
        else:
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
