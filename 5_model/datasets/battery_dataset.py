"""Battery HI dataset loader and cell-level train/val/test splitter.

Each pkl under _4_data_hi/{MIT,HUST}/ is a per-cell DataFrame:
  rows = cycles
  cols = [meta(4)] + [global HIs(15)] + [segment HIs(396)]

The split is performed at the cell level (not cycle level) to prevent
data leakage across cells in generalization evaluation.
"""

from __future__ import annotations
import pathlib
import pickle
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Must be imported before any pickle.load() to enable numpy 1.x/2.x compatibility
import utils.compat  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tqdm_utils import tqdm

from utils.hi_groups import HIGroupInfo, META_COLS
from datasets.normalization import FeatureNormalizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform-safe DataLoader worker count
# ---------------------------------------------------------------------------

def _safe_num_workers(requested: int) -> int:
    """Return a safe num_workers for DataLoader.

    Windows spawn-based multiprocessing is unreliable inside conda run or
    when called without a proper __main__ guard.  Default to 0 when the
    interpreter is in a non-standard context, otherwise honour the request.
    """
    if sys.platform == "win32" and requested > 0:
        # Allow >0 only when running as __main__ (i.e., python script.py)
        if not any("train.py" in a or "test.py" in a or "predict.py" in a
                   for a in sys.argv):
            return 0
    return requested


# ---------------------------------------------------------------------------
# Per-cell loader (used in parallel)
# ---------------------------------------------------------------------------

def _load_cell(
    pkl_path: pathlib.Path,
    dataset_name: str,
    min_cycles: int,
) -> Optional[pd.DataFrame]:
    """Load one cell pkl file and return a DataFrame (or None if too short)."""
    with open(pkl_path, "rb") as f:
        df: pd.DataFrame = pickle.load(f)

    if len(df) < min_cycles:
        return None

    df = df.copy()
    if "dataset" not in df.columns:
        df.insert(0, "dataset", dataset_name)
    if "cell_id" not in df.columns:
        df.insert(1, "cell_id", pkl_path.stem)
    return df


# ---------------------------------------------------------------------------
# Data loading  (parallelised with ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def load_all_cells(
    data_dir: str | pathlib.Path,
    datasets: List[str] = ("MIT", "HUST"),
    min_cycles: int = 10,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Load all per-cell pkl files in parallel and return a concatenated DataFrame.

    I/O-bound loading is parallelised with a ThreadPoolExecutor.
    A tqdm bar shows per-cell progress.

    Args:
        data_dir:    Root directory containing MIT/ and HUST/ subdirectories.
        datasets:    Which dataset sub-folders to scan.
        min_cycles:  Minimum number of cycles required to keep a cell.
        num_workers: Number of parallel I/O threads.
    """
    root = pathlib.Path(data_dir)

    # Collect all (path, dataset_name) tasks
    tasks: List[Tuple[pathlib.Path, str]] = []
    for ds in datasets:
        ds_dir = root / ds
        if not ds_dir.exists():
            logger.warning("Dataset directory not found: %s", ds_dir)
            continue
        pkls = sorted(ds_dir.glob("*.pkl"))
        if not pkls:
            logger.warning("No pkl files in %s", ds_dir)
            continue
        tasks.extend((p, ds) for p in pkls)

    if not tasks:
        raise FileNotFoundError(f"No pkl files found under {data_dir}")

    frames: List[pd.DataFrame] = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(_load_cell, path, ds, min_cycles): (path, ds)
            for path, ds in tasks
        }
        pbar = tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Loading cells",
            unit="cell",
            leave=True,
        )
        for future in pbar:
            path, ds = future_map[future]
            try:
                df = future.result()
            except Exception as exc:
                logger.error("Failed to load %s: %s", path, exc)
                skipped += 1
                continue

            if df is None:
                skipped += 1
            else:
                frames.append(df)
                pbar.set_postfix({"loaded": len(frames), "skip": skipped})

    if not frames:
        raise FileNotFoundError(f"No data loaded from {data_dir}")

    combined = pd.concat(frames, ignore_index=True)

    ds_counts = {ds: combined[combined["dataset"] == ds]["cell_id"].nunique()
                 for ds in datasets}
    logger.info(
        "Loaded %d cycles from %d cells  |  %s",
        len(combined),
        combined["cell_id"].nunique(),
        "  ".join(f"{ds}:{n}" for ds, n in ds_counts.items()),
    )
    return combined


# ---------------------------------------------------------------------------
# Cell-level split
# ---------------------------------------------------------------------------

def split_cells(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_dataset: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split at cell level (not cycle level).

    Stratification ensures each dataset is proportionally represented
    in train/val/test.
    """
    rng = np.random.RandomState(seed)

    def _split_group(cell_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(cell_ids)
        idx = rng.permutation(n)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        return (
            cell_ids[idx[:n_train]],
            cell_ids[idx[n_train: n_train + n_val]],
            cell_ids[idx[n_train + n_val:]],
        )

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    if stratify_by_dataset:
        for ds in df["dataset"].unique():
            cells = df[df["dataset"] == ds]["cell_id"].unique()
            tr, va, te = _split_group(cells)
            train_ids.extend(tr)
            val_ids.extend(va)
            test_ids.extend(te)
    else:
        all_cells = df["cell_id"].unique()
        tr, va, te = _split_group(all_cells)
        train_ids, val_ids, test_ids = list(tr), list(va), list(te)

    train_df = df[df["cell_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["cell_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["cell_id"].isin(test_ids)].reset_index(drop=True)

    logger.info(
        "Split  train: %d cells / %d cycles  |  "
        "val: %d cells / %d cycles  |  test: %d cells / %d cycles",
        len(train_ids), len(train_df),
        len(val_ids),   len(val_df),
        len(test_ids),  len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BatteryHIDataset(Dataset):
    """PyTorch Dataset for pre-computed HI features.

    All tensors are pre-built in __init__ so __getitem__ is a pure index
    operation with no I/O or Python overhead — num_workers>0 is safe.

    Each item contains:
      x_global:       (n_global,) normalised global HIs
      group_features: list of K (n_k,) tensors per segment-category group
      nan_masks:      list of K (n_k,) float tensors  (1=valid, 0=was NaN)
      target:         scalar capacity_Ah
    """

    def __init__(
        self,
        df: pd.DataFrame,
        hi_info: HIGroupInfo,
        normalizer: FeatureNormalizer,
    ) -> None:
        feature_groups = {"__global__": hi_info.global_his}
        for seg, cat in hi_info.group_keys:
            name = hi_info.group_name(seg, cat)
            feature_groups[name] = hi_info.groups[name]

        normed, masks = normalizer.transform(df, feature_groups)

        self._x_global = torch.tensor(normed["__global__"], dtype=torch.float32)

        self._group_features: List[torch.Tensor] = []
        self._nan_masks: List[torch.Tensor] = []
        for seg, cat in hi_info.group_keys:
            name = hi_info.group_name(seg, cat)
            self._group_features.append(torch.tensor(normed[name], dtype=torch.float32))
            self._nan_masks.append(torch.tensor(masks[name], dtype=torch.float32))

        self._targets  = torch.tensor(df["capacity_Ah"].values, dtype=torch.float32)
        self._cell_ids = df["cell_id"].values
        self._datasets = df["dataset"].values
        self._cycles   = df["cycle"].values.astype(np.int32)
        self._n_groups = len(hi_info.group_keys)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "x_global":      self._x_global[idx],
            "group_features": [gf[idx] for gf in self._group_features],
            "nan_masks":      [m[idx]  for m  in self._nan_masks],
            "target":         self._targets[idx],
            "cell_id":        self._cell_ids[idx],
            "dataset":        self._datasets[idx],
            "cycle":          int(self._cycles[idx]),
        }


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """Stack per-sample list tensors into batched tensors."""
    n_groups = len(batch[0]["group_features"])
    return {
        "x_global": torch.stack([b["x_global"] for b in batch]),
        "group_features": [
            torch.stack([b["group_features"][k] for b in batch])
            for k in range(n_groups)
        ],
        "nan_masks": [
            torch.stack([b["nan_masks"][k] for b in batch])
            for k in range(n_groups)
        ],
        "target":  torch.stack([b["target"]  for b in batch]),
        "cell_id": [b["cell_id"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "cycle":   [b["cycle"]   for b in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    data_dir: str | pathlib.Path,
    hi_info: HIGroupInfo,
    cfg: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureNormalizer]:
    """End-to-end: load → split → normalise → wrap in DataLoaders.

    Returns (train_loader, val_loader, test_loader, normalizer).
    """
    io_workers = cfg["data"].get("io_workers", 4)

    df = load_all_cells(
        data_dir,
        datasets=cfg["data"]["datasets"],
        min_cycles=cfg["data"].get("min_cycles_per_cell", 10),
        num_workers=io_workers,
    )

    train_df, val_df, test_df = split_cells(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["split_seed"],
    )

    feature_groups = {"__global__": hi_info.global_his}
    for seg, cat in hi_info.group_keys:
        name = hi_info.group_name(seg, cat)
        feature_groups[name] = hi_info.groups[name]

    normalizer = FeatureNormalizer()
    normalizer.fit(train_df, feature_groups)

    batch_size  = cfg["training"]["batch_size"]
    dl_workers  = _safe_num_workers(cfg["data"].get("num_workers", 0))
    pin_memory  = torch.cuda.is_available() and dl_workers > 0

    def _make_loader(df_split: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = BatteryHIDataset(df_split, hi_info, normalizer)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=dl_workers,
            pin_memory=pin_memory,
            persistent_workers=(dl_workers > 0),
            drop_last=False,
        )

    return (
        _make_loader(train_df, shuffle=True),
        _make_loader(val_df,   shuffle=False),
        _make_loader(test_df,  shuffle=False),
        normalizer,
    )
