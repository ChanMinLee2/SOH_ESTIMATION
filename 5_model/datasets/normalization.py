"""Feature normalization fitted on training cells only."""

from __future__ import annotations
import warnings
import numpy as np
import pickle
import pathlib
from typing import Dict, List, Tuple, Optional
import pandas as pd


class FeatureNormalizer:
    """StandardScaler per feature, fitted only on non-NaN training values.

    After fitting, NaN values are imputed to 0.0 (which equals the mean
    after z-scoring), preserving the NaN mask separately.
    """

    def __init__(self) -> None:
        self._means: Optional[Dict[str, np.ndarray]] = None
        self._stds: Optional[Dict[str, np.ndarray]] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> "FeatureNormalizer":
        """Fit mean/std from training DataFrame for each feature group.

        feature_groups: dict of group_name → list of column names.
                        Also includes key '__global__' for global HIs.
        """
        self._means = {}
        self._stds = {}

        for group_name, cols in feature_groups.items():
            vals = df[cols].values.astype(np.float64)  # (N, n_feat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means = np.nanmean(vals, axis=0)
                stds = np.nanstd(vals, axis=0)
            # Columns that are entirely NaN: fall back to mean=0, std=1
            means = np.where(np.isnan(means), 0.0, means)
            stds = np.where(np.isnan(stds) | (stds < 1e-8), 1.0, stds)
            self._means[group_name] = means.astype(np.float32)
            self._stds[group_name] = stds.astype(np.float32)

        self._fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Transform features and return (normalized_arrays, nan_masks).

        Returns:
            normed:   dict of group_name → float32 array (N, n_feat), NaN → 0.0
            masks:    dict of group_name → float32 array (N, n_feat), 1=valid 0=was_NaN
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        normed: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}

        for group_name, cols in feature_groups.items():
            vals = df[cols].values.astype(np.float32)  # (N, n_feat)
            mask = (~np.isnan(vals)).astype(np.float32)

            m = self._means[group_name]
            s = self._stds[group_name]
            vals_z = (vals - m) / s
            vals_z = np.nan_to_num(vals_z, nan=0.0)  # NaN → mean (=0 after scaling)

            normed[group_name] = vals_z
            masks[group_name] = mask

        return normed, masks

    def fit_transform(
        self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        self.fit(df, feature_groups)
        return self.transform(df, feature_groups)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"means": self._means, "stds": self._stds}, f)

    @classmethod
    def load(cls, path: pathlib.Path) -> "FeatureNormalizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._means = state["means"]
        obj._stds = state["stds"]
        obj._fitted = True
        return obj
