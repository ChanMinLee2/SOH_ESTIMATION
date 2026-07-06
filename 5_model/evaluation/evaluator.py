"""Evaluator: runs DFRModel on a DataLoader and saves metrics, predictions,
routing statistics, and figures to _5_data_model/.
"""

from __future__ import annotations
import logging
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.tqdm_utils import tqdm

from models.dfr_model import DFRModel
from utils.metrics import compute_metrics, routing_stats
from utils.visualization import plot_prediction_scatter, plot_routing_heatmap
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates a trained DFRModel and saves all outputs.

    Args:
        model:      Trained DFRModel.
        output_dir: Base output directory (_5_data_model/).
        device:     torch device string.
    """

    def __init__(
        self,
        model: DFRModel,
        output_dir: pathlib.Path,
        device: str = "cpu",
    ) -> None:
        self.model      = model
        self.device     = device
        self.output_dir = output_dir

        self.metrics_dir = output_dir / "metrics"
        self.pred_dir    = output_dir / "predictions"
        self.routing_dir = output_dir / "routing"
        self.fig_dir     = output_dir / "figures"
        for d in [self.metrics_dir, self.pred_dir, self.routing_dir, self.fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "test",
        metric_names: Optional[List[str]] = None,
    ) -> Dict:
        """Run evaluation with a tqdm progress bar.

        Args:
            loader:       DataLoader (test or val).
            split_name:   Label used in output filenames (e.g., 'test').
            metric_names: Subset of metrics to compute; None = all.

        Returns:
            Dict with 'metrics', 'routing', 'predictions'.
        """
        self.model.eval()
        self.model.to(self.device)

        all_pred, all_true = [], []
        all_gates = []
        all_cell_ids, all_datasets, all_cycles = [], [], []

        pbar = tqdm(
            loader,
            desc=f"Evaluating [{split_name}]",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch in pbar:
            x_global = batch["x_global"].to(self.device)
            gf = [t.to(self.device) for t in batch["group_features"]]
            nm = [t.to(self.device) for t in batch["nan_masks"]]
            target = batch["target"].to(self.device)

            batch_dev = {"x_global": x_global, "group_features": gf, "nan_masks": nm}
            pred, gates = self.model(batch_dev, hard=True)

            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())
            all_gates.append((gates > 0.5).float().cpu().numpy())
            all_cell_ids.extend(batch["cell_id"])
            all_datasets.extend(batch["dataset"])
            all_cycles.extend(batch["cycle"])

        y_pred    = np.concatenate(all_pred)
        y_true    = np.concatenate(all_true)
        gates_arr = np.concatenate(all_gates, axis=0)   # (N, n_groups)

        group_names = self.model.group_names()

        metrics = compute_metrics(y_true, y_pred, names=metric_names)
        r_stats = routing_stats(gates_arr, group_names)

        # Print summary line after bar
        summary = " | ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        logger.info("[%s] %s", split_name, summary)
        logger.info(
            "[%s] mean_active=%.1f/%d  sparsity=%.3f",
            split_name,
            r_stats["mean_active_groups"], len(group_names),
            r_stats["sparsity"],
        )

        pred_df = pd.DataFrame({
            "cell_id":       all_cell_ids,
            "dataset":       all_datasets,
            "cycle":         all_cycles,
            "capacity_true": y_true,
            "capacity_pred": y_pred,
            "error":         y_pred - y_true,
        })
        for i, name in enumerate(group_names):
            pred_df[f"gate_{name}"] = gates_arr[:, i]

        self._save_metrics(metrics, r_stats, split_name)
        self._save_predictions(pred_df, split_name)
        self._save_routing(gates_arr, group_names, split_name)
        self._save_figures(
            y_true, y_pred, metrics, gates_arr,
            group_names, np.array(all_datasets), split_name,
        )

        return {"metrics": metrics, "routing": r_stats, "predictions": pred_df}

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save_metrics(self, metrics: Dict, r_stats: Dict, split: str) -> None:
        save_json({**metrics, **r_stats}, self.metrics_dir / f"{split}_metrics.json")

    def _save_predictions(self, df: pd.DataFrame, split: str) -> None:
        df.to_csv(self.pred_dir / f"{split}_predictions.csv", index=False)

    def _save_routing(
        self, gates_arr: np.ndarray, group_names: List[str], split: str
    ) -> None:
        pd.DataFrame({
            "group": group_names,
            "activation_rate": gates_arr.mean(axis=0),
        }).to_csv(self.routing_dir / f"{split}_routing_stats.csv", index=False)

    def _save_figures(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict,
        gates_arr: np.ndarray,
        group_names: List[str],
        datasets: np.ndarray,
        split: str,
    ) -> None:
        try:
            plot_prediction_scatter(
                y_true, y_pred, metrics,
                self.fig_dir / f"{split}_scatter.png",
                dataset_labels=datasets,
            )
            plot_routing_heatmap(
                gates_arr.mean(axis=0),
                group_names,
                self.fig_dir / f"{split}_routing_heatmap.png",
                title=f"Routing Gate Activation ({split} set)",
            )
        except Exception as e:
            logger.warning("Could not save evaluation figures: %s", e)
