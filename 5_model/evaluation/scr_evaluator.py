"""
SCR Evaluator.

Produces per run folder:
  figures/
    scatter_test.png
    capacity_curve_*.png
    confusion_matrix_test.png
  metrics/
    metrics.json          capacity + breakdown + scenario + efficiency
  routing/
    routing_heatmap.png   scen(6) x HI(65) activation map
    routing_table.csv     selected HI list per scenario
  predictions/
    test_predictions.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import compute_metrics
from utils.hi_schema import (
    get_hi_cost_vector, get_hi_cols_for_seg,
    N_HI, N_SEGS, N_LEVELS, SEGMENTS,
)
from datasets.segment_dataset import SegmentDataset, SegmentNormalizer


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_LEVEL_NAMES = ["Low", "Mid", "High"]
_COST_VEC = np.array(get_hi_cost_vector("dis_hi"), dtype=np.float32)  # (65,)


class SCREvaluator:

    def __init__(
        self,
        model: nn.Module,
        normalizer: SegmentNormalizer,
        device: torch.device,
        figures_dir: Path,
        rep_cells: Sequence[str] = (),
    ):
        self.model = model.to(device)
        self.normalizer = normalizer
        self.device = device
        self.figures_dir = figures_dir
        self.rep_cells = list(rep_cells)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_dataset(self, ds: SegmentDataset, batch_size: int = 512) -> dict:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
        self.model.eval()

        preds_norm, trues_norm = [], []
        level_preds, level_trues = [], []
        seg_idxs, directions = [], []
        probe_zs, scen_zs = [], []

        for batch in loader:
            batch_d = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch_d)

            preds_norm.append(out["cap_pred"].cpu().numpy())
            trues_norm.append(batch["target"].numpy())
            level_preds.append(out["level_logits"].argmax(1).cpu().numpy())
            level_trues.append(batch["level"].numpy())
            seg_idxs.append(batch["seg_idx"].numpy())
            directions.append(batch["direction"].numpy())
            probe_zs.append(out["probe_z"].cpu().numpy())   # (B, N_HI)
            scen_zs.append(out["scen_z"].cpu().numpy())     # (B, N_HI)

        preds_norm   = np.concatenate(preds_norm)
        trues_norm   = np.concatenate(trues_norm)
        level_preds  = np.concatenate(level_preds)
        level_trues  = np.concatenate(level_trues)
        seg_idxs     = np.concatenate(seg_idxs)
        directions   = np.concatenate(directions)
        probe_zs     = np.concatenate(probe_zs)   # (N, 65)
        scen_zs      = np.concatenate(scen_zs)    # (N, 65)

        cap_pred_raw = self.normalizer.inverse_target(preds_norm)
        cap_true_raw = self.normalizer.inverse_target(trues_norm)

        return {
            "cap_pred_raw":  cap_pred_raw,
            "cap_true_raw":  cap_true_raw,
            "cap_pred_norm": preds_norm,
            "cap_true_norm": trues_norm,
            "level_pred":    level_preds,
            "level_true":    level_trues,
            "seg_idx":       seg_idxs,
            "direction":     directions,
            "probe_z":       probe_zs,
            "scen_z":        scen_zs,
            "cell_ids":      ds.cell_ids,
            "cycles":        ds.cycles,
            "seg_names":     ds.seg_names,
            "cap_raw":       ds.capacity_raw,
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        train_ds: SegmentDataset,
        val_ds: SegmentDataset,
        test_ds: SegmentDataset,
        batch_size: int = 512,
    ) -> dict[str, dict]:
        results = {}
        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            pred = self.predict_dataset(ds, batch_size)
            metrics = compute_metrics(pred["cap_true_raw"], pred["cap_pred_raw"])
            results[split_name] = {**metrics, **pred}
            print(f"[eval] {split_name}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        self._plot_scatter(results["test"], tag="test")
        self._plot_capacity_curves(results["test"])
        self._plot_confusion_matrix(results["test"], tag="test")
        return results

    # ------------------------------------------------------------------
    # Save metrics JSON (capacity + breakdown + scenario + efficiency)
    # ------------------------------------------------------------------
    def save_metrics(self, results: dict, metrics_dir: Path) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for split in ("train", "val", "test"):
            p = results[split]
            out[split] = {
                "capacity":  self._capacity_metrics(p),
                "breakdown": self._compute_breakdown(p),
                "scenario":  self._compute_scenario_metrics(p),
                "efficiency": self._compute_efficiency(p),
            }
        path = metrics_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[eval] saved {path}")

    def _capacity_metrics(self, p: dict) -> dict:
        m = compute_metrics(p["cap_true_raw"], p["cap_pred_raw"])
        return {k: float(v) for k, v in m.items()}

    def _compute_breakdown(self, p: dict) -> dict:
        t, pred = p["cap_true_raw"], p["cap_pred_raw"]
        d = p["direction"]
        lv = p["level_true"]

        out: dict = {}
        for tag, sel in [("charge", d > 0), ("discharge", d < 0)]:
            if sel.sum() > 1:
                m = compute_metrics(t[sel], pred[sel])
                out[tag] = {k: float(v) for k, v in m.items()}

        for i, name in enumerate(_LEVEL_NAMES):
            sel = lv == i
            if sel.sum() > 1:
                m = compute_metrics(t[sel], pred[sel])
                out[f"level_{name.lower()}"] = {k: float(v) for k, v in m.items()}
        return out

    def _compute_scenario_metrics(self, p: dict) -> dict:
        y_true = p["level_true"]
        y_pred = p["level_pred"]
        acc = float((y_true == y_pred).mean())

        per_class = []
        for i in range(N_LEVELS):
            sel = y_true == i
            per_class.append(float((y_pred[sel] == i).mean()) if sel.sum() > 0 else float("nan"))

        cm = np.zeros((N_LEVELS, N_LEVELS), dtype=int)
        for t, pr in zip(y_true, y_pred):
            cm[t][pr] += 1

        return {
            "accuracy": acc,
            "per_class_accuracy": {_LEVEL_NAMES[i]: per_class[i] for i in range(N_LEVELS)},
            "confusion_matrix": cm.tolist(),
        }

    def _compute_efficiency(self, p: dict) -> dict:
        probe_act = (p["probe_z"] > 0)   # (N, 65) bool
        scen_act  = (p["scen_z"] > 0)    # (N, 65) bool
        computed  = probe_act | scen_act  # union = actually computed

        avg_probe    = float(probe_act.sum(axis=1).mean())
        avg_scen     = float(scen_act.sum(axis=1).mean())
        avg_computed = float(computed.sum(axis=1).mean())
        avg_cost     = float((computed.astype(np.float32) @ _COST_VEC).mean())
        max_cost     = float(_COST_VEC.sum())

        return {
            "avg_probe_his":    avg_probe,
            "avg_scen_his":     avg_scen,
            "avg_computed_his": avg_computed,
            "avg_cost":         avg_cost,
            "max_cost":         max_cost,
            "cost_reduction_pct": float((1 - avg_cost / max_cost) * 100),
        }

    # ------------------------------------------------------------------
    # Save predictions CSV
    # ------------------------------------------------------------------
    def save_predictions(self, pred_dict: dict, predictions_dir: Path) -> None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        path = predictions_dir / "test_predictions.csv"

        probe_active_n = (pred_dict["probe_z"] > 0).sum(axis=1)
        scen_active_n  = (pred_dict["scen_z"]  > 0).sum(axis=1)

        rows = zip(
            pred_dict["cell_ids"],
            pred_dict["cycles"],
            pred_dict["seg_names"],
            pred_dict["cap_true_raw"].tolist(),
            pred_dict["cap_pred_raw"].tolist(),
            pred_dict["level_true"].tolist(),
            pred_dict["level_pred"].tolist(),
            probe_active_n.tolist(),
            scen_active_n.tolist(),
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "cell_id", "cycle", "seg_name",
                "cap_true_Ah", "cap_pred_Ah", "error_Ah",
                "level_true", "level_pred",
                "probe_active_n", "scen_active_n",
            ])
            for cell, cyc, seg, tr, pr, lt, lp, pa, sa in rows:
                w.writerow([
                    cell, cyc, seg,
                    f"{tr:.6f}", f"{pr:.6f}", f"{pr - tr:.6f}",
                    lt, lp, pa, sa,
                ])
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Routing heatmap + table
    # ------------------------------------------------------------------
    def plot_routing_heatmap(
        self,
        routing_dir: Path,
        probe_sel: list[int] | None = None,
        scen_sel: dict[int, list[int]] | None = None,
    ) -> None:
        routing_dir.mkdir(parents=True, exist_ok=True)

        # Use JSON-provided selections if available; fall back to model gate values
        if probe_sel is None:
            probe_sel = self.model.get_selected_probe_his()
        if scen_sel is None:
            scen_sel = self.model.get_selected_scen_his()

        act_matrix = np.zeros((N_SEGS, N_HI), dtype=np.float32)
        for s in range(N_SEGS):
            for i in scen_sel.get(s, []):
                act_matrix[s, i] = 1.0

        probe_row = np.zeros(N_HI, dtype=np.float32)
        for i in probe_sel:
            probe_row[i] = 1.0

        # Full matrix: probe row + 6 scen rows  → (7, 65)
        full_matrix = np.vstack([probe_row[np.newaxis, :], act_matrix])
        row_labels  = ["probe"] + SEGMENTS

        # HI short names (strip seg suffix, keep category+key)
        hi_cols = get_hi_cols_for_seg("dis_hi")  # reference seg
        hi_short = [c.rsplit("_dis_hi", 1)[0] for c in hi_cols]

        # --- PNG ---
        if _HAS_MPL:
            fig, ax = plt.subplots(figsize=(max(14, N_HI * 0.18), 4))
            im = ax.imshow(full_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_xticks(range(N_HI))
            ax.set_xticklabels(hi_short, rotation=90, fontsize=5.5)
            ax.set_title("SCR Routing: probe + scenario × HI activation")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            fig.tight_layout()
            path_png = routing_dir / "routing_heatmap.png"
            fig.savefig(path_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[eval] saved {path_png}")

        # --- CSV ---
        path_csv = routing_dir / "routing_table.csv"
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gate"] + hi_short)
            for row_name, row in zip(row_labels, full_matrix):
                w.writerow([row_name] + [int(v) for v in row])
        print(f"[eval] saved {path_csv}")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self, pred_dict: dict, tag: str = "test") -> None:
        if not _HAS_MPL:
            return
        y_true = pred_dict["level_true"]
        y_pred = pred_dict["level_pred"]

        cm = np.zeros((N_LEVELS, N_LEVELS), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        acc = (y_true == y_pred).mean()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(N_LEVELS)); ax.set_xticklabels(_LEVEL_NAMES)
        ax.set_yticks(range(N_LEVELS)); ax.set_yticklabels(_LEVEL_NAMES)
        ax.set_xlabel("Predicted level"); ax.set_ylabel("True level")
        ax.set_title(f"Level confusion ({tag})  acc={acc:.3f}")
        for i in range(N_LEVELS):
            for j in range(N_LEVELS):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        path = self.figures_dir / f"confusion_matrix_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    def _plot_scatter(self, pred_dict: dict, tag: str = "test") -> None:
        if not _HAS_MPL:
            return
        p = pred_dict["cap_pred_raw"]
        t = pred_dict["cap_true_raw"]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(t, p, s=5, alpha=0.4)
        lim = [min(t.min(), p.min()) * 0.98, max(t.max(), p.max()) * 1.02]
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlabel("True capacity (Ah)")
        ax.set_ylabel("Predicted capacity (Ah)")
        ax.set_title(f"SCR {tag} — pred vs true")
        ax.set_xlim(lim); ax.set_ylim(lim)
        path = self.figures_dir / f"scatter_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Capacity curve plots for representative cells
    # 2-row × 3-col layout: row0=Charge / row1=Discharge
    # Each row: [capacity curve | abs error | rel error]
    # Pred lines are split by segment sub-type (Low / Mid / High) rather
    # than averaged, so misrouted predictions are visible as separate lines.
    # ------------------------------------------------------------------
    def _plot_capacity_curves(self, pred_dict: dict) -> None:
        if not _HAS_MPL:
            return

        cell_ids   = np.array(pred_dict["cell_ids"])
        cycles     = np.array(pred_dict["cycles"])
        cap_true   = pred_dict["cap_true_raw"]
        cap_pred   = pred_dict["cap_pred_raw"]
        directions = pred_dict["direction"]
        seg_idxs   = pred_dict["seg_idx"]

        # (seg_idx, level_idx 0-2, display label)
        _DIR_SEGS = {
            "Charge":    [(0, 0, "Low"), (1, 1, "Mid"), (2, 2, "High")],
            "Discharge": [(3, 2, "High"), (4, 1, "Mid"), (5, 0, "Low")],
        }
        _LV_COLOR = {0: "tab:red", 1: "tab:green", 2: "tab:orange"}
        _LV_LS    = {0: "--", 1: "-.", 2: ":"}

        for cell in self.rep_cells:
            sel = cell_ids == cell
            if sel.sum() == 0:
                print(f"[eval] rep cell '{cell}' not found in test split, skipping")
                continue

            c_cyc  = cycles[sel]
            c_true = cap_true[sel]
            c_pred = cap_pred[sel]
            c_dir  = directions[sel]
            c_seg  = seg_idxs[sel]

            uniq_cyc = np.unique(c_cyc)

            fig, axes = plt.subplots(2, 3, figsize=(17, 8))
            fig.suptitle(cell, fontsize=12, y=1.01)

            for row, (dir_name, seg_list) in enumerate(_DIR_SEGS.items()):
                is_charge = (dir_name == "Charge")
                dir_mask  = (c_dir > 0) if is_charge else (c_dir < 0)

                ax_cap, ax_err, ax_rel = axes[row, 0], axes[row, 1], axes[row, 2]

                # True capacity per cycle for this direction
                d_cyc  = c_cyc[dir_mask]
                d_true = c_true[dir_mask]
                true_line = np.array([
                    d_true[d_cyc == cy].mean() if (d_cyc == cy).any() else np.nan
                    for cy in uniq_cyc
                ])

                ax_cap.plot(uniq_cyc, true_line, "b-", label="True", linewidth=1.5)

                for seg_idx, lv_idx, lv_name in seg_list:
                    s_mask = dir_mask & (c_seg == seg_idx)
                    if s_mask.sum() == 0:
                        continue
                    s_cyc  = c_cyc[s_mask]
                    s_pred = c_pred[s_mask]
                    pred_line = np.array([
                        s_pred[s_cyc == cy].mean() if (s_cyc == cy).any() else np.nan
                        for cy in uniq_cyc
                    ])

                    color = _LV_COLOR[lv_idx]
                    ls    = _LV_LS[lv_idx]
                    ax_cap.plot(uniq_cyc, pred_line, ls, color=color,
                                label=f"Pred-{lv_name}", linewidth=1.2)

                    err_abs = np.abs(pred_line - true_line)
                    err_rel = err_abs / np.where(true_line == 0, 1.0, np.abs(true_line)) * 100
                    ax_err.plot(uniq_cyc, err_abs, ls, color=color,
                                label=lv_name, linewidth=1.0)
                    ax_rel.plot(uniq_cyc, err_rel, ls, color=color,
                                label=lv_name, linewidth=1.0)

                ax_cap.set_xlabel("Cycle"); ax_cap.set_ylabel("Capacity (Ah)")
                ax_cap.set_title(f"{dir_name} — capacity curve")
                ax_cap.legend(fontsize=8)
                ax_err.set_xlabel("Cycle"); ax_err.set_ylabel("|Error| (Ah)")
                ax_err.set_title(f"{dir_name} — absolute error")
                ax_err.legend(fontsize=8)
                ax_rel.set_xlabel("Cycle"); ax_rel.set_ylabel("Relative error (%)")
                ax_rel.set_title(f"{dir_name} — relative error (%)")
                ax_rel.legend(fontsize=8)

            fig.tight_layout()
            safe_name = cell.replace("/", "_")
            path = self.figures_dir / f"capacity_curve_{safe_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[eval] saved {path}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
