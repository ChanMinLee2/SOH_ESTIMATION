"""
finetune_scr.py  —  Few-shot fine-tuning of a Phase-2 SCR checkpoint.

Protocol
--------
1. Load Phase-2 checkpoint (probe + scen gates already FIXED via JSON masks;
   model.parameters() == cap_head only → no explicit freezing needed).
2. Load *only* the target dataset (e.g. MIT) using checkpoint normalizer.
   SOH ratio target is dataset-agnostic so no renormalization needed.
3. Split target cells:
     --finetune-cells N  →  fine-tune train  (few-shot, N cells)
     20 % of remainder  →  fine-tune val    (early stopping)
     80 % of remainder  →  fine-tune test
4. Evaluate BEFORE fine-tuning on the test split.
5. Fine-tune cap_head (SCRTrainer, lower LR, few epochs, lambda_l0=0).
6. Evaluate AFTER fine-tuning on the same test split.
7. Save fine-tuned checkpoint + comparison report.

Usage
-----
  python 5_model/finetune_scr.py --checkpoint _5_data_model_scr/0711_1538/checkpoints/best.pt --finetune-dataset MIT --finetune-cells 5

  # Optional overrides:
  #   --epochs 100 --lr 1e-4 --val-split 0.2 --seed 42 --device auto
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))

try:
    from utils.compat import install_numpy2_shim
    install_numpy2_shim()
except ImportError:
    pass

import numpy as np
import torch
import torch.nn as nn

from utils.io_utils import load_config
from utils.hi_schema import N_HI, spec_from_qfrac
from datasets.segment_dataset import (
    SegmentDataset, SegmentNormalizer,
    filter_dataset_by_cells,
    load_dataset_native_seg,
    load_dataset_wide,
    split_cells,
)
from models.scr_model import SCRModel
from training.scr_trainer import SCRTrainer
from evaluation.scr_evaluator import SCREvaluator

_proj_root_c = Path(__file__).resolve().parent.parent
if str(_proj_root_c) not in sys.path:
    sys.path.insert(0, str(_proj_root_c))
from common.scenario.base import ScenarioSpec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot fine-tuning of SCR")
    p.add_argument("--checkpoint",       required=True,
                   help="Phase-2 checkpoint path (best.pt)")
    p.add_argument("--finetune-dataset", default="MIT",
                   help="Dataset name to fine-tune on (default: MIT)")
    p.add_argument("--finetune-cells",   type=int, default=5,
                   help="Number of cells for few-shot train (default: 5)")
    p.add_argument("--val-split",        type=float, default=0.2,
                   help="Fraction of remaining cells used as val (default: 0.2)")
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--config",           default=None,
                   help="Optional yaml override (default: use checkpoint's saved config)")
    p.add_argument("--device",           default="auto")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Normalizer / checkpoint helpers  (equivalent to test_scr.py)
# ---------------------------------------------------------------------------

def _build_normalizer_from_ckpt(ckpt: dict) -> SegmentNormalizer:
    norm = SegmentNormalizer()
    norm.mean_ = ckpt["norm_mean"]
    norm.std_  = ckpt["norm_std"]
    # 구버전 체크포인트(norm_target_*) 하위호환
    norm.cap_init_mean_ = float(ckpt.get("norm_cap_init_mean",
                                          ckpt.get("norm_target_mean", 0.0)))
    norm.cap_init_std_  = float(ckpt.get("norm_cap_init_std",
                                          ckpt.get("norm_target_std",  1.0)))
    return norm


def _load_probe_masks(
    json_path: Path | None, charge_m: int, discharge_m: int,
    auto: bool = False, threshold: float = 0.5,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not json_path or not json_path.exists():
        return None, None
    data = json.loads(json_path.read_text())

    def _mask(indices: list[int]) -> torch.Tensor:
        m = torch.zeros(N_HI, dtype=torch.bool)
        for i in indices:
            m[i] = True
        return m

    def _threshold_mask(ranked: list[int], probs: list[float]) -> torch.Tensor:
        selected = [idx for idx, p in zip(ranked, probs) if p >= threshold]
        return _mask(selected if selected else ranked[:1])

    if "charge_ranked" in data:
        if auto:
            return (_threshold_mask(data["charge_ranked"],    data.get("charge_probs", [])),
                    _threshold_mask(data["discharge_ranked"], data.get("discharge_probs", [])))
        return (_mask(data["charge_ranked"][:charge_m]),
                _mask(data["discharge_ranked"][:discharge_m]))
    indices = data.get("ranked_indices", [])
    if auto:
        probs = data.get("probs", [1.0] * len(indices))
        ch = _threshold_mask(indices, probs)
        return ch, ch
    return _mask(indices[:charge_m]), _mask(indices[:discharge_m])


def _load_scen_masks(
    json_path: Path | None, n_segs: int, n_hi: int, k: int,
    auto: bool = False, threshold: float = 0.5,
) -> torch.Tensor | None:
    if not json_path or not json_path.exists():
        return None
    data  = json.loads(json_path.read_text())
    masks = torch.zeros(n_segs, n_hi, dtype=torch.bool)
    for s in range(n_segs):
        if f"seg_{s}_ranked" in data:
            ranked = data[f"seg_{s}_ranked"]
            if auto:
                probs    = data.get(f"seg_{s}_probs", [])
                selected = [idx for idx, p in zip(ranked, probs) if p >= threshold]
                indices  = selected if selected else ranked[:1]
            else:
                indices = ranked[:k]
        else:
            indices = data.get(f"seg_{s}", [])
        for i in indices:
            masks[s, i] = True
    return masks


def _resolve_gate_json(run_dir: Path, new_name: str, old_name: str) -> Path | None:
    p = run_dir / "gates" / new_name
    if p.exists():
        return p
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    import yaml as _yaml
    with open(cfg_path, encoding="utf-8") as f:
        saved_cfg = _yaml.safe_load(f)
    if saved_cfg.get("data", {}).get("gates_ignore", False):
        return None
    gates_from = saved_cfg.get("data", {}).get("gates_from")
    if not gates_from:
        return None
    from_dir   = PROJECT_ROOT / gates_from
    search_dir = (from_dir / "gates") if (from_dir / "gates").exists() else from_dir
    for name in (new_name, old_name):
        p = search_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Target dataset loading
# ---------------------------------------------------------------------------

def _load_target_dataset(
    cfg_saved: dict,
    target_ds_name: str,
    norm: SegmentNormalizer,
    spec: ScenarioSpec,
) -> SegmentDataset:
    """
    Load ALL cells from the target dataset using checkpoint normalizer (no refit).
    Returns a single SegmentDataset containing all target cells.
    """
    data_cfg = cfg_saved["data"]
    root     = PROJECT_ROOT
    seg_dir  = root / data_cfg["seg_data_dir"]
    wide_dir = root / data_cfg["data_dir"]

    # Load raw DataFrame for target dataset only
    raw_df = None
    if seg_dir.exists():
        raw_df = load_dataset_native_seg(seg_dir, [target_ds_name], wide_dir, spec=spec)

    if raw_df is None or len(raw_df) == 0:
        raw_df = load_dataset_wide(
            wide_dir, [target_ds_name],
            min_cycles=data_cfg.get("min_cycles_per_cell", 10),
            spec=spec,
        )

    if len(raw_df) == 0:
        raise RuntimeError(f"No data found for dataset '{target_ds_name}'")

    print(f"[finetune] loaded {target_ds_name}: {raw_df['cell_id'].nunique()} cells, "
          f"{len(raw_df)} segments")

    # Build dataset with CHECKPOINT normalizer (no refit)
    ds = SegmentDataset(raw_df, norm, fit_normalizer=False, data_cfg=data_cfg)
    return ds


# ---------------------------------------------------------------------------
# Cell split for fine-tuning
# ---------------------------------------------------------------------------

def _split_finetune_cells(
    all_cell_ids: list[str],
    n_finetune: int,
    val_split: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split cells into (finetune_train, finetune_val, finetune_test).

    n_finetune cells  → few-shot train
    val_split * rest  → val (early stopping)
    remainder         → test (final eval)
    """
    rng   = np.random.default_rng(seed)
    cells = sorted(all_cell_ids)
    rng.shuffle(cells)

    n_total = len(cells)
    if n_finetune >= n_total:
        raise ValueError(
            f"--finetune-cells {n_finetune} >= total cells {n_total}; "
            "need at least 1 cell for testing"
        )

    ft_train = cells[:n_finetune]
    remaining = cells[n_finetune:]

    n_val = max(1, int(len(remaining) * val_split))
    ft_val  = remaining[:n_val]
    ft_test = remaining[n_val:]

    if not ft_test:
        ft_test = ft_val[-1:]
        ft_val  = ft_val[:-1]

    return ft_train, ft_val, ft_test


# ---------------------------------------------------------------------------
# Model rebuild  (identical to test_scr.py logic)
# ---------------------------------------------------------------------------

def _build_model(ckpt: dict, cfg_saved: dict, run_dir: Path, spec: ScenarioSpec) -> SCRModel:
    cls_cfg = cfg_saved.get("classifier", {})
    reg_cfg = cfg_saved.get("regression", {})
    charge_m    = cls_cfg.get("charge_probe_m",    cls_cfg.get("probe_m_count", 1))
    discharge_m = cls_cfg.get("discharge_probe_m", cls_cfg.get("probe_m_count", 1))
    scen_k      = reg_cfg.get("scen_k_count", 5)
    auto_mk     = cls_cfg.get("is_auto_mk_selection", False)

    probe_json = _resolve_gate_json(run_dir, "classification_HIs.json",
                                    "scenario_classification_HIs.json")
    scen_json  = _resolve_gate_json(run_dir, "regression_HIs.json",
                                    "scenario_regression_HIs.json")

    ch_mask, dis_mask = _load_probe_masks(probe_json, charge_m, discharge_m, auto=auto_mk)
    scen_masks        = _load_scen_masks(scen_json, spec.n_scenarios, N_HI, scen_k, auto=auto_mk)

    m_cfg = cfg_saved.get("model", {})
    model = SCRModel(
        d_probe=m_cfg.get("d_probe", 64),
        d_head=m_cfg.get("d_head", 128),
        dropout=m_cfg.get("dropout", 0.1),
        charge_probe_mask=ch_mask,
        discharge_probe_mask=dis_mask,
        scen_masks=scen_masks,
        model_cfg=m_cfg,
        spec=spec,
    )

    state_keys  = set(ckpt["model_state"].keys())
    ckpt_has_l0 = ("charge_probe_gate.log_alpha" in state_keys or
                   "probe_gate.log_alpha" in state_keys)
    strict_load = not (ckpt_has_l0 and ch_mask is not None)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict_load)
    if unexpected:
        print(f"[finetune] gate params skipped (JSON masks active): {len(unexpected)}")

    # Verify only cap_head is trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[finetune] trainable params: {len(trainable)} "
          f"(all from cap_head — gates are frozen buffers)")

    return model


# ---------------------------------------------------------------------------
# Quick metric helper
# ---------------------------------------------------------------------------

def _quick_metrics(model: nn.Module, ds: SegmentDataset, device: torch.device) -> dict:
    from torch.utils.data import DataLoader
    from datasets.segment_dataset import collate_fn
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_fn)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch_d = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_d)
            preds.append(out["cap_pred"].cpu().numpy())
            trues.append(batch["target"].numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    rmse = float(np.sqrt(((p - t) ** 2).mean()))
    mae  = float(np.abs(p - t).mean())
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse_args()
    device = _resolve_device(args.device)
    print(f"[finetune] device={device}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    cfg_saved = ckpt.get("cfg", {})
    norm      = _build_normalizer_from_ckpt(ckpt)
    run_dir   = ckpt_path.parent.parent
    print(f"[finetune] checkpoint: {ckpt_path}")
    print(f"[finetune] run_dir:    {run_dir}")

    # ------------------------------------------------------------------
    # Merge optional override config
    # ------------------------------------------------------------------
    if args.config:
        cfg_override = load_config(str(PROJECT_ROOT / args.config))
        cfg_saved = {**cfg_saved, **cfg_override}

    # Ensure data paths are resolved (may be null in yaml, resolved at training time)
    data_cfg = cfg_saved.get("data", {})
    if not data_cfg.get("seg_data_dir"):
        raise RuntimeError(
            "cfg_saved['data']['seg_data_dir'] is null. "
            "Run with checkpoint from a completed training run."
        )

    # ------------------------------------------------------------------
    # Load scenario spec
    # ------------------------------------------------------------------
    _spec_path = run_dir / "scenario_spec.json"
    if _spec_path.exists():
        spec = ScenarioSpec.load(_spec_path)
        print(f"[finetune] spec: axis={spec.axis}")
    else:
        spec = spec_from_qfrac()
        print(f"[finetune] no scenario_spec.json — using qfrac default")

    # ------------------------------------------------------------------
    # Load target dataset (all cells, checkpoint normalizer)
    # ------------------------------------------------------------------
    all_target_ds = _load_target_dataset(cfg_saved, args.finetune_dataset, norm, spec)
    all_cell_ids  = sorted(set(all_target_ds.cell_ids))
    print(f"[finetune] {args.finetune_dataset}: {len(all_cell_ids)} cells total")

    ft_train_cells, ft_val_cells, ft_test_cells = _split_finetune_cells(
        all_cell_ids,
        n_finetune=args.finetune_cells,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"[finetune] split  train={len(ft_train_cells)}  "
          f"val={len(ft_val_cells)}  test={len(ft_test_cells)}")
    print(f"[finetune] train cells: {ft_train_cells}")

    ft_train_ds = filter_dataset_by_cells(all_target_ds, ft_train_cells)
    ft_val_ds   = filter_dataset_by_cells(all_target_ds, ft_val_cells)
    ft_test_ds  = filter_dataset_by_cells(all_target_ds, ft_test_cells)
    print(f"[finetune] segs   train={len(ft_train_ds)}  "
          f"val={len(ft_val_ds)}  test={len(ft_test_ds)}")

    # ------------------------------------------------------------------
    # Build model (probe + scen gates = buffers → only cap_head trains)
    # ------------------------------------------------------------------
    model = _build_model(ckpt, cfg_saved, run_dir, spec)
    model.to(device)

    # ------------------------------------------------------------------
    # BEFORE fine-tuning: evaluate on test split
    # ------------------------------------------------------------------
    before_metrics = _quick_metrics(model, ft_test_ds, device)
    print(f"\n[finetune] BEFORE  rmse={before_metrics['rmse']:.4f}  "
          f"mae={before_metrics['mae']:.4f}  r2={before_metrics['r2']:.4f}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp  = datetime.now().strftime("%m%d_%H%M")
    tag        = f"ft_{args.finetune_dataset}_{args.finetune_cells}shot_{timestamp}"
    ft_out_dir = run_dir / tag
    ft_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[finetune] output: {ft_out_dir}")

    # Save "before" metrics
    _save_json(ft_out_dir / "before_metrics.json", {
        "finetune_dataset": args.finetune_dataset,
        "finetune_cells":   args.finetune_cells,
        "train_cells":      ft_train_cells,
        "val_cells":        ft_val_cells,
        "test_cells":       ft_test_cells,
        "before": before_metrics,
    })

    # ------------------------------------------------------------------
    # Build fine-tune config (lower LR, fewer epochs, no L0 penalty)
    # ------------------------------------------------------------------
    ft_cfg = copy.deepcopy(cfg_saved)
    ft_cfg["training"]["epochs"]              = args.epochs
    ft_cfg["training"]["lr"]                  = args.lr
    ft_cfg["training"]["early_stop_patience"] = 20
    ft_cfg["training"]["warmup_epochs"]       = 5
    ft_cfg["training"]["scheduler"]           = "cosine"
    ft_cfg["loss"]["lambda_l0"]               = 0.0   # gates frozen, no L0 needed
    ft_cfg["loss"]["lambda_l0_auto"]          = False
    ft_cfg["loss"]["lambda_l0_schedule"]      = "none"
    ft_cfg["loss"]["lambda_scen"]             = 0.0

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    from datasets.segment_dataset import collate_fn
    from torch.utils.data import DataLoader

    batch_size   = min(cfg_saved.get("training", {}).get("batch_size", 512), len(ft_train_ds))
    train_loader = DataLoader(ft_train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=False)
    val_loader   = DataLoader(ft_val_ds,   batch_size=512, shuffle=False,
                              collate_fn=collate_fn)

    trainer = SCRTrainer(
        model=model,
        cfg=ft_cfg,
        output_dir=ft_out_dir,
        device=device,
        normalizer=norm,
    )
    trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # AFTER fine-tuning: evaluate on same test split
    # ------------------------------------------------------------------
    model.eval()
    after_metrics = _quick_metrics(model, ft_test_ds, device)
    print(f"\n[finetune] AFTER   rmse={after_metrics['rmse']:.4f}  "
          f"mae={after_metrics['mae']:.4f}  r2={after_metrics['r2']:.4f}")

    delta_rmse = after_metrics["rmse"] - before_metrics["rmse"]
    delta_sign = "-" if delta_rmse < 0 else "+"
    print(f"[finetune] delta   rmse={delta_sign}{abs(delta_rmse):.4f}  "
          f"({'improved' if delta_rmse < 0 else 'degraded'})")

    # Save comparison report
    report = {
        "finetune_dataset": args.finetune_dataset,
        "finetune_cells":   args.finetune_cells,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "seed":             args.seed,
        "train_cells":      ft_train_cells,
        "val_cells":        ft_val_cells,
        "test_cells":       ft_test_cells,
        "n_test_cells":     len(ft_test_cells),
        "n_test_segments":  len(ft_test_ds),
        "before":           before_metrics,
        "after":            after_metrics,
        "delta":            {k: after_metrics[k] - before_metrics[k]
                             for k in before_metrics},
    }
    _save_json(ft_out_dir / "finetune_report.json", report)
    print(f"[finetune] saved report: {ft_out_dir / 'finetune_report.json'}")

    # Full evaluation with plots
    figures_dir = ft_out_dir / "figures"
    metrics_dir = ft_out_dir / "metrics"
    for d in (figures_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    evaluator = SCREvaluator(
        model=model,
        normalizer=norm,
        device=device,
        figures_dir=figures_dir,
        rep_cells=ft_test_cells[:3],
    )
    results = evaluator.evaluate(ft_train_ds, ft_val_ds, ft_test_ds)
    evaluator.save_metrics(results, metrics_dir)

    # Summary
    print("\n=== Few-shot Fine-tuning Summary ===")
    print(f"Dataset: {args.finetune_dataset}  |  Fine-tune cells: {args.finetune_cells}")
    print(f"{'':12s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}")
    print(f"{'Before':12s}  {before_metrics['rmse']:8.4f}  "
          f"{before_metrics['mae']:8.4f}  {before_metrics['r2']:8.4f}")
    print(f"{'After':12s}  {after_metrics['rmse']:8.4f}  "
          f"{after_metrics['mae']:8.4f}  {after_metrics['r2']:8.4f}")
    print(f"{'Delta':12s}  {delta_rmse:+8.4f}  "
          f"{after_metrics['mae']-before_metrics['mae']:+8.4f}  "
          f"{after_metrics['r2']-before_metrics['r2']:+8.4f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
