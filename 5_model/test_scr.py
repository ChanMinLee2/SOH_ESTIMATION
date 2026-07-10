"""
test_scr.py  —  SCR evaluation entry point.

Loads a Phase 2 checkpoint and evaluates on test split.
Produces:
  - per-split metrics table (RMSE / MAE / R2 / MAPE)
  - scatter plot: pred vs true
  - capacity curve plots for representative cells

Usage:
  python 5_model/test_scr.py
  python 5_model/test_scr.py --checkpoint _5_data_model_scr/0708_1720/checkpoints/best.pt
  python 5_model/test_scr.py --rep-cells b1c0 b1c1 1-1
"""

from __future__ import annotations

import argparse
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

import json
import torch
import numpy as np

from utils.io_utils import load_config
from datasets.segment_dataset import build_datasets, SegmentNormalizer
from models.scr_model import SCRModel
from evaluation.scr_evaluator import SCREvaluator
from utils.hi_schema import N_HI, N_SEGS, SEGMENTS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SCR model")
    p.add_argument("--config",     default="5_model/config/scr.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rep-cells",  nargs="+", default=None)
    p.add_argument("--device",     default="auto")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _build_normalizer_from_ckpt(ckpt: dict) -> SegmentNormalizer:
    norm = SegmentNormalizer()
    norm.mean_        = ckpt["norm_mean"]
    norm.std_         = ckpt["norm_std"]
    norm.target_mean_ = ckpt["norm_target_mean"]
    norm.target_std_  = ckpt["norm_target_std"]
    return norm


def main() -> None:
    args   = _parse_args()
    device = _resolve_device(args.device)
    print(f"[test] device={device}")

    cfg = load_config(str(PROJECT_ROOT / args.config))

    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"]
    ckpt_path  = (Path(args.checkpoint) if args.checkpoint
                  else _find_latest_checkpoint(output_dir))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[test] checkpoint: {ckpt_path}")

    ckpt      = torch.load(ckpt_path, map_location="cpu")
    cfg_saved = ckpt.get("cfg", cfg)

    # ------------------------------------------------------------------
    # 데이터 재구성
    # ------------------------------------------------------------------
    # cap_init 관련 옵션은 현재 yaml 기준으로 적용 (cfg_saved가 구버전이면 키 없음)
    for k in ("use_initial_capacity", "nominal_capacities", "datasets"):
        if k in cfg.get("data", {}):
            cfg_saved.setdefault("data", {})[k] = cfg["data"][k]

    train_ds, val_ds, test_ds, _ = build_datasets(cfg_saved)
    norm = _build_normalizer_from_ckpt(ckpt)
    for ds in (train_ds, val_ds, test_ds):
        _reapply_norm(ds, norm)

    # ------------------------------------------------------------------
    # run_dir 결정 + gate JSON 탐색
    # ------------------------------------------------------------------
    run_dir = ckpt_path.parent.parent

    probe_json = _resolve_gate_json(run_dir, "classification_HIs.json",
                                    "scenario_classification_HIs.json")
    scen_json  = _resolve_gate_json(run_dir, "regression_HIs.json",
                                    "scenario_regression_HIs.json")

    # m / k
    cls_cfg     = cfg_saved.get("classifier", cfg.get("classifier", {}))
    reg_cfg     = cfg_saved.get("regression", cfg.get("regression", {}))
    charge_m    = cfg.get("classifier", {}).get("charge_probe_m",
                  cls_cfg.get("charge_probe_m", cls_cfg.get("probe_m_count", 1)))
    discharge_m = cfg.get("classifier", {}).get("discharge_probe_m",
                  cls_cfg.get("discharge_probe_m", cls_cfg.get("probe_m_count", 1)))
    scen_k      = cfg.get("regression", {}).get("scen_k_count",
                  reg_cfg.get("scen_k_count", 5))

    # gate 마스크 로드
    ch_mask, dis_mask = _load_probe_masks(probe_json, charge_m, discharge_m)
    scen_masks        = _load_scen_masks(scen_json, N_SEGS, N_HI, scen_k)

    if probe_json: print(f"[test] probe gate JSON: {probe_json}")
    if scen_json:  print(f"[test] scen  gate JSON: {scen_json}")
    if not probe_json and not scen_json:
        print("[test] gates JSON 없음 — L0 게이트 그대로 사용")

    # ------------------------------------------------------------------
    # 모델 재구성
    # ------------------------------------------------------------------
    state_keys   = set(ckpt["model_state"].keys())
    ckpt_has_l0  = ("charge_probe_gate.log_alpha" in state_keys or
                    "probe_gate.log_alpha" in state_keys)   # legacy compat

    m_cfg = cfg_saved.get("model", cfg.get("model", {}))
    model = SCRModel(
        d_probe=m_cfg.get("d_probe", 64),
        d_head=m_cfg.get("d_head", 128),
        dropout=m_cfg.get("dropout", 0.1),
        charge_probe_mask=ch_mask,
        discharge_probe_mask=dis_mask,
        scen_masks=scen_masks,
    )
    strict_load = not (ckpt_has_l0 and ch_mask is not None)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict_load)
    if unexpected:
        print(f"[test] gate 파라미터 무시 (JSON 마스크 사용): {len(unexpected)}개")
    model.eval()

    # ------------------------------------------------------------------
    # 대표 셀 선정
    # ------------------------------------------------------------------
    eval_cfg  = cfg.get("evaluation", {})
    rep_cells = (args.rep_cells
                 or eval_cfg.get("rep_cells")
                 or _pick_rep_cells(test_ds, cfg_saved,
                                    eval_cfg.get("rep_cells_per_dataset", 1)))
    print(f"[test] rep_cells: {rep_cells}")

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    figures_dir     = run_dir / "figures"
    metrics_dir     = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    routing_dir     = run_dir / "routing"
    for d in (figures_dir, metrics_dir, predictions_dir, routing_dir):
        d.mkdir(parents=True, exist_ok=True)

    evaluator = SCREvaluator(
        model=model,
        normalizer=norm,
        device=device,
        figures_dir=figures_dir,
        rep_cells=rep_cells,
    )
    results = evaluator.evaluate(train_ds, val_ds, test_ds)
    evaluator.save_metrics(results, metrics_dir)

    # Routing heatmap
    probe_sel = _selected_from_masks(ch_mask, dis_mask)   # union for heatmap
    scen_sel  = ({s: scen_masks[s].nonzero(as_tuple=False).squeeze(1).tolist()
                  for s in range(N_SEGS)} if scen_masks is not None else None)
    evaluator.plot_routing_heatmap(routing_dir,
                                   probe_sel=probe_sel,
                                   scen_sel=scen_sel)
    evaluator.save_predictions(results["test"], predictions_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== SCR Evaluation Summary ===")
    header = f"{'split':8s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}  {'MAPE':>8s}"
    print(header)
    print("-" * len(header))
    for split in ("train", "val", "test"):
        m_res = results[split]
        print(f"{split:8s}  {m_res['rmse']:8.4f}  {m_res['mae']:8.4f}"
              f"  {m_res['r2']:8.4f}  {m_res.get('mape', float('nan')):8.2f}")

    probe_his = model.get_selected_probe_his()
    scen_his  = model.get_selected_scen_his()
    avg_scen  = np.mean([len(v) for v in scen_his.values()])
    print(f"\nActive charge probe HIs    : {len(probe_his['charge'])}/{N_HI}")
    print(f"Active discharge probe HIs : {len(probe_his['discharge'])}/{N_HI}")
    print(f"Avg scen HIs/seg           : {avg_scen:.1f}/{N_HI}")
    for s, idxs in scen_his.items():
        print(f"  {SEGMENTS[s]:10s}: {len(idxs)} HIs selected")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: Path) -> Path:
    if not output_dir.exists():
        return output_dir / "checkpoints" / "final.pt"
    run_dirs   = [d for d in output_dir.iterdir()
                  if d.is_dir() and len(d.name) == 9 and d.name[4] == "_"]
    candidates = [d / "checkpoints" / "final.pt" for d in run_dirs
                  if (d / "checkpoints" / "final.pt").exists()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    old = [d / "scr_final.pt" for d in run_dirs if (d / "scr_final.pt").exists()]
    if old:
        return max(old, key=lambda p: p.stat().st_mtime)
    return output_dir / "checkpoints" / "final.pt"


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


def _load_probe_masks(
    json_path: Path | None, charge_m: int, discharge_m: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not json_path or not json_path.exists():
        return None, None
    data = json.loads(json_path.read_text())

    def _mask(indices: list[int]) -> torch.Tensor:
        m = torch.zeros(N_HI, dtype=torch.bool)
        for i in indices:
            m[i] = True
        return m

    if "charge_ranked" in data:
        return (_mask(data["charge_ranked"][:charge_m]),
                _mask(data["discharge_ranked"][:discharge_m]))
    # backward compat: single ranked_indices
    indices = data.get("ranked_indices", [])
    return _mask(indices[:charge_m]), _mask(indices[:discharge_m])


def _load_scen_masks(
    json_path: Path | None, n_segs: int, n_hi: int, k: int
) -> torch.Tensor | None:
    if not json_path or not json_path.exists():
        return None
    data  = json.loads(json_path.read_text())
    masks = torch.zeros(n_segs, n_hi, dtype=torch.bool)
    for s in range(n_segs):
        indices = (data.get(f"seg_{s}_ranked", [])[:k]
                   if f"seg_{s}_ranked" in data
                   else data.get(f"seg_{s}", []))
        for i in indices:
            masks[s, i] = True
    return masks


def _selected_from_masks(
    ch_mask: torch.Tensor | None, dis_mask: torch.Tensor | None
) -> list[int] | None:
    """Union of charge + discharge probe masks → routing heatmap의 probe row."""
    if ch_mask is None and dis_mask is None:
        return None
    union = torch.zeros(N_HI, dtype=torch.bool)
    if ch_mask  is not None: union |= ch_mask
    if dis_mask is not None: union |= dis_mask
    return union.nonzero(as_tuple=False).squeeze(1).tolist()


def _pick_rep_cells(test_ds, cfg: dict, n_per_dataset: int = 1) -> list[str]:
    data_cfg   = cfg["data"]
    seg_dir    = PROJECT_ROOT / data_cfg["seg_data_dir"]
    test_cells = sorted(set(test_ds.cell_ids))
    picked: list[str] = []
    for ds_name in data_cfg.get("datasets", []):
        ds_dir = seg_dir / ds_name
        if not ds_dir.exists():
            continue
        ds_cell_set = {p.stem for p in ds_dir.glob("*.pkl")}
        candidates  = [c for c in test_cells if c in ds_cell_set]
        picked.extend(candidates[:n_per_dataset])
    if not picked:
        picked = test_cells[:n_per_dataset * 2]
    return picked


def _reapply_norm(ds, norm: SegmentNormalizer) -> None:
    ds.target = torch.tensor(
        norm.transform_target(ds.capacity_raw), dtype=torch.float32
    )
    # cap_init도 동일 normalizer로 재정규화 (build_datasets가 refitting한 normalizer와 다를 수 있음)
    cap_init_raw = norm.inverse_target(ds.cap_init.numpy())
    ds.cap_init = torch.tensor(
        norm.transform_target(cap_init_raw), dtype=torch.float32
    )


if __name__ == "__main__":
    main()
