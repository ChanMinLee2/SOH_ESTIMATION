"""
test_scr.py  —  SCR evaluation entry point.

Loads trained SCR checkpoint and evaluates on test split.
Produces:
  - per-split metrics table (RMSE / MAE / R2 / MAPE)
  - scatter plot: pred vs true
  - capacity curve plots for representative cells (b1c0, 1-1)

Usage:
  python 5_model/test_scr.py
  python 5_model/test_scr.py --checkpoint _5_data_model_scr/scr_final.pt
  python 5_model/test_scr.py --rep-cells b1c0 1-1
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
    p.add_argument("--checkpoint", default=None,
                   help="Path to scr_final.pt (default: output_dir/scr_final.pt)")
    p.add_argument("--rep-cells",  nargs="+", default=None,
                   help="Representative cells for capacity curves")
    p.add_argument("--device",     default="auto")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _build_normalizer_from_ckpt(ckpt: dict) -> SegmentNormalizer:
    norm = SegmentNormalizer()
    norm.mean_ = ckpt["norm_mean"]
    norm.std_ = ckpt["norm_std"]
    norm.target_mean_ = ckpt["norm_target_mean"]
    norm.target_std_ = ckpt["norm_target_std"]
    return norm


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"[test] device={device}")

    cfg = load_config(str(PROJECT_ROOT / args.config))

    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"]
    ckpt_path = (Path(args.checkpoint) if args.checkpoint
                 else _find_latest_checkpoint(output_dir))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[test] checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_saved = ckpt.get("cfg", cfg)  # prefer saved config

    # ------------------------------------------------------------------
    # Rebuild datasets (same split seed → same cell assignment)
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds, _ = build_datasets(cfg_saved)

    # Restore normalizer from checkpoint
    norm = _build_normalizer_from_ckpt(ckpt)
    # Re-apply saved normalizer stats to all datasets
    for ds in (train_ds, val_ds, test_ds):
        ds.x_hi    # already normalised during build_datasets — we need to re-normalise
        # The datasets were normalised with their own fit; overwrite with ckpt norm
        _reapply_norm(ds, norm)

    # ------------------------------------------------------------------
    # Reconstruct model — JSON gates 우선, 없으면 체크포인트 타입에 맞춰 재구성
    # ------------------------------------------------------------------
    # run_dir = checkpoints/ 의 부모 (= MMDD_HHMM/)
    run_dir = ckpt_path.parent.parent

    state_keys = set(ckpt["model_state"].keys())
    ckpt_has_l0 = "probe_gate.log_alpha" in state_keys

    # gates/ JSON이 있으면 항상 그대로 HI를 고정해서 평가
    probe_json = _resolve_gate_json(
        run_dir, "classification_HIs.json", "scenario_classification_HIs.json"
    )
    scen_json = _resolve_gate_json(
        run_dir, "regression_HIs.json", "scenario_regression_HIs.json"
    )

    m = cfg_saved.get("classifier", {}).get("probe_m_count",
        cfg.get("classifier", {}).get("probe_m_count", N_HI))
    k = cfg_saved.get("regression", {}).get("scen_k_count",
        cfg.get("regression", {}).get("scen_k_count", N_HI))
    probe_mask = _load_bool_mask(probe_json, N_HI, m) if probe_json else None
    scen_masks = _load_scen_masks(scen_json, N_SEGS, N_HI, k) if scen_json else None

    if probe_json:
        print(f"[test] probe gate JSON: {probe_json}")
    if scen_json:
        print(f"[test] scen  gate JSON: {scen_json}")
    if not probe_json and not scen_json:
        print("[test] gates JSON 없음 — L0 게이트 그대로 사용")

    m_cfg = cfg_saved["model"]
    model = SCRModel(
        d_probe=m_cfg["d_probe"],
        d_head=m_cfg["d_head"],
        dropout=m_cfg.get("dropout", 0.1),
        probe_mask=probe_mask,
        scen_masks=scen_masks,
    )
    # Phase 1 체크포인트(L0 키)를 fixed-mask 모델로 로드하는 경우:
    # gate 파라미터 키가 맞지 않으므로 strict=False로 MLP/head만 로드
    strict_load = not (ckpt_has_l0 and probe_mask is not None)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict_load)
    if unexpected:
        print(f"[test] 체크포인트 gate 파라미터 무시 (JSON 마스크 사용): {len(unexpected)}개")
    model.eval()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    rep_cells = args.rep_cells or _pick_rep_cells(test_ds, cfg_saved)
    print(f"[test] rep_cells: {rep_cells}")

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

    # Routing heatmap: probe_mask/scen_masks에서 인덱스 변환 (JSON 기반)
    routing_probe_sel = probe_mask.nonzero(as_tuple=False).squeeze(1).tolist() if probe_mask is not None else None
    routing_scen_sel  = {s: scen_masks[s].nonzero(as_tuple=False).squeeze(1).tolist()
                         for s in range(N_SEGS)} if scen_masks is not None else None
    evaluator.plot_routing_heatmap(routing_dir,
                                   probe_sel=routing_probe_sel,
                                   scen_sel=routing_scen_sel)

    evaluator.save_predictions(results["test"], predictions_dir)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n=== SCR Evaluation Summary ===")
    header = f"{'split':8s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}  {'MAPE':>8s}"
    print(header)
    print("-" * len(header))
    for split in ("train", "val", "test"):
        m = results[split]
        print(f"{split:8s}  {m['rmse']:8.4f}  {m['mae']:8.4f}  {m['r2']:8.4f}  {m.get('mape', float('nan')):8.2f}")

    # Active HI counts
    model_eval = model.to("cpu")
    n_probe = len(model_eval.get_selected_probe_his())
    scen_sel = model_eval.get_selected_scen_his()
    avg_scen = np.mean([len(v) for v in scen_sel.values()])
    print(f"\nActive probe HIs : {n_probe}/{N_HI}")
    print(f"Avg scen HIs/seg : {avg_scen:.1f}/{N_HI}")
    for s, idxs in scen_sel.items():
        print(f"  {SEGMENTS[s]:10s}: {len(idxs)} HIs selected")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: Path) -> Path:
    """
    output_dir 아래 MMDD_HHMM 폴더 중 가장 최근 checkpoints/final.pt 반환.
    없으면 하위호환: scr_final.pt 탐색.
    """
    if not output_dir.exists():
        return output_dir / "checkpoints" / "final.pt"
    run_dirs = [d for d in output_dir.iterdir()
                if d.is_dir() and len(d.name) == 9 and d.name[4] == "_"]
    # 신규 구조 우선
    candidates = [d / "checkpoints" / "final.pt" for d in run_dirs
                  if (d / "checkpoints" / "final.pt").exists()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    # 구형 구조 fallback
    old = [d / "scr_final.pt" for d in run_dirs if (d / "scr_final.pt").exists()]
    if old:
        return max(old, key=lambda p: p.stat().st_mtime)
    return output_dir / "checkpoints" / "final.pt"


def _pick_rep_cells(test_ds, cfg: dict) -> list[str]:
    """
    Test split에서 datasets 순서대로 각 1개 셀을 자동 선택한다.
    seg_data_dir 아래 {dataset}/ 폴더에 있는 pkl 파일명으로 소속 데이터셋을 판별.
    """
    data_cfg = cfg["data"]
    seg_dir = PROJECT_ROOT / data_cfg["seg_data_dir"]
    test_cells = sorted(set(test_ds.cell_ids))

    picked: list[str] = []
    for ds_name in data_cfg.get("datasets", []):
        ds_dir = seg_dir / ds_name
        if not ds_dir.exists():
            continue
        ds_cell_set = {p.stem for p in ds_dir.glob("*.pkl")}
        candidates = [c for c in test_cells if c in ds_cell_set]
        if candidates:
            picked.append(candidates[0])

    if not picked:
        picked = test_cells[:2]  # fallback
    return picked


def _resolve_gate_json(run_dir: Path, new_name: str, old_name: str) -> Path | None:
    """
    gates JSON 경로를 두 단계로 탐색한다.
    1) run_dir/gates/{new_name}  — 자기 run에서 직접 학습/복사된 경우
    2) run_dir/config.yaml의 gates_from 경로  — 이전 run gates 재사용한 경우
       gates_ignore=true 이면 탐색 중단
    """
    # 1. 자기 run 폴더 우선
    p = run_dir / "gates" / new_name
    if p.exists():
        return p

    # 2. 저장된 config.yaml 참조
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

    from_dir = PROJECT_ROOT / gates_from
    search_dir = (from_dir / "gates") if (from_dir / "gates").exists() else from_dir
    for name in (new_name, old_name):
        p = search_dir / name
        if p.exists():
            return p
    return None


def _load_probe_indices(json_path: Path, m: int | None = None) -> list[int] | None:
    if not json_path or not json_path.exists():
        return None
    import json as _json
    data = _json.loads(json_path.read_text())
    if "ranked_indices" in data:
        return data["ranked_indices"][:m] if m is not None else data["ranked_indices"]
    return data.get("selected_indices", [])


def _load_scen_indices(json_path: Path, n_segs: int, k: int | None = None) -> dict[int, list[int]] | None:
    if not json_path or not json_path.exists():
        return None
    import json as _json
    data = _json.loads(json_path.read_text())
    out = {}
    for s in range(n_segs):
        if f"seg_{s}_ranked" in data:
            out[s] = data[f"seg_{s}_ranked"][:k] if k is not None else data[f"seg_{s}_ranked"]
        else:
            out[s] = data.get(f"seg_{s}", [])
    return out


def _load_bool_mask(json_path: Path, n: int, m: int | None = None) -> torch.Tensor | None:
    """
    새 형식: {"ranked_indices": [...]}  → 상위 m개 True
    구 형식: {"selected_indices": [...]}  → 그대로 사용 (backward compat)
    m=None 이면 전체 사용
    """
    if not json_path.exists():
        return None
    import json as _json
    data = _json.loads(json_path.read_text())
    if "ranked_indices" in data:
        indices = data["ranked_indices"][:m] if m is not None else data["ranked_indices"]
    else:
        indices = data.get("selected_indices", [])
    mask = torch.zeros(n, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    return mask


def _load_scen_masks(json_path: Path, n_segs: int, n_hi: int, k: int | None = None) -> torch.Tensor | None:
    """
    새 형식: {"seg_0_ranked": [...]}  → 상위 k개 True
    구 형식: {"seg_0": [...]}  → 그대로 사용 (backward compat)
    k=None 이면 전체 사용
    """
    if not json_path.exists():
        return None
    import json as _json
    data = _json.loads(json_path.read_text())
    masks = torch.zeros(n_segs, n_hi, dtype=torch.bool)
    for s in range(n_segs):
        if f"seg_{s}_ranked" in data:
            indices = data[f"seg_{s}_ranked"][:k] if k is not None else data[f"seg_{s}_ranked"]
        else:
            indices = data.get(f"seg_{s}", [])
        for i in indices:
            masks[s, i] = True
    return masks


def _reapply_norm(ds, norm: SegmentNormalizer) -> None:
    """
    Re-normalise a SegmentDataset's x_hi and target using the checkpoint normalizer.
    This ensures val/test use train-fitted stats rather than their own fit.
    """
    import numpy as _np
    # x_hi: we stored normalised values already via the local norm;
    # but since val/test were built with the same norm object (fit on train only),
    # they are already correct.  Only needed if we reload from disk separately.
    ds.target = torch.tensor(
        norm.transform_target(ds.capacity_raw), dtype=torch.float32
    )


if __name__ == "__main__":
    main()
