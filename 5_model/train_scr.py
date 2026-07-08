"""
train_scr.py  —  SCR training entry point.

Workflow:
  1. Load config (config/scr.yaml or --config arg)
  2. Build datasets (wide pkl reshape or native seg format)
  3. Check for cached JSON masks → load fixed probe/scen masks if present
  4. Build SCRModel and SCRTrainer
  5. Train
  6. After training: extract selected HI indices → save JSON if not in cache-mode
  7. Save final checkpoint

Usage:
  python 5_model/train_scr.py
  python 5_model/train_scr.py --config 5_model/config/scr.yaml
  python 5_model/train_scr.py --probe-m 5 --scen-k 10   # override fixed counts
"""

from __future__ import annotations

import argparse
import json
import math
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
from torch.utils.data import DataLoader

from utils.io_utils import load_config, save_json
from datasets.segment_dataset import build_datasets, collate_fn
from models.scr_model import SCRModel
from training.scr_trainer import SCRTrainer
from utils.hi_schema import N_HI, N_SEGS, SEGMENTS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SCR model")
    p.add_argument("--config",     default="5_model/config/scr.yaml")
    p.add_argument("--probe-m",    type=int, default=None, help="Override probe_m_count")
    p.add_argument("--scen-k",     type=int, default=None, help="Override scen_k_count")
    p.add_argument("--gates-from", default=None,
                   help="이전 run 폴더 경로 (gates/*.json 또는 scenario_*.json 자동 탐색). "
                        "미지정 시 scr.yaml의 gates_from 사용")
    p.add_argument("--no-gates", action="store_true",
                   help="gates_from 무시하고 L0부터 재학습 (yaml의 gates_ignore와 동일)")
    p.add_argument("--device",     default="auto")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Gates directory resolver
# ---------------------------------------------------------------------------

def _resolve_gates_dir(cli_arg: str | None, cfg: dict) -> Path | None:
    """
    CLI --gates-from > yaml gates_from > None 순서로 gates 폴더를 결정한다.
    run 폴더 안에 gates/ 서브폴더가 있으면 그쪽 반환, 없으면 run 폴더 자체 반환.
    """
    raw = cli_arg or cfg.get("data", {}).get("gates_from")
    if not raw:
        return None
    run_dir = PROJECT_ROOT / raw
    gates_subdir = run_dir / "gates"
    return gates_subdir if gates_subdir.exists() else run_dir


def _find_json(gates_dir: Path, new_name: str, old_name: str) -> Path | None:
    """신버전 파일명 우선, 없으면 구버전(scenario_ 접두사) 탐색."""
    for name in (new_name, old_name):
        p = gates_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# JSON mask helpers
# ---------------------------------------------------------------------------

def _load_probe_mask_from_json(json_path: Path, m: int) -> torch.Tensor | None:
    """
    (N_HI,) bool tensor 반환.
    새 형식: {"ranked_indices": [...65개 정렬]}  → 상위 m개만 True
    구 형식: {"selected_indices": [...이미 절삭]}  → 그대로 사용 (backward compat)
    """
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    if "ranked_indices" in data:
        indices = data["ranked_indices"][:m]
    else:
        indices = data.get("selected_indices", [])
    mask = torch.zeros(N_HI, dtype=torch.bool)
    for i in indices:
        mask[i] = True
    print(f"[train] probe mask: top-{m} from ranking → {sorted(indices)}")
    return mask


def _load_scen_masks_from_json(json_path: Path, k: int) -> torch.Tensor | None:
    """
    (N_SEGS, N_HI) bool tensor 반환.
    새 형식: {"seg_0_ranked": [...65개 정렬], ...}  → 시나리오별 상위 k개만 True
    구 형식: {"seg_0": [...이미 절삭], ...}          → 그대로 사용 (backward compat)
    """
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    masks = torch.zeros(N_SEGS, N_HI, dtype=torch.bool)
    for s in range(N_SEGS):
        if f"seg_{s}_ranked" in data:
            indices = data[f"seg_{s}_ranked"][:k]
        else:
            indices = data.get(f"seg_{s}", [])
        for i in indices:
            masks[s, i] = True
    print(f"[train] scen masks: top-{k}/scenario")
    return masks


def _ranked_indices(gate) -> tuple[list[int], list[float]]:
    """gate_prob 내림차순으로 전체 HI 인덱스와 확률 반환."""
    prob = gate.gate_prob().detach().cpu()
    sorted_idx = prob.argsort(descending=True).tolist()
    sorted_prob = [round(float(prob[i]), 6) for i in sorted_idx]
    return sorted_idx, sorted_prob


def _save_probe_mask_to_json(
    model: SCRModel,
    json_path: Path,
    hi_cols_ref: list[str],
) -> None:
    """Phase 1: gate_prob 기준 전체 랭킹을 저장. m은 Phase 1b/test 시점에 yaml로 결정."""
    ranked, probs = _ranked_indices(model.probe_gate)
    out = {
        "ranked_indices": ranked,
        "ranked_names":   [hi_cols_ref[i] for i in ranked],
        "gate_probs":     probs,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[train] Saved probe HI ranking -> {json_path}  (전체 {len(ranked)}개 랭킹)")


def _save_scen_masks_to_json(
    model: SCRModel,
    json_path: Path,
    hi_cols_by_seg: dict[int, list[str]],
) -> None:
    """Phase 1: 시나리오별 gate_prob 기준 전체 랭킹을 저장. k는 Phase 1b/test 시점에 yaml로 결정."""
    out = {}
    for s in range(N_SEGS):
        ranked, probs = _ranked_indices(model.scen_gates[s])
        out[f"seg_{s}_ranked"]   = ranked
        out[f"seg_{s}_names"]    = [hi_cols_by_seg[s][i] for i in ranked]
        out[f"seg_{s}_probs"]    = probs
        out[f"seg_{s}_seg_name"] = SEGMENTS[s]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[train] Saved scen HI ranking -> {json_path}  (시나리오별 전체 {N_HI}개 랭킹)")


# ---------------------------------------------------------------------------
# Dynamic lambda_l0
# ---------------------------------------------------------------------------

_REF_LAMBDA = 0.01  # 기준값: ~10 probe HI, ~10 scen HI 선택에 대응
_REF_M      = 10    # 위 lambda 기준 probe HI 수 (경험값)
_REF_K      = 10    # 위 lambda 기준 scen  HI 수 (경험값)


def _compute_lambda_l0(m: int, k: int) -> float:
    """
    목표 m/k에 비례해 lambda_l0을 동적으로 계산한다.

    원리: m/k가 작을수록(더 적은 HI 원함) L0 페널티를 강하게 줘야
    L0 게이트가 자연스럽게 그 근방에서 수렴한다.
    기준(_REF_M=10, _REF_K=10)에서 벗어나는 비율의 기하평균으로 스케일링.

    예시 (REF_LAMBDA=0.01 기준):
      m=1,  k=5  → scale=√(10×2)≈4.5  → lambda≈0.045
      m=5,  k=5  → scale=√(2×2)=2.0   → lambda≈0.020
      m=5,  k=10 → scale=√(2×1)≈1.4   → lambda≈0.014
      m=10, k=20 → scale=√(1×0.5)≈0.7 → lambda≈0.007
    """
    probe_scale = _REF_M / max(m, 1)
    scen_scale  = _REF_K / max(k, 1)
    scale = math.sqrt(probe_scale * scen_scale)
    return round(max(1e-4, min(_REF_LAMBDA * scale, 0.5)), 5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"[train] device={device}")

    cfg = load_config(str(PROJECT_ROOT / args.config))

    # Override from CLI
    if args.probe_m is not None:
        cfg["classifier"]["probe_m_count"] = args.probe_m
    if args.scen_k is not None:
        cfg["regression"]["scen_k_count"] = args.scen_k

    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"] / timestamp
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "gates").mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {output_dir}")

    # config.yaml 복사본 저장
    import shutil
    shutil.copy(str(PROJECT_ROOT / args.config), str(output_dir / "config.yaml"))

    # 이번 run의 저장 경로 (항상 새 폴더의 gates/)
    probe_json_out = output_dir / "gates" / "classification_HIs.json"
    scen_json_out  = output_dir / "gates" / "regression_HIs.json"

    # 이전 run에서 로드할 gates 디렉토리 (CLI > yaml > None)
    # --no-gates 또는 yaml gates_ignore=true 이면 강제 무시
    gates_ignore = args.no_gates or cfg["data"].get("gates_ignore", False)
    gates_dir = None if gates_ignore else _resolve_gates_dir(args.gates_from, cfg)
    probe_json_in = scen_json_in = None
    if gates_ignore:
        print("[train] gates_ignore=true: L0부터 재학습")
    elif gates_dir:
        probe_json_in = _find_json(gates_dir, "classification_HIs.json", "scenario_classification_HIs.json")
        scen_json_in  = _find_json(gates_dir, "regression_HIs.json",     "scenario_regression_HIs.json")
        print(f"[train] gates_from: {gates_dir}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds, norm = build_datasets(cfg)

    tr_cfg = cfg["training"]
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds, batch_size=tr_cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr_cfg["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=pin,
    )

    # ------------------------------------------------------------------
    # JSON cache → fixed masks
    # ------------------------------------------------------------------

    m = cfg["classifier"]["probe_m_count"]
    k = cfg["regression"]["scen_k_count"]
    probe_mask = _load_probe_mask_from_json(probe_json_in, m) if probe_json_in else None
    scen_masks = _load_scen_masks_from_json(scen_json_in, k)  if scen_json_in  else None

    if probe_mask is not None:
        print(f"[train] Using cached probe mask from {probe_json_in} (m={probe_mask.sum()})")
    if scen_masks is not None:
        print(f"[train] Using cached scen masks from {scen_json_in}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    m_cfg = cfg["model"]
    model = SCRModel(
        d_probe=m_cfg["d_probe"],
        d_head=m_cfg["d_head"],
        dropout=m_cfg["dropout"],
        probe_mask=probe_mask,
        scen_masks=scen_masks,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] SCRModel  trainable params: {n_params:,}")

    # ------------------------------------------------------------------
    # Dynamic lambda_l0 (Phase 1 전용: L0 게이트가 살아있을 때만)
    # Phase 1b (probe_mask != None) 는 SCRLoss에서 l0=0으로 자동 처리되므로 불필요
    # ------------------------------------------------------------------
    if probe_mask is None and cfg["loss"].get("lambda_l0_auto", False):
        m = cfg["classifier"]["probe_m_count"]
        k = cfg["regression"]["scen_k_count"]
        auto_lambda = _compute_lambda_l0(m, k)
        cfg["loss"]["lambda_l0"] = auto_lambda
        print(f"[train] lambda_l0_auto: m={m}, k={k} → lambda_l0={auto_lambda:.5f}")
    else:
        print(f"[train] lambda_l0={cfg['loss']['lambda_l0']:.5f}"
              + (" (Phase 1b: L0=0 자동 적용)" if probe_mask is not None else ""))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = SCRTrainer(model, cfg, output_dir, device, normalizer=norm)
    model = trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # Save JSON masks — 항상 새 run 폴더에 저장 (test_scr.py가 gates/ 에서 찾음)
    # 캐시에서 불러온 경우: 원본 JSON 복사
    # L0 학습한 경우: 모델에서 추출해 저장
    # ------------------------------------------------------------------
    from utils.hi_schema import get_hi_cols_for_seg
    hi_cols_dis_hi = get_hi_cols_for_seg("dis_hi")

    if probe_mask is not None and probe_json_in is not None:
        probe_json_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(probe_json_in, probe_json_out)
        print(f"[train] Copied probe gate JSON -> {probe_json_out}")
    else:
        _save_probe_mask_to_json(model, probe_json_out, hi_cols_dis_hi)

    if scen_masks is not None and scen_json_in is not None:
        scen_json_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(scen_json_in, scen_json_out)
        print(f"[train] Copied scen gate JSON  -> {scen_json_out}")
    else:
        hi_by_seg = {s: get_hi_cols_for_seg(SEGMENTS[s]) for s in range(N_SEGS)}
        _save_scen_masks_to_json(model, scen_json_out, hi_by_seg)

    # ------------------------------------------------------------------
    # Final checkpoint (best 모델 복원 후 저장)
    # ------------------------------------------------------------------
    from training.scr_trainer import _save_model as _save_ckpt
    ckpt_path = output_dir / "checkpoints" / "final.pt"
    _save_ckpt(model, ckpt_path, cfg, norm)
    print(f"[train] Saved final checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
