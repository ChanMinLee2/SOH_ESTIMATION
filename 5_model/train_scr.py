"""
train_scr.py  —  SCR training entry point.

Two-phase training:

  --phase 1 (Gate learning)
    L0 penalty로 최적 HI 서브셋을 탐색한다.
    charge_probe_gate / discharge_probe_gate 가 각 방향에서 m개를 선정하고,
    6개 scen_gate 가 시나리오별 k개를 선정한다.
    결과: gates/classification_HIs.json + gates/regression_HIs.json + gates/gate_probs.png

  --phase 2 (Classification + Regression 정밀 학습)
    Phase 1 JSON을 로드해 gate를 고정하고, probe_mlp + cap_head만 학습한다.
    L0 페널티 없음 (순수 MSE + CE 최소화).
    결과: checkpoints/ + figures/ + metrics/ + predictions/ + routing/

Usage:
  python 5_model/train_scr.py --phase 1
  python 5_model/train_scr.py --phase 2
  python 5_model/train_scr.py --phase 2 --gates-from _5_data_model_scr/0708_1533
  python 5_model/train_scr.py --phase 1 --charge-m 3 --discharge-m 1 --scen-k 5

Legacy alias (backward compat):
  gates_from in yaml  → equivalent to --phase 2 (미지정 시)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

try:
    from utils.compat import install_numpy2_shim
    install_numpy2_shim()
except ImportError:
    pass

import torch
from torch.utils.data import DataLoader

from utils.io_utils import load_config, save_config, save_json
from datasets.segment_dataset import build_datasets, collate_fn, FastTensorLoader
from models.scr_model import SCRModel
from training.scr_trainer import SCRTrainer
from utils.hi_schema import N_HI, spec_from_qfrac, EXCLUDE_STAT_LEAK

_proj_root_common = Path(__file__).resolve().parent.parent
if str(_proj_root_common) not in sys.path:
    sys.path.insert(0, str(_proj_root_common))
from common.scenario import get_segmenter as _get_segmenter
from common.scenario.base import ScenarioSpec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SCR model")
    p.add_argument("--config",      default="5_model/config/scr.yaml")
    p.add_argument("--phase",       type=int, choices=[1, 2], default=None,
                   help="1=Gate 학습(HI 서브셋 선정), 2=분류·회귀 정밀 학습")
    p.add_argument("--charge-m",    type=int, default=None,
                   help="Phase 2에서 충전 probe 상위 m개 사용 (yaml charge_probe_m 오버라이드)")
    p.add_argument("--discharge-m", type=int, default=None,
                   help="Phase 2에서 방전 probe 상위 m개 사용 (yaml discharge_probe_m 오버라이드)")
    p.add_argument("--scen-k",      type=int, default=None,
                   help="시나리오별 scen HI 수 오버라이드")
    p.add_argument("--gates-from",  default=None,
                   help="Phase 2 시 이전 run 폴더 경로 (gates JSON 자동 탐색). "
                        "미지정 시 yaml의 gates_from 사용")
    p.add_argument("--synergy-groups-json", default=None, dest="synergy_groups_json",
                   help="Phase 1 전용: build_synergy_groups.py 산출물(synergy_groups_*.json) 경로. "
                        "주어지면 scen_gates가 시나리오별로 그룹 단위 계층 게이트"
                        "(GroupedHardConcreteGate)로 학습됨 — 미지정 시 기존과 동일(개별 게이트)")
    p.add_argument("--device",      default="auto")
    # 시나리오 축
    p.add_argument("--seg-axis",    default=None,
                   help="세그멘테이션 축 (qfrac|protocol|vwindow|rcs|cluster). "
                        "미지정 시 yaml scenario.axis 또는 qfrac 사용")
    p.add_argument("--axis-config", default=None,
                   help="축 파라미터 JSON 문자열 (예: '{\"n_windows\": 4}')")
    p.add_argument("--seed",        type=int, default=None,
                   help="재현성 시드 — 모델 초기화/torch·numpy·random RNG (yaml training.seed 오버라이드)")
    p.add_argument("--split-seed",  type=int, default=None,
                   help="train/val/test 셀 분할 시드 (yaml data.split_seed 오버라이드)")
    p.add_argument("--exclude-cv",  action="store_true", dest="exclude_cv",
                   help="hi_correlation.py --exclude-cv 로 추출된 '_ccOnly' 경로 사용 "
                        "(data_dir/seg_data_dir 미지정 시 자동 경로에 접미사 추가). "
                        "yaml data.exclude_cv 로도 설정 가능")
    p.add_argument("--skip-shape",  action="store_true", dest="skip_shape",
                   help="2_preprocess/preprocess.py --skip-shape 로 만든 필터7(형상 이상치) "
                        "미적용 데이터 사용 — hi_correlation.py --skip-shape 로 추출된 "
                        "'_noshape' 경로를 그대로 사용(data_dir/seg_data_dir 미지정 시 자동 "
                        "접미사 추가, --exclude-cv와 동일 패턴). yaml data.skip_shape 로도 설정 가능")
    p.add_argument("--with-raw-cnn", action="store_true", dest="with_raw_cnn",
                   help="Phase 2 전용(REGRESSION_UPGRADE.md §5/§8): 회귀 헤드에 raw V/|I| "
                        "CNN 임베딩 융합(yaml model.with_raw_cnn 오버라이드). Phase 1은 "
                        "무조건 비활성화되므로 이 플래그를 줘도 무시됨")
    p.add_argument("--raw-cnn-pretrained-from", default=None, dest="raw_cnn_pretrained_from",
                   help="--with-raw-cnn과 함께: classifier clf_best.pt 경로를 주면 그 RawCNN을 "
                        "얼려서 재사용(방안 a). 미지정 시 랜덤 초기화 후 Phase2와 함께 학습(방안 b). "
                        "(yaml model.raw_cnn_pretrained_from 오버라이드)")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Phase 결정 헬퍼
# ---------------------------------------------------------------------------

def _resolve_phase(args: argparse.Namespace, cfg: dict) -> int:
    """
    --phase > yaml gates_from 유무 순서로 phase를 결정한다.
    """
    if args.phase is not None:
        return args.phase
    # gates_from이 설정되어 있으면 Phase 2로
    if cfg.get("data", {}).get("gates_from"):
        print("[train] yaml gates_from 설정 감지 → Phase 2 (legacy behavior)")
        return 2
    # 기본: Phase 1
    print("[train] phase 미지정 → Phase 1 (기본)")
    return 1


# ---------------------------------------------------------------------------
# Gates directory / JSON resolver
# ---------------------------------------------------------------------------

def _resolve_gates_dir(cli_arg: str | None, cfg: dict) -> Path | None:
    raw = cli_arg or cfg.get("data", {}).get("gates_from")
    if not raw:
        return None
    run_dir = PROJECT_ROOT / raw
    gates_sub = run_dir / "gates"
    return gates_sub if gates_sub.exists() else run_dir


def _find_json(gates_dir: Path, new_name: str, old_name: str) -> Path | None:
    for name in (new_name, old_name):
        p = gates_dir / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# JSON mask loaders — dual probe (charge + discharge)
# ---------------------------------------------------------------------------

def _load_probe_masks_from_json(
    json_path: Path, charge_m: int, discharge_m: int,
    auto: bool = False, threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    신형식: {"charge_ranked": [...], "charge_probs": [...], "discharge_ranked": [...], ...}
    구형식: {"ranked_indices": [...]}  → 충/방전 모두 동일하게 적용 (backward compat)

    auto=False : charge_ranked[:charge_m], discharge_ranked[:discharge_m] 슬라이싱
    auto=True  : gate_prob >= threshold 인 HI만 선택 (개수 자동 결정)
    Returns (charge_mask, discharge_mask) each (N_HI,) bool
    """
    if not json_path.exists():
        return None, None
    data = json.loads(json_path.read_text())

    def _to_mask(indices: list[int]) -> torch.Tensor:
        mask = torch.zeros(N_HI, dtype=torch.bool)
        for i in indices:
            mask[i] = True
        return mask

    def _threshold_mask(ranked: list[int], probs: list[float]) -> torch.Tensor:
        selected = [idx for idx, p in zip(ranked, probs) if p >= threshold]
        if not selected:
            selected = ranked[:1]
        return _to_mask(selected)

    if "charge_ranked" in data:
        if auto:
            ch_mask  = _threshold_mask(data["charge_ranked"],    data.get("charge_probs", []))
            dis_mask = _threshold_mask(data["discharge_ranked"], data.get("discharge_probs", []))
            print(f"[train] probe masks (auto thr={threshold}): "
                  f"charge {ch_mask.sum().item()}, discharge {dis_mask.sum().item()}")
        else:
            ch_mask  = _to_mask(data["charge_ranked"][:charge_m])
            dis_mask = _to_mask(data["discharge_ranked"][:discharge_m])
            print(f"[train] probe masks (new format): charge top-{charge_m}, discharge top-{discharge_m}")
    else:
        # backward compat: single ranking → apply to both
        indices = data.get("ranked_indices", [])
        if auto:
            probs    = data.get("probs", [1.0] * len(indices))
            ch_mask  = _threshold_mask(indices, probs)
            dis_mask = _threshold_mask(indices, probs)
            print(f"[train] probe masks (legacy auto thr={threshold}): "
                  f"charge {ch_mask.sum().item()}, discharge {dis_mask.sum().item()}")
        else:
            ch_mask  = _to_mask(indices[:charge_m])
            dis_mask = _to_mask(indices[:discharge_m])
            print(f"[train] probe masks (legacy format): charge top-{charge_m}, discharge top-{discharge_m}")

    return ch_mask, dis_mask


def _load_scen_masks_from_json(
    json_path: Path, k: int,
    auto: bool = False, threshold: float = 0.5,
    n_scenarios: int = 6, n_hi: int | None = None,
) -> torch.Tensor | None:
    if not json_path.exists():
        return None
    _n_hi = n_hi or N_HI
    data  = json.loads(json_path.read_text())
    masks = torch.zeros(n_scenarios, _n_hi, dtype=torch.bool)
    for s in range(n_scenarios):
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

    if auto:
        avg_k = masks.sum(dim=1).float().mean().item()
        print(f"[train] scen masks (auto thr={threshold}): avg {avg_k:.1f}/seg")
    else:
        print(f"[train] scen masks: top-{k}/scenario")
    return masks


# ---------------------------------------------------------------------------
# JSON savers — dual probe
# ---------------------------------------------------------------------------

def _load_synergy_group_ids(
    json_path: Path,
    n_scenarios: int,
    scenario_names: list[str],
) -> dict[int, list[int]]:
    """build_synergy_groups.py의 seg_{s}_groups(HI 인덱스 묶음)를 GroupedHardConcreteGate용
    group_ids({scenario_idx: [group_id per HI]})로 변환. 시너지 그룹 생성 시점과 지금 학습이
    같은 SOH_EXCLUDE_STAT_LEAK 설정을 썼는지(=HI 개수가 N_HI와 일치하는지)를 검증한다 —
    안 맞으면 인덱스가 다른 HI를 가리키게 되어 조용히 잘못된 그룹으로 학습될 수 있다."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: dict[int, list[int]] = {}
    for s in range(n_scenarios):
        key = f"seg_{s}_groups"
        if key not in data:
            print(f"[train] synergy-groups-json에 시나리오 {s}({scenario_names[s]}) 없음 "
                  f"— 이 시나리오는 그룹 없이 개별 게이트로 학습")
            continue
        groups = data[key]
        n_hi_json = sum(len(g) for g in groups)
        if n_hi_json != N_HI:
            raise ValueError(
                f"synergy-groups-json 시나리오 {s}({scenario_names[s]})의 HI 개수({n_hi_json})가 "
                f"현재 N_HI({N_HI})와 다릅니다 — SOH_EXCLUDE_STAT_LEAK 설정이 그룹 생성 시점과 "
                f"다른 것으로 보입니다. 같은 설정으로 build_synergy_groups.py를 다시 실행하세요."
            )
        group_ids = [-1] * N_HI
        for g_idx, members in enumerate(groups):
            for m in members:
                group_ids[m] = g_idx
        out[s] = group_ids
    return out


def _ranked_indices(gate) -> tuple[list[int], list[float]]:
    prob       = gate.gate_prob().detach().cpu()
    sorted_idx = prob.argsort(descending=True).tolist()
    sorted_prob = [round(float(prob[i]), 6) for i in sorted_idx]
    return sorted_idx, sorted_prob


def _save_probe_masks_to_json(
    model: SCRModel,
    json_path: Path,
    hi_cols_ref: list[str],
) -> None:
    """Phase 1: charge/discharge probe gate_prob 전체 랭킹을 저장."""
    ch_ranked,  ch_probs  = _ranked_indices(model.charge_probe_gate)
    dis_ranked, dis_probs = _ranked_indices(model.discharge_probe_gate)
    out = {
        "charge_ranked":    ch_ranked,
        "charge_names":     [hi_cols_ref[i] for i in ch_ranked],
        "charge_probs":     ch_probs,
        "discharge_ranked": dis_ranked,
        "discharge_names":  [hi_cols_ref[i] for i in dis_ranked],
        "discharge_probs":  dis_probs,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[train] Saved probe HI ranking → {json_path}  (charge/discharge 각 {len(ch_ranked)}개 랭킹)")


def _save_scen_masks_to_json(
    model: SCRModel,
    json_path: Path,
    hi_cols_by_seg: dict[int, list[str]],
    gates=None,
) -> None:
    """Phase 1: 시나리오별 gate_prob 전체 랭킹을 저장.

    gates: 기본 None이면 model.scen_gates(raw HI) 사용 — 기존과 100% 동일 동작.
    model.scen_kernel_gates를 넘기면 커널 융합 HI 블록(build_kernel_group_features.py)의
    랭킹을 같은 형식으로 저장할 수 있다(phase1_trainer_v2.py에서 재사용)."""
    gates = gates if gates is not None else model.scen_gates
    out = {}
    seg_names = model.spec.scenario_names
    for s in range(model.n_scenarios):
        ranked, probs = _ranked_indices(gates[s])
        out[f"seg_{s}_ranked"]   = ranked
        out[f"seg_{s}_names"]    = [hi_cols_by_seg[s][i] for i in ranked]
        out[f"seg_{s}_probs"]    = probs
        out[f"seg_{s}_seg_name"] = seg_names[s]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_hi_out = len(next(iter(hi_cols_by_seg.values())))
    print(f"[train] Saved scen HI ranking → {json_path}  (시나리오별 {n_hi_out}개 랭킹)")


# ---------------------------------------------------------------------------
# Phase 1 시각화 — gate_prob bar chart
# ---------------------------------------------------------------------------

def _plot_gate_probs(
    model: SCRModel,
    output_path: Path,
    hi_cols_ref: list[str],
    charge_m: int,
    discharge_m: int,
    scen_k: int,
) -> None:
    """
    8개 서브플롯: charge probe / discharge probe / 6 scen gates
    x축: HI 인덱스, y축: gate_prob. threshold(m/k) 기준선 표시.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[train] matplotlib 미설치 — gate_probs.png 생략")
        return

    gates_info = [
        ("Charge Probe",    model.charge_probe_gate,    charge_m,    "steelblue"),
        ("Discharge Probe", model.discharge_probe_gate, discharge_m, "darkorange"),
    ]
    seg_names = model.spec.scenario_names
    for s in range(model.n_scenarios):
        gates_info.append((f"Scen: {seg_names[s]}", model.scen_gates[s], scen_k, "seagreen"))

    n_plots = len(gates_info)   # 8
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    axes = axes.flatten()

    is_grouped = any(hasattr(gate, "group_index") for _, gate, _, _ in gates_info)

    for ax, (title, gate, threshold, color) in zip(axes, gates_info):
        prob = gate.gate_prob().detach().cpu().numpy()
        idx  = np.arange(len(prob))
        sorted_idx = np.argsort(prob)[::-1]
        sorted_prob = prob[sorted_idx]

        if hasattr(gate, "group_index"):
            # 그룹 계층 게이트 — 막대를 그룹별 색으로 칠해서 같은 그룹 멤버가 랭킹에서
            # 뭉쳐 있는지(=그룹이 실제로 같이 움직인다) 한눈에 보이게 하고, 그룹 자체의
            # 게이트 확률(멤버 오프셋 제외)을 점선으로 겹쳐 그린다.
            group_idx = gate.group_index.detach().cpu().numpy()
            sorted_groups = group_idx[sorted_idx]
            cmap = plt.cm.tab20(np.linspace(0, 1, max(gate.n_groups, 1)))
            bar_colors = cmap[sorted_groups % 20]
            ax.bar(range(len(sorted_prob)), sorted_prob, color=bar_colors, alpha=0.85)
            group_prob = gate.group_gate_prob().detach().cpu().numpy()
            ax.plot(range(len(sorted_prob)), group_prob[sorted_groups],
                    color="black", linestyle=":", linewidth=1.0, alpha=0.7,
                    label=f"group_gate_prob ({gate.n_groups} groups)")
        else:
            ax.bar(range(len(sorted_prob)), sorted_prob, color=color, alpha=0.7)

        if threshold <= len(sorted_prob):
            cutoff = float(sorted_prob[threshold - 1]) if threshold > 0 else 1.0
            ax.axvline(x=threshold - 0.5, color="red", linestyle="--", linewidth=1.2,
                       label=f"top-{threshold} cutoff")
            ax.axhline(y=cutoff, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("HI rank", fontsize=8)
        ax.set_ylabel("gate_prob", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        # 상위 5개 이름 표시
        for rank in range(min(5, len(sorted_idx))):
            hi_name = hi_cols_ref[sorted_idx[rank]]
            short   = hi_name.split("_dis_hi")[0].split("_chg_lo")[0]
            ax.text(rank, sorted_prob[rank] + 0.01, short,
                    rotation=90, fontsize=5, ha="center", va="bottom")

    _suptitle = "Phase 1 — Gate Probability by HI (sorted desc)"
    if is_grouped:
        _suptitle += "  [scen gates: grouped — bar color=synergy group, dotted=group_gate_prob]"
    fig.suptitle(_suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Saved gate_prob plot → {output_path}")


# ---------------------------------------------------------------------------
# Dynamic lambda_l0
# ---------------------------------------------------------------------------

_REF_LAMBDA = 0.01
_REF_M      = 10
_REF_K      = 10


def _compute_lambda_l0(m: int, k: int) -> float:
    probe_scale = _REF_M / max(m, 1)
    scen_scale  = _REF_K / max(k, 1)
    scale = math.sqrt(probe_scale * scen_scale)
    return round(max(1e-4, min(_REF_LAMBDA * scale, 0.5)), 5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse_args()
    device = _resolve_device(args.device)
    print(f"[train] device={device}")

    cfg   = load_config(str(PROJECT_ROOT / args.config))
    phase = _resolve_phase(args, cfg)
    print(f"[train] ═══ Phase {phase} ═══")

    # 시드 적용 — 이전엔 --seed가 cfg["training"]["seed"]에 저장만 되고 실제로
    # torch/numpy/random에 전달되지 않아 사실상 no-op이었다(2026-08-12 발견,
    # docs/260811_RESULTS.md 참고). 모델 초기화(HardConcreteGate log_alpha,
    # cap_head 가중치 등)의 재현성을 위해 여기서 직접 시드를 건다.
    _seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed")
    if _seed is not None:
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        cfg.setdefault("training", {})["seed"] = _seed
        print(f"[train] seed={_seed} (torch/numpy/random 전부 적용)")

    # split-seed 오버라이드 — train/val/test 셀 분할 (yaml에만 있던 data.split_seed의
    # CLI 대체 경로, 2026-08-12 추가)
    if args.split_seed is not None:
        cfg.setdefault("data", {})["split_seed"] = args.split_seed
        print(f"[train] split_seed={args.split_seed}")

    # with_raw_cnn/raw_cnn_pretrained_from CLI 오버라이드 (run_pipeline.py가 분류기 run_dir을
    # 자동 주입할 때 사용 — yaml에 값이 없어도 파이프라인 한 번에 이어붙일 수 있게 함).
    _m_cfg_cli = cfg.setdefault("model", {})
    if args.with_raw_cnn:
        _m_cfg_cli["with_raw_cnn"] = True
    if args.raw_cnn_pretrained_from is not None:
        _m_cfg_cli["raw_cnn_pretrained_from"] = args.raw_cnn_pretrained_from

    # raw_mlp 모드: HI/게이트 없이 raw_v/raw_i(flatten)만으로 회귀하는 베이스라인
    # (MODEL_DIRECTION.md B-4 계열 — "HI 추출 자체의 기여도" 측정용)
    _is_raw_mode = cfg.get("model", {}).get("regression_model") == "raw_mlp"
    # with_raw_cnn: 회귀 헤드에 raw CNN 임베딩 융합(REGRESSION_UPGRADE.md §5/§8, 방안2) — Phase 2 전용.
    # with_raw_flat: 회귀 헤드에 raw를 압축 없이 flatten 융합(REGRESSION_UPGRADE.md §2, 방안1) — Phase 2 전용.
    # 둘 다 Phase 1은 아래에서 _p1_model_cfg가 강제로 False 처리한다(게이트 선정과 무관한 옵션).
    _use_raw_cnn  = bool(cfg.get("model", {}).get("with_raw_cnn", False))
    _use_raw_flat = bool(cfg.get("model", {}).get("with_raw_flat", False))

    # CLI 오버라이드
    cls_cfg = cfg.setdefault("classifier", {})
    reg_cfg = cfg.setdefault("regression", {})
    if args.charge_m    is not None: cls_cfg["charge_probe_m"]    = args.charge_m
    if args.discharge_m is not None: cls_cfg["discharge_probe_m"] = args.discharge_m
    if args.scen_k      is not None: reg_cfg["scen_k_count"]      = args.scen_k

    charge_m    = cls_cfg.get("charge_probe_m",    cls_cfg.get("probe_m_count", 1))
    discharge_m = cls_cfg.get("discharge_probe_m", cls_cfg.get("probe_m_count", 1))
    scen_k      = reg_cfg.get("scen_k_count", 5)
    auto_mk     = cls_cfg.get("is_auto_mk_selection", False)

    # 시나리오 축 spec 결정 (CLI > yaml scenario.axis > qfrac)
    _axis_name = (args.seg_axis
                  or cfg.get("scenario", {}).get("axis", "qfrac"))
    _axis_cfg_raw = args.axis_config or cfg.get("scenario", {}).get("axis_config", None)
    _axis_cfg: dict = json.loads(_axis_cfg_raw) if isinstance(_axis_cfg_raw, str) else (_axis_cfg_raw or {})
    # CLI로 준 --seg-axis/--axis-config가 저장되는 config.yaml에도 보이도록 반영
    cfg.setdefault("scenario", {})["axis"] = _axis_name
    cfg["scenario"]["axis_config"] = _axis_cfg
    _segmenter = _get_segmenter(_axis_name, {_axis_name: _axis_cfg})
    spec = _segmenter.get_spec()
    print(f"[train] seg-axis={_axis_name}  n_scenarios={spec.n_scenarios}  n_classes={spec.n_classes}")

    # Phase 2: gates_from의 scenario_spec.json으로 spec을 여기서 바로 확정한다.
    # (2026-07-31 버그 수정) 이전엔 이 재할당이 build_datasets/spec.save() 이후, Phase 2
    # 분기 안에서야 일어났다 — 그 사이에 이미 spec.save()가 "지금 코드 기준" spec을
    # 파일로 저장해버려서, 저장된 scenario_spec.json이 실제로 이 run의 학습에 쓰인
    # spec(gates_from 시점 것)과 어긋날 수 있었다. 시나리오 축 코드(예: _ROUTING)가
    # gates_from run 이후 바뀌면, test_scr.py가 나중에 그 "어긋난" 파일을 읽어 "구"
    # 모델(옛 routing으로 학습됨)을 "신" routing으로 잘못 해석해 평가하게 되고, hard
    # 라우팅·분류 정확도·routing_gap_pct가 전부 깨진다(oracle은 seg_idx를 직접 쓰므로
    # 영향 없음 — docs/260731_RESULTS.md에서 실측으로 확인됨). build_datasets/spec.save
    # 전에 최종 spec을 먼저 확정해 이 어긋남 자체를 없앤다.
    _gates_dir: "Path | None" = None
    if phase == 2:
        _gates_dir = _resolve_gates_dir(args.gates_from, cfg)
        if _gates_dir is None:
            raise RuntimeError(
                "Phase 2는 gates JSON이 필요합니다. "
                "--gates-from <run_dir> 또는 yaml의 gates_from을 설정하세요."
            )
        _p2_spec_path = _gates_dir.parent / "scenario_spec.json"
        if _p2_spec_path.exists():
            spec = ScenarioSpec.load(_p2_spec_path)
            print(f"[train] Phase 2 spec loaded: {_p2_spec_path}")

    # random_segment=True 면 태그에 _random-L{seg_len_pts} suffix (hi_correlation._rand_suffix 와 동일)
    _rand_sfx = (f"_random-L{int(_axis_cfg.get('seg_len_pts', 20))}"
                 if _axis_cfg.get("random_segment", False) else "")
    # min_pts 기본값(10)이 아니면 접미사 (hi_correlation._qfw_tag 와 동일 규칙, 2026-08-10)
    _min_pts = int(_axis_cfg.get("min_pts", 10))
    _minpts_sfx = f"_minpts{_min_pts}" if _min_pts != 10 else ""
    # assign="none"(no_scen 대조군, docs/260816_RESULTS.md §5)이면 접미사 —
    # hi_correlation._qfw_tag 와 동일 규칙. 이게 없으면 spec은 n_scenarios=2로 구성되는데
    # 실제로는 position_bin(6-시나리오) 캐시를 읽어버려 segment_id(0~5)가 spec의
    # n_scenarios=2 매핑 테이블(_id_to_lvl 등, segment_dataset.py)에 없는 값이 되고,
    # 그 결과 level 컬럼이 NaN → astype(int64)에서 pandas.errors.IntCastingNaNError가 난다.
    _assign_sfx = "" if _axis_cfg.get("assign", "position_bin") == "position_bin" else "_noscen"
    # q_frac_wide: n1/n2/n_samples 별 하위 디렉터리 결정
    if _axis_name == "q_frac_wide":
        _n1 = int(round(_axis_cfg.get("n1", 0.4) * 100))
        _n2 = int(round(_axis_cfg.get("n2", 0.2) * 100))
        _ns = int(_axis_cfg.get("n_samples", 4))
        _axis_dir = f"q_frac_wide/n1-{_n1}%_n2-{_n2}%_N-{_ns}{_rand_sfx}{_minpts_sfx}{_assign_sfx}"
    elif _axis_name == "q_frac_ref":
        # q_frac_ref: n1/n2/n_samples(q_frac_wide와 동일 규칙) + ref_lag/noise_amp
        # (hi_correlation._qfref_tag 와 동일 규칙)
        _n1 = int(round(_axis_cfg.get("n1", 0.4) * 100))
        _n2 = int(round(_axis_cfg.get("n2", 0.2) * 100))
        _ns = int(_axis_cfg.get("n_samples", 4))
        _lag = int(_axis_cfg.get("ref_lag", 0))
        _noise = int(round(_axis_cfg.get("noise_amp", 0.03) * 100))
        _nmode = str(_axis_cfg.get("noise_mode", "ou"))
        _period = int(round(_axis_cfg.get("noise_period_cycles", 200.0)))
        _axis_dir = (f"q_frac_ref/n1-{_n1}%_n2-{_n2}%_N-{_ns}{_rand_sfx}{_minpts_sfx}{_assign_sfx}"
                     f"_lag-{_lag}_noise-{_noise}%_{_nmode}-{_period}")
    elif _axis_name == "q_abs":
        # q_abs: mid_start/mid_end/seg_len/n_samples 별 하위 디렉터리 (hi_correlation._qabs_tag 와 동일)
        _ms = int(round(_axis_cfg.get("mid_start", 0.20) * 100))
        _me = int(round(_axis_cfg.get("mid_end", 0.50) * 100))
        _sl = int(round(_axis_cfg.get("seg_len", 0.15) * 100))
        _ns = int(_axis_cfg.get("n_samples", 4))
        _axis_dir = f"q_abs/ms-{_ms}%_me-{_me}%_sl-{_sl}%_N-{_ns}{_rand_sfx}"
    elif _axis_name == "vqslope":
        # vqslope: mode(dva/ica)·n_samples 별 하위 디렉터리
        _vqmode = str(_axis_cfg.get("mode", "dva")).lower()
        _ns     = int(_axis_cfg.get("n_samples", 1))
        _axis_dir = f"vqslope/{_vqmode}_N-{_ns}{_rand_sfx}"
    else:
        _axis_dir = _axis_name

    # --exclude-cv: hi_correlation.py --exclude-cv 로 추출된 '_ccOnly' 경로 사용
    # (CLI > yaml data.exclude_cv). 이 run의 cfg는 그대로 checkpoint에 저장되므로
    # test_scr.py/finetune_scr.py는 별도 처리 없이 checkpoint에 담긴 경로를 그대로 재사용한다.
    _exclude_cv = args.exclude_cv or bool(cfg.get("data", {}).get("exclude_cv", False))
    if _exclude_cv:
        _axis_dir = f"{_axis_dir}_ccOnly"

    # --skip-shape: hi_correlation.py --skip-shape 로 추출된 '_noshape' 경로 사용
    # (CLI > yaml data.skip_shape). hi_correlation.py의 접미사 순서(_ccOnly 다음 _noshape,
    # hi_correlation.py:2419/2423 확인)와 동일하게 맞춘다.
    _skip_shape = args.skip_shape or bool(cfg.get("data", {}).get("skip_shape", False))
    if _skip_shape:
        _axis_dir = f"{_axis_dir}_noshape"

    # 데이터 경로: null → scenario.axis 기반 자동 결정
    # 2026-08-08: _4_data_hi를 D 드라이브로 이동 — 절대경로로 주면 segment_dataset.py의
    # `PROJECT_ROOT / data_cfg["seg_data_dir"]` join에서 절대경로 쪽이 우선(pathlib 표준
    # 동작)해 PROJECT_ROOT(C:)는 무시되고 D: 경로가 그대로 쓰인다.
    _data_cfg = cfg["data"]
    if not _data_cfg.get("seg_data_dir"):
        _data_cfg["seg_data_dir"] = f"{DATA_4_HI_ROOT_STR}/{_axis_dir}/seg"
    if not _data_cfg.get("data_dir"):
        _data_cfg["data_dir"] = f"{DATA_4_HI_ROOT_STR}/{_axis_dir}/cycle"
    _data_cfg["exclude_cv"] = _exclude_cv  # checkpoint에 저장되는 cfg에 명시 (재현/추적용)
    _data_cfg["skip_shape"] = _skip_shape  # checkpoint에 저장되는 cfg에 명시 (재현/추적용)
    if args.gates_from is not None:
        _data_cfg["gates_from"] = args.gates_from  # CLI --gates-from도 저장되는 config.yaml에 반영
    print(f"[train] data_dir={_data_cfg['data_dir']}")
    print(f"[train] seg_data_dir={_data_cfg['seg_data_dir']}")
    print(f"[train] exclude_cv={_exclude_cv}")

    # seed 오버라이드
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    _AXIS_SHORT  = {"qfrac": "qfr", "protocol": "prot", "vwindow": "vwin",
                    "rcs": "rcs", "cluster": "clst", "q_frac_wide": "qfw",
                    "q_abs": "qabs", "vqslope": "vqs"}
    _MODEL_SHORT = {"mlp": "mlp", "transformer": "tr", "i_transformer": "itr",
                    "resnet_tab": "res", "ft_transformer": "ftt"}
    _axis_short  = _AXIS_SHORT.get(_axis_name, _axis_name[:4])
    if _axis_name == "q_frac_wide":
        _axis_short += f"_{_n1}%_{_n2}%"
    elif _axis_name == "q_frac_ref":
        _axis_short += f"_{_n1}%_{_n2}%"
    elif _axis_name == "q_abs":
        _axis_short += f"_{_ms}-{_me}%"
    elif _axis_name == "vqslope":
        _axis_short += f"_{_vqmode}"
    if _rand_sfx:
        _axis_short += "_rand"
    # SOH_EXCLUDE_STAT_LEAK=1로 프로세스를 띄운 경우 — N_HI가 64/66으로 달라져
    # 체크포인트 구조 자체가 다르므로(hi_schema.py 모듈 docstring 참고) run 디렉터리명에도
    # 반드시 남겨 나중에 --checkpoint로 잘못 섞어 로드하는 사고를 방지한다.
    if EXCLUDE_STAT_LEAK:
        _axis_short += "_noleak"
    _reg_model   = cfg.get("model", {}).get("regression_model", "mlp")
    _model_short = _MODEL_SHORT.get(_reg_model, _reg_model[:3])
    _phase_tag   = f"p{args.phase}" if args.phase else "p?"
    _suffix      = f"_{_phase_tag}_{_model_short}_{_axis_short}" if args.phase == 2 else f"_{_phase_tag}_{_axis_short}"
    timestamp  = datetime.now().strftime("%m%d_%H%M") + _suffix
    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"] / timestamp
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "gates").mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {output_dir}")
    print(f"[train] N_HI={N_HI} (SOH_EXCLUDE_STAT_LEAK={'1' if EXCLUDE_STAT_LEAK else '0'})")

    # run_dir에 spec 저장 (test_scr.py 재사용)
    spec.save(output_dir / "scenario_spec.json")

    # 원본 yaml을 그대로 복사하지 않고, CLI 오버라이드(--n1/--n2/--scen-k/--exclude-cv 등)가
    # 반영된 실제 실행 시점의 cfg를 저장 — 나중에 run_dir만 보고도 실행 조건을 알 수 있게 함.
    # exclude_stat_leak은 CLI 인자가 아니라 환경변수(SOH_EXCLUDE_STAT_LEAK)로만 제어되므로
    # (hi_schema.py 모듈 docstring 참고) 여기서 명시적으로 기록해 두지 않으면 저장된
    # config.yaml만 봐서는 그 run의 N_HI가 64였는지 66이었는지 알 수 없다.
    cfg.setdefault("data", {})["exclude_stat_leak"] = EXCLUDE_STAT_LEAK
    save_config(cfg, output_dir / "config.yaml")

    probe_json_out = output_dir / "gates" / "classification_HIs.json"
    scen_json_out  = output_dir / "gates" / "regression_HIs.json"

    # ------------------------------------------------------------------
    # 데이터
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds, norm = build_datasets(cfg, spec=spec)

    tr_cfg = cfg["training"]
    # FastTensorLoader: 사전 구축 텐서 슬라이싱 → per-sample collate 오버헤드 제거.
    # SCRModel 회귀 forward는 기본적으로 x_raw 미사용 → 학습 로더에서 제외(메모리/전송 절감).
    # raw_mlp 모드는 x_raw가 곧 유일한 모델 입력이라, with_raw_cnn/with_raw_flat 모드는
    # cap_head 융합에 x_raw가 필요하므로 반드시 포함해야 한다.
    _include_raw = _is_raw_mode or _use_raw_cnn or _use_raw_flat
    train_loader = FastTensorLoader(train_ds, tr_cfg["batch_size"], shuffle=True,
                                    include_raw=_include_raw)
    val_loader   = FastTensorLoader(val_ds,   tr_cfg["batch_size"], shuffle=False,
                                    include_raw=_include_raw)

    # ------------------------------------------------------------------
    # raw_mlp 모드: HI/게이트 완전히 우회 — Phase 1/2 구분이 무의미하므로
    # 두 phase 모두 동일하게 RawMLPModel을 처음부터 학습한다(비용이 작아 재학습 허용).
    # gates/ 디렉터리는 STEPS 호환을 위해 빈 채로 남긴다(probe_json_out/scen_json_out 미생성).
    # ------------------------------------------------------------------
    if _is_raw_mode:
        from models.raw_mlp_model import RawMLPModel

        print("[train] raw_mlp 모드 — HI/게이트 없이 raw_v/raw_i(flatten)만으로 직접 회귀 "
              "(베이스라인, MODEL_DIRECTION.md B-4 계열)")
        m_cfg = cfg.get("model", {})
        model = RawMLPModel(
            d_head=m_cfg.get("d_head", 128),
            dropout=m_cfg.get("dropout", 0.1),
            model_cfg=m_cfg,
            spec=spec,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] RawMLPModel  trainable params: {n_params:,}")

        trainer = SCRTrainer(model, cfg, output_dir, device, normalizer=norm)
        model   = trainer.fit(train_loader, val_loader)

        from training.scr_trainer import _save_model as _save_ckpt
        ckpt_path = output_dir / "checkpoints" / "final.pt"
        _save_ckpt(model, ckpt_path, cfg, norm)
        print(f"[train] Saved final checkpoint → {ckpt_path}")
        print(f"[train] Phase {phase} 완료 (raw_mlp). run dir: {output_dir}")
        return

    # ------------------------------------------------------------------
    # Phase 1: L0 gate 학습
    # ------------------------------------------------------------------
    if phase == 1:
        print(f"[train] Phase 1: charge_m={charge_m}, discharge_m={discharge_m}, scen_k={scen_k}")
        print("[train] L0 게이트 학습 — charge/discharge probe + 6 scen gates 독립 탐색")

        _lambda_scen = cfg.get("loss", {}).get("lambda_scen", 0.0)
        _with_probe_mlp = _lambda_scen > 0
        if _with_probe_mlp:
            print(f"[train] dual-objective: MSE + CE(lambda_scen={_lambda_scen}) — probe_mlp 활성화")
        else:
            print("[train] single-objective: MSE only — probe_mlp 비활성화")

        # Phase 1은 항상 MLPHead(design intent) — regression_model 선택은 Phase 2 전용이므로
        # 강제로 "mlp"만 적용하되, mlp_hidden_dims 등 MLPHead 세부 설정은 반영한다
        # (기존엔 model_cfg를 아예 안 넘겨서 mlp_hidden_dims가 무시되고 항상 [d_head, d_head//2]
        # 기본값으로 cap_head가 지어짐 — Phase 2/test_scr.py의 구성과 어긋나는 버그였음).
        # with_raw_cnn/with_raw_flat도 같은 이유로 Phase 1에서는 강제 비활성화한다 —
        # 게이트(HI 서브셋) 선정과 무관한 Phase 2 전용 옵션이라 Phase 1에 흘러들어가면 안 된다.
        _p1_model_cfg = {**cfg["model"], "regression_model": "mlp",
                          "with_raw_cnn": False, "with_raw_flat": False}

        _scen_group_ids = None
        if args.synergy_groups_json:
            _scen_group_ids = _load_synergy_group_ids(
                Path(args.synergy_groups_json), spec.n_scenarios, spec.scenario_names,
            )
            _n_grouped = sum(len(set(g)) for g in _scen_group_ids.values())
            print(f"[train] synergy-groups-json 적용: {args.synergy_groups_json} "
                  f"({len(_scen_group_ids)}/{spec.n_scenarios}개 시나리오, 총 그룹 {_n_grouped}개 "
                  f"— scen_gates가 GroupedHardConcreteGate로 학습됨)")

        model = SCRModel(
            d_probe=cfg["model"]["d_probe"],
            d_head=cfg["model"]["d_head"],
            dropout=cfg["model"]["dropout"],
            spec=spec,
            with_probe_mlp=_with_probe_mlp,
            model_cfg=_p1_model_cfg,
            scen_group_ids=_scen_group_ids,
            # 마스크 없음 → HardConcreteGate 활성화
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] SCRModel  trainable params: {n_params:,}")

        if cfg["loss"].get("lambda_l0_auto", False):
            avg_m = (charge_m + discharge_m) / 2
            auto_lambda = _compute_lambda_l0(int(avg_m), scen_k)
            cfg["loss"]["lambda_l0"] = auto_lambda
            print(f"[train] lambda_l0_auto: charge_m={charge_m}, discharge_m={discharge_m}, "
                  f"scen_k={scen_k} → lambda_l0={auto_lambda:.5f}")
        else:
            print(f"[train] lambda_l0={cfg['loss']['lambda_l0']:.5f}")

        trainer = SCRTrainer(model, cfg, output_dir, device, normalizer=norm)
        model   = trainer.fit(train_loader, val_loader)

        # JSON 저장
        from utils.hi_schema import get_hi_cols_for_seg
        hi_cols_ref    = get_hi_cols_for_seg("dis_hi")
        hi_cols_by_seg = {s: get_hi_cols_for_seg(model.spec.scenario_names[s])
                          for s in range(model.n_scenarios)}

        _save_probe_masks_to_json(model, probe_json_out, hi_cols_ref)
        _save_scen_masks_to_json(model, scen_json_out, hi_cols_by_seg)

        # gate_prob 시각화
        _plot_gate_probs(
            model,
            output_dir / "gates" / "gate_probs.png",
            hi_cols_ref,
            charge_m, discharge_m, scen_k,
        )

    # ------------------------------------------------------------------
    # Phase 2: 고정 게이트 재학습 (분류 + 회귀 정밀 학습)
    # ------------------------------------------------------------------
    else:  # phase == 2
        # gates_dir/spec은 위(spec 최초 생성 직후)에서 이미 확정했다 — 여기서 재계산하지 않는다.
        gates_dir = _gates_dir
        probe_json_in = _find_json(gates_dir, "classification_HIs.json",
                                   "scenario_classification_HIs.json")
        scen_json_in  = _find_json(gates_dir, "regression_HIs.json",
                                   "scenario_regression_HIs.json")
        print(f"[train] gates_from: {gates_dir}")

        charge_mask, discharge_mask = _load_probe_masks_from_json(
            probe_json_in, charge_m, discharge_m, auto=auto_mk,
        ) if probe_json_in else (None, None)
        scen_masks = _load_scen_masks_from_json(
            scen_json_in, scen_k, auto=auto_mk,
            n_scenarios=spec.n_scenarios,
        ) if scen_json_in else None

        if charge_mask is None or discharge_mask is None:
            raise RuntimeError(f"probe JSON 로드 실패: {probe_json_in}")
        if scen_masks is None:
            raise RuntimeError(f"scen JSON 로드 실패: {scen_json_in}")

        print(f"[train] Phase 2: charge_m={charge_mask.sum()}, "
              f"discharge_m={discharge_mask.sum()}, scen_k={scen_k}")
        print("[train] 고정 마스크 — probe_mlp + cap_head만 학습")

        m_cfg = cfg.get("model", {})
        reg_model = m_cfg.get("regression_model", "mlp")
        model = SCRModel(
            d_probe=m_cfg.get("d_probe", 64),
            d_head=m_cfg.get("d_head", 128),
            dropout=m_cfg.get("dropout", 0.1),
            charge_probe_mask=charge_mask,
            discharge_probe_mask=discharge_mask,
            scen_masks=scen_masks,
            model_cfg=m_cfg,
            spec=spec,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] SCRModel  trainable params: {n_params:,}  (cap_head: {reg_model})")
        print("[train] lambda_l0=0 (Phase 2: L0 페널티 자동 비활성)")

        trainer = SCRTrainer(model, cfg, output_dir, device, normalizer=norm)
        model   = trainer.fit(train_loader, val_loader)

        # gates JSON 복사 (test_scr.py가 gates/ 에서 탐색)
        shutil.copy(probe_json_in, probe_json_out)
        shutil.copy(scen_json_in,  scen_json_out)
        print(f"[train] Copied gate JSONs → {output_dir / 'gates'}/")

    # ------------------------------------------------------------------
    # Laplace UQ (Phase 2 전용, uq.enabled=true 시)
    # ------------------------------------------------------------------
    uq_cfg = cfg.get("uq", {})
    uq = None
    if phase == 2 and uq_cfg.get("enabled", False):
        print("[train] ── Laplace UQ 적합 시작 ──")
        uq = trainer.fit_laplace(train_loader, val_loader, uq_cfg)
        uq.save(output_dir / "checkpoints" / "laplace_uq.pt")

    # ------------------------------------------------------------------
    # Final checkpoint
    # ------------------------------------------------------------------
    from training.scr_trainer import _save_model as _save_ckpt
    ckpt_path = output_dir / "checkpoints" / "final.pt"
    _save_ckpt(model, ckpt_path, cfg, norm)
    print(f"[train] Saved final checkpoint → {ckpt_path}")
    print(f"[train] Phase {phase} 완료. run dir: {output_dir}")


if __name__ == "__main__":
    main()
