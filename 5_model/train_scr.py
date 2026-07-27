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

Legacy aliases (backward compat):
  --no-gates  → equivalent to --phase 1
  gates_from in yaml  → equivalent to --phase 2
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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
from datasets.segment_dataset import build_datasets, collate_fn, FastTensorLoader
from models.scr_model import SCRModel
from training.scr_trainer import SCRTrainer
from utils.hi_schema import N_HI, spec_from_qfrac

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
    # legacy aliases
    p.add_argument("--no-gates",    action="store_true",
                   help="[legacy] --phase 1과 동일")
    p.add_argument("--device",      default="auto")
    # 시나리오 축
    p.add_argument("--seg-axis",    default=None,
                   help="세그멘테이션 축 (qfrac|protocol|vwindow|rcs|cluster). "
                        "미지정 시 yaml scenario.axis 또는 qfrac 사용")
    p.add_argument("--axis-config", default=None,
                   help="축 파라미터 JSON 문자열 (예: '{\"n_windows\": 4}')")
    p.add_argument("--seed",        type=int, default=None,
                   help="재현성 시드 (yaml 오버라이드)")
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
    --phase > --no-gates(legacy) > yaml gates_from 유무 순서로 phase를 결정한다.
    """
    if args.phase is not None:
        return args.phase
    if args.no_gates:
        print("[train] --no-gates detected → Phase 1 (legacy alias)")
        return 1
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
) -> None:
    """Phase 1: 시나리오별 gate_prob 전체 랭킹을 저장."""
    out = {}
    seg_names = model.spec.scenario_names
    for s in range(model.n_scenarios):
        ranked, probs = _ranked_indices(model.scen_gates[s])
        out[f"seg_{s}_ranked"]   = ranked
        out[f"seg_{s}_names"]    = [hi_cols_by_seg[s][i] for i in ranked]
        out[f"seg_{s}_probs"]    = probs
        out[f"seg_{s}_seg_name"] = seg_names[s]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[train] Saved scen HI ranking → {json_path}  (시나리오별 {N_HI}개 랭킹)")


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

    for ax, (title, gate, threshold, color) in zip(axes, gates_info):
        prob = gate.gate_prob().detach().cpu().numpy()
        idx  = np.arange(len(prob))
        sorted_idx = np.argsort(prob)[::-1]
        sorted_prob = prob[sorted_idx]

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

    fig.suptitle("Phase 1 — Gate Probability by HI (sorted desc)", fontsize=13, fontweight="bold")
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
    _segmenter = _get_segmenter(_axis_name, {_axis_name: _axis_cfg})
    spec = _segmenter.get_spec()
    print(f"[train] seg-axis={_axis_name}  n_scenarios={spec.n_scenarios}  n_classes={spec.n_classes}")

    # random_segment=True 면 태그에 _random-L{seg_len_pts} suffix (hi_correlation._rand_suffix 와 동일)
    _rand_sfx = (f"_random-L{int(_axis_cfg.get('seg_len_pts', 20))}"
                 if _axis_cfg.get("random_segment", False) else "")
    # q_frac_wide: n1/n2/n_samples 별 하위 디렉터리 결정
    if _axis_name == "q_frac_wide":
        _n1 = int(round(_axis_cfg.get("n1", 0.4) * 100))
        _n2 = int(round(_axis_cfg.get("n2", 0.2) * 100))
        _ns = int(_axis_cfg.get("n_samples", 4))
        _axis_dir = f"q_frac_wide/n1-{_n1}%_n2-{_n2}%_N-{_ns}{_rand_sfx}"
    elif _axis_name == "vqslope":
        # vqslope: mode(dva/ica)·n_samples 별 하위 디렉터리
        _vqmode = str(_axis_cfg.get("mode", "dva")).lower()
        _ns     = int(_axis_cfg.get("n_samples", 1))
        _axis_dir = f"vqslope/{_vqmode}_N-{_ns}{_rand_sfx}"
    else:
        _axis_dir = _axis_name

    # 데이터 경로: null → scenario.axis 기반 자동 결정
    _data_cfg = cfg["data"]
    if not _data_cfg.get("seg_data_dir"):
        _data_cfg["seg_data_dir"] = f"_4_data_hi/{_axis_dir}/seg"
    if not _data_cfg.get("data_dir"):
        _data_cfg["data_dir"] = f"_4_data_hi/{_axis_dir}/cycle"
    print(f"[train] data_dir={_data_cfg['data_dir']}")
    print(f"[train] seg_data_dir={_data_cfg['seg_data_dir']}")

    # seed 오버라이드
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    _AXIS_SHORT  = {"qfrac": "qfr", "protocol": "prot", "vwindow": "vwin",
                    "rcs": "rcs", "cluster": "clst", "q_frac_wide": "qfw",
                    "vqslope": "vqs"}
    _MODEL_SHORT = {"mlp": "mlp", "transformer": "tr", "i_transformer": "itr",
                    "resnet_tab": "res", "ft_transformer": "ftt"}
    _axis_short  = _AXIS_SHORT.get(_axis_name, _axis_name[:4])
    if _axis_name == "q_frac_wide":
        _axis_short += f"_{_n1}%_{_n2}%"
    elif _axis_name == "vqslope":
        _axis_short += f"_{_vqmode}"
    if _rand_sfx:
        _axis_short += "_rand"
    _reg_model   = cfg.get("model", {}).get("regression_model", "mlp")
    _model_short = _MODEL_SHORT.get(_reg_model, _reg_model[:3])
    _phase_tag   = f"p{args.phase}" if args.phase else "p?"
    _suffix      = f"_{_phase_tag}_{_model_short}_{_axis_short}" if args.phase == 2 else f"_{_phase_tag}_{_axis_short}"
    timestamp  = datetime.now().strftime("%m%d_%H%M") + _suffix
    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"] / timestamp
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "gates").mkdir(parents=True, exist_ok=True)
    print(f"[train] run dir: {output_dir}")

    # run_dir에 spec 저장 (test_scr.py 재사용)
    spec.save(output_dir / "scenario_spec.json")

    shutil.copy(str(PROJECT_ROOT / args.config), str(output_dir / "config.yaml"))

    probe_json_out = output_dir / "gates" / "classification_HIs.json"
    scen_json_out  = output_dir / "gates" / "regression_HIs.json"

    # ------------------------------------------------------------------
    # 데이터
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds, norm = build_datasets(cfg, spec=spec)

    tr_cfg = cfg["training"]
    # FastTensorLoader: 사전 구축 텐서 슬라이싱 → per-sample collate 오버헤드 제거.
    # SCRModel 회귀 forward는 x_raw 미사용 → 학습 로더에서 제외 (메모리/전송 절감).
    train_loader = FastTensorLoader(train_ds, tr_cfg["batch_size"], shuffle=True)
    val_loader   = FastTensorLoader(val_ds,   tr_cfg["batch_size"], shuffle=False)

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

        model = SCRModel(
            d_probe=cfg["model"]["d_probe"],
            d_head=cfg["model"]["d_head"],
            dropout=cfg["model"]["dropout"],
            spec=spec,
            with_probe_mlp=_with_probe_mlp,
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
        gates_dir     = _resolve_gates_dir(args.gates_from, cfg)
        probe_json_in = None
        scen_json_in  = None

        if gates_dir:
            probe_json_in = _find_json(gates_dir, "classification_HIs.json",
                                       "scenario_classification_HIs.json")
            scen_json_in  = _find_json(gates_dir, "regression_HIs.json",
                                       "scenario_regression_HIs.json")
            print(f"[train] gates_from: {gates_dir}")
        else:
            raise RuntimeError(
                "Phase 2는 gates JSON이 필요합니다. "
                "--gates-from <run_dir> 또는 yaml의 gates_from을 설정하세요."
            )

        # Phase 2 spec 결정: gates_from/scenario_spec.json 우선, 없으면 현재 CLI spec
        _p2_spec_path = gates_dir.parent / "scenario_spec.json"
        if _p2_spec_path.exists():
            spec = ScenarioSpec.load(_p2_spec_path)
            print(f"[train] Phase 2 spec loaded: {_p2_spec_path}")
        # (else: spec already set from CLI / yaml above)

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
