"""
model_lib/utils/gate_io.py — SCRModel 게이트 확률 JSON 저장/로드 + 시각화.

2026-09-24: model_lib/legacy/train_scr.py(Stage0, 삭제됨)에서 v4가 실제로 계속 쓰는
5개 함수만 옮겨온 것 — 8_train/train.py(저장)와 model_lib/tools/visualize_results.py
(synergy 그룹 ID 로드)가 여기서 import한다. train_scr.py의 나머지(--phase 1/2 CLI,
Stage0 학습 루프 등)는 git 히스토리에만 남아있다.
"""

from __future__ import annotations

import json
from pathlib import Path

from models.scr_model import SCRModel
from utils.hi_schema import N_HI


def _load_synergy_group_ids(
    json_path: Path,
    n_scenarios: int,
    scenario_names: list[str],
) -> dict[int, list[int]]:
    """synergy.py의 seg_{s}_groups(HI 인덱스 묶음)를 GroupedHardConcreteGate용
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
                f"다른 것으로 보입니다. 같은 설정으로 synergy.py를 다시 실행하세요."
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
    model.scen_kernel_gates를 넘기면 커널 융합 HI 블록(kernel.py)의
    랭킹을 같은 형식으로 저장할 수 있다(train.py에서 재사용)."""
    gates = gates if gates is not None else model.scen_gates
    out = {}
    seg_names = model.spec.scenario_names
    gate_group_map = getattr(model, "_gate_group_map", None)  # n_gate_groups(2026-09-17 안건2):
        # scenario_idx -> 축소된 게이트 뱅크 인덱스. None(기본)이면 s 그대로(기존과 동일).
    for s in range(model.n_scenarios):
        g = int(gate_group_map[s]) if gate_group_map is not None else s
        ranked, probs = _ranked_indices(gates[g])
        out[f"seg_{s}_ranked"]   = ranked
        out[f"seg_{s}_names"]    = [hi_cols_by_seg[s][i] for i in ranked]
        out[f"seg_{s}_probs"]    = probs
        out[f"seg_{s}_seg_name"] = seg_names[s]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    n_hi_out = len(next(iter(hi_cols_by_seg.values())))
    print(f"[train] Saved scen HI ranking → {json_path}  (시나리오별 {n_hi_out}개 랭킹)")


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
    gate_group_map = getattr(model, "_gate_group_map", None)  # n_gate_groups(2026-09-17 안건2)
    for s in range(model.n_scenarios):
        g = int(gate_group_map[s]) if gate_group_map is not None else s
        gates_info.append((f"Scen: {seg_names[s]}", model.scen_gates[g], scen_k, "seagreen"))

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
