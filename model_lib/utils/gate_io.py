"""
model_lib/utils/gate_io.py — SCRModel 게이트 확률 JSON 저장/로드.

2026-09-24: model_lib/legacy/train_scr.py(Stage0, 삭제됨)에서 v4가 실제로 계속 쓰는
함수들만 옮겨온 것 — 8_train/train.py(저장)와 model_lib/tools/visualize_results.py
(synergy 그룹 ID 로드)가 여기서 import한다. train_scr.py의 나머지(--phase 1/2 CLI,
Stage0 학습 루프 등)는 git 히스토리에만 남아있다.

2026-10-02: 시각화 함수 `_plot_gate_probs`는 8_train/plot.py로 옮겼다 — 이 파일은
이제 JSON 저장/로드 전용(다른 Step들이 plot.py를 스텝 폴더 로컬로 두는 관례와
맞추기 위함, 6_synergy/plot.py·7_kernel/plot.py 참고). train.py는
`from plot import _plot_gate_probs, _plot_loss_curves`로 가져다 쓴다.
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
