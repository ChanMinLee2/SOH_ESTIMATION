"""
8_train/train.py

Stage1(체크포인트 선택 기준 변경) + Stage2(temperature annealing)를 적용한
독립 Phase1 트레이너.

2026-10-02: --output-dir를 제외한 모든 CLI 인자를 제거했다

실행은 그냥:
    python 8_train/train.py

2026-10-02: main()을 단일 책임 원칙에 따라 7개 서브함수로 분리했다(동작 변화
없는 순수 구조 정리) — t1_set_params_and_config(파라미터/컨피그) ->
t2_load_pkl_json(output_dir 확정 + kernel/interaction 경로 resolve + 데이터셋/
pkl/json 로드) -> t3_build_tensor_masks(redundancy_mask/shared_hi_mask) ->
t4_build_model_and_hparams(로더/SCRModel/SCRLoss/L0 스케줄러/optimizer) ->
t5_define_output_paths(gates/logs 디렉터리 + config.yaml + 초기 summary) ->
t6_run_training_loop(에폭 반복) -> t7_save_results(최종 체크포인트 복원 +
게이트 JSON/플롯 + 최종 summary).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# data_directories/parameters/common/models/datasets/training/evaluation/utils는
# pip install -e .로 어디서든 바로 import된다(pyproject.toml 참고) — sys.path 조작 불필요.

# Windows 콘솔이 cp949일 때 em-dash 등 특수문자 print가 UnicodeEncodeError로 죽는 문제
# 방지(lambda_sweep.py/plot_lambda_sweep.py와 동일 패턴) — 짧은 스모크런처럼 체크포인트가
# 한 번도 안 뽑히는 예외 경로의 경고 메시지에서 실제로 이 문제로 죽는 걸 확인해서 추가.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import numpy as np
import torch

from utils.io_utils import save_config  # noqa: E402
from utils.hi_schema import N_HI, get_hi_cols_for_seg, EXCLUDE_STAT_LEAK, EXCLUDE_DQDV_LEAK  # noqa: E402
from utils.metrics import rmse as _rmse, r2 as _r2  # noqa: E402
from utils.tqdm_utils import trange, write as tqdm_write  # noqa: E402
from datasets.segment_dataset import build_datasets, FastTensorLoader  # noqa: E402
from models.scr_model import SCRModel  # noqa: E402
from training.scr_loss import SCRLoss  # noqa: E402
from training.scr_trainer import L0LambdaScheduler  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

# 게이트 확률 JSON 저장/로드는 model_lib/utils/gate_io.py 단일 소스(2026-09-24 —
# model_lib/legacy/train_scr.py(Stage0)에서 v4가 실제 쓰는 부분만 옮기고 파일은 삭제).
from utils.gate_io import (  # noqa: E402
    _save_probe_masks_to_json, _save_scen_masks_to_json,
)
# 시각화는 전부 8_train/plot.py(이 폴더 로컬, 2026-10-02에 _plot_gate_probs를
# gate_io.py에서 이쪽으로 이동 — 6_synergy/plot.py·7_kernel/plot.py와 동일 관례).
from plot import _plot_loss_curves, _plot_gate_probs  # noqa: E402
import parameters as P  # noqa: E402 — 축 설정 단일 소스(P.ACTIVE_AXIS_CONFIG)

# run_pipeline.py의 P1V2_RUNS_DIR과 동일 경로(2026-09-23 재배치) — 단독 실행 시(--output-dir
# 미지정) 폴백으로만 쓰인다. run_pipeline.py를 거치면 항상 --output-dir이 명시되므로 이
# 상수는 실제로 안 쓰인다.
RESULTS_DIR = PROJECT_ROOT / "model_lib" / "results" / "p1v2_runs"


def _parse_args() -> argparse.Namespace:
    # 2026-10-02: --output-dir 하나만 남기고 전부 제거 — 나머지는 이미 parameters.py가
    # 단일 소스인 값의 CLI 통로였을 뿐이다(hi_correlation.py/interaction.py/synergy.py/
    # kernel.py 정리와 동일 원칙). --output-dir만 예외인 이유, kernel-features-pkl/
    # combined-redundancy-json/interaction-json 해석이 바뀐 이유는 모듈 docstring 참고.
    p = argparse.ArgumentParser(description="Phase1 v2 — 체크포인트 기준 변경 + temperature annealing")
    p.add_argument("--output-dir", default=None, dest="output_dir",
                   help="run 산출물(gates/checkpoints/logs/config.yaml 등)을 저장할 폴더를 "
                        "직접 지정 — 주어지면 이 경로를 그대로 쓰고, RESULTS_DIR 밑에 "
                        "{timestamp}_p1v2_{tag}_seed{seed} 폴더를 새로 만들지 않는다. "
                        "run_pipeline.py가 Step 6~9(상호작용/시너지/커널/평가) 산출물과 같은 "
                        "폴더에 학습 결과를 모으려고 씀 — kernel-features-pkl/interaction-json "
                        "자동탐색도 이 폴더를 기준으로 한다(모듈 docstring 참고). 단독 실행 시 "
                        "기본(미지정)이면 기존과 100%% 동일하게 타임스탬프 폴더를 새로 만든다.")
    return p.parse_args()


def _resolve_kernel_paths(
    kernel_features_pkl: str | None, combined_redundancy_json: str | None,
    kernel_pkl_out: Path, kernel_redundancy_out: Path, default_pkl: str,
) -> tuple[str, str | None]:
    """run_pipeline.py의 동명 함수와 동일 로직(2026-10-02 이식, 원래는 run_pipeline.py에만
    있어서 train.py 단독 실행 시 이 fallback이 적용되지 않았었다 — 모듈 docstring 참고).
    parameters.py 값이 명시적으로 주어지면 최우선, 없으면 자동 경로(이번 run의 output_dir
    안, Step 6~7이 방금/이전에 만든 파일)가 실존하면 그걸, 그것도 없으면 default_pkl(v4
    정식 고정 경로)로 최종 fallback한다."""
    if kernel_features_pkl is not None:
        resolved_pkl = kernel_features_pkl
    elif kernel_pkl_out.exists():
        resolved_pkl = str(kernel_pkl_out)
    else:
        resolved_pkl = default_pkl
    if combined_redundancy_json is not None:
        resolved_redundancy = combined_redundancy_json
    elif kernel_redundancy_out.exists():
        resolved_redundancy = str(kernel_redundancy_out)
    else:
        resolved_redundancy = None
    return resolved_pkl, resolved_redundancy


def _resolve_interaction_path(
    interaction_json: str | None, interaction_out: Path, default_json: str,
) -> str | None:
    """run_pipeline.py의 동명 함수와 동일 로직(2026-10-02 이식) — 3단 우선순위:
    1) parameters.py: ACTIVE_INTERACTION_JSON이 명시적으로 주어지면(빈 문자열 포함) 그
    값 그대로(빈 문자열이면 아예 전달 안 함, v0/v2/v3 재현용) 2) 이번 run의 output_dir
    안에 Step 5(자동 경로) 산출물이 있으면 그걸 3) 그것도 없으면 default_json(v4 정식
    고정 경로)로 최종 fallback."""
    if interaction_json is not None:
        return interaction_json or None
    if interaction_out.exists():
        return str(interaction_out)
    return default_json


def _kernel_source(resolved: str | None, explicit: str | None, auto_path: Path) -> str:
    """resolved 값이 _resolve_kernel_paths/_resolve_interaction_path의 3단 우선순위 중
    어디서 나왔는지 콘솔 출력용으로 되짚는다(2026-10-02 신설) — 새 상태를 따로 들고
    다니지 않고 resolved를 auto_path 문자열과 직접 비교해서 판별."""
    if explicit is not None:
        return "parameters.py 명시값"
    if resolved == str(auto_path):
        return "자동탐색(이번 run_dir)"
    return "legacy v4 기본값"


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _apply_kernel_features(
    datasets: list, pkl_path: Path, spec, combined_redundancy: dict | None = None,
) -> tuple[dict[int, list[str]], list[int], dict[int, list[float]]]:
    """kernel.py 산출물을 로드해 각 dataset에 x_kernel(정규화된
    RBF 커널 융합값)을 새로 붙인다 — x_hi(raw HI)는 건드리지 않는다(v2: 대체가 아니라
    추가). FastTensorLoader가 ds.x_kernel 존재 여부를 보고 자동으로 배치에 포함시킨다
    (datasets/segment_dataset.py 참고).

    combined_redundancy: kernel.py의 3차(결합 raw+kernel, 시나리오별)
    다중공선성 배제 결과 dict({seg_name: {"removed_kernel_names": [...], ...}}, 요구사항2,
    --combined-redundancy-json). 주어지면 그 시나리오의 로컬 목록에서 해당 커널을 아예
    제외한다(그만큼 K_s가 줄어듦 — raw HI 쪽 배제는 이 함수가 아니라 _build_redundancy_mask()가
    만드는 게이트 레벨 redundancy_mask가 담당, 2026-09-21부터 입력 레벨 nan_mask 이중
    마스킹은 분류기 오염 버그로 제거됨).

    train으로만 fit된 모델을 val/test에도 그대로 적용(predict만)하고, 정규화도 train
    mean/std를 val/test에 그대로 적용(fit 안 함)하므로 누수는 없다.
    반환값이 v1(위 커밋 이전)과 다르다 — (시나리오별 이름 목록 dict, 시나리오별 개수
    리스트, 시나리오별 L0 비용 목록 dict) 3-튜플. 호출자는 둘째를 SCRModel(kernel_hi_counts=...)
    로, 첫째를 게이트 JSON 저장(gates/regression_kernel_HIs.json)에, 셋째를
    SCRModel(kernel_hi_costs=...)로 넘겨 scr_loss.py의 커널 L0 페널티에 쓴다.

    2026-09-18(L0 비용 가중치 추가): 각 커널 HI의 비용은 그 커널을 만든 멤버 raw HI들의
    카테고리 비용(hi_schema.CATEGORY_COSTS)의 평균이다 — kernel.py가
    pkl 저장 시점에 f["cost"]로 미리 계산해둔 값을 그대로 읽는다(구 pkl 호환: 없으면 1.0).
    """
    with open(pkl_path, "rb") as fh:
        artifact = pickle.load(fh)

    features = artifact["features"]
    seg_name_to_idx = {n: i for i, n in enumerate(spec.scenario_names)}

    removed_kernel_by_scen: dict[str, set] = {}
    if combined_redundancy:
        for seg_name, info in combined_redundancy.get("by_scenario", combined_redundancy).items():
            removed_kernel_by_scen[seg_name] = set(info.get("removed_kernel_names", []))

    # 시나리오별 own 커널만 모아 로컬 순서를 고정(원래 pkl 순서 유지) -- 결합 다중공선성
    # 배제로 탈락한 건 여기서 그냥 제외한다(그만큼 K_s가 줄어듦, 요구사항2).
    feats_by_scen: dict[int, list[dict]] = {s: [] for s in range(spec.n_scenarios)}
    n_excluded = 0
    for f in features:
        if f["name"] in removed_kernel_by_scen.get(f["scenario"], ()):
            n_excluded += 1
            continue
        feats_by_scen[seg_name_to_idx[f["scenario"]]].append(f)

    kernel_hi_counts = [len(feats_by_scen[s]) for s in range(spec.n_scenarios)]
    max_k = max(kernel_hi_counts) if kernel_hi_counts else 0
    names_by_scen = {s: [f["name"] for f in feats_by_scen[s]] for s in range(spec.n_scenarios)}
    costs_by_scen = {
        s: [float(f.get("cost", 1.0)) for f in feats_by_scen[s]]
        for s in range(spec.n_scenarios)
    }

    for ds in datasets:
        x = (ds.x_hi * ds.nan_mask).numpy()  # NaN 위치 0으로 (fit 시점과 동일 처리)
        scen_idx_np = ds.scen_idx.numpy()
        x_kernel = np.zeros((x.shape[0], max_k), dtype=np.float32)
        for s in range(spec.n_scenarios):
            row_mask = scen_idx_np == s
            if not row_mask.any():
                continue
            x_scen = x[row_mask]
            for local_j, f in enumerate(feats_by_scen[s]):
                pred = f["model"].predict(x_scen[:, f["members"]])
                x_kernel[row_mask, local_j] = (pred - f["mean"]) / f["std"]  # train 통계로 z-score
        ds.x_kernel = torch.from_numpy(x_kernel.astype(np.float32))

    avg_r2 = float(np.mean([f["train_r2"] for f in features])) if features else 0.0
    print(f"[p1v2] kernel-features-pkl 적용: {pkl_path} "
          f"(커널 HI 시나리오별 폭 {kernel_hi_counts}(max={max_k}, x_hi {N_HI}개와 별도 추가) "
          f"-- 게이트 구조 자체가 own-scenario로 제한됨, 평균 train R^2={avg_r2:.4f})"
          + (f" [combined-redundancy: {n_excluded}개 컬럼 로컬 목록에서 사전 제외]"
             if combined_redundancy else ""))
    return names_by_scen, kernel_hi_counts, costs_by_scen


def _build_redundancy_mask(combined_redundancy: dict, spec) -> torch.Tensor:
    """kernel.py의 3차 결합(raw+kernel) 다중공선성 배제 결과 중 raw HI
    쪽을 SCRModel(redundancy_mask=...)용 bool 텐서 (n_scenarios, N_HI)로 만든다(2026-09-18,
    요구사항2를 raw HI에도 게이트 구조로 강제 — scr_model.py의 _apply_scen_gate가 이 마스크를
    scen_gates 출력에 곱해 False 위치는 log_alpha와 무관하게 항상 0으로 만든다). True=허용,
    False=그 시나리오에서 배제된 raw HI."""
    by_scenario = combined_redundancy.get("by_scenario", combined_redundancy)
    seg_name_to_idx = {n: i for i, n in enumerate(spec.scenario_names)}
    mask = torch.ones(spec.n_scenarios, N_HI, dtype=torch.bool)
    for seg_name, info in by_scenario.items():
        removed_raw_idx = info.get("removed_raw_idx", [])
        if not removed_raw_idx:
            continue
        s = seg_name_to_idx[seg_name]
        mask[s, removed_raw_idx] = False
    return mask


def _gate_saturation_fraction(model: SCRModel) -> float:
    """게이트 확률이 애매한 [0.1,0.9] 구간에 있는 비율 — 낮을수록 더 확실하게 이산화됨."""
    gates = [model.charge_probe_gate, model.discharge_probe_gate, *model.scen_gates]
    if model.scen_kernel_gates is not None:
        gates += list(model.scen_kernel_gates)
    if model.shared_gate is not None:
        gates.append(model.shared_gate)  # v4: scen_gates가 shared 몫만큼 좁아진 대신
            # shared_gate가 그 몫을 담당하므로, 얘를 빼면 포화도가 실제보다 낮게(더 좋게)
            # 잘못 나온다 — 전체 게이트 파라미터 집합에 반드시 포함해야 함.
    probs = [gate.gate_prob().detach().cpu() for gate in gates]
    p = torch.cat(probs)
    return float(((p > 0.1) & (p < 0.9)).float().mean().item())


def _save_scen_masks_with_shared(model: SCRModel, json_path, hi_cols_by_seg: dict[int, list[str]]) -> None:
    """v4 전용: model.shared_gate가 있으면 scen_gates[s](specific 폭)와 shared_gate를
    원래 컬럼 순서로 재조립해서, 기존과 동일한 seg_{s}_ranked/names/probs/seg_name
    스키마로 저장한다(하위 분석 스크립트가 무수정으로 읽을 수 있게). shared_gate가
    없으면(shared_hi_mask 미지정) train_scr.py의 원본 함수로 그대로 위임."""
    if model.shared_gate is None:
        _save_scen_masks_to_json(model, json_path, hi_cols_by_seg)
        return

    n_hi = len(next(iter(hi_cols_by_seg.values())))
    shared_idx = model._shared_idx.detach().cpu()
    specific_idx = model._specific_idx.detach().cpu()
    shared_prob = model.shared_gate.gate_prob().detach().cpu()

    out = {}
    seg_names = model.spec.scenario_names
    for s in range(model.n_scenarios):
        full_prob = torch.zeros(n_hi)
        full_prob[shared_idx] = shared_prob
        if len(specific_idx) > 0:
            full_prob[specific_idx] = model.scen_gates[s].gate_prob().detach().cpu()
        ranked = full_prob.argsort(descending=True).tolist()
        probs = [round(float(full_prob[i]), 6) for i in ranked]
        out[f"seg_{s}_ranked"] = ranked
        out[f"seg_{s}_names"] = [hi_cols_by_seg[s][i] for i in ranked]
        out[f"seg_{s}_probs"] = probs
        out[f"seg_{s}_seg_name"] = seg_names[s]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[p1v2] Saved scen HI ranking(shared_gate 반영) -> {json_path} (시나리오별 {n_hi}개 랭킹)")


def t1_set_params_and_config() -> SimpleNamespace:
    """1) 파라미터 및 컨피그 세팅 — parameters.py에서 그대로 읽는 실행 파라미터를 전부
    모으고, cfg(P1_MODEL_CONFIG 사본)에 반영한 뒤 세그먼터 spec까지 만든다."""
    seed = P.ACTIVE_SEED if P.ACTIVE_SEED is not None else P.FIXED_DEFAULT_SEED
    split_seed = P.ACTIVE_SPLIT_SEED if P.ACTIVE_SPLIT_SEED is not None else P.FIXED_DEFAULT_SEED
    seg_axis = P.FIXED_SEG_AXIS
    axis_cfg = dict(P.ACTIVE_AXIS_CONFIG)
    train_cycle_frac = P.FIXED_TRAIN_CYCLE_FRAC if P.FIXED_TRAIN_CYCLE_FRAC is not None else 1.0
    data_dir = P.FIXED_CANONICAL_DATA_DIR
    seg_data_dir = P.FIXED_CANONICAL_SEG_DATA_DIR
    beta_min = P.FIXED_BETA_MIN if P.FIXED_BETA_MIN is not None else 0.1
    device_str = P.FIXED_DEVICE or "auto"
    max_epochs_override = P.ACTIVE_MAX_EPOCHS
    patience = P.ACTIVE_PATIENCE if P.ACTIVE_PATIENCE is not None else 60
    batch_size_override = P.ACTIVE_BATCH_SIZE
    tag = P.ACTIVE_P1_TAG
    lambda_l0_override = P.ACTIVE_LAMBDA_L0_OVERRIDE
    l0_warmup_epochs_override = P.FIXED_L0_WARMUP_EPOCHS_OVERRIDE
    l0_norm_constant = P.FIXED_L0_NORM_CONSTANT
    hi_cost_weighted_l0 = P.ACTIVE_HI_COST_WEIGHTED_L0
    val_rmse_epsilon = P.FIXED_VAL_RMSE_EPSILON

    device = _resolve_device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = copy.deepcopy(P.P1_MODEL_CONFIG)

    cfg["scenario"] = {"axis": seg_axis, "axis_config": axis_cfg}
    cfg["data"]["split_seed"] = split_seed
    cfg["data"]["train_cycle_frac"] = train_cycle_frac
    cfg["data"]["data_dir"] = data_dir
    cfg["data"]["seg_data_dir"] = seg_data_dir

    cls_cfg = cfg.setdefault("classifier", {})
    reg_cfg = cfg.setdefault("regression", {})
    if P.FIXED_CHARGE_PROBE_M is not None: cls_cfg["charge_probe_m"] = P.FIXED_CHARGE_PROBE_M
    if P.FIXED_DISCHARGE_PROBE_M is not None: cls_cfg["discharge_probe_m"] = P.FIXED_DISCHARGE_PROBE_M
    if P.FIXED_SCEN_K_COUNT is not None: reg_cfg["scen_k_count"] = P.FIXED_SCEN_K_COUNT
    charge_m = cls_cfg.get("charge_probe_m", 10)
    discharge_m = cls_cfg.get("discharge_probe_m", 10)
    scen_k = reg_cfg.get("scen_k_count", 5)

    spec = get_segmenter(seg_axis, {seg_axis: axis_cfg}).get_spec()

    return SimpleNamespace(
        seed=seed, split_seed=split_seed, tag=tag, device=device, cfg=cfg, spec=spec,
        charge_m=charge_m, discharge_m=discharge_m, scen_k=scen_k, beta_min=beta_min,
        max_epochs_override=max_epochs_override, patience=patience,
        batch_size_override=batch_size_override, lambda_l0_override=lambda_l0_override,
        l0_warmup_epochs_override=l0_warmup_epochs_override, l0_norm_constant=l0_norm_constant,
        hi_cost_weighted_l0=hi_cost_weighted_l0, val_rmse_epsilon=val_rmse_epsilon,
    )


def t2_load_pkl_json(args: argparse.Namespace, params: SimpleNamespace) -> SimpleNamespace:
    """2) pkl/json 로드 — output_dir를 데이터 로딩 전에 먼저 확정하고(아래 kernel/
    interaction 자동탐색이 이 경로를 기준으로 하기 때문, 모듈 docstring 참고),
    kernel-features-pkl/combined-redundancy-json/interaction-json을 3단 우선순위로
    resolve해 콘솔에 보여준 뒤, 세그먼트 데이터셋(train/val/test)과 실제 pkl/json
    내용까지 로드한다."""
    tag, seed = params.tag, params.seed

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%masks%d_%H%M")
        output_dir = RESULTS_DIR / f"{timestamp}_p1v2_{tag}_seed{seed}"

    kernel_tag = P.FIXED_KERNEL_TAG or f"{tag}_kernel"
    interaction_tag = P.FIXED_INTERACTION_TAG or f"{tag}_interaction"
    kernel_pkl_auto = output_dir / f"kernel_group_features_{kernel_tag}.pkl"
    kernel_redundancy_auto = output_dir / f"kernel_group_features_{kernel_tag}_combined_redundancy.json"
    interaction_auto = output_dir / f"hi_scenario_interaction_{interaction_tag}.json"

    kernel_features_pkl, combined_redundancy_json = _resolve_kernel_paths(
        P.ACTIVE_KERNEL_FEATURES_PKL, P.ACTIVE_COMBINED_REDUNDANCY_JSON,
        kernel_pkl_auto, kernel_redundancy_auto, P.FIXED_LEGACY_V4_KERNEL_FEATURES_PKL,
    )
    interaction_json = _resolve_interaction_path(
        P.ACTIVE_INTERACTION_JSON, interaction_auto, P.FIXED_LEGACY_V4_INTERACTION_JSON,
    )
    if combined_redundancy_json and not kernel_features_pkl:
        raise RuntimeError(
            "combined_redundancy_json은 kernel_features_pkl 없이 쓸 수 없습니다(그 산출물이 "
            "커널 features pkl을 만든 뒤 이어서 생성되는 후속 산출물이라) — parameters.py: "
            "ACTIVE_KERNEL_FEATURES_PKL/ACTIVE_COMBINED_REDUNDANCY_JSON을 확인하세요."
        )

    # 데이터 로딩(수 분 소요 가능) 전에 바로 찍는다 — 3단 우선순위(명시값/자동탐색/legacy
    # fallback) 중 어디서 왔는지도 같이 보여줘서, 뭘 쓰는지 학습이 끝날 때까지 안 기다리고
    # 바로 확인할 수 있게 한다(2026-10-02 신설, _kernel_source는 모듈 상단 함수).
    print(f"[p1v2] kernel-features-pkl     : {kernel_features_pkl} "
          f"[{_kernel_source(kernel_features_pkl, P.ACTIVE_KERNEL_FEATURES_PKL, kernel_pkl_auto)}]")
    print(f"[p1v2] combined-redundancy-json: "
          + (f"{combined_redundancy_json} "
             f"[{_kernel_source(combined_redundancy_json, P.ACTIVE_COMBINED_REDUNDANCY_JSON, kernel_redundancy_auto)}]"
             if combined_redundancy_json else "(미사용)"))
    print(f"[p1v2] interaction-json        : "
          + (f"{interaction_json} "
             f"[{_kernel_source(interaction_json, P.ACTIVE_INTERACTION_JSON, interaction_auto)}]"
             if interaction_json else "(미사용)"))

    train_ds, val_ds, test_ds, _norm = build_datasets(params.cfg, spec=params.spec)

    kernel_names_by_scen = None
    kernel_hi_counts = None
    kernel_costs_by_scen = None
    combined_redundancy = None
    if combined_redundancy_json:
        combined_redundancy = json.loads(Path(combined_redundancy_json).read_text(encoding="utf-8"))
    if kernel_features_pkl:
        kernel_names_by_scen, kernel_hi_counts, kernel_costs_by_scen = _apply_kernel_features(
            [train_ds, val_ds, test_ds], Path(kernel_features_pkl), params.spec,
            combined_redundancy=combined_redundancy,
        )

    return SimpleNamespace(
        output_dir=output_dir, kernel_features_pkl=kernel_features_pkl,
        combined_redundancy_json=combined_redundancy_json, interaction_json=interaction_json,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, combined_redundancy=combined_redundancy,
        kernel_names_by_scen=kernel_names_by_scen, kernel_hi_counts=kernel_hi_counts,
        kernel_costs_by_scen=kernel_costs_by_scen,
    )


def t3_build_tensor_masks(params: SimpleNamespace, data: SimpleNamespace) -> SimpleNamespace:
    """3) 텐서 마스킹 — kernel.py의 3차 결합(raw+kernel) 다중공선성 배제 결과를
    redundancy_mask 텐서로, interaction.py의 유의성 판정을 shared_hi_mask 텐서로 만든다."""
    redundancy_mask = None
    if data.combined_redundancy is not None:
        # 2026-09-21: 입력 레벨 nan_mask 이중 마스킹(_apply_combined_redundancy_raw)은
        # 삭제 — forward()가 x = x_hi * nan_mask를 probe_x/scen_x 양쪽에 공유해서 쓰는데,
        # 이 함수가 회귀 전용으로 만든 nan_mask 제로잉이 그대로 probe_x(분류기 입력)까지
        # 오염시키고 있었다(실측: 이 로직 추가 전 run은 분류기 정확도 98.65%, 추가 후
        # 36~67%로 붕괴). redundancy_mask(게이트 출력 강제)만으로 scen_x 쪽 정확성은
        # 이미 완전히 보장되므로(로그 알파와 무관하게 항상 0), 입력 단은 건드리지 않는다.
        redundancy_mask = _build_redundancy_mask(data.combined_redundancy, params.spec)
        print(f"[p1v2] combined-redundancy-json 적용: {data.combined_redundancy_json} "
              f"(게이트 출력 0-강제만 적용 — 입력 레벨 마스킹은 분류기 오염 버그로 제거됨, "
              f"{int((~redundancy_mask).sum().item())}개 (시나리오,HI) 조합 배제)")

    shared_hi_mask = None
    if data.interaction_json:
        interaction_data = json.loads(Path(data.interaction_json).read_text(encoding="utf-8"))
        ref_seg_name = params.spec.scenario_names[0]
        ref_cols = get_hi_cols_for_seg(ref_seg_name)
        suffix = f"_{ref_seg_name}"
        concepts_in_order = [c[: -len(suffix)] if c.endswith(suffix) else c for c in ref_cols]
        per_hi = interaction_data["per_hi"]
        shared_hi_mask = torch.tensor(
            [not per_hi.get(c, {"significant": False})["significant"] for c in concepts_in_order],
            dtype=torch.bool,
        )
        n_shared = int(shared_hi_mask.sum().item())
        print(f"[p1v2] interaction-json 적용: {data.interaction_json} "
              f"({n_shared}/{len(shared_hi_mask)}개 HI -> shared_gate, "
              f"{len(shared_hi_mask) - n_shared}개 -> 기존 scen_gates)")

    return SimpleNamespace(redundancy_mask=redundancy_mask, shared_hi_mask=shared_hi_mask)


def t4_build_model_and_hparams(params: SimpleNamespace, data: SimpleNamespace, masks: SimpleNamespace) -> SimpleNamespace:
    """4) 모델 선언 및 하이퍼파라미터 세팅 — 배치 로더, SCRModel/SCRLoss, lambda_l0/
    L0 스케줄러, optimizer/scheduler까지 전부 구성한다."""
    cfg = params.cfg
    tr_cfg = cfg["training"]
    if params.batch_size_override is not None:
        print(f"[p1v2] batch_size 오버라이드: {tr_cfg['batch_size']} -> {params.batch_size_override}")
        tr_cfg["batch_size"] = params.batch_size_override
    train_loader = FastTensorLoader(data.train_ds, tr_cfg["batch_size"], shuffle=True)
    val_loader = FastTensorLoader(data.val_ds, tr_cfg["batch_size"], shuffle=False)

    lambda_scen = cfg.get("loss", {}).get("lambda_scen", 0.0)
    with_probe_mlp = lambda_scen > 0
    # regression_model은 cfg["model"]에 저장된 값(v4는 항상 "mlp") 그대로 쓴다 — 다른
    # 아키텍처(transformer 등)는 sanity-check용으로만 쓰이던 오버라이드라 제거함.
    # (2026-09-27: with_raw_cnn/with_raw_flat 강제 오버라이드 삭제 — SCRModel이 이제
    # model_cfg에서 두 키를 아예 읽지 않으므로 의미 없는 no-op이었다.)
    p1_model_cfg = dict(cfg["model"])

    model = SCRModel(
        d_probe=cfg["model"]["d_probe"], d_head=cfg["model"]["d_head"], dropout=cfg["model"]["dropout"],
        spec=params.spec, with_probe_mlp=with_probe_mlp, model_cfg=p1_model_cfg,
        shared_hi_mask=masks.shared_hi_mask,
        kernel_hi_counts=data.kernel_hi_counts,
        kernel_hi_costs=data.kernel_costs_by_scen,
        redundancy_mask=masks.redundancy_mask,
    ).to(params.device)

    loss_cfg = cfg["loss"]
    loss_fn = SCRLoss(lambda_scen=lambda_scen, lambda_l0=loss_cfg["lambda_l0"],
                       l0_norm_constant=params.l0_norm_constant,
                       hi_cost_weighted=params.hi_cost_weighted_l0).to(params.device)
    if params.l0_norm_constant is not None:
        print(f"[p1v2] l0-norm-constant 적용: _l0_penalty를 n_scenarios 대신 "
              f"{params.l0_norm_constant}로 나눔")
    print(f"[p1v2] hi-cost-weighted-l0: {params.hi_cost_weighted_l0} "
          f"({'CATEGORY_COSTS 가중' if params.hi_cost_weighted_l0 else '균일 비용 1.0(기본)'})")

    if params.lambda_l0_override is not None:
        loss_cfg["lambda_l0"] = params.lambda_l0_override
        print(f"[p1v2] lambda_l0_override: {params.lambda_l0_override} (lambda_l0_auto/yaml 값 무시)")
    elif loss_cfg.get("lambda_l0_auto", False):
        avg_m = (params.charge_m + params.discharge_m) / 2
        probe_scale = 10 / max(avg_m, 1)
        scen_scale = 10 / max(params.scen_k, 1)
        auto_lambda = round(max(1e-4, min(0.01 * math.sqrt(probe_scale * scen_scale), 0.5)), 5)
        loss_cfg["lambda_l0"] = auto_lambda
        print(f"[p1v2] lambda_l0_auto: charge_m={params.charge_m} discharge_m={params.discharge_m} "
              f"scen_k={params.scen_k} -> {auto_lambda}")

    epochs = params.max_epochs_override if params.max_epochs_override is not None else tr_cfg["epochs"]
    if params.max_epochs_override is not None:
        print(f"[p1v2] epochs 상한 오버라이드: {tr_cfg['epochs']} -> {epochs}")
    warmup_ep = tr_cfg.get("warmup_epochs", 10)
    if params.l0_warmup_epochs_override is not None:
        loss_cfg["lambda_l0_warmup_epochs"] = params.l0_warmup_epochs_override
        print(f"[p1v2] l0_warmup_epochs_override: {params.l0_warmup_epochs_override} (yaml 값 무시)")
    l0_scheduler = L0LambdaScheduler(target=loss_cfg["lambda_l0"], loss_cfg=loss_cfg, total_epochs=epochs)
    l0_warmup_ep = loss_cfg.get("lambda_l0_warmup_epochs", 50)
    l0_ramp_ep = loss_cfg.get("lambda_l0_ramp_epochs", 50)
    l0_fully_ramped_ep = l0_warmup_ep + l0_ramp_ep  # 이 에폭부터 체크포인트 후보로 인정

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_ep, 1), eta_min=1e-6)

    beta_default = 2.0 / 3.0

    return SimpleNamespace(
        tr_cfg=tr_cfg, train_loader=train_loader, val_loader=val_loader, model=model,
        loss_fn=loss_fn, loss_cfg=loss_cfg, epochs=epochs, warmup_ep=warmup_ep,
        l0_scheduler=l0_scheduler, l0_warmup_ep=l0_warmup_ep, l0_ramp_ep=l0_ramp_ep,
        l0_fully_ramped_ep=l0_fully_ramped_ep, optimizer=optimizer, scheduler=scheduler,
        beta_default=beta_default,
    )


def t5_define_output_paths(params: SimpleNamespace, data: SimpleNamespace, hyperparams: SimpleNamespace) -> SimpleNamespace:
    """5) 출력 경로 및 형식 정의 — gates/logs/checkpoints 디렉터리, scenario_spec.json,
    train_log_v2.csv 헤더, config.yaml, 초기 p1v2_summary.json(status=in_progress)까지
    학습 루프 시작 전에 미리 만들어둔다."""
    output_dir = data.output_dir
    (output_dir / "gates").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoints" / "best_by_saturation.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    params.spec.save(output_dir / "scenario_spec.json")

    log_path = output_dir / "logs" / "train_log_v2.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lambda_l0,lambda_scen,beta,tr_rmse,tr_r2,tr_mse,tr_ce,tr_l0,"
                 "val_rmse,val_r2,gate_saturation,is_selected\n")

    # 2026-09-18: config.yaml/p1v2_summary.json을 여기(학습 루프 시작 전)에서도 한 번 써둔다
    # — 원래는 학습이 끝난 뒤(맨 아래)에만 썼는데, 여기서 미리 써두면
    # checkpoints/best_by_saturation.pt가 한 번이라도 저장된 시점부터는
    # (=epoch>=l0_fully_ramped_ep 도달) 언제 중단해도 바로 테스트 가능
    # — 맨 아래의 최종 write가 best_epoch/gate_saturation을 채워 그대로 덮어쓰므로 정상
    # 종료 시 동작은 100% 기존과 동일.
    cfg = params.cfg
    cfg.setdefault("data", {})["exclude_stat_leak"] = EXCLUDE_STAT_LEAK
    cfg["data"]["exclude_dqdv_leak"] = EXCLUDE_DQDV_LEAK
    save_config(cfg, output_dir / "config.yaml")
    _early_summary = {
        "tag": params.tag, "seed": params.seed, "split_seed": params.split_seed,
        "lambda_l0_used": hyperparams.loss_cfg["lambda_l0"],
        "lambda_l0_warmup_epochs_used": hyperparams.l0_warmup_ep,
        "selected_epoch": None, "gate_saturation": None,  # 아직 모름 -- 학습 끝나면 채워짐
        "beta_min": params.beta_min, "l0_fully_ramped_epoch": hyperparams.l0_fully_ramped_ep,
        "output_dir": str(output_dir),
        "n_hi": N_HI, "exclude_stat_leak": EXCLUDE_STAT_LEAK, "exclude_dqdv_leak": EXCLUDE_DQDV_LEAK,
        "synergy_groups_json": None,  # v0~v2 하위호환용 고정 필드(아래 최종 write와 동일)
        "kernel_features_pkl": data.kernel_features_pkl,
        "combined_redundancy_json": data.combined_redundancy_json,
        "interaction_json": data.interaction_json,
        "l0_norm_constant": params.l0_norm_constant,
        "hi_cost_weighted_l0": params.hi_cost_weighted_l0,
        "status": "in_progress",  # 최종 write에서는 이 키가 아예 빠짐(=완료) -- get()으로만 읽는
            # 기존 소비자(test.py 등)에는 영향 없음
    }
    (output_dir / "p1v2_summary.json").write_text(
        json.dumps(_early_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return SimpleNamespace(ckpt_path=ckpt_path, log_path=log_path)


def t6_run_training_loop(params: SimpleNamespace, hyperparams: SimpleNamespace, paths: SimpleNamespace) -> SimpleNamespace:
    """6) 학습 에포크 반복문 — Stage1(L0 완전 램프 이후 val_rmse 우선 + gate_saturation
    fallback 체크포인트 선택)/Stage2(temperature annealing) 적용 메인 루프."""
    device = params.device
    tr_cfg = hyperparams.tr_cfg
    model = hyperparams.model
    loss_fn = hyperparams.loss_fn
    optimizer = hyperparams.optimizer
    scheduler = hyperparams.scheduler
    train_loader = hyperparams.train_loader
    val_loader = hyperparams.val_loader
    warmup_ep = hyperparams.warmup_ep
    l0_scheduler = hyperparams.l0_scheduler
    l0_warmup_ep = hyperparams.l0_warmup_ep
    l0_ramp_ep = hyperparams.l0_ramp_ep
    l0_fully_ramped_ep = hyperparams.l0_fully_ramped_ep
    beta_default = hyperparams.beta_default
    beta_min = params.beta_min
    epochs = hyperparams.epochs
    ckpt_path = paths.ckpt_path
    log_path = paths.log_path

    best_sat = float("inf")  # 2026-09-18부터 1순위 아님 — val_rmse가 epsilon 이내 동률일 때만 tie-break
    best_val_rmse = float("inf")  # 2026-09-18부터 체크포인트 선택 1순위(--val-rmse-epsilon)
    best_epoch = -1
    no_improve = 0  # L0 완전 램프 이후, best 갱신 없이 지난 에폭 수

    for epoch in trange(epochs, desc=f"[p1v2:{params.tag}] seed={params.seed}"):
        if epoch < warmup_ep:
            lr = tr_cfg["lr"] * (epoch + 1) / warmup_ep
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        eff_l0 = l0_scheduler.get(epoch)
        if epoch < l0_warmup_ep:
            beta_now = beta_default
        elif epoch < l0_fully_ramped_ep:
            frac = (epoch - l0_warmup_ep) / max(l0_ramp_ep, 1)
            beta_now = beta_default + (beta_min - beta_default) * frac
        else:
            beta_now = beta_min
        loss_fn.lambda_l0 = eff_l0
        for gate in [model.charge_probe_gate, model.discharge_probe_gate, *model.scen_gates]:
            gate.BETA = beta_now

        # ---- train epoch ----
        model.train()
        tr_preds, tr_targets = [], []
        tr_mse_sum = tr_ce_sum = tr_l0_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            losses = loss_fn(out, batch, model)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.get("grad_clip", 1.0))
            optimizer.step()
            tr_preds.append(out["cap_pred"].detach().cpu())
            tr_targets.append(batch["target"].cpu())
            # epoch별 손실 항 비중 플롯(loss_curves.png)용 — 배치 평균으로 집계
            tr_mse_sum += losses["mse"].item()
            tr_ce_sum += losses["ce"].item()
            tr_l0_sum += losses["l0"].item()
            n_batches += 1
        tr_p, tr_t = torch.cat(tr_preds).numpy(), torch.cat(tr_targets).numpy()
        tr_rmse_v, tr_r2_v = float(_rmse(tr_t, tr_p)), float(_r2(tr_t, tr_p))
        tr_mse_v = tr_mse_sum / n_batches
        tr_ce_v = tr_ce_sum / n_batches
        tr_l0_v = tr_l0_sum / n_batches

        if epoch >= warmup_ep:
            scheduler.step()

        # ---- val epoch ----
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                val_preds.append(out["cap_pred"].cpu())
                val_targets.append(batch["target"].cpu())
        val_p, val_t = torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy()
        val_rmse_v, val_r2_v = float(_rmse(val_t, val_p)), float(_r2(val_t, val_p))

        sat = _gate_saturation_fraction(model)

        # Stage1: 체크포인트 선택 = "L0가 완전히 램프된 이후" 구간에서 val_rmse 우선,
        # sat은 fallback(2026-09-18, 기준 변경 — 기존 sat-1순위 방식은 docs 2026-09-04
        # 결정 참고). val_rmse가 best보다 --val-rmse-epsilon **이상** 좋아지면 그 epoch을
        # "진짜 개선"으로 채택하고, 그게 아니면(개선폭이 epsilon 미만이거나 오히려 나빠져도)
        # sat이 best보다 낮은 epoch을 채택한다 — epoch마다 val_rmse가 수백 분의 1 수준으로
        # 출렁이는데(실측 표준편차 ~0.00025) 매번 그 노이즈만으로 best가 계속 갈아치워지는
        # 걸 막으면서도, sat이 실제로 더 낮아진(게이트가 더 이산화된) epoch은 val_rmse가
        # 다소 나빠도 놓치지 않기 위함.
        is_selected = False
        if epoch >= l0_fully_ramped_ep:
            rmse_diff = val_rmse_v - best_val_rmse
            is_better = (rmse_diff <= -params.val_rmse_epsilon) or (sat < best_sat)
            if is_better:
                best_sat = sat
                best_val_rmse = val_rmse_v
                best_epoch = epoch
                is_selected = True
                no_improve = 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "gate_saturation": sat,
                            "val_rmse": val_rmse_v}, ckpt_path)
            else:
                no_improve += 1

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{eff_l0:.6f},{loss_fn.lambda_scen:.6f},{beta_now:.4f},"
                    f"{tr_rmse_v:.6f},{tr_r2_v:.6f},{tr_mse_v:.6f},{tr_ce_v:.6f},{tr_l0_v:.6f},"
                    f"{val_rmse_v:.6f},{val_r2_v:.6f},{sat:.6f},{int(is_selected)}\n")

        if (epoch + 1) % 10 == 0 or is_selected:
            _msg = (f"epoch {epoch+1:4d}  lambda_l0={eff_l0:.4f}  beta={beta_now:.3f}  "
                    f"tr_r2={tr_r2_v:.4f}  val_r2={val_r2_v:.4f}  sat={sat:.3f}")
            tqdm_write(_msg + (" *selected*" if is_selected else ""))

        # 조기종료: best 갱신 없이 --patience 에폭이 지나면 중단. 이후 남은 에폭을
        # 더 돌아도 이미 저장된 best 체크포인트가 바뀌지 않으므로(항상 진짜 best만
        # 저장) 결과에는 영향 없이 시간만 절약된다 — SCRTrainer의 patience와 동일 원리.
        if params.patience > 0 and no_improve >= params.patience:
            best_ep_str = str(best_epoch + 1) if best_epoch >= 0 else "없음(한 번도 개선 안 됨)"
            tqdm_write(f"[p1v2] 조기종료: epoch {epoch+1} (best epoch={best_ep_str}, "
                       f"{params.patience}에폭 연속 val_rmse/sat 개선 없음)")
            break

    if best_epoch < 0:
        tqdm_write("[p1v2] 경고: L0 완전 램프 이후 구간에서 val_rmse/sat이 한 번도 개선되지 않음 — "
                   "마지막으로 돈 에폭을 그대로 채택합니다(epochs를 늘리거나 beta_min을 더 낮춰보세요).")
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "gate_saturation": sat,
                    "val_rmse": val_rmse_v}, ckpt_path)
        best_epoch = epoch

    return SimpleNamespace(best_epoch=best_epoch, best_sat=best_sat)


def t7_save_results(params: SimpleNamespace, data: SimpleNamespace, hyperparams: SimpleNamespace,
                     paths: SimpleNamespace, loop: SimpleNamespace) -> None:
    """7) 결과 저장 — 최종 선택된 체크포인트를 복원해 게이트 JSON/플롯을 저장하고,
    최종 p1v2_summary.json(status 키 없음=완료)을 덮어쓴다."""
    ckpt_path = paths.ckpt_path
    model = hyperparams.model
    spec = params.spec
    output_dir = data.output_dir

    # ---- 최종 선택 체크포인트 복원 후 게이트 JSON 저장 (기존 함수 그대로 재사용) ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # raw HI(x_hi) 랭킹은 커널 피처 사용 여부와 무관하게 항상 같은 방식으로 저장(v2:
    # 커널은 raw를 대체하지 않고 추가하므로).
    hi_cols_ref = get_hi_cols_for_seg("dis_hi")
    hi_cols_by_seg = {s: get_hi_cols_for_seg(spec.scenario_names[s]) for s in range(spec.n_scenarios)}
    _save_probe_masks_to_json(model, output_dir / "gates" / "classification_HIs.json", hi_cols_ref)
    _save_scen_masks_with_shared(model, output_dir / "gates" / "regression_HIs.json", hi_cols_by_seg)

    if data.kernel_names_by_scen is not None:
        # 2026-09-18부터 커널 피처는 시나리오별 로컬 인덱스 공간(폭 K_s, own 커널만) —
        # kernel_names_by_scen[s]가 이미 그 시나리오 전용 이름 목록(길이 K_s)이다.
        # K_s=0인 시나리오는 scr_model.py가 폭1 더미 게이트를 만들어두므로 이름도 1개
        # 채워준다(실제로 선택될 일은 없음 — 입력이 존재하지 않는 슬롯).
        kernel_cols_by_seg = {
            s: (names if names else ["_unused_slot"])
            for s, names in data.kernel_names_by_scen.items()
        }
        _save_scen_masks_to_json(
            model, output_dir / "gates" / "regression_kernel_HIs.json", kernel_cols_by_seg,
            gates=model.scen_kernel_gates,
        )

    _plot_gate_probs(
        model, output_dir / "gates" / "gate_probs.png", hi_cols_ref,
        params.charge_m, params.discharge_m, params.scen_k,
    )
    _plot_loss_curves(paths.log_path, output_dir / "logs" / "loss_curves.png")

    # config.yaml은 학습 루프 시작 전에 이미 써둠(t5_define_output_paths 참고) — cfg가 그
    # 이후 안 바뀌므로 여기서 다시 쓸 필요 없음.
    summary = {
        "tag": params.tag, "seed": params.seed, "split_seed": params.split_seed,
        "lambda_l0_used": hyperparams.loss_cfg["lambda_l0"],
        "lambda_l0_warmup_epochs_used": hyperparams.l0_warmup_ep,
        "selected_epoch": loop.best_epoch, "gate_saturation": loop.best_sat,
        "beta_min": params.beta_min, "l0_fully_ramped_epoch": hyperparams.l0_fully_ramped_ep,
        "output_dir": str(output_dir),
        "n_hi": N_HI, "exclude_stat_leak": EXCLUDE_STAT_LEAK, "exclude_dqdv_leak": EXCLUDE_DQDV_LEAK,
        "synergy_groups_json": None,  # v0~v2 그룹 게이트 스키마 하위호환용 고정 필드
            # (model_lib/tools/visualize_results.py가 과거 run과 나란히 읽을 수 있어야 함) —
            # v4는 kernel HI + interaction_json(shared_gate)으로 완전히 대체돼 이 트레이너
            # 자체는 더 이상 만들지 않는다(2026-09-24 로직 제거).
        "kernel_features_pkl": data.kernel_features_pkl,
        "combined_redundancy_json": data.combined_redundancy_json,
        "interaction_json": data.interaction_json,
        "l0_norm_constant": params.l0_norm_constant,
        "hi_cost_weighted_l0": params.hi_cost_weighted_l0,
    }
    (output_dir / "p1v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[p1v2] 선택된 epoch={loop.best_epoch} (gate_saturation={loop.best_sat:.4f})")
    print(f"[p1v2] run dir: {output_dir}")


def main() -> None:
    args = _parse_args()
    params = t1_set_params_and_config()
    data = t2_load_pkl_json(args, params)
    masks = t3_build_tensor_masks(params, data)
    hyperparams = t4_build_model_and_hparams(params, data, masks)
    paths = t5_define_output_paths(params, data, hyperparams)
    loop = t6_run_training_loop(params, hyperparams, paths)
    t7_save_results(params, data, hyperparams, paths, loop)


if __name__ == "__main__":
    main()
