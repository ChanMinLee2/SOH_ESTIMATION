"""9_eval/scenario_attribution.py — 방식 B(풀링 모델 + 사후 attribution) 1회성 분석 스크립트.

docs/phase1_lab/SCENARIO_VALIDATE.md의 Stage 2 아이디어("게이트로 시나리오를 분리하지
않고, 전체 데이터로 풀링 학습한 모델에서 사후적으로 시나리오별 HI 중요도를 추출")를
noscen run(`ACTIVE_AXIS_CONFIG["assign"]="none"`로 학습 — 시나리오 조건화 전혀 없이
방향 2개로만 풀링 학습됨)에 직접 적용한다.

noscen run은 학습 때 6-way 시나리오 라벨을 전혀 안 쓰지만, Step4 추출 시 세그먼트별
q_frac_lo는 그대로 남아있다(SegmentDataset.q_frac_lo, assign과 무관하게 항상 보존됨) —
이 q_frac_lo + n1(축 경계)으로 common/scenario/q_frac_wide.py::_zone_of_segment()를
그대로 재사용해 "모델은 몰랐던" 진짜 6-way 시나리오 라벨을 사후에 복원할 수 있다.

방법 (permutation importance, 방향 내부에서만 셔플):
  1. 체크포인트를 로드하고 test set 전체에 대해 baseline 예측을 1회 구한다.
  2. raw HI 64개 각각에 대해, 그 HI 열만 "같은 방향(충전/방전) 샘플끼리" 무작위로
     섞은 뒤 다시 예측해 MSE 증가량을 측정한다(= permutation importance).
  3. 각 샘플을 사후 복원한 진짜 6-way 시나리오로 묶어, "시나리오별 HI 중요도" 행렬을 만든다.

noscen 모델은 시나리오를 아예 모르는데도 시나리오별로 중요도 패턴이 갈린다면,
그건 "시나리오 정보 자체가 raw HI-SOH 관계에 실재한다"는, 게이트 불안정성과
무관한 깨끗한 증거가 된다(SCENARIO_VALIDATE.md §I-1이 요구하는 바로 그 분리).

실행: python 9_eval/scenario_attribution.py --run-dir model_lib/results/p1v2_runs/1015_2302_p1v2_noscen_no_probe_x_for_reg_seed42
출력: <run-dir>/scenario_attribution/importance_matrix.csv, heatmap.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "8_train"))
sys.path.insert(0, str(PROJECT_ROOT / "9_eval"))

# ── N_HI(raw HI leakage 제외 범위)는 SOH_EXCLUDE_STAT_LEAK/SOH_EXCLUDE_DQDV_LEAK
# 환경변수를 utils.hi_schema가 import 시점에 읽어서 고정한다(run_pipeline.py가 평소
# train.py/test.py를 "그 run의 config.yaml: data.exclude_*_leak" 값으로 미리 env를
# 세팅한 뒤 별도 프로세스로 띄우는 것과 동일한 이유) — 이 스크립트는 그 subprocess
# 레이어 없이 직접 실행되므로, 무거운 프로젝트 모듈(utils.hi_schema/models.scr_model
# 등)을 import하기 전에 --run-dir의 config.yaml을 먼저 가볍게 들여다보고 똑같은 env를
# 세팅해야 한다 — 안 그러면 체크포인트(N_HI=64로 학습됨)와 지금 막 생성한 모델
# (기본 env라면 N_HI=66)의 텐서 shape가 어긋나 load_state_dict가 실패한다.
_run_dir_arg = None
for _i, _a in enumerate(sys.argv):
    if _a == "--run-dir" and _i + 1 < len(sys.argv):
        _run_dir_arg = sys.argv[_i + 1]
    elif _a.startswith("--run-dir="):
        _run_dir_arg = _a.split("=", 1)[1]
if _run_dir_arg:
    import yaml
    _run_dir_path = Path(_run_dir_arg)
    if not _run_dir_path.is_absolute():
        _run_dir_path = PROJECT_ROOT / _run_dir_path
    _cfg_peek = yaml.safe_load((_run_dir_path / "config.yaml").read_text(encoding="utf-8"))
    os.environ["SOH_EXCLUDE_STAT_LEAK"] = "1" if _cfg_peek["data"].get("exclude_stat_leak") else "0"
    os.environ["SOH_EXCLUDE_DQDV_LEAK"] = "1" if _cfg_peek["data"].get("exclude_dqdv_leak") else "0"
    print(f"[scenario_attribution] N_HI env 선반영: SOH_EXCLUDE_STAT_LEAK="
          f"{os.environ['SOH_EXCLUDE_STAT_LEAK']}, SOH_EXCLUDE_DQDV_LEAK="
          f"{os.environ['SOH_EXCLUDE_DQDV_LEAK']} (run의 config.yaml 기준)")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import parameters as P  # noqa: E402
from test import (  # noqa: E402 (중복 구현 금지 — 로딩 로직은 test.py 재사용)
    _resolve_run_and_config, _load_checkpoint, _build_dataset_and_kernel,
    _build_shared_mask, _build_and_load_model, _KernelAugmentedDataset,
)
from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402
from common.scenario.q_frac_ref import QFracRefSegmenter  # noqa: E402

# test.py::_resolve_run_and_config는 P.ACTIVE_DATA_DIR/ACTIVE_SEG_DATA_DIR이 설정돼
# 있으면 run 자신의 config.yaml에 저장된 data_dir을 그걸로 덮어쓴다(여러 run을 같은
# 축으로 재평가하려는 용도) — 이 스크립트는 그 축 전역값과 무관하게 "그 run이 실제로
# 학습된 축"을 정확히 써야 하므로(noscen run은 전용 축 디렉터리를 씀), 여기서 무조건
# None으로 고정해 오버라이드를 비활성화한다.
P.ACTIVE_DATA_DIR = None
P.ACTIVE_SEG_DATA_DIR = None

ROUTING = [[2, 1, 0], [5, 4, 3]]  # routing[dir_idx][latent] = scenario_id (q_frac_wide.py와 동일)
SCEN_NAMES = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]


def _true_scenario_labels(test_ds, axis_cfg: dict) -> np.ndarray:
    """q_frac_lo(+n2) + n1 존 경계로 진짜 6-way 시나리오를 사후 복원한다."""
    seg = QFracRefSegmenter(n1=axis_cfg["n1"], n2=axis_cfg["n2"])
    n2 = axis_cfg["n2"]
    q_lo = np.asarray(test_ds.q_frac_lo, dtype=float)
    direction = test_ds.direction.numpy()
    latent = np.array([seg._zone_of_segment(q, q + n2) for q in q_lo], dtype=int)
    dir_idx = (direction < 0).astype(int)  # 0=charge(+1), 1=discharge(-1)
    scen_id = np.array([ROUTING[d][l] for d, l in zip(dir_idx, latent)], dtype=int)
    return scen_id


def _build_batch(ds, idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {
        "x_hi": ds.x_hi[idx].to(device),
        "nan_mask": ds.nan_mask[idx].to(device),
        "direction": ds.direction[idx].to(device),
        "scen_idx": ds.scen_idx[idx].to(device),
        "cap_init": ds.cap_init[idx].to(device),
    }
    if hasattr(ds, "x_kernel"):
        batch["x_kernel"] = ds.x_kernel[idx].to(device)
    if hasattr(ds, "x_kernel_probe"):
        batch["x_kernel_probe"] = ds.x_kernel_probe[idx].to(device)
    return batch


@torch.no_grad()
def _predict_sq_err(model, ds, target: torch.Tensor, device: torch.device,
                     batch_size: int = 65536) -> np.ndarray:
    n = len(ds)
    sq_err = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        batch = _build_batch(ds, idx, device)
        pred = model.predict(batch)["cap_pred"].cpu().numpy()
        sq_err[idx] = (pred - target[idx].numpy()) ** 2
    return sq_err


def main() -> None:
    ap = argparse.ArgumentParser(description="방식 B: 풀링 모델 사후 attribution (시나리오별 HI 중요도)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    run = _resolve_run_and_config(args.run_dir)
    ckpt_info = _load_checkpoint(run.run_dir)

    # p1v2_summary.json에 기록된 kernel/redundancy/interaction 경로는 학습 당시 cwd
    # 기준이라, 그 뒤 프로젝트 레이아웃이 바뀐 legacy run(예: 구 "5_model/..." 접두사)
    # 에서는 존재하지 않는 경로를 가리킬 수 있다(test.py::_resolve_summary_path의
    # 일반 폴백도 못 잡는 케이스). 이 세 산출물은 전부 run_dir 안에 그대로 남아있으므로,
    # summary.json을 거치지 않고 run_dir 자체를 글롭해서 ACTIVE_* 오버라이드로 직접
    # 지정한다 — _build_dataset_and_kernel/_build_shared_mask는 P.ACTIVE_*가 있으면
    # summary.json보다 그걸 최우선으로 쓴다.
    _kernel_pkls = sorted(run.run_dir.glob("kernel_group_features_*_kernel.pkl"))
    _redund_jsons = sorted(run.run_dir.glob("kernel_group_features_*_combined_redundancy.json"))
    _interact_jsons = sorted(run.run_dir.glob("hi_scenario_interaction_*_interaction.json"))
    if _kernel_pkls:
        P.ACTIVE_KERNEL_FEATURES_PKL = str(_kernel_pkls[0])
        print(f"[scenario_attribution] kernel-features-pkl 로컬 발견: {_kernel_pkls[0]}")
    if _redund_jsons:
        P.ACTIVE_COMBINED_REDUNDANCY_JSON = str(_redund_jsons[0])
        print(f"[scenario_attribution] combined-redundancy-json 로컬 발견: {_redund_jsons[0]}")
    if _interact_jsons:
        P.ACTIVE_INTERACTION_JSON = str(_interact_jsons[0])
        print(f"[scenario_attribution] interaction-json 로컬 발견: {_interact_jsons[0]}")

    data = _build_dataset_and_kernel(run.cfg, run.summary)
    shared_hi_mask = _build_shared_mask(data.spec, run.summary)
    model = _build_and_load_model(run.cfg, data, shared_hi_mask, run.device, ckpt_info.ckpt)

    test_ds = data.test_ds
    if hasattr(test_ds, "x_kernel"):
        test_ds = _KernelAugmentedDataset(test_ds)

    axis_cfg = run.cfg["scenario"]["axis_config"]
    true_scen = _true_scenario_labels(test_ds, axis_cfg)
    direction = test_ds.direction.numpy()

    ref_seg = data.spec.scenario_names[0]
    suffix = f"_{ref_seg}"
    hi_names = [n[: -len(suffix)] if n.endswith(suffix) else n
                for n in get_hi_cols_for_seg(ref_seg)]
    n_hi = len(hi_names)
    print(f"[scenario_attribution] N_HI={n_hi}, test_n={len(test_ds)}")

    target = test_ds.target if hasattr(test_ds, "target") else test_ds._base.target
    base_sq_err = _predict_sq_err(model, test_ds, target, run.device)
    print(f"[scenario_attribution] baseline MSE={base_sq_err.mean():.6f}")

    rng = np.random.default_rng(args.seed)
    chg_rows = np.where(direction > 0)[0]
    dis_rows = np.where(direction < 0)[0]

    importance = np.zeros((n_hi, len(SCEN_NAMES)), dtype=np.float64)
    counts = np.zeros((n_hi, len(SCEN_NAMES)), dtype=np.int64)

    x_hi_orig = test_ds.x_hi if hasattr(test_ds, "x_hi") else test_ds._base.x_hi

    for j in range(n_hi):
        perturbed = x_hi_orig.clone()
        for rows in (chg_rows, dis_rows):
            if len(rows) < 2:
                continue
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            perturbed[rows, j] = x_hi_orig[shuffled, j]

        if hasattr(test_ds, "x_hi"):
            orig_col = test_ds.x_hi
            test_ds.x_hi = perturbed
        else:
            orig_col = test_ds._base.x_hi
            test_ds._base.x_hi = perturbed
        try:
            perm_sq_err = _predict_sq_err(model, test_ds, target, run.device)
        finally:
            if hasattr(test_ds, "x_hi"):
                test_ds.x_hi = orig_col
            else:
                test_ds._base.x_hi = orig_col

        delta = perm_sq_err - base_sq_err
        for s_id, s_name in enumerate(SCEN_NAMES):
            mask = true_scen == s_id
            if mask.any():
                importance[j, s_id] = delta[mask].mean()
                counts[j, s_id] = int(mask.sum())
        if (j + 1) % 10 == 0 or j == n_hi - 1:
            print(f"[scenario_attribution] {j + 1}/{n_hi} HI 완료")

    out_dir = run.run_dir / "scenario_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(importance, index=hi_names, columns=SCEN_NAMES)
    df.to_csv(out_dir / "importance_matrix.csv", encoding="utf-8")
    counts_df = pd.DataFrame(counts, index=hi_names, columns=SCEN_NAMES)
    counts_df.to_csv(out_dir / "sample_counts.csv", encoding="utf-8")
    print(f"[scenario_attribution] 저장: {out_dir / 'importance_matrix.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, max(8, n_hi * 0.18)))
        order = df.mean(axis=1).sort_values(ascending=False).index
        im = ax.imshow(df.loc[order].values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(SCEN_NAMES)))
        ax.set_xticklabels(SCEN_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(n_hi))
        ax.set_yticklabels(order, fontsize=6)
        _assign = axis_cfg.get("assign", "position_bin")
        _model_desc = "pooled/noscen model" if _assign == "none" else f"scen-gated model (assign={_assign})"
        ax.set_title(f"Scenario-wise permutation importance ({_model_desc})\n{run.run_dir.name}",
                     fontsize=9)
        fig.colorbar(im, ax=ax, label="MSE increase when permuted")
        fig.tight_layout()
        out_png = out_dir / "heatmap.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[scenario_attribution] 저장: {out_png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
