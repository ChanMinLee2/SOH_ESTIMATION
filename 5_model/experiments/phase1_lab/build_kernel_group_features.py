"""
5_model/experiments/phase1_lab/build_kernel_group_features.py

build_synergy_groups.py가 만든 그룹(다중공선성 배제 + 편상관계수 시너지 필터를 통과한
시나리오별 HI 묶음, 크기 2 이상만 대상)을 "그룹당 새 HI 하나"로 물리적으로 융합하는 스크립트.
그룹 멤버 HI들 -> SOH를 RBF 커널(Nystroem 근사 + Ridge)로 fit해서, 그 예측값을 그룹의
"커널 HI"로 쓴다 — 편상관계수(선형)로는 못 잡는 비선형 시너지를 명시적으로 캡처하기 위함.
raw HI는 대체하지 않고 그대로 둔 채 별도 블록으로 "추가"한다(phase1_trainer_v2.py가
scr_model.py의 독립 게이트 scen_kernel_gates에 연결 — 설계 배경/이력은
docs/260820_RESULTS.md 참고).

이 스크립트가 하는 다중공선성 관리는 2단계뿐이다:
  1. (build_synergy_groups.py가 이미 함) 그룹 *내부* raw HI 중복 배제.
  2. 이 스크립트: 커널 HI *끼리*(시나리오 다른 그룹끼리도 원본 HI가 겹치면 커널값이
     비슷할 수 있어 전체 train pooled로 재검사) 다중공선성 배제(--redundancy-threshold).
  raw HI와 그걸로 만든 커널 HI 사이의 다중공선성은 검사하지 않는다(알려진 한계,
  docs/260820_RESULTS.md 참고) — scen_gates(raw)/scen_kernel_gates(kernel)가 별개
  게이트라 원칙적으로는 각자 걸러낼 여지가 있지만 명시적 보장은 아니다.

출력(pickle, JSON이 아닌 이유: sklearn 파이프라인 객체를 그대로 저장해 재적용해야 함):
  {
    "tag": str, "n_features": int,
    "alpha": float, "gamma": float|None, "n_components": int,
    "redundancy_threshold": float, "max_features": int|None,
    "features": [
      {"name": str, "scenario": str, "members": [raw HI idx...],
       "member_names": [...], "train_r2": float, "model": Pipeline,
       "mean": float, "std": float}, ...
    ],
  }

사용 예(--seg-axis/--axis-config/--data-dir/--seg-data-dir은 표준 조합이면 생략 가능 —
기본값 자동 적용, 다른 조합이면 넷 다 같이 오버라이드):
  python 5_model/experiments/phase1_lab/build_kernel_group_features.py \
      --model-config 5_model/config/main_qfref_S_p60.yaml \
      --synergy-groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_noleak.json \
      --split-seed 42 --tag k25_full_N2_kernel
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from log_utils import append_log_entry, current_command_str  # noqa: E402

# 루트는 data_directories.py의 DATA_4_HI_ROOT_STR에서 가져온다(build_synergy_groups.py/
# lambda_sweep.py와 동일 이유 — PC마다 실제 드라이브가 다름).
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

# seg-axis/axis-config도 이 세션 전체에서 한 번도 안 바뀐 고정값 — 위 데이터 경로와 세트로
# 묶인 값이라(다른 조합이면 데이터 경로도 같이 바뀌어야 함) 다른 조합을 쓰려면 셋 다
# 함께 오버라이드해야 한다.
DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})

try:
    from tqdm import tqdm as _tqdm

    def tqdm(iterable=None, **kwargs):
        return _tqdm(iterable, **kwargs)

    def tqdm_write(msg: str) -> None:
        _tqdm.write(msg)
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter([])

    def tqdm_write(msg: str) -> None:
        print(msg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="시너지 그룹(크기 2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, "
                     "추가로 넣음) + 2차 다중공선성 배제 + 정규화 통계 저장"
    )
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="cycle pkl 경로")
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR, help="seg pkl 경로")
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--synergy-groups-json", required=True,
                    help="build_synergy_groups.py 산출물 경로 — 이 그룹들을 융합 대상으로 씀")
    p.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge 정규화 강도 (기본 1.0)")
    p.add_argument("--gamma", type=float, default=None,
                    help="RBF 커널 폭 (기본 None -> sklearn 기본값 1/n_features)")
    p.add_argument("--n-components", type=int, default=100,
                    help="Nystroem 랜드마크(근사 차원) 개수 — 시나리오당 표본이 수만~수십만 "
                         "행이라 KernelRidge의 O(n^2) 그람 행렬 대신 Nystroem 근사를 쓴다 "
                         "(기본 100, 그룹 표본 수보다 크면 자동으로 표본 수까지 줄어듦)")
    p.add_argument("--redundancy-threshold", type=float, default=0.9,
                    help="2차 다중공선성 배제 기준(커널 HI끼리) — build_synergy_groups.py와 "
                         "동일 임계값 재사용")
    p.add_argument("--max-features", type=int, default=None,
                    help="최종 커널 HI 개수 상한(기본 None=무제한, 다중공선성 배제 통과한 "
                         "건 전부 유지). 주면 시나리오별 쿼터 라운드로빈으로 그 개수까지만 "
                         "남김(특정 시나리오가 전역 랭킹에서 전부 밀려나는 것 방지)")
    p.add_argument("--min-raw-partial-corr", type=float, default=None,
                    dest="min_raw_partial_corr",
                    help="v3 전용: 커널 예측값을 자기 그룹의 raw 멤버로 조건화한 편상관계수가 "
                         "이 값 미만이면 그 커널 후보를 버린다(raw로 이미 설명되는 걸 "
                         "커널로 한 번 더 만든 것에 불과하다는 뜻이라 raw-커널 간 중복으로 "
                         "간주). 기본 None=필터 비활성(기존 v1/v2 동작과 100% 동일). "
                         "build_synergy_groups.py의 그룹 성장 문턱(0.02)을 그대로 재사용해도 "
                         "되고 별도 값을 줘도 됨 — 이 값 자체가 별도로 튜닝된 적은 없음.")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _load_train_split(args) -> tuple:
    from utils.io_utils import load_config
    from datasets.segment_dataset import build_datasets
    from common.scenario import get_segmenter
    from utils.hi_schema import get_hi_cols_for_seg

    cfg = load_config(args.model_config)
    cfg.setdefault("data", {})
    cfg["data"]["data_dir"] = args.data_dir
    cfg["data"]["seg_data_dir"] = args.seg_data_dir
    cfg["data"]["datasets"] = args.datasets
    cfg["data"]["split_seed"] = args.split_seed

    axis_cfg = json.loads(args.axis_config)
    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()
    train_ds, _val_ds, _test_ds, _norm = build_datasets(cfg, spec=spec)

    x_all = (train_ds.x_hi * train_ds.nan_mask).numpy()  # NaN 위치는 0으로 (forward()와 동일 처리)
    y_all = train_ds.target.numpy()
    seg_idx_all = train_ds.seg_idx.numpy()
    names_by_seg = {s: get_hi_cols_for_seg(name) for s, name in enumerate(spec.scenario_names)}
    return x_all, y_all, seg_idx_all, spec, names_by_seg


def _fit_group_kernel(
    x_group: np.ndarray, y: np.ndarray, alpha: float, gamma: float | None,
    n_components: int, random_state: int,
):
    """Nystroem(RBF 근사) + Ridge. 시나리오당 표본이 수만~수십만 행이라 KernelRidge의
    O(n^2) 그람 행렬은 메모리가 안 감당된다(예: n=278186 -> 288GiB) — Nystroem이
    랜드마크 n_components개만 서브샘플해 커널을 저차원으로 근사한 뒤 그 위에서 선형
    회귀(Ridge)를 푸는, 대규모 데이터의 표준적인 RBF 커널 근사 방식. 개념(비선형 RBF
    조합)은 KernelRidge와 동일하고 .predict() 인터페이스도 동일하다."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline

    n_components_eff = min(n_components, x_group.shape[0])
    model = make_pipeline(
        Nystroem(kernel="rbf", gamma=gamma, n_components=n_components_eff, random_state=random_state),
        Ridge(alpha=alpha),
    )
    model.fit(x_group, y)
    pred = model.predict(x_group)
    return model, float(r2_score(y, pred))


def _residualize(y: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
    """build_synergy_groups.py의 동명 함수와 동일 로직(중복 재구현이지만 두 스크립트가
    서로 import하는 관계가 아니라 독립 유지 — 로직이 5줄짜리라 모듈 결합보다 낫다고 판단)."""
    if conditioning.shape[1] == 0:
        return y
    A = np.column_stack([conditioning, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _raw_conditioned_partial_corr(y: np.ndarray, kernel_pred: np.ndarray, x_group: np.ndarray) -> float:
    """v3 전용: 커널 예측값이 '자기 그룹의 raw 멤버로 이미 설명되는 부분'을 빼고도
    SOH와 관계가 남는지 검사. 낮으면(raw로 이미 설명됨) 커널이 raw 대비 새 정보를
    거의 안 준다는 뜻 — raw-커널 간 중복으로 간주해 후보에서 제외한다."""
    ry = _residualize(y, x_group)
    rk = _residualize(kernel_pred, x_group)
    if np.std(ry) < 1e-8 or np.std(rk) < 1e-8:
        return 0.0
    c = np.corrcoef(ry, rk)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _round_robin_select(
    kept: list[int], candidates: list[dict], cap: int,
) -> list[int]:
    """시나리오별 쿼터 라운드로빈으로 kept(다중공선성 배제를 통과한 후보 인덱스)에서
    최대 cap개를 고른다. 전역 train_r2 랭킹으로 한 번에 자르면 특정 시나리오의 그룹이
    전부 R^2가 낮아 최종본에 하나도 안 남을 수 있다(build_synergy_groups.py로 어렵게 찾은
    그 시나리오 그룹 정보가 통째로 버려짐) — 시나리오마다 "남은 후보 중 최선" 하나씩
    돌아가며 채워 최소 floor(cap/n_scenarios)개는 보장한다."""
    by_scenario: dict[int, list[int]] = {}
    for i in kept:
        by_scenario.setdefault(candidates[i]["scenario_idx"], []).append(i)
    for s in by_scenario:
        by_scenario[s].sort(key=lambda i: -candidates[i]["train_r2"])

    selected: list[int] = []
    scenario_order = sorted(by_scenario.keys())
    while len(selected) < cap and any(by_scenario[s] for s in scenario_order):
        for s in scenario_order:
            if not by_scenario[s]:
                continue
            selected.append(by_scenario[s].pop(0))
            if len(selected) >= cap:
                break
    return selected


def main() -> None:
    args = _parse_args()
    x_all, y_all, seg_idx_all, spec, names_by_seg = _load_train_split(args)

    groups_data = json.loads(Path(args.synergy_groups_json).read_text(encoding="utf-8"))

    candidates: list[dict] = []
    n_skipped_size1 = 0
    for s, seg_name in enumerate(tqdm(spec.scenario_names, desc="그룹 -> 커널 HI 피팅", unit="scenario")):
        key = f"seg_{s}_groups"
        if key not in groups_data:
            continue
        sel = seg_idx_all == s
        if sel.sum() < 20:
            tqdm_write(f"[kernel] {seg_name}: 표본 부족({int(sel.sum())}) — 스킵")
            continue
        x_scen, y_scen = x_all[sel], y_all[sel]

        groups = groups_data[key]
        n_fit = 0
        n_skipped_raw_dup = 0
        for gi, members in enumerate(groups):
            # v3.1(global_dedup) 산출물의 "attached"(사전 가지치기로 탈락해 이 그룹에
            # 사후 편입된 HI)는 여기서 일부러 안 쓴다 — attached는 대표와 항상 |raw
            # corr|>=0.9인 거의 동일 신호라 커널 입력에 추가해도 새 정보가 거의 없는 반면,
            # 그룹마다 커널 입력 폭이 2~29개로 들쭉날쭉해져 (a) RBF 커널이 서로 거의
            # 동일한 컬럼 수십 개로 학습되는 통계적으로 불안정한 상황을 만들고 (b) Level2
            # gap 비교 시 그룹 크기 자체가 교란변수가 된다(실측: 재비교에서 확인됨). attached는
            # groups json에 "이 그룹 소속"이라는 라벨로만 남아 다중공선성 배제 보장(0건)엔
            # 그대로 기여한다 — raw HI 커버리지도 x_hi가 이미 64개 전부 독립적으로 갖고
            # 있어 손실이 없다(커널 융합은 대체가 아니라 추가이므로).
            if len(members) < 2:
                # 크기 1 그룹 = raw HI가 이미 x_hi에 그대로 있으므로 커널 융합 대상에서
                # 제외(안 그러면 자기 자신의 단조 변환에 가까운 슬롯만 하나 더 늘어남).
                n_skipped_size1 += 1
                continue
            x_group = x_scen[:, members]
            model, r2 = _fit_group_kernel(
                x_group, y_scen, args.alpha, args.gamma, args.n_components, args.split_seed,
            )
            if args.min_raw_partial_corr is not None:
                kernel_pred = model.predict(x_group)
                pc = _raw_conditioned_partial_corr(y_scen, kernel_pred, x_group)
                if abs(pc) < args.min_raw_partial_corr:
                    n_skipped_raw_dup += 1
                    continue
            candidates.append({
                "name": f"kernel_{seg_name}_g{gi}",
                "scenario": seg_name,
                "scenario_idx": s,
                "members": [int(m) for m in members],
                "member_names": [names_by_seg[s][i] for i in members],
                "model": model,
                "train_r2": r2,
            })
            n_fit += 1
        extra = f", raw-중복 {n_skipped_raw_dup}개 탈락" if args.min_raw_partial_corr is not None else ""
        tqdm_write(f"[kernel] {seg_name}: 그룹 {len(groups)}개(크기1 {len(groups) - n_fit - n_skipped_raw_dup}개 스킵{extra}) "
                    f"-> 커널 HI {n_fit}개 피팅 완료")

    if not candidates:
        raise RuntimeError("생성된 커널 후보가 없습니다 — synergy-groups-json 내용을 확인하세요.")

    # ------------------------------------------------------------------
    # 2차 다중공선성 배제: 전체 train(모든 시나리오 pooled)에서 커널 값끼리 상관 계산.
    # 그룹 내부 중복은 build_synergy_groups.py가 이미 걸렀지만, 시나리오가 다른 그룹끼리는
    # (원본 HI가 겹치면) 커널 변환 후에도 비슷한 값이 나올 수 있어 여기서 다시 검사.
    # ------------------------------------------------------------------
    kernel_vals = np.zeros((x_all.shape[0], len(candidates)), dtype=np.float64)
    for j, c in enumerate(candidates):
        kernel_vals[:, j] = c["model"].predict(x_all[:, c["members"]])

    corr = np.corrcoef(kernel_vals, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["train_r2"])
    kept: list[int] = []
    for i in order:
        if all(abs(corr[i, k]) < args.redundancy_threshold for k in kept):
            kept.append(i)
    n_dropped_corr = len(candidates) - len(kept)

    # --max-features가 주어졌을 때만 시나리오별 쿼터 라운드로빈으로 캡을 건다(안 주면
    # 다중공선성 배제를 통과한 건 전부 유지 — 더 이상 N_HI 폭에 욱여넣을 필요가 없으므로).
    cap = args.max_features if args.max_features is not None else len(kept)
    final_idx = _round_robin_select(kept, candidates, cap)
    n_dropped_cap = max(0, len(kept) - len(final_idx))
    final = [candidates[i] for i in final_idx]

    # ------------------------------------------------------------------
    # 정규화 통계 — 커널 예측값은 SOH를 직접 예측하도록 fit돼 스케일이 SOH 자체
    # (대략 0.7~1.05)를 따른다. 원래 x_hi는 z-score(평균0/표준편차1)라 스케일이 전혀
    # 다름 — 여기서 train 기준 mean/std를 구해 저장해두고, 적용 시점(phase1_trainer_v2.py)
    # 에서 (v - mean) / std로 표준화한다(val/test엔 이 train 통계를 그대로 적용, fit은 안 함
    # — 누수 없음).
    # ------------------------------------------------------------------
    final_idx_arr = np.array([candidates.index(f) for f in final])
    final_vals = kernel_vals[:, final_idx_arr]
    means = final_vals.mean(axis=0)
    stds = final_vals.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)  # 상수에 가까운 피처 0-division 방지
    for f, m, sd in zip(final, means, stds):
        f["mean"] = float(m)
        f["std"] = float(sd)

    n_final_by_scenario = {
        spec.scenario_names[s]: sum(1 for f in final if f["scenario_idx"] == s)
        for s in range(spec.n_scenarios)
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"kernel_group_features_{args.tag}.pkl"
    artifact = {
        "tag": args.tag,
        "n_features": len(final),
        "alpha": args.alpha,
        "gamma": args.gamma,
        "n_components": args.n_components,
        "redundancy_threshold": args.redundancy_threshold,
        "max_features": args.max_features,
        "min_raw_partial_corr": args.min_raw_partial_corr,
        "features": [
            {k: v for k, v in f.items() if k != "scenario_idx"} for f in final
        ],
    }
    with open(out_path, "wb") as fh:
        pickle.dump(artifact, fh)

    avg_r2 = float(np.mean([f["train_r2"] for f in final]))
    print(f"\n[kernel] 후보 {len(candidates)}개(크기1 그룹 {n_skipped_size1}개 스킵) "
          f"-> 2차 다중공선성 배제로 {n_dropped_corr}개 제거 "
          f"-> {'상한(' + str(args.max_features) + ')으로 ' + str(n_dropped_cap) + '개 추가 제거 -> ' if args.max_features else ''}"
          f"최종 {len(final)}개, 평균 train R^2={avg_r2:.4f}")
    print("[kernel] 시나리오별 최종 커널 HI 개수: " +
          ", ".join(f"{k}={v}" for k, v in n_final_by_scenario.items()))
    print(f"[kernel] 저장: {out_path}")

    append_log_entry(
        tag=f"kernel_group_features_{args.tag}",
        purpose="시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) "
                "+ 2차 다중공선성 배제 + 정규화 통계 저장",
        command=current_command_str(),
        result_files=[str(out_path)],
        key_metrics=(f"후보 {len(candidates)}개 -> 최종 {len(final)}개, "
                     f"평균 train R^2={avg_r2:.4f}, 시나리오별 개수={n_final_by_scenario}"),
        interpretation=(
            "이 pkl은 phase1_trainer_v2.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 "
            "두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 "
            "— raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 "
            "상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호."
        ),
    )


if __name__ == "__main__":
    main()
