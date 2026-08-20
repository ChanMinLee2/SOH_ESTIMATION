"""
5_model/experiments/phase1_lab/build_synergy_groups.py

Phase1 학습 이전에 실행하는 "시너지 그룹" 사전 구성 스크립트 (기존 5_model 코드 무변경).

설계 배경 (세션 논의 요약):
  - 다중공선성 제거를 먼저 하고 시너지를 나중에 찾으면, "클러스터 대표"를 뭘로 뽑을지가
    시너지 정보 없이 정해져야 하는 순환 논리 문제가 생긴다.
  - 그래서 순서를 뒤집지 않고, 편상관계수(partial correlation) 필터 하나로
    "다중공선성 배제"와 "시너지 후보 발굴"을 동시에 수행한다:
      - 이미 그룹에 있는 멤버와 거의 같은 정보(|raw corr| >= redundancy-threshold)인 후보는
        애초에 후보에서 제외 (= 다중공선성 배제, Stage4의 클러스터 임계값과 동일 기준 재사용)
      - 살아남은 후보 중, 그룹으로 conditioning했을 때도 target과의 관계가 여전히/더 강하게
        남는(편상관계수가 큰) 후보를 추가 (= 시너지 있는 조합 우선)
  - 시간복잡도: 그룹 성장 단계마다 전체 후보를 다 정밀 검사(회귀 기반 편상관계수)하지 않고,
    먼저 "그 seed와의 단순 상관계수"로 상위 --prefilter-top-m개만 추린 뒤에만 정밀 계산 —
    O(N^2 * S) -> O(N^2) 필터 + O(N * M * S) 본검사로 완화(N=HI 후보 수, S=표본 수, M=prefilter 폭).

알고리즘 (시나리오별로 독립 수행):
  1. 전체 HI를 target과의 단순 상관계수 |r| 내림차순으로 정렬 -> seed 순서.
  2. 아직 어느 그룹에도 안 속한 seed를 하나씩 꺼내 새 그룹 시작.
  3. 그 그룹을 최대 --max-group-size(기본 4)까지 그리디로 채움:
     a. 미배정 후보 중, 지금 그룹의 어느 멤버와도 |raw corr| < --redundancy-threshold(기본 0.9)인
        것만 남김 (다중공선성 배제).
     b. 그중 |raw corr(candidate, target)|가 큰 상위 --prefilter-top-m개만 추림 (저비용 필터).
     c. 그 M개에 대해서만 편상관계수(그룹 멤버 전체로 target/후보를 회귀한 잔차의 상관)를 계산 —
        가장 큰 후보를 채택. 채택 기준(|편상관계수|) < --min-partial-corr면 이 그룹은 그만 채움.
  4. 모든 HI가 정확히 하나의 그룹에 배정될 때까지 반복 (약한 HI는 크기 1짜리 그룹으로 남음).

출력은 기존 gates JSON과 같은 "seg_{s}_..." 키 컨벤션을 따른다 — 나중에 Phase1이 이 파일을
그대로 읽어 그룹 단위로 후보를 제한하도록 확장할 때 다른 스크립트와 동일한 패턴을 쓸 수 있게.

사용 예:
  python 5_model/experiments/phase1_lab/build_synergy_groups.py \
      --model-config 5_model/config/main_qfref_S_p60.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":2}' \
      --data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle" \
      --seg-data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg" \
      --split-seed 42 --tag k25_full_N2_groups
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from log_utils import append_log_entry, current_command_str  # noqa: E402

# analyze_hi_synergy.py와 동일한 이유로 torch 의존 import(load_config/build_datasets/
# get_segmenter)는 모듈 최상단에 두지 않는다 — 이 스크립트는 멀티프로세싱을 안 쓰지만,
# 규칙을 통일해두면 나중에 병렬화가 필요해져도 안전하다.
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
    p = argparse.ArgumentParser(description="Phase1 이전 HI 시너지 그룹 사전 구성 (다중공선성 배제 필터 통합)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True)
    p.add_argument("--data-dir", required=True, help="cycle pkl 경로")
    p.add_argument("--seg-data-dir", required=True, help="seg pkl 경로")
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-group-size", type=int, default=4,
                   help="그룹당 최대 HI 개수 (기본 4)")
    p.add_argument("--redundancy-threshold", type=float, default=0.9,
                   help="|raw corr| >= 이 값이면 같은 그룹에 같이 못 들어감 (Stage4 클러스터 threshold와 동일 기준)")
    p.add_argument("--min-partial-corr", type=float, default=0.02,
                   help="그룹 성장을 멈추는 기준 — 편상관계수 절댓값이 이보다 작으면 더 안 채움")
    p.add_argument("--prefilter-top-m", type=int, default=15,
                   help="그룹 성장 단계마다 정밀 검사(편상관계수)할 후보 수 상한 — "
                        "먼저 단순 상관계수로 이 개수만 추린 뒤에만 정밀 계산 (시간복잡도 완화)")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _load_all_scenarios(args) -> tuple:
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

    x_all = train_ds.x_hi.numpy()
    y_all = train_ds.target.numpy()
    seg_idx_all = train_ds.seg_idx.numpy()
    # 시나리오별 실제 HI 공식 이름 (get_hi_cols_for_seg는 seg 접미사만 다르고 순서는
    # 항상 동일 — hi_schema.py 참고) — hi_00 같은 자리표시자 대신 diff_dqdv_area_chg_lo처럼
    # 바로 읽을 수 있는 이름을 쓰기 위해 시나리오별로 하나씩 만들어둔다.
    names_by_seg = {s: get_hi_cols_for_seg(name) for s, name in enumerate(spec.scenario_names)}
    return x_all, y_all, seg_idx_all, spec, names_by_seg


# ---------------------------------------------------------------------------
# 편상관계수 (그룹 전체로 conditioning) — 작은 회귀 잔차의 상관
# ---------------------------------------------------------------------------

def _residualize(y: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
    """conditioning 열들(+절편)로 y를 회귀한 뒤 잔차 반환. conditioning이 비어있으면 y 그대로."""
    if conditioning.shape[1] == 0:
        return y
    A = np.column_stack([conditioning, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _partial_corr(y: np.ndarray, candidate: np.ndarray, group_x: np.ndarray) -> float:
    if group_x.shape[1] == 0:
        c = np.corrcoef(candidate, y)[0, 1]
        return 0.0 if np.isnan(c) else float(c)
    ry = _residualize(y, group_x)
    rc = _residualize(candidate, group_x)
    if np.std(ry) < 1e-8 or np.std(rc) < 1e-8:
        return 0.0
    c = np.corrcoef(ry, rc)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


# ---------------------------------------------------------------------------
# 그리디 그룹 구성
# ---------------------------------------------------------------------------

def build_groups(
    x: np.ndarray,
    y: np.ndarray,
    max_group_size: int,
    redundancy_threshold: float,
    min_partial_corr: float,
    prefilter_top_m: int,
) -> list[dict]:
    n_hi = x.shape[1]

    # 시드 순서: 단순 상관계수(부호 있음, 정렬은 절댓값 기준) — 필터에도 재사용
    marg_signed = np.zeros(n_hi)
    for i in range(n_hi):
        c = np.corrcoef(x[:, i], y)[0, 1]
        marg_signed[i] = 0.0 if np.isnan(c) else c
    seed_order = list(np.argsort(-np.abs(marg_signed)))

    # 원시 상관행렬 — 다중공선성 배제 가드용 (한 번만 계산, O(N^2 * S))
    raw_corr = np.corrcoef(x, rowvar=False)
    raw_corr = np.nan_to_num(raw_corr, nan=0.0)

    assigned = [False] * n_hi
    groups: list[dict] = []

    for seed in seed_order:
        if assigned[seed]:
            continue
        members = [int(seed)]
        scores = [float(marg_signed[seed])]
        assigned[seed] = True

        while len(members) < max_group_size:
            group_x = x[:, members]

            # a) 다중공선성 배제: 현재 그룹 어느 멤버와도 |raw corr| < threshold인 미배정 후보만
            eligible = [
                c for c in range(n_hi)
                if not assigned[c]
                and all(abs(raw_corr[c, m]) < redundancy_threshold for m in members)
            ]
            if not eligible:
                break

            # b) 저비용 사전 필터: 단순 |상관계수(candidate, target)| 상위 M개만 정밀 검사 대상으로
            eligible.sort(key=lambda c: -abs(marg_signed[c]))
            shortlist = eligible[:prefilter_top_m]

            # c) 정밀 검사: 그룹 전체로 conditioning한 편상관계수
            best_cand, best_score = None, min_partial_corr
            for cand in shortlist:
                pc = _partial_corr(y, x[:, cand], group_x)
                if abs(pc) > abs(best_score):
                    best_score, best_cand = pc, cand

            if best_cand is None:
                break
            members.append(int(best_cand))
            scores.append(float(best_score))
            assigned[best_cand] = True

        groups.append({"members": members, "scores": scores})

    return groups


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    x_all, y_all, seg_idx_all, spec, names_by_seg = _load_all_scenarios(args)

    report: dict = {"tag": args.tag, "max_group_size": args.max_group_size,
                     "redundancy_threshold": args.redundancy_threshold,
                     "min_partial_corr": args.min_partial_corr,
                     "prefilter_top_m": args.prefilter_top_m}

    all_group_sizes: list[int] = []
    for s, seg_name in enumerate(tqdm(spec.scenario_names, desc="시나리오별 그룹 구성", unit="scenario")):
        sel = seg_idx_all == s
        x_scen, y_scen = x_all[sel], y_all[sel]
        if x_scen.shape[0] < 20:
            tqdm_write(f"[groups] {seg_name}: 표본 부족({x_scen.shape[0]}) — 스킵")
            continue

        groups = build_groups(
            x_scen, y_scen,
            max_group_size=args.max_group_size,
            redundancy_threshold=args.redundancy_threshold,
            min_partial_corr=args.min_partial_corr,
            prefilter_top_m=args.prefilter_top_m,
        )
        groups.sort(key=lambda g: -abs(g["scores"][0]))  # seed 개별 중요도 순으로 그룹 정렬

        report[f"seg_{s}_seg_name"] = seg_name
        report[f"seg_{s}_groups"] = [g["members"] for g in groups]
        report[f"seg_{s}_group_names"] = [[names_by_seg[s][i] for i in g["members"]] for g in groups]
        report[f"seg_{s}_group_scores"] = [g["scores"] for g in groups]

        sizes = [len(g["members"]) for g in groups]
        all_group_sizes += sizes
        n_multi = sum(1 for sz in sizes if sz > 1)
        tqdm_write(
            f"[groups] {seg_name}: HI {x_scen.shape[1]}개 -> 그룹 {len(groups)}개 "
            f"(2개 이상 묶인 그룹 {n_multi}개, 최대크기 {max(sizes)}, 평균크기 {np.mean(sizes):.2f})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"synergy_groups_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[groups] 저장: {out_path}")

    mean_size = float(np.mean(all_group_sizes)) if all_group_sizes else 0.0
    n_hi_total = len(all_group_sizes)
    n_groups_total = sum(1 for s in range(spec.n_scenarios) if f"seg_{s}_groups" in report
                          for _ in report[f"seg_{s}_groups"])
    append_log_entry(
        tag=f"synergy_groups_{args.tag}",
        purpose="Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)",
        command=current_command_str(),
        result_files=[str(out_path)],
        key_metrics=f"전체 HI {n_hi_total}개 -> 그룹 {n_groups_total}개, 평균 그룹 크기 {mean_size:.2f}",
        interpretation=(
            "평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), "
            "4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 "
            "함께 보면 이 그룹 구조가 타당한지 교차검증 가능."
        ),
    )


if __name__ == "__main__":
    main()
