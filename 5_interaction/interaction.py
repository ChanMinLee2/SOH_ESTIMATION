"""
5_interaction/interaction.py

HI x 시나리오 상호작용 검정. 모델과 완전히 분리된 절차로, "이 HI가 시나리오마다
SOH와의 관계가 실제로 다른가"를 확인한다. 학습을 안 거치므로 게이트가 우연히
그렇게 갈랐는지, 데이터 자체에 진짜 구조가 있는지 독립적으로 확인 가능.

이 스크립트의 산출물(JSON)은:
  1) v4의 shared_gate/scen_gates 라우팅 결정(유의미한 HI만 시나리오별 게이트 유지)
  2) plot.py(같은 폴더, 셀 단위 분포 시각화)의 입력
에 쓰인다. train split만 사용(val/test 누수 없음) — 데이터 로더(_load_all_scenarios,
아래)는 이 스크립트가 정의하고 synergy.py(Step 6)가 그대로 가져다 쓴다(중복 구현
금지 원칙 — 2026-09-30 기존엔 반대 방향이었는데 Step 실행 순서와 맞춰 교정했다).

(2026-10-01: "HI 값 자체의 시나리오 구분력"(상관계수 차이와는 다른 질문 —
시나리오 분류기 feature 선택용)을 η²로 재는 별도 기능을 한 라운드 추가했다가
바로 제거했다 — 이 스크립트는 v4 게이트 결정이라는 단일 질문에만 집중한다.
구분력 분석이 필요해지면 별도 스크립트로 새로 만들 것, docs/REFACTORING.md 참고.)

2026-09-30 방법론 변경: 원래는 Fisher z-변환 상관계수 동일성 검정(시나리오
15쌍 각각에 대해 "r_a = r_b"를 검정, 최소 p-value를 15쌍 중 최악 쌍으로 선택,
64~66개 HI에 Benjamini-Hochberg 다중비교 보정)을 썼는데, 실제 판정에는 쓰이지
않는 죽은 계산이었다 — 최종 판정은 처음부터 std_r_across_scenarios(효과크기)
기준이었고, p-value는 세그먼트 행 개수(시나리오당 수십만 건)를 표본 크기로 써서
사실상 항상 유의하게 나왔다(실측 66/66). 근본 원인은 "표본이 커서"가 아니라
**유사반복(pseudo-replication)** — 같은 셀에서 나온 세그먼트 수백~수천 개가
서로 강하게 상관된 비독립 표본인데 이를 독립 표본처럼 취급해 표준오차가 수천 배
과소추정됐다. 그래서 Fisher z 검정 전체(및 BH 보정)를 제거하고, 대신 **셀 단위
직접 계산**으로 교체했다: 셀 하나하나에서(다른 셀과 안 섞고) 독립적으로
std_r_across_scenarios를 구해 셀간 분포의 95% CI를 보는 방식 — 심슨의 역설류
집계 착시를 걸러내면서도 유사반복 문제가 없다(셀이 곧 독립 단위).

2026-09-30: --out-dir를 제외한 모든 CLI 인자를 제거했다(hi_correlation.py
2026-09-29 정리와 동일 원칙) — seg-axis/axis-config/data-dir/seg-data-dir/
datasets/split-seed/shuffle-from/shuffle-seed 전부 parameters.py에서만
읽는다(ACTIVE_*/FIXED_INTERACTION_*/FIXED_CANONICAL_*/FIXED_SHUFFLE_SEED/
FIXED_INTERACTION_SHUFFLE_FROM). --out-dir만 예외인 이유: run_pipeline.py가
여러 스텝(5~9)이 공유하는 실험 폴더(run_dir, 실행 시각 타임스탬프 포함)를
계산해서 넘겨주는 값이라 parameters.py만으로는 복원할 수 없다. 실행은 그냥:
    python 5_interaction/interaction.py

main()이 호출하는 핵심 함수 순서(아래 번호는 main() 본문의 동일 번호 주석과
대응 — 두 모드는 상호배타적, FIXED_INTERACTION_SHUFFLE_FROM 값에 따라 둘 중
하나만 실행됨):

일반 모드(FIXED_INTERACTION_SHUFFLE_FROM=None, 기본):
  1) _load_all_scenarios              — (synergy.py 공용) train split 데이터 로드
  2) _compute_per_cell_std_r          — 셀마다 독립적으로 std_r_across_scenarios 계산
  3) _cell_level_ci                   — 셀간 분포의 평균/95% CI(정규근사)
  4) _build_per_hi_result             — HI별 결과 dict 조립(유의성 판정 포함)
  5) _save_json                       — payload를 hi_scenario_interaction_{tag}.json으로 저장
  6) _print_top5                      — 콘솔에 cell_level_std_r_mean 상위 5개 요약 출력

(2026-10-01: 전체 풀링 기준 _compute_per_scenario_correlations/_find_most_
different_pair를 제거했다 — 판정이 cell_level_std_r_mean/cell_level_confirmed
으로 완전히 이전된 뒤로 풀링 r_by_scenario/worst_pair는 출력 JSON에만 남아있는
순수 진단 필드였는데, 그마저도 "풀링값은 체계적으로 깎인다"는 걸 이미 알고 있는
상태라 오히려 오해를 살 수 있어 코드와 출력 필드(r_by_scenario/std_r_across_
scenarios/worst_pair/worst_pair_delta_r) 모두 삭제했다.)

v4-ctrl 모드(FIXED_INTERACTION_SHUFFLE_FROM 설정 시 — 셀 단위 계산 생략, 조기 종료):
  1) _shuffle_significant — 참조 파일의 공유/특이 *개수*만 유지하고 배정만 무작위 재배정
  2) _save_json           — 일반 모드와 동일한 저장 함수 재사용
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import copy
from datasets.segment_dataset import build_datasets
from common.scenario import get_segmenter
from utils.hi_schema import get_hi_cols_for_seg

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
RESULTS_DIR = PROJECT_ROOT / "model_lib" / "results"

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import parameters as P  # noqa: E402 — 축/실행/통계 파라미터 단일 소스

def _parse_args() -> argparse.Namespace:
    # 2026-09-30: --out-dir 하나만 남기고 전부 제거 — 나머지는 이미 parameters.py가
    # 단일 소스인 값의 CLI 통로였을 뿐이다(hi_correlation.py 2026-09-29 정리와 동일
    # 원칙). --out-dir만 예외인 이유는 모듈 docstring 참고(run_pipeline.py가 여러
    # 스텝이 공유하는 실험 폴더를 넘겨주는 용도라 parameters.py로 복원 불가).
    p = argparse.ArgumentParser(description="HI x 시나리오 상호작용 검정 (효과크기 + 셀 단위 확인, train만 사용)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="산출물 저장 위치(기본: results/) — run_pipeline.py가 Step 9 학습 "
                        "run 폴더로 넘길 때 씀(2026-09-19).")
    return p.parse_args()


def _load_all_scenarios(args) -> tuple:
    """train split 전체를 로드해 (x_all, y_all, scen_idx_all, spec, names_by_seg,
    cell_ids)를 반환, args는 data_dir/seg_data_dir/datasets/split_seed/axis_config(JSON 문자열)/seg_axis
    속성을 가진 아무 객체나 가능(둘 다 SimpleNamespace로 호출)."""

    cfg = copy.deepcopy(P.P1_MODEL_CONFIG)
    cfg["data"]["data_dir"] = args.data_dir
    cfg["data"]["seg_data_dir"] = args.seg_data_dir
    cfg["data"]["datasets"] = args.datasets
    cfg["data"]["split_seed"] = args.split_seed

    axis_cfg = json.loads(args.axis_config)
    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()
    train_ds, _val_ds, _test_ds, _norm = build_datasets(cfg, spec=spec)

    x_all = train_ds.x_hi.numpy()
    y_all = train_ds.target.numpy()
    scen_idx_all = train_ds.scen_idx.numpy()
    # 이 스크립트의 셀 단위 직접 계산(풀링 집계 착시 방지, docs/REFACTORING.md
    # 2026-09-30 참고)이 세그먼트를 셀별로 묶어야 해서 cell_ids도 반환한다 —
    # x_all/y_all/scen_idx_all과 행 순서가 동일(SegmentDataset 생성 시 df 순서 그대로).
    cell_ids = np.asarray(train_ds.cell_ids)
    # 시나리오별 실제 HI 공식 이름 (get_hi_cols_for_seg는 seg 접미사만 다르고 순서는
    # 항상 동일 — hi_schema.py 참고) — hi_00 같은 자리표시자 대신 diff_dqdv_area_chg_lo처럼
    # 바로 읽을 수 있는 이름을 쓰기 위해 시나리오별로 하나씩 만들어둔다.
    names_by_seg = {s: get_hi_cols_for_seg(name) for s, name in enumerate(spec.scenario_names)}
    return x_all, y_all, scen_idx_all, spec, names_by_seg, cell_ids


def _shuffle_significant(ref_path: str, seed: int) -> dict:
    """참조 interaction json의 공유/특이 *개수*는 유지하고 배정만 무작위로 섞는다
    (synergy.py의 build_groups_shuffled와 동일 원칙 — 대조군은 실제 산출물과
    피처/구조 규모가 같아야 순수 효과 비교가 성립한다)."""
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    per_hi = ref["per_hi"]
    concepts = list(per_hi.keys())
    n_sig = sum(1 for v in per_hi.values() if v["significant"])

    rng = np.random.RandomState(seed)
    shuffled_sig = set(rng.choice(len(concepts), size=n_sig, replace=False).tolist())

    result = {}
    for i, c in enumerate(concepts):
        v = dict(per_hi[c])  # 실제 계산값(r_by_scenario 등)은 그대로 복사, 진단용으로 남김
        v["significant"] = i in shuffled_sig
        result[c] = v
    return result, ref


def _pearson_corr_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x의 각 열과 y의 Pearson r을 한 번에 벡터화 계산(np.corrcoef를 열마다 따로
    부르는 것과 동일한 값) — 셀×시나리오 조합마다(수백 번) 66개 HI를 전부 다시
    계산해야 해서 열별 반복 호출로는 느려 벡터화했다. 분산이 0이면(분모 0)
    NaN 대신 0.0으로 대체."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * yc[:, None]).sum(axis=0) / denom
    return np.nan_to_num(r, nan=0.0)


def _compute_per_cell_std_r(
    x_all: np.ndarray, y_all: np.ndarray, scen_idx_all: np.ndarray,
    cell_ids: np.ndarray, n_scen: int, min_segs_per_cell: int,
) -> np.ndarray:
    """셀 하나하나에서(다른 셀과 안 섞고) 독립적으로 시나리오별 r을 구해
    std_r_across_scenarios를 계산 — 전체를 풀링했을 때만 보이는 집계 착시(심슨의
    역설류)가 아니라 개별 셀 수준에서도 이 효과가 진짜인지 확인하기 위함
    (2026-09-30, Fisher z 검정 대체 — 모듈 docstring "방법론 변경" 참고).

    6개 시나리오 중 하나라도 그 셀의 세그먼트 수가 min_segs_per_cell 미만이면
    상관계수가 불안정해지므로 그 셀 전체를 계산에서 제외한다.

    반환: (n_cells_used, n_hi) 배열 — 포함된 셀 수는 len(반환값)로 알 수 있음.
    """
    n_hi = x_all.shape[1]
    unique_cells = np.unique(cell_ids)
    rows = []
    for cell in unique_cells:
        cell_mask = cell_ids == cell
        r_this_cell = np.empty((n_scen, n_hi))
        ok = True
        for s in range(n_scen):
            sel = cell_mask & (scen_idx_all == s)
            if int(sel.sum()) < min_segs_per_cell:
                ok = False
                break
            r_this_cell[s] = _pearson_corr_vec(x_all[sel], y_all[sel])
        if ok:
            rows.append(np.std(r_this_cell, axis=0))
    return np.array(rows) if rows else np.empty((0, n_hi))


def _cell_level_ci(per_cell_std_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """셀별 std_r 행렬(n_cells_used, n_hi) → HI별(열별) 평균과 95% CI.

    정규근사(mean ± 1.96*SE, SE=표본표준편차/sqrt(n_cells))를 쓴다 — 셀 수가
    보통 수십~백 단위라 t분포가 이론적으로 더 정확하지만 이 정도 n에서 z와
    차이가 미미하다. 반환: (mean, ci_lower, ci_upper), 각 (n_hi,).
    """
    n = per_cell_std_r.shape[0]
    mean = per_cell_std_r.mean(axis=0)
    se = per_cell_std_r.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _cohen_r_band(value: float) -> str:
    """Cohen(1988) 상관계수 효과크기 관행 구간 라벨 — 리포팅용(판정에는 min_effect_size만 씀)."""
    if value >= 0.5:
        return "large"
    if value >= 0.3:
        return "medium"
    if value >= 0.1:
        return "small"
    return "negligible"


def _build_per_hi_result(
    concepts: list[str], cell_mean: np.ndarray, cell_ci_lower: np.ndarray,
    cell_ci_upper: np.ndarray, min_effect_size: float,
) -> dict:
    """HI(concept)별 결과 dict 조립 — 셀 단위 std_r 평균/95% CI, 최종 유의성
    판정까지 전부 포함.

    최종 판정 "significant"는 두 조건을 **모두** 만족해야 한다(2026-10-01, 모듈
    docstring "방법론 변경" 참고):
      1) effect_size_meaningful — 셀 단위로 독립 계산한 std_r의 평균(cell_mean)이
         Cohen(1988) 상관계수 효과크기 관행 중 "작음(small)" 문턱(min_effect_size,
         기본 0.1) 이상. 2026-09-30엔 이 문턱을 전체 풀링 std_r_across_scenarios에
         비교해서 근거가 약했다(풀링이 셀 간 베이스라인 차이 때문에 값을 체계적으로
         깎아, 66개 HI 전부 풀링값 < 셀 단위 평균, 평균 2.6배 — "0.1이 틀렸다"가
         아니라 "0.1을 잘못된 깎인 값에 비교했다"가 진짜 문제였음). 중간에 순열
         (permutation) 귀무기준선으로 교체를 시도했으나, 셀 수(120개)가 충분히
         많아 거의 모든 HI가 통과해(최소 7.7배) "우연이 아니다"엔 답했지만 "크기가
         충분하다"엔 답을 못 해 폐기 — Cohen 관행값을 cell_mean에 그대로 재적용
         하는 쪽으로 되돌렸다(docs/REFACTORING.md 참고).
      2) cell_level_confirmed — 셀 단위 std_r의 95% CI 하한이 0보다 큼(풀링 집계
         착시가 아니라 개별 셀 수준에서도 재현되는 효과)
    """
    result = {}
    for i, concept in enumerate(concepts):
        effect_size_meaningful = bool(cell_mean[i] >= min_effect_size)
        cell_level_confirmed = bool(cell_ci_lower[i] > 0)
        result[concept] = {
            "cell_level_std_r_mean": float(cell_mean[i]),
            "cell_level_std_r_ci_lower": float(cell_ci_lower[i]),
            "cell_level_std_r_ci_upper": float(cell_ci_upper[i]),
            "effect_size_band": _cohen_r_band(float(cell_mean[i])),
            "effect_size_meaningful": effect_size_meaningful,
            "cell_level_confirmed": cell_level_confirmed,
            "significant": bool(effect_size_meaningful and cell_level_confirmed),
        }
    return result


def _save_json(out_dir_arg: str | None, tag: str, payload: dict) -> Path:
    """payload를 hi_scenario_interaction_{tag}.json으로 저장(일반 모드/v4-ctrl 모드 공용).
    out_dir_arg가 없으면 RESULTS_DIR(model_lib/results/)에 저장."""
    out_dir = Path(out_dir_arg) if out_dir_arg else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hi_scenario_interaction_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _print_top5(result: dict) -> None:
    """콘솔에 셀 단위 시나리오 간 편차(cell_level_std_r_mean) 가장 큰 HI 5개 요약 출력."""
    top5 = sorted(result.items(), key=lambda kv: -kv[1]["cell_level_std_r_mean"])[:5]
    print("\n[interaction] 시나리오 간 편차(cell_level_std_r_mean) 가장 큰 HI 5개:")
    for name, v in top5:
        print(f"  {name:30s} cell_mean={v['cell_level_std_r_mean']:.4f}"
              f"({v['effect_size_band']}) "
              f"cell_ci=[{v['cell_level_std_r_ci_lower']:.4f}, {v['cell_level_std_r_ci_upper']:.4f}] "
              f"significant={v['significant']}")


def main() -> None:
    args = _parse_args()

    # parameters.py에서 그대로 읽는 실행 파라미터 — --out-dir 외엔 전부 여기서 해석
    # (main() 위 docstring 참고).
    seg_axis = P.FIXED_SEG_AXIS
    axis_config = json.dumps(P.ACTIVE_AXIS_CONFIG)
    data_dir = P.FIXED_CANONICAL_DATA_DIR
    seg_data_dir = P.FIXED_CANONICAL_SEG_DATA_DIR
    datasets = P.FIXED_CANONICAL_DATASETS
    split_seed = P.ACTIVE_SPLIT_SEED if P.ACTIVE_SPLIT_SEED is not None else P.FIXED_DEFAULT_SEED
    min_segs_per_cell = P.FIXED_INTERACTION_MIN_SEGS_PER_CELL
    min_effect_size = P.FIXED_INTERACTION_MIN_EFFECT_SIZE
    shuffle_from = P.FIXED_INTERACTION_SHUFFLE_FROM
    shuffle_seed = P.FIXED_SHUFFLE_SEED
    tag = P.FIXED_INTERACTION_TAG or f"{P.ACTIVE_P1_TAG}_interaction"

    # v4-ctrl 모드 — FIXED_INTERACTION_SHUFFLE_FROM이 설정된 경우만 타는 조기 종료
    # 분기(셀 단위 계산 생략). 번호는 모듈 docstring의 "v4-ctrl 모드" 목록과 대응.
    if shuffle_from:
        print(f"[interaction] v4-ctrl 모드: {shuffle_from}의 공유/특이 개수는 그대로 두고 "
              f"배정만 무작위 재배정(shuffle-seed={shuffle_seed}) — 셀 단위 계산 생략")
        # 1) _shuffle_significant — 참조 파일의 공유/특이 개수만 유지하고 배정만 무작위 재배정
        result, ref = _shuffle_significant(shuffle_from, shuffle_seed)
        n_sig = sum(1 for v in result.values() if v["significant"])
        payload = {
            "tag": tag, "min_effect_size": ref.get("min_effect_size"),
            "n_scenarios": ref.get("n_scenarios"), "scenario_names": ref.get("scenario_names"),
            "n_hi": ref.get("n_hi"), "n_significant": n_sig,
            "per_hi": result,
            "shuffle_from": shuffle_from, "shuffle_seed": shuffle_seed,
            "note": "v4-ctrl: significant 배정이 실제 검정이 아니라 무작위(개수만 참조 파일과 "
                    "동일하게 유지)다 — r_by_scenario 등 나머지 필드는 참조 파일의 실제 계산값을 "
                    "그대로 복사한 진단용 정보이며 이 배정 결정에는 안 쓰였다.",
        }
        # 2) _save_json — 일반 모드와 동일한 저장 함수 재사용
        out_path = _save_json(args.out_dir, tag, payload)
        print(f"[interaction] 저장: {out_path} (공유 {payload['n_hi'] - n_sig}개 / "
              f"특이 {n_sig}개, 무작위 배정)")
        return

    # 일반 모드 — 아래 번호는 모듈 docstring의 "일반 모드" 목록과 대응.
    # 1) _load_all_scenarios — train split 데이터 로드(cell_ids도 함께 — 셀 단위 계산용)
    _loader_args = SimpleNamespace(
        data_dir=data_dir, seg_data_dir=seg_data_dir, datasets=datasets,
        split_seed=split_seed, axis_config=axis_config, seg_axis=seg_axis,
    )
    x_all, y_all, scen_idx_all, spec, names_by_seg, cell_ids = _load_all_scenarios(_loader_args)
    n_hi = x_all.shape[1]
    n_scen = spec.n_scenarios

    # concept 이름(시나리오 접미사 없는 raw HI 개념 이름) — seg_0 기준으로 접미사만 제거
    seg0_names = names_by_seg[0]
    seg0_suffix = f"_{spec.scenario_names[0]}"
    concepts = [n[:-len(seg0_suffix)] if n.endswith(seg0_suffix) else n for n in seg0_names]

    # 2) _compute_per_cell_std_r — 셀마다 독립적으로 std_r_across_scenarios 계산
    per_cell_std_r = _compute_per_cell_std_r(
        x_all, y_all, scen_idx_all, cell_ids, n_scen, min_segs_per_cell)
    n_cells_used = int(per_cell_std_r.shape[0])

    # 3) _cell_level_ci — 셀간 분포의 평균/95% CI(정규근사)
    cell_mean, cell_ci_lower, cell_ci_upper = _cell_level_ci(per_cell_std_r)

    # 4) _build_per_hi_result — HI별 결과 dict 조립(유의성 판정 포함)
    result = _build_per_hi_result(concepts, cell_mean, cell_ci_lower, cell_ci_upper, min_effect_size)

    n_effect = sum(1 for v in result.values() if v["effect_size_meaningful"])
    n_sig = sum(1 for v in result.values() if v["significant"])
    payload = {
        "tag": tag, "min_effect_size": min_effect_size, "min_segs_per_cell": min_segs_per_cell,
        "n_scenarios": n_scen, "scenario_names": spec.scenario_names, "n_hi": n_hi,
        "n_cells_used_for_cell_level_ci": n_cells_used,
        "n_effect_size_meaningful": n_effect, "n_significant": n_sig,
        "per_hi": result,
        "note": "significant=True(=effect_size_meaningful AND cell_level_confirmed)인 HI만 "
                "v4에서 기존 scen_gates에 남기고, 나머지는 shared_gate로 통합한다. "
                "effect_size_meaningful은 셀 단위로 독립 계산한 std_r의 평균(cell_level_std_r_mean)"
                "이 Cohen(1988) 상관계수 효과크기 관행 중 '작음' 문턱(min_effect_size, 기본 0.1) "
                "이상인지를 본다",
    }

    # 5) _save_json — payload를 hi_scenario_interaction_{tag}.json으로 저장
    out_path = _save_json(args.out_dir, tag, payload)

    print(f"[interaction] HI {n_hi}개 중 효과크기(cell_mean>={min_effect_size}) 통과: "
          f"{n_effect}개 / 셀 단위 95% CI 확인(하한>0)까지 통과(최종 판정): {n_sig}개 "
          f"(셀 {n_cells_used}개 사용)")
    print(f"[interaction] 저장: {out_path}")
    # 6) _print_top5 — 콘솔에 cell_level_std_r_mean 상위 5개 요약 출력
    _print_top5(result)


if __name__ == "__main__":
    main()
