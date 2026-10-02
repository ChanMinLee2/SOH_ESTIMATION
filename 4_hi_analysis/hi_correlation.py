"""
hi_correlation.py

_4_data_hi/clean 에서 HI(Health Indicator)를 사이클별로 추출하고 방전 용량(capacity_Ah)과의
Spearman 상관계수를 계산·시각화 — run_pipeline.py Step 4.

2026-09-29: main/plot/logics/constants 4파일 분리 — 이 파일은 진입점(main())만
갖는다. 값은 constants.py, 실행 로직(추출/상관분석/결과 저장)은 logics.py, 그림은
plot.py 소유. hi_compute.py(HI 계산 레지스트리, 여러 tools/ 스크립트가 공유하는
독립 모듈이라 이 스텝 전용 로직이 아님)는 tools/hi_compute.py로 이동했다.
tools/ 스크립트(profile_hi_timing.py 등)가 `from hi_correlation import X` 식으로
여러 이름을 참조하므로, 이 파일은 그 이름들을 constants.py/logics.py/
tools.hi_compute에서 재수출한다(아래 import 블록).

main()이 logics.py 함수를 호출하는 순서(아래 번호는 main() 본문의 동일 번호
주석과 대응 — 흐름만 훑고 싶으면 이 목록 + main()의 번호 주석만 보면 된다):
  1) build_hi_groups            — 세그먼트 이름 → HI_GROUPS/ALL_HI_KEYS 빌드
  2) _print_run_config          — 실행 조건(축·파라미터·경로) 요약 출력
  3) load_cache                 — 캐시 경로 계산 + 캐시 hit면 즉시 로드
  4) extract_all_datasets       — (캐시 miss일 때만) 데이터셋별 전체 셀 HI 추출(병렬)
  5) save_extraction_results    — (〃) 셀 단위 pkl/샘플 CSV/ScenarioSpec/coverage 저장
  6) build_flat_correlation_df  — (〃) 상관분석용 wide df로 재구성 + 캐시 저장
  7) compute_correlations       — HI별 Spearman ρ(HI, capacity_Ah) 계산
  8) compute_extraction_stats   — 데이터셋별 셀/사이클/세그먼트 수 기초 통계
  9) save_step4_result          — correlation.csv/extraction_stats.csv/manifest.json 저장
  (이후 plot.py::_plot_step4_from_result_dir가 step_4_result/ 경로만 받아 전체 플롯을 그림 —
   logics.py가 아니라 plot.py 소유라 위 번호에는 포함하지 않음)

2026-10-01: 예전엔 3)~6)이 load_or_extract() 하나로 묶여있었다(캐시 로드/전체
추출/결과 저장/wide df 재구성이 한 함수 안에서 전부 일어남) — "로드"와 "추출"과
"저장"은 서로 다른 책임이라는 지적을 받아 load_cache/extract_all_datasets/
save_extraction_results/build_flat_correlation_df 4개로 나눴다. axis_dir도
load_cache()가 이미 계산해 반환하므로, 예전처럼 resolve_cache_and_axis_dir()를
8) 직전에 한 번 더 부르지 않는다(중복 호출 제거).

입력 : _4_data_hi/clean/{MIT,HUST,TJU,CALCE}/*.pkl
출력 : _4_data_hi/{axis}/cycle/{DS}/{cell_id}.pkl 
        (dataset/cell_id/cycle/capacity_Ah만 —
         2026-09-28부로 완전 사이클 Global HI는 계산하지 않는다, 아래 "HI 구조" 참고.
         capacity_Ah 컬럼만은 계속 필요 — model_lib/datasets/segment_dataset.py가 세그먼트별
         부분 capacity_Ah를 이 사이클 총량으로 대체하는 데 씀.)
       _4_data_hi/{axis}/seg/{DS}/{cell_id}.pkl   (세그먼트 포맷 — Step 5 이후가 실제로 읽는 것)
       _4_data_hi/{axis}/step_4_result/  

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2026-09-24: q_frac_ref가 유일한 세그멘테이션 축이다(과거 실험용 축 protocol/vwindow/
rcs/cluster/q_abs/vqslope/full_cycle/test_rs는 common/scenario/에서 삭제 — git
히스토리에 남아있으니 필요하면 그쪽에서 복원).

2026-09-29: 이 스크립트는 CLI 인자를 전혀 받지 않는다 — 워커 수/캐시 강제 재추출
여부/데이터셋 그룹/세그멘테이션 축/축 파라미터(n1/n2/ref_lag/noise_amp/...), 전부
parameters.py(ACTIVE_WORKERS/ACTIVE_FORCE_EXTRACT/FIXED_DATASET_GROUP/
FIXED_SEG_AXIS/ACTIVE_AXIS_CONFIG)에서만 읽는다(run_pipeline.py도 동일한 값을
읽어 쓴다 — 단일 소스). exclude_cv(CC-only 구간 제외)/skip_shape(clean_noshape
입력으로 전환) 옵션은 앞으로 쓰지 않기로 해서 같은 날 제거했다(git 히스토리에서
복원 가능). 실행은 그냥:
    python 4_hi_analysis/hi_correlation.py

실험 조건을 바꾸려면 parameters.py를 직접 고친 뒤 실행한다. 여러 조건을 순차로
돌리려면(예: n1을 바꿔가며 5케이스), 매 케이스 전에 parameters.py를 패치하고
실행 후 원래 값으로 복원하는 드라이버 스크립트를 쓴다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HI 구조 (docs/NEW_HIS.md 참조)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Segment (n_seg × 66): 통계 S01–S20 / 미분 D01–D20 / LFP L01–L20 / Morph M01–M06
  (완전 사이클 Global HI, 옛 G01–G15는 2026-09-28부로 계산하지 않음 — 진단/시각화
  전용이었고 모델 학습 입력도 아니었다. git 히스토리에서 복원 가능.)
  세그먼트 이름: 축마다 다름 (q_frac_ref: dis_hi/dis_mid/dis_lo/chg_lo/chg_mid/chg_hi)
  키 명명: stat_{k}_{seg} / diff_{k}_{seg} / lfp_{k}_{seg} / morph_{k}_{seg}
"""

import os

import pandas as pd

import parameters as P

import constants as C
import logics as L
import plot as PLOT

# ── tools/ 스크립트 하위호환 재수출 (2026-09-29 분리 전과 동일하게 `from
# hi_correlation import X`가 계속 동작해야 함 — profile_hi_timing.py/
# seg_corr_analysis.py/seg_diagnose.py가 이 이름들을 참조) ────────────────────
from constants import (  # noqa: F401
    STEP_DIR, MIT_DIR, HUST_DIR, TJU_DIR, CALCE_DIR, CACHE_PATH, HI_ROOT,
    DATASET_GROUPS, STAT_KEYS, DIFF_KEYS, LFP_KEYS, MORPH_KEYS,
    ALL_SEGS, CHG_SEGS, DIS_SEGS, ALL_HI_KEYS, HI_LABELS, HI_GROUPS, HI_GROUP_TAG,
)
from logics import (  # noqa: F401
    _ds_dir, _build_hi_groups, _add_phase, _strip_seg_suffix,
    _qfw_tag, _qfref_tag, extract_dataset_cells, compute_correlations,
    _extract_one_cell,
)
from tools.hi_compute import (  # noqa: F401
    _seg_stat, _seg_diff, _seg_lfp, _seg_morph_curves, _peak_fwhm_asym,
)

from common.scenario import get_segmenter as _get_seg_hi


def main():
    # 2026-09-29: 이 스크립트의 실행 파라미터는 parameters.py에서만 읽는다 — 예전엔
    # --n1/--n2/--axis-config/--dataset-group 등 CLI로 parameters.py를 우회해 값을
    # 바꿀 수 있었지만(대부분 PowerShell JSON 인용 우회 목적), 실험 조건을 바꾸려면
    # 이제 parameters.py 자체를 고치고 실행한다. 여러 조건을 순차 실행하려면 매
    # 케이스 전에 parameters.py를 패치하고 실행 후 원복하는 드라이버 스크립트를
    # 쓴다(docs/REFACTORING.md 2026-09-29 항목). 
    workers = min(P.ACTIVE_WORKERS, os.cpu_count() or 1)
    force = P.ACTIVE_FORCE_EXTRACT
    dataset_group = P.FIXED_DATASET_GROUP
    _axis = P.FIXED_SEG_AXIS
    _axis_cfg: dict = dict(P.ACTIVE_AXIS_CONFIG)

    # 1) build_hi_groups — 세그먼트 이름 집합에 따라 ALL_HI_KEYS를 매번 새로
    # 빌드한다(q_frac_ref 세그먼트 이름 기준 — constants.py의 초기값을 그대로
    # 신뢰하지 않음).
    _seg_names = _get_seg_hi(_axis, {_axis: _axis_cfg}).get_spec().scenario_names
    _, all_hi_keys, _ = L.build_hi_groups(_seg_names)
    print(f"[hi] 세그먼트 이름: {_seg_names}")

    # 2) _print_run_config — 실행 조건 요약 출력(추출 진입 전)
    L._print_run_config(_axis, _axis_cfg, workers, force)

    _ds_names = C.DATASET_GROUPS[dataset_group]

    # 3) load_cache — 캐시 경로 계산 + 캐시 hit면 즉시 로드(df는 hit일 때만 채워짐)
    cache_path, axis_dir, df = L.load_cache(_axis, _axis_cfg, dataset_group, force=force)

    if df is None:
        # 4) extract_all_datasets — 데이터셋별 전체 셀 HI 추출(병렬)
        seg_per_ds, cyc_per_ds, cov_per_ds = L.extract_all_datasets(
            dataset_group, n_workers=workers, axis=_axis, axis_cfg=_axis_cfg)

        # 5) save_extraction_results — 셀 단위 pkl/샘플 CSV/ScenarioSpec/coverage 저장
        L.save_extraction_results(seg_per_ds, cyc_per_ds, cov_per_ds,
                                   _axis, _axis_cfg, axis_dir, dataset_group)

        # 6) build_flat_correlation_df — 상관분석용 wide df로 재구성 + 캐시 저장
        seg_all = pd.concat([seg_per_ds[ds] for ds in _ds_names], ignore_index=True)
        cyc_all = pd.concat([cyc_per_ds[ds] for ds in _ds_names], ignore_index=True)
        print("  총 사이클: " + " / ".join(f"{ds} {len(cyc_per_ds[ds]):,}" for ds in _ds_names)
              + "  (세그먼트 인스턴스: "
              + " / ".join(f"{ds} {len(seg_per_ds[ds]):,}" for ds in _ds_names) + ")")
        df = L.build_flat_correlation_df(seg_all, cyc_all)
        df.to_pickle(cache_path)
        print(f"  캐시 저장: {cache_path}")

    print(f"\n총 사이클: {len(df):,}")

    # 7) compute_correlations — HI별 Spearman ρ(HI, capacity_Ah) 계산
    print("\n=== Spearman ρ 계산 ===")
    corr = L.compute_correlations(df, all_hi_keys, datasets=_ds_names)

    # ── Step4 결과 폴더(step_4_result/) — 2026-09-29 신규 ──────────────────
    # 상관계수 CSV/추출 통계 CSV/매니페스트를 축-설정 디렉터리(cycle/seg 캐시와
    # 같은 부모) 안에 저장한다. 큰 데이터(df, 셀별 pkl)는 이미 axis-config로
    # 키가 매겨진 전역 캐시에 있으므로 복제하지 않는다. axis_dir은 3)에서 이미
    # 계산해뒀으므로 resolve_cache_and_axis_dir()를 여기서 다시 부르지 않는다.
    step_dir = C.HI_ROOT / axis_dir / "step_4_result"

    # 8) compute_extraction_stats — 데이터셋별 셀/사이클/세그먼트 수 기초 통계
    stats_df = L.compute_extraction_stats(axis_dir, _ds_names, cache_path, len(df))
    manifest = {
        "axis": _axis, "axis_cfg": _axis_cfg, "axis_dir": axis_dir,
        "seg_names": list(_seg_names), "cache_path": str(cache_path),
        "dataset_group": dataset_group, "datasets": _ds_names,
        "total_rows": len(df),
    }

    # 9) save_step4_result — correlation.csv/extraction_stats.csv/manifest.json 저장
    L.save_step4_result(step_dir, corr, stats_df, manifest)

    # (logics.py 단계 끝 — 이후는 plot.py::_plot_step4_from_result_dir가 step_4_result/
    # 경로만 받아 전체 플롯을 그린다. 그 폴더의 manifest.json/correlation.csv를
    # 다시 읽어서 그리고, 같은 폴더에 저장) ─────────────────────────────────
    print(f"\n=== Plot 생성: {step_dir} ===")
    try:
        PLOT._plot_step4_from_result_dir(step_dir)
    except Exception as _e:
        print(f"[경고] 플롯 생성 실패(캐시/CSV는 정상 저장됨): {_e}")

    print("완료!")


if __name__ == "__main__":
    main()
