"""
run_pipeline.py

LFP SOH Prediction 전체 파이프라인 실행기.
데이터 전처리(Step 1~5)부터 모델 학습/평가(Step 6~7)까지 지원.

2026-08-15: Step 4(HI 추출)가 예전엔 항상 `--force`로 캐시를 무시하고 재추출했다
(코드/파라미터를 바꾸고 전체 파이프라인을 처음부터 돌릴 때 낡은 캐시를 실수로 쓰는 걸
막기 위함). 하지만 `python run_pipeline.py 4 --to-step 4 ...`처럼 캐시만 미리
만들어두려는 실행에서도 매번 강제 재추출이 되는 게 비효율적이라, 기본값을
"캐시 있으면 재사용"으로 바꾸고 강제 재추출은 `--force-extract`로 명시할 때만
하도록 뒤집었다 — `hi_correlation.py` 직접 실행과 동일한 기본 동작이 됐다. 단,
`random`/`random_grid`/`protocol`/`vwindow`/`cluster` 등 축 파라미터가 캐시 파일명에
안 들어가는 축(`load_or_extract`의 `else` 분기)은 axis_config 값만 바꾸고 축 이름은
그대로면 옛 캐시를 조용히 재사용할 수 있으니, 그런 축의 파라미터를 바꿀 땐
`--force-extract`를 꼭 같이 줘야 한다.

2026-09-03: Step 6(SCR Phase 1)을 원본 `train_scr.py --phase 1`("Stage0")에서
`5_model/experiments/phase1_lab/phase1_trainer_v2.py`(v0~v5 게이트 안정화 계보)로
교체하고, 기본 레시피를 v4로 맞췄다(`--model-config 5_model/config/main_qfref_S.yaml`
+ kernel_v3 features + N2 interaction json, docs/260827_RESULTS.md 기준). Step 6은
`--phase1-model-config`/`--kernel-features-pkl`/`--interaction-json`/
`--specific-group-ids-json`/`--p1-tag`로 다른 버전(v0/v2/v3/v5)도 재현 가능하다.
phase1_trainer_v2.py는 `--seed`/`--split-seed`가 필수 인자라 미지정 시 42로 자동
채워지고, `--exclude-cv`/`--skip-shape` 플래그는 아예 없어 Step 6에는 전달되지 않는다
(경고만 출력). 또한 v0~v5 체크포인트는 전부 `SOH_EXCLUDE_STAT_LEAK=1`(N_HI=64) 기준이라
Step 6~7 하위 프로세스 환경에 자동으로 이 값을 심는다.

2026-09-03(같은 날, 후속): 구 Step 7(시나리오 분류기, `train_classifier.py`)과
구 Step 8(SCR Phase 2, `train_scr.py --phase 2`)을 파이프라인에서 완전히 제거했다.
phase1_trainer_v2.py(Step 6)가 probe게이트+시나리오게이트+cap_head를 전부 포함한
**단일 통합 모델**을 한 번에 학습하므로, 원래 2단계로 나뉘어 있던 "시나리오 분류 →
그 분류 결과로 회귀 헤드 미세조정"이라는 구조 자체가 더 이상 없다 — Phase 1 학습
결과물이 곧 최종 산출물이다. 이에 따라 구 Step 9(평가, `test_scr.py`)도
`5_model/experiments/phase1_lab/test_phase1_checkpoint.py`로 교체해 Step 6이 만든
run_dir을 직접 평가하는 Step 7로 재배치했다. **별도 분류기 학습 스텝은 없어졌지만
hard/soft 라우팅 평가 자체는 사라지지 않았다** — phase1_trainer_v2.py가 기본으로 쓰는
lambda_scen>0 설정에서는 SCRModel 안에 probe_mlp라는 dual-objective 분류 헤드가 회귀와
함께 CE로 학습되고(scr_model.py), test_phase1_checkpoint.py가 이 probe_mlp를
SCREvaluator에 그대로 라우팅 분류기로 연결해 oracle/hard/soft를 전부 평가한다(2026-09-03
복원, 별도 학습 스텝 불필요) — lambda_scen=0인 체크포인트만 oracle 단독으로 떨어진다.
스텝 번호가 1~7로 당겨졌으므로(구 6/9 → 신 6/7), 예전 `--to-step 8`이나
`--gates-from`/`--with-raw-cnn`/`--skip-classifier` 같은 Phase2·분류기 전용 옵션을
쓰던 스크립트/문서는 갱신이 필요하다.

사용:
  python run_pipeline.py                          # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2                        # Step 2부터 재실행
  python run_pipeline.py 6                        # 학습+평가만 (Step 6~7, v4 기본)
  python run_pipeline.py 6 --to-step 6            # 학습만(평가 제외)
  python run_pipeline.py 7                        # 평가만(직전 Phase 1 run 자동 탐색)
  python run_pipeline.py 3 --workers 8
  python run_pipeline.py 6 --seed 0 --split-seed 0 --p1-tag p1v4_seed0
  python run_pipeline.py 4 --to-step 4 --force-extract --seg-axis random_grid --axis-config '{...}'  # 캐시 무시하고 강제 재추출
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# phase1_trainer_v2.py는 "{MMDD_HHMM}_p1v2_{tag}_seed{seed}" 형식으로 저장한다.
P1V2_RUNS_DIR = ROOT / "5_model" / "experiments" / "phase1_lab" / "results" / "p1v2_runs"

# v4의 실제 학습 레시피(docs/260827_RESULTS.md "v4 정식 학습 결과" 절 그대로) — 다른
# 버전(v0/v2/v3/v5)으로 돌리고 싶으면 --kernel-features-pkl/--interaction-json/
# --specific-group-ids-json을 CLI로 덮어쓰면 된다(v0=둘 다 비우고 --synergy-groups-json,
# v2/v3=--interaction-json만 빼고 kernel만, v5=--specific-group-ids-json 추가).
P1V4_KERNEL_FEATURES_PKL = ("5_model/experiments/phase1_lab/results/"
                             "kernel_group_features_k25_full_N2_kernel_v3.pkl")
P1V4_INTERACTION_JSON = ("5_model/experiments/phase1_lab/results/"
                          "hi_scenario_interaction_k25_full_N2.json")

# (번호, 이름, 스크립트 경로, 기본 추가 인자, --workers 지원 여부)
STEPS = [
    (1, "데이터 변환",          "1_convert/convert_unified.py",    ["--dataset", "all"], True),
    (2, "이상 사이클 제거",     "2_preprocess/preprocess.py",       [],                   True),
    (3, "무결성 검사",          "3_integrity/check_integrity.py",   [],                   True),
    (4, "HI 상관 분석",         "4_hi_analysis/hi_correlation.py",  [],                   True),
    (5, "HI 세그먼트 시각화",   "4_hi_analysis/hi_segment_viz.py",  [],                   True),
    (6, "SCR Phase 1 학습(v4)", "5_model/experiments/phase1_lab/phase1_trainer_v2.py",     [], False),
    (7, "Phase 1 평가",         "5_model/experiments/phase1_lab/test_phase1_checkpoint.py", [], False),
]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m}분 {s}초" if m else f"{s}초"


def _snapshot_p1v2_run_dirs() -> set[Path]:
    """현재 P1V2_RUNS_DIR 내 디렉터리 집합 스냅샷 (phase1_trainer_v2.py 전용)."""
    if not P1V2_RUNS_DIR.exists():
        return set()
    return {d for d in P1V2_RUNS_DIR.iterdir() if d.is_dir()}


def _find_new_p1v2_run_dir(before: set[Path]) -> Path | None:
    """스냅샷 이후 새로 생긴 p1v2 run 디렉터리 반환. 없으면 가장 최신 디렉터리."""
    if not P1V2_RUNS_DIR.exists():
        return None
    after = {d for d in P1V2_RUNS_DIR.iterdir() if d.is_dir()}
    new = after - before
    if new:
        return max(new, key=lambda d: d.stat().st_mtime)
    all_dirs = sorted(after, key=lambda d: d.stat().st_mtime)
    return all_dirs[-1] if all_dirs else None


# ─────────────────────────────────────────────────────────────────────────────
# 스텝 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_step(
    num: int,
    name: str,
    script: str,
    extra_args: list,
    use_workers: bool,
    workers: int,
    model_config: str | None,
    config_flag: str | None = None,
    extra_env: dict | None = None,
) -> bool:
    cmd = [sys.executable, str(ROOT / script)] + extra_args
    if use_workers:
        cmd += ["--workers", str(workers)]
    elif config_flag and config_flag not in extra_args:
        cmd += [config_flag, model_config]

    print(f"\n{'='*60}")
    print(f"  Step {num}  {name}")
    print(f"  $ {' '.join(str(a) for a in cmd)}")
    print(f"{'='*60}")

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0

    ok = result.returncode == 0
    label = "[OK]" if ok else "[FAIL]"
    print(f"\n  {label} Step {num} {'완료' if ok else '실패'}  "
          f"(exit={result.returncode}, {_fmt_time(elapsed)})")
    return ok


def _ask_continue(num: int) -> bool:
    try:
        ans = input(f"\n  Step {num} 실패. 계속 진행하시겠습니까? [y/N]: ").strip().lower()
    except EOFError:
        ans = "n"
    return ans == "y"


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LFP SOH 파이프라인 실행기 (데이터 전처리 → Phase 1 통합 학습/평가)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="스텝 목록:\n" + "\n".join(
            f"  {n}  {name}" for n, name, _, _, _ in STEPS
        ),
    )
    parser.add_argument(
        "from_step", nargs="?", type=int, default=1, metavar="FROM_STEP",
        help=f"시작 스텝 번호 (기본: 1, 범위: 1~{len(STEPS)})",
    )
    parser.add_argument(
        "--to-step", type=int, default=None, metavar="TO_STEP",
        help=f"종료 스텝 번호 포함 (미지정 시 끝까지, 범위: 1~{len(STEPS)})",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="데이터 스텝(1~5)에 전달할 병렬 프로세스 수 (기본: 8)",
    )
    parser.add_argument(
        "--force-extract", action="store_true", dest="force_extract",
        help="Step 4(HI 추출) 캐시를 무시하고 강제 재추출. 기본은 캐시가 있으면 재사용 "
             "(2026-08-15부터 — 예전엔 항상 강제 재추출이었음). 코드/파라미터를 바꾼 뒤나, "
             "random/random_grid처럼 캐시 파일명에 axis_config 값이 안 들어가는 축의 "
             "파라미터만 바꿨을 때는 이 플래그를 꼭 같이 줘야 한다(안 그러면 옛 캐시를 "
             "조용히 재사용함).",
    )
    parser.add_argument(
        "--phase1-model-config", default="5_model/config/main_qfref_S.yaml",
        dest="phase1_model_config",
        help="Step 6(phase1_trainer_v2.py) 모델 설정 파일 (기본: main_qfref_S.yaml — "
             "v4 학습에 실제로 쓰인 설정).",
    )
    parser.add_argument(
        "--kernel-features-pkl", default=P1V4_KERNEL_FEATURES_PKL,
        dest="kernel_features_pkl",
        help="Step 6 전용 — 커널 특징 pkl 경로 (기본: v4가 쓴 kernel_v3 파일). "
             "빈 문자열('')이면 전달하지 않음(v0 재현 등).",
    )
    parser.add_argument(
        "--interaction-json", default=P1V4_INTERACTION_JSON,
        dest="interaction_json",
        help="Step 6/7 전용 — HI x 시나리오 상호작용 JSON 경로 (기본: v4가 쓴 파일). "
             "학습(6)과 평가(7) 양쪽에 동일하게 전달된다(v4 체크포인트는 이 파일 경로가 "
             "p1v2_summary.json에 자동 기록되지 않아 평가 시에도 다시 필요함). "
             "빈 문자열('')이면 전달하지 않음(v0/v2/v3 재현 등).",
    )
    parser.add_argument(
        "--specific-group-ids-json", default=None, dest="specific_group_ids_json",
        help="Step 6/7 전용 — v5(그룹 게이팅) 계보를 재현할 때만 지정. 기본은 미사용(v4 방식).",
    )
    parser.add_argument(
        "--p1-tag", default="p1v4_full", dest="p1_tag",
        help="Step 6 phase1_trainer_v2.py의 --tag (run 디렉터리 이름에 들어감, 기본: p1v4_full)",
    )
    parser.add_argument(
        "--rep-cells", nargs="+", default=None, dest="rep_cells",
        help="Step 7 평가 시 용량곡선 비교 플랏을 그릴 셀 ID(들) (미지정 시 데이터셋별 1개 자동 선정)",
    )
    parser.add_argument(
        "--export-for-visualize", action="store_true", dest="export_for_visualize",
        help="Step 7 평가 결과를 visualize_results.py가 읽을 수 있는 "
             "metrics/predictions/routing 파일로 run_dir에 추가 저장",
    )
    parser.add_argument(
        "--seg-axis", default=None, metavar="AXIS",
        help="세그멘테이션 축 (Step 4~6에 전달). 예: qfrac, q_frac_wide, q_frac_ref",
    )
    parser.add_argument(
        "--axis-config", default=None, metavar="JSON",
        help="축 파라미터 JSON (Step 4~6에 전달). 예: '{\"n1\": 0.4, \"n2\": 0.2, \"n_samples\": 4}'",
    )
    # 단축 인자 (PowerShell JSON 우회) — Step 4~6에 그대로 전달
    parser.add_argument("--n1",        type=float, default=None,
                        help="q_frac_wide 구간 크기 (--axis-config 대체, PowerShell 호환)")
    parser.add_argument("--n2",        type=float, default=None,
                        help="q_frac_wide 세그먼트 길이 (--axis-config 대체)")
    parser.add_argument("--n2-start",  type=float, default=None, dest="n2_start",
                        help="q_frac_ref n2 범위 모드 하한 — 세그먼트 길이를 고정하지 않고 "
                             "{n2_start, +n2_step, ..., n2_end} 격자에서 랜덤 추첨(커버리지 100%% "
                             "타일링). --n2-end와 반드시 함께 (--axis-config 대체)")
    parser.add_argument("--n2-end",    type=float, default=None, dest="n2_end",
                        help="q_frac_ref n2 범위 모드 상한 (--n2-start와 함께, --axis-config 대체)")
    parser.add_argument("--n2-step",   type=float, default=None, dest="n2_step",
                        help="q_frac_ref n2 격자 간격 (기본 0.1, --axis-config 대체)")
    parser.add_argument("--n-samples", type=int,   default=None, dest="n_samples",
                        help="q_frac_wide/vqslope 구간당 세그먼트 수 (--axis-config 대체)")
    # q_frac_ref 전용 단축 인자 (n1/n2/n_samples는 q_frac_wide와 공유해 위 인자 그대로 씀)
    parser.add_argument("--ref-lag",   type=int, default=None, dest="ref_lag",
                        help="q_frac_ref 레퍼런스 지연 사이클 수 (기본 0=q_frac_wide와 동등, --axis-config 대체)")
    parser.add_argument("--noise-amp", type=float, default=None, dest="noise_amp",
                        help="q_frac_ref 레퍼런스 노이즈 최대 진폭, 분수 (기본 0.03=±3%%, --axis-config 대체)")
    parser.add_argument("--noise-mode", type=str, default=None, dest="noise_mode",
                        choices=["ou", "sine"],
                        help="q_frac_ref 노이즈 드리프트 방식 ou(기본)|sine(구버전) (--axis-config 대체)")
    parser.add_argument("--noise-period", type=float, default=None, dest="noise_period_cycles",
                        help="q_frac_ref 노이즈 평균회귀 특성시간/파장(사이클 수, 기본 200, --axis-config 대체)")
    parser.add_argument("--min-pts", type=int, default=None, dest="min_pts",
                        help="q_frac_wide/q_frac_ref 세그먼트 최소 포인트 수(기본 10, --axis-config 대체). "
                             "기본값과 다르면 '_minptsN' 접미사 경로에 별도 저장됨")
    parser.add_argument("--calibration-period", type=int, default=None, dest="calibration_period",
                        help="q_frac_ref 레퍼런스 재보정 주기(사이클 수) — N사이클마다 드리프트를 "
                             "리셋(docs/260903_RESULTS.md §1). 미지정 시 재보정 없음(기존 동작). "
                             "권장값 100 (--axis-config 대체)")
    parser.add_argument("--calibration-mode", type=str, default=None, dest="calibration_mode",
                        choices=["drift_only", "full"],
                        help="재보정 시 무엇을 리셋할지 — drift_only(기본, OU만) | full(바이어스까지) "
                             "(--axis-config 대체)")
    parser.add_argument("--calibration-jitter", type=int, default=None, dest="calibration_jitter",
                        help="재보정 주기를 ±jitter 사이클 흔듦(기본 0=정확히 주기대로). "
                             "calibration_jitter < calibration_period 필요 (--axis-config 대체)")
    parser.add_argument("--data-dir", default=None, dest="p1_data_dir",
                        help="Step 6(phase1_trainer_v2.py) 전용 cycle pkl 경로 오버라이드. "
                             "phase1_trainer_v2.py는 --axis-config만으로 데이터 경로를 자동 계산하지 "
                             "않고 항상 자기 자신의 기본 경로(정식 q_frac_ref 캐논 설정)로 fallback "
                             "하므로, n2 범위 모드·calibration 등 캐논이 아닌 축 설정으로 학습하려면 "
                             "**반드시** 이 옵션과 --seg-data-dir을 함께 줘야 한다(안 주면 축 설정과 "
                             "무관하게 조용히 캐논 데이터로 학습됨).")
    parser.add_argument("--seg-data-dir", default=None, dest="p1_seg_data_dir",
                        help="Step 6 전용 seg pkl 경로 오버라이드. --data-dir와 항상 같이 줄 것 "
                             "(위 설명 참고).")
    parser.add_argument("--exclude-cv", action="store_true", dest="exclude_cv",
                        help="충전 세그먼트 HI에서 CC→CV 전환 이후 구간 제외 (Step 4 전달; "
                             "Step 6=phase1_trainer_v2.py는 이 플래그가 없어 미전달, 경고만 출력). "
                             "Step 4는 결과를 '_ccOnly' 접미사 경로에 저장한다.")
    parser.add_argument("--skip-shape", action="store_true", dest="skip_shape",
                        help="전처리 필터7(형상 이상치 제거) 비활성화 (Step 2/4 전달; "
                             "Step 6=phase1_trainer_v2.py는 이 플래그가 없어 미전달, 경고만 출력). "
                             "Step 2는 _4_data_hi/clean_noshape/에 저장하고, Step 4는 그 데이터로 "
                             "'_noshape' 접미사 경로에 추출한다.")
    # m/k 오버라이드 (Step 6에 그대로 전달 — phase1_trainer_v2.py --charge-m/--discharge-m/--scen-k)
    parser.add_argument("--charge-m",    type=int, default=None,
                        help="충전 probe 상위 m개 (yaml charge_probe_m 오버라이드, Step 6 전달)")
    parser.add_argument("--discharge-m", type=int, default=None,
                        help="방전 probe 상위 m개 (yaml discharge_probe_m 오버라이드, Step 6 전달)")
    parser.add_argument("--scen-k",      type=int, default=None,
                        help="시나리오별 scen HI 수 (yaml scen_k_count 오버라이드, Step 6 전달)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="재현성 시드 — 모델 초기화 torch/numpy/random RNG (Step 6 전달, 기본 42)")
    parser.add_argument("--split-seed",  type=int, default=None,
                        help="train/val/test 셀 분할 시드 (Step 6 전달, 기본 42)")
    args = parser.parse_args()

    # 단축 인자 → args.axis_config(JSON) 로 합침. 이후 기존 --axis-config 전달 로직이
    # 모든 하위 스텝(4~6)에 올바른 JSON을 넘긴다. subprocess는 shell 없이 인자를 그대로
    # 전달하므로 PowerShell 따옴표 벗김 문제가 발생하지 않는다.
    if (args.n1 is not None or args.n2 is not None or args.n_samples is not None
            or args.ref_lag is not None or args.noise_amp is not None
            or args.noise_mode is not None or args.noise_period_cycles is not None
            or args.min_pts is not None or args.n2_start is not None
            or args.n2_end is not None or args.n2_step is not None
            or args.calibration_period is not None or args.calibration_mode is not None
            or args.calibration_jitter is not None):
        import json as _json
        _quick: dict = {}
        if args.n1        is not None: _quick["n1"]        = args.n1
        if args.n2        is not None: _quick["n2"]        = args.n2
        if args.n2_start  is not None: _quick["n2_start"]  = args.n2_start
        if args.n2_end    is not None: _quick["n2_end"]    = args.n2_end
        if args.n2_step   is not None: _quick["n2_step"]   = args.n2_step
        if args.n_samples is not None: _quick["n_samples"] = args.n_samples
        if args.ref_lag   is not None: _quick["ref_lag"]   = args.ref_lag
        if args.noise_amp is not None: _quick["noise_amp"] = args.noise_amp
        if args.noise_mode is not None: _quick["noise_mode"] = args.noise_mode
        if args.noise_period_cycles is not None: _quick["noise_period_cycles"] = args.noise_period_cycles
        if args.min_pts is not None: _quick["min_pts"] = args.min_pts
        if args.calibration_period is not None: _quick["calibration_period"] = args.calibration_period
        if args.calibration_mode   is not None: _quick["calibration_mode"]   = args.calibration_mode
        if args.calibration_jitter is not None: _quick["calibration_jitter"] = args.calibration_jitter
        args.axis_config = _json.dumps(_quick)

    to_step = args.to_step if args.to_step is not None else len(STEPS)

    if not (1 <= args.from_step <= len(STEPS)):
        parser.error(f"from_step 은 1~{len(STEPS)} 사이여야 합니다.")
    if not (1 <= to_step <= len(STEPS)):
        parser.error(f"--to-step 은 1~{len(STEPS)} 사이여야 합니다.")
    if args.from_step > to_step:
        parser.error("from_step 이 to_step 보다 클 수 없습니다.")

    selected = [s for s in STEPS if args.from_step <= s[0] <= to_step]

    print("\n" + "="*60)
    print("  LFP SOH Prediction — 전체 파이프라인")
    print("="*60)
    print(f"  스텝 범위   : {args.from_step} → {to_step}")
    print(f"  병렬 워커   : {args.workers}  (데이터 스텝 전용)")
    if any(s[0] == 6 for s in selected):
        print(f"  Phase1 설정 : {args.phase1_model_config}  (Step 6, phase1_trainer_v2.py)")
        print(f"  Phase1 tag  : {args.p1_tag}")
        print(f"  kernel-pkl  : {args.kernel_features_pkl or '(미사용)'}")
        print(f"  interaction : {args.interaction_json or '(미사용)'}")
        if args.specific_group_ids_json:
            print(f"  group-ids   : {args.specific_group_ids_json}  (v5 계보)")
    if args.seg_axis:
        print(f"  seg-axis    : {args.seg_axis}")
    if args.axis_config:
        print(f"  axis-config : {args.axis_config}")
    if args.exclude_cv:
        print(f"  exclude-cv  : True")
    if args.skip_shape:
        print(f"  skip-shape  : True")
    if args.charge_m is not None:
        print(f"  charge-m    : {args.charge_m}")
    if args.discharge_m is not None:
        print(f"  discharge-m : {args.discharge_m}")
    if args.scen_k is not None:
        print(f"  scen-k      : {args.scen_k}")
    if args.force_extract:
        print(f"  force-extract: True  (Step4 캐시 무시)")
    print(f"  실행 스텝   :")
    for n, name, _, _, _ in selected:
        print(f"    Step {n}  {name}")
    print("="*60)

    total_t0 = time.time()
    failed: list[int] = []

    # Phase 1 run_dir 핸드오프 (Step 6 → Step 7)
    p1_run_dir: Path | None = None
    snapshot: set[Path] = set()

    for num, name, script, extra, use_workers in selected:
        step_extra = list(extra)

        # ── 축 정보 주입 (Step 4~6) ──────────────────────────────────────────
        if num in (4, 5, 6):
            if args.seg_axis:
                step_extra += ["--seg-axis", args.seg_axis]
            if args.axis_config:
                step_extra += ["--axis-config", args.axis_config]

        # ── 강제 재추출 옵션 주입 (Step 4만 — 기본은 캐시 재사용, 2026-08-15) ──
        if num == 4 and args.force_extract:
            step_extra += ["--force"]

        # ── CV 제외 옵션 주입 (Step 4=추출만 — '_ccOnly' 경로 인지 필요).
        #    Step 6(phase1_trainer_v2.py)은 --exclude-cv 플래그 자체가 없어 제외 — 아래
        #    Step 6 전용 블록에서 경고만 출력한다. ──
        if num == 4 and args.exclude_cv:
            step_extra += ["--exclude-cv"]

        # ── shape filter 비활성화 옵션 주입 (Step 2=전처리 자체가 필터7 스킵,
        #    Step 4=추출 — '_noshape' 경로 인지 필요).
        #    Step 6은 --skip-shape 플래그가 없어 제외(아래 Step 6 전용 블록에서 경고) ──
        if num in (2, 4) and args.skip_shape:
            step_extra += ["--skip-shape"]

        # ── m/k/시드 오버라이드 주입 (Step 6=Phase1) ─────────────────────────
        if num == 6:
            if args.charge_m is not None:
                step_extra += ["--charge-m", str(args.charge_m)]
            if args.discharge_m is not None:
                step_extra += ["--discharge-m", str(args.discharge_m)]
            if args.scen_k is not None:
                step_extra += ["--scen-k", str(args.scen_k)]
            # phase1_trainer_v2.py는 --seed/--split-seed가 required=True라 항상 값을
            # 넘겨야 한다 — 미지정 시 42로 채운다.
            _seed = args.seed if args.seed is not None else 42
            _split_seed = args.split_seed if args.split_seed is not None else 42
            if args.seed is None or args.split_seed is None:
                print(f"\n  [안내] Step 6(phase1_trainer_v2.py)는 --seed/--split-seed가 "
                      f"필수 인자라 미지정 값을 기본 42로 채웁니다 "
                      f"(seed={_seed}, split-seed={_split_seed}).")
            step_extra += ["--seed", str(_seed), "--split-seed", str(_split_seed)]

            step_extra += ["--tag", args.p1_tag]
            if args.kernel_features_pkl:
                step_extra += ["--kernel-features-pkl", args.kernel_features_pkl]
            if args.interaction_json:
                step_extra += ["--interaction-json", args.interaction_json]
            if args.specific_group_ids_json:
                step_extra += ["--specific-group-ids-json", args.specific_group_ids_json]

            # phase1_trainer_v2.py는 --axis-config만으로 데이터 경로를 자동 계산하지
            # 않는다 — --data-dir/--seg-data-dir을 안 주면 자기 자신의 기본값(정식
            # q_frac_ref 캐논 경로)으로 항상 fallback한다. n2 범위 모드·calibration처럼
            # 캐논이 아닌 축 설정을 쓰면서 이걸 빠뜨리면, scenario_spec.json은 그
            # 설정을 반영해 만들어지는데 실제로 로드되는 pkl은 캐논 데이터라는
            # "spec과 데이터 불일치"가 조용히 발생한다 — 반드시 명시적으로 확인.
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]
            _non_canon = (args.n2_start is not None or args.calibration_period is not None
                          or (args.axis_config and args.axis_config != "{}"))
            if _non_canon and not (args.p1_data_dir and args.p1_seg_data_dir):
                print("\n  [경고] --seg-axis/--axis-config가 정식(캐논) q_frac_ref 설정과 다른데 "
                      "--data-dir/--seg-data-dir을 안 줬습니다 — Step 6이 이 축 설정을 반영한 "
                      "scenario_spec.json은 만들면서, 실제 pkl 데이터는 phase1_trainer_v2.py의 "
                      "기본 캐논 경로에서 그대로 읽어버립니다(spec-데이터 불일치, 조용히 틀린 "
                      "결과). Step 4로 미리 추출한 경로를 --data-dir/--seg-data-dir로 명시하세요.")

            if args.exclude_cv or args.skip_shape:
                print("\n  [경고] phase1_trainer_v2.py는 --exclude-cv/--skip-shape 옵션이 "
                      "없습니다(train_scr.py 전용 플래그) — Step 6에는 전달하지 않습니다. "
                      "해당 변형 데이터로 Phase 1을 학습하려면 --data-dir/--seg-data-dir을 "
                      "phase1_trainer_v2.py에 직접 지정하는 별도 실행이 필요합니다.")
            if os.environ.get("SOH_EXCLUDE_STAT_LEAK") != "1":
                print("\n  [안내] SOH_EXCLUDE_STAT_LEAK=1 을 Step 6 하위 프로세스 환경에 "
                      "자동 설정합니다(v0~v5 체크포인트 계보는 전부 N_HI=64 기준 — 이 값이 "
                      "없으면 66으로 계산돼 shape 불일치가 납니다). Step 7도 동일 계보를 "
                      "이어가야 하므로 같은 값을 물려받습니다.")

            # Phase 1 학습 전 스냅샷 (phase1_trainer_v2.py는 p1v2_runs/ 전용 디렉터리 사용)
            snapshot = _snapshot_p1v2_run_dirs()

        # ── Step 7 전용: 평가할 run_dir + v4/v5 재구성에 필요한 인자 주입 ──────
        if num == 7:
            run_src = str(p1_run_dir) if p1_run_dir else None
            if run_src is None:
                latest = _find_new_p1v2_run_dir(set())
                run_src = str(latest) if latest else None
            if run_src:
                step_extra += ["--run-dir", run_src]
                print(f"\n  → run-dir (평가 대상): {run_src}")
            else:
                print("\n  [경고] Phase 1 run 디렉터리를 찾을 수 없습니다. "
                      "--run-dir을 직접 지정하려면 test_phase1_checkpoint.py를 따로 실행하세요.")
            # v4는 interaction_json이 p1v2_summary.json에 자동 기록되지 않으므로 학습 때와
            # 동일한 값을 평가에도 다시 넘겨야 한다(스크립트 자체 docstring 참고).
            if args.interaction_json:
                step_extra += ["--interaction-json", args.interaction_json]
            if args.specific_group_ids_json:
                step_extra += ["--specific-group-ids-json", args.specific_group_ids_json]
            if args.rep_cells:
                step_extra += ["--rep-cells", *args.rep_cells]
            if args.export_for_visualize:
                step_extra += ["--export-for-visualize"]

        # v0~v5 체크포인트 계보는 N_HI=64(SOH_EXCLUDE_STAT_LEAK=1) 기준으로 통일돼 있어야
        # 하므로, phase1_trainer_v2.py를 쓰는 Step 6과 그걸 평가하는 Step 7에 동일 값을 준다.
        _config_flag = "--model-config" if num == 6 else None
        _step_model_config = args.phase1_model_config if num == 6 else None
        _extra_env = {"SOH_EXCLUDE_STAT_LEAK": "1"} if num in (6, 7) else None
        ok = run_step(num, name, script, step_extra, use_workers, args.workers,
                       _step_model_config, config_flag=_config_flag, extra_env=_extra_env)

        # ── Phase 1 후: 신규 run dir 기록 (phase1_trainer_v2.py → p1v2_runs/) ──────
        if num == 6:
            p1_run_dir = _find_new_p1v2_run_dir(snapshot)
            if p1_run_dir:
                print(f"  → Phase 1 run dir: {p1_run_dir}")
            else:
                print("  [경고] Phase 1 run 디렉터리를 감지하지 못했습니다.")

        if not ok:
            failed.append(num)
            if not _ask_continue(num):
                print("  파이프라인 중단.")
                sys.exit(1)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    if failed:
        print(f"  완료 (실패 스텝: {failed})  총 {_fmt_time(total_elapsed)}")
    else:
        print(f"  전체 완료  총 {_fmt_time(total_elapsed)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
