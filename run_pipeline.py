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

2026-09-18: build_synergy_groups.py(HI 시너지 그룹 구성)/build_kernel_group_features.py
(커널 HI 피처 생성)를 신규 Step 6/7로 편입했다 — 예전엔 이 둘을 파이프라인 밖에서 손으로
순서대로 돌려서 만든 산출물의 경로를 Step 6(학습)에 --kernel-features-pkl로 손수 연결해야
했는데, --max-group-size 같은 파라미터를 스윕할 때마다 axis-config/data-dir/split-seed를
세 스크립트에 각각 똑같이 맞춰줘야 해서 실수하기 쉬웠다. 이제 이 값들을 파이프라인이 한
번만 받아 세 스크립트 모두에 동일하게 전달하고, Step 7의 출력(kernel pkl +
combined_redundancy json)을 파일 존재 여부로 감지해 Step 8에 자동 연결한다(둘 다
--tag만으로 출력 경로가 결정되는 스크립트라 p1v2_runs/처럼 스냅샷-diff가 필요 없음).
기존 학습(Step 6)/평가(Step 7)는 Step 8/9로 밀렸다 — 예전에도 한 번 있었던 갱신(구
6/9 → 신 6/7)과 같은 종류의 변경이라, "Step 6/7"을 언급하는 이 시점 이전 문서는 그
시점 기준 번호로 남겨두고 갱신하지 않는다.

사용:
  python run_pipeline.py                          # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2                        # Step 2부터 재실행
  python run_pipeline.py 8                        # 학습+평가만 (Step 8~9, v4 기본 — Step 6~7 산출물 재사용)
  python run_pipeline.py 8 --to-step 8            # 학습만(평가 제외)
  python run_pipeline.py 9                        # 평가만(직전 Phase 1 run 자동 탐색)
  python run_pipeline.py 3 --workers 8
  python run_pipeline.py 8 --seed 0 --split-seed 0 --p1-tag p1v4_seed0
  python run_pipeline.py 4 --to-step 4 --force-extract --seg-axis random_grid --axis-config '{...}'  # 캐시 무시하고 강제 재추출
  # 시너지 그룹→커널 HI→학습→평가를 max-group-size=2로 전부 새로 만들어서 실행:
  python run_pipeline.py 6 --max-group-size 2 --p1-tag p1v4_scen_g2 --seg-axis q_frac_ref --axis-config '{...}' --data-dir ... --seg-data-dir ...
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows 콘솔이 cp949일 때(특히 파이프/리다이렉트로 stdout이 콘솔이 아니게 되는 경우,
# 예: `| Tee-Object -FilePath ...`) em-dash 등 특수문자 print가 UnicodeEncodeError로
# 죽는 문제 방지(phase1_trainer_v2.py/lambda_sweep.py와 동일 패턴, 2026-09-13 —
# `--include-stat-leak` 안내문 직전 배너 print에서 실제로 이 문제로 죽는 걸 확인해서 추가).
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent

# phase1_trainer_v2.py는 "{MMDD_HHMM}_p1v2_{tag}_seed{seed}" 형식으로 저장한다.
P1V2_RUNS_DIR = ROOT / "5_model" / "experiments" / "phase1_lab" / "results" / "p1v2_runs"

# v4의 실제 학습 레시피(docs/260827_RESULTS.md "v4 정식 학습 결과" 절 그대로) — 다른
# 버전(v0/v2/v3/v5)으로 돌리고 싶으면 --kernel-features-pkl/--interaction-json/
# --specific-group-ids-json을 CLI로 덮어쓰면 된다(v0=둘 다 비우고 --synergy-groups-json,
# v2/v3=--interaction-json만 빼고 kernel만, v5=--specific-group-ids-json 추가).
# Step 6~7(시너지 그룹/커널 HI 재생성)을 선택 범위에서 빼면(예: `python run_pipeline.py 8`)
# 이 고정 경로가 그대로 Step 8의 기본 --kernel-features-pkl로 쓰인다 — Step 6~7을
# 포함시키면 그 결과물이 자동으로 이 기본값을 덮어쓴다(아래 _synergy_out_path/
# _kernel_out_path 참고).
P1V4_KERNEL_FEATURES_PKL = ("5_model/experiments/phase1_lab/results/"
                             "kernel_group_features_k25_full_N2_kernel_v3.pkl")
P1V4_INTERACTION_JSON = ("5_model/experiments/phase1_lab/results/"
                          "hi_scenario_interaction_k25_full_N2.json")

# (번호, 이름, 스크립트 경로, 기본 추가 인자, --workers 지원 여부)
# 2026-09-18: Step 6~7(시너지 그룹 구성/커널 HI 피처 생성)을 신규 편입 — 예전엔
# build_synergy_groups.py/build_kernel_group_features.py를 파이프라인 밖에서 손으로 3단
# 체인(그룹→커널→학습)으로 돌려야 했는데, --max-group-size 같은 파라미터를 스윕할 때마다
# axis-config/data-dir/seed를 세 스크립트에 각각 손으로 맞추는 게 실수하기 쉬워서
# 편입했다. 기존 Step 6/7(학습/평가)은 Step 8/9로 밀림 — 예전에도 한 번 있었던
# 갱신(구 6/9 → 신 6/7)과 같은 종류의 변경이라, "Step 6/7"을 언급하는 과거 문서는
# 그 시점 기준 번호로 남겨두고 갱신하지 않는다(역사적 기록).
STEPS = [
    (1, "데이터 변환",             "1_convert/convert_unified.py",    ["--dataset", "all"], True),
    (2, "이상 사이클 제거",        "2_preprocess/preprocess.py",       [],                   True),
    (3, "무결성 검사",             "3_integrity/check_integrity.py",   [],                   True),
    (4, "HI 상관 분석",            "4_hi_analysis/hi_correlation.py",  [],                   True),
    (5, "HI 세그먼트 시각화",      "4_hi_analysis/hi_segment_viz.py",  [],                   True),
    (6, "HI 시너지 그룹 구성",     "5_model/experiments/phase1_lab/build_synergy_groups.py",        [], False),
    (7, "커널 HI 피처 생성",       "5_model/experiments/phase1_lab/build_kernel_group_features.py", [], False),
    (8, "SCR Phase 1 학습(v4)",    "5_model/experiments/phase1_lab/phase1_trainer_v2.py",     [], False),
    (9, "Phase 1 평가",            "5_model/experiments/phase1_lab/test_phase1_checkpoint.py", [], False),
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


# build_synergy_groups.py/build_kernel_group_features.py는 p1v2_runs/(타임스탬프 디렉터리)와
#달리 --tag만으로 출력 경로가 완전히 결정된다(RESULTS_DIR/f"synergy_groups_{tag}.json" 등)
# — 그래서 p1v2 run처럼 스냅샷-diff로 "새로 생긴 걸 찾을" 필요가 없고, tag만 알면 Step 8이
# 쓸 경로를 미리 계산할 수 있다.
_KERNEL_RESULTS_DIR = ROOT / "5_model" / "experiments" / "phase1_lab" / "results"


def _synergy_out_path(tag: str) -> Path:
    return _KERNEL_RESULTS_DIR / f"synergy_groups_{tag}.json"


def _kernel_out_paths(tag: str) -> tuple[Path, Path]:
    """(kernel pkl, combined_redundancy json) 경로."""
    base = _KERNEL_RESULTS_DIR / f"kernel_group_features_{tag}"
    return Path(f"{base}.pkl"), Path(f"{base}_combined_redundancy.json")


def _resolve_kernel_paths(
    args, kernel_pkl_out: Path, kernel_redundancy_out: Path, default_pkl: str,
) -> tuple[str, str | None]:
    """Step 8용 kernel-features-pkl/combined-redundancy-json 최종 해석 — 명시적 CLI 값이
    최우선, 없으면 자동 경로가 실존하면 그걸, 그것도 없으면 default_pkl(v4 정식 고정 경로)로
    최종 fallback한다. **파일 존재 여부를 매번 새로 검사**하므로, Step 6~7을 같은 파이프라인
    실행 안에서 방금 돌렸을 때도(그 시점엔 아직 파일이 없다가 Step 7이 끝나야 생김) Step 8
    직전에 다시 부르면 정확히 잡힌다 — main() 맨 위에서 미리보기로 한 번(그때는 파일이 아직
    없을 수 있음) + Step 8 블록에서 다시 한 번(진짜 이때 값) 호출하는 이유."""
    if args.kernel_features_pkl is not None:
        resolved_pkl = args.kernel_features_pkl
    elif kernel_pkl_out.exists():
        resolved_pkl = str(kernel_pkl_out)
    else:
        resolved_pkl = default_pkl
    if args.combined_redundancy_json is not None:
        resolved_redundancy = args.combined_redundancy_json
    elif kernel_redundancy_out.exists():
        resolved_redundancy = str(kernel_redundancy_out)
    else:
        resolved_redundancy = None
    return resolved_pkl, resolved_redundancy


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
        help="Step 6~8(HI 시너지 그룹/커널 HI/phase1_trainer_v2.py) 공통 모델 설정 파일 "
             "(기본: main_qfref_S.yaml — v4 학습에 실제로 쓰인 설정).",
    )
    parser.add_argument(
        "--kernel-features-pkl", default=None,
        dest="kernel_features_pkl",
        help="Step 8 전용 — 커널 특징 pkl 경로. 미지정 시: Step 6~7을 이번에 돌렸거나(또는 "
             "이전에 돌려서) --kernel-tag 기준 자동 경로에 파일이 있으면 그걸 쓰고, 없으면 "
             f"v4 정식 kernel_v3 파일({P1V4_KERNEL_FEATURES_PKL})로 최종 fallback한다. "
             "빈 문자열('')을 명시하면 아예 전달하지 않음(v0 재현 등).",
    )
    parser.add_argument(
        "--interaction-json", default=P1V4_INTERACTION_JSON,
        dest="interaction_json",
        help="Step 8/9 전용 — HI x 시나리오 상호작용 JSON 경로 (기본: v4가 쓴 파일). "
             "학습(8)과 평가(9) 양쪽에 동일하게 전달된다(v4 체크포인트는 이 파일 경로가 "
             "p1v2_summary.json에 자동 기록되지 않아 평가 시에도 다시 필요함). "
             "빈 문자열('')이면 전달하지 않음(v0/v2/v3 재현 등).",
    )
    parser.add_argument(
        "--specific-group-ids-json", default=None, dest="specific_group_ids_json",
        help="Step 8/9 전용 — v5(그룹 게이팅) 계보를 재현할 때만 지정. 기본은 미사용(v4 방식).",
    )
    parser.add_argument(
        "--combined-redundancy-json", default=None, dest="combined_redundancy_json",
        help="Step 8 전용 — build_kernel_group_features.py의 결합(raw+kernel) 다중공선성 "
             "배제 산출물(*_combined_redundancy.json, v4 요구사항2, 2026-09-18). 미지정이고 "
             "Step 6~7 결과물(kernel-tag 기준 자동 경로)이 존재하면 그걸 자동으로 쓴다.",
    )
    # ── Step 6(HI 시너지 그룹 구성, build_synergy_groups.py) 전용 ──────────────────
    parser.add_argument(
        "--max-group-size", type=int, default=4, dest="max_group_size",
        help="Step 6 전용 — 시너지 그룹 최대 크기(그룹당 raw HI 개수 상한, 기본 4). 줄이면"
             "(2~3) 그룹당 RBF 커널 입력 차원이 줄어, 시나리오당 샘플이 적을 때(scen처럼 "
             "6분할) 커널 과적합 위험을 낮출 수 있다는 가설을 검증하는 용도(2026-09-18).",
    )
    parser.add_argument(
        "--synergy-redundancy-threshold", type=float, default=0.9,
        dest="synergy_redundancy_threshold",
        help="Step 6 전용 — build_synergy_groups.py --redundancy-threshold(그룹 내부 raw HI "
             "다중공선성 배제 기준, 기본 0.9).",
    )
    parser.add_argument(
        "--min-partial-corr", type=float, default=0.02, dest="min_partial_corr",
        help="Step 6 전용 — 그룹 성장 시 편상관계수 최소 개선 문턱(기본 0.02).",
    )
    parser.add_argument(
        "--prefilter-top-m", type=int, default=15, dest="prefilter_top_m",
        help="Step 6 전용 — 그룹 성장 후보 사전 필터링 상위 M개(기본 15).",
    )
    parser.add_argument(
        "--global-dedup", action="store_true", dest="global_dedup",
        help="Step 6 전용 — 그룹 성장 시작 전 raw HI끼리 전역 다중공선성을 연결요소 방식으로 "
             "먼저 정리(build_synergy_groups.py v3.1 — 기존 v4 정식 레시피가 사용한 방식).",
    )
    parser.add_argument(
        "--synergy-tag", default=None, dest="synergy_tag",
        help="Step 6 출력 태그(synergy_groups_{tag}.json으로 저장, Step 7이 같은 태그로 "
             "자동 탐색). 미지정 시 --p1-tag에서 자동 파생(f'{p1_tag}_groups').",
    )
    # ── Step 7(커널 HI 피처 생성, build_kernel_group_features.py) 전용 ─────────────
    parser.add_argument(
        "--kernel-synergy-groups-json", default=None, dest="kernel_synergy_groups_json",
        help="Step 7 전용 — build_kernel_group_features.py --synergy-groups-json을 이 값으로 "
             "고정(예: 기존 v3 시너지 그룹 파일 재사용). 미지정 시 Step 6의 --synergy-tag 기준 "
             "자동 경로를 쓴다(Step 6을 이번에 안 돌렸으면 그 경로에 파일이 이미 있어야 함).",
    )
    parser.add_argument(
        "--kernel-alpha", type=float, default=1.0, dest="kernel_alpha",
        help="Step 7 전용 — Ridge 정규화 강도(기본 1.0).",
    )
    parser.add_argument(
        "--kernel-gamma", type=float, default=None, dest="kernel_gamma",
        help="Step 7 전용 — RBF 커널 폭(기본 None=sklearn 기본값 1/n_features).",
    )
    parser.add_argument(
        "--kernel-n-components", type=int, default=100, dest="kernel_n_components",
        help="Step 7 전용 — Nystroem 랜드마크(근사 차원) 개수(기본 100).",
    )
    parser.add_argument(
        "--kernel-redundancy-threshold", type=float, default=0.9,
        dest="kernel_redundancy_threshold",
        help="Step 7 전용 — 2차(커널끼리, pooled) 다중공선성 배제 기준(기본 0.9).",
    )
    parser.add_argument(
        "--kernel-max-features", type=int, default=None, dest="kernel_max_features",
        help="Step 7 전용 — 최종 커널 HI 개수 상한(기본 None=무제한).",
    )
    parser.add_argument(
        "--min-raw-partial-corr", type=float, default=None, dest="min_raw_partial_corr",
        help="Step 7 전용 — raw-커널 간 중복 필터 문턱(기본 None=비활성). 기존 v3 레시피는 0.02.",
    )
    parser.add_argument(
        "--kernel-tag", default=None, dest="kernel_tag",
        help="Step 7 출력 태그(kernel_group_features_{tag}.pkl/.json, Step 8이 같은 태그로 "
             "자동 탐색). 미지정 시 --p1-tag에서 자동 파생(f'{p1_tag}_kernel').",
    )
    # ── Step 8(SCR Phase 1 학습) 전용 — phase1_trainer_v2.py 나머지 옵션 pass-through
    #    (2026-09-18, 이 시점까지 추가된 v4 아키텍처 실험 플래그 전부) ──────────────
    parser.add_argument(
        "--train-cycle-frac", type=float, default=None, dest="train_cycle_frac",
        help="Step 8 전용 — train split 사이클 서브샘플링 비율(기본 None=1.0, 전체 사용).",
    )
    parser.add_argument(
        "--beta-min", type=float, default=None, dest="beta_min",
        help="Step 8 전용 — BETA anneal 종착값(기본 None=phase1_trainer_v2.py 자체 기본 0.1).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None, dest="p1_max_epochs",
        help="Step 8 전용 — 학습 에폭 상한(기본 None=yaml training.epochs).",
    )
    parser.add_argument(
        "--patience", type=int, default=None, dest="p1_patience",
        help="Step 8 전용 — 조기종료 patience(기본 None=phase1_trainer_v2.py 자체 기본 60).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="p1_batch_size",
        help="Step 8 전용 — 배치 크기 오버라이드(기본 None=yaml training.batch_size).",
    )
    parser.add_argument(
        "--l0-warmup-epochs-override", type=int, default=None, dest="l0_warmup_epochs_override",
        help="Step 8 전용 — loss.lambda_l0_warmup_epochs 강제 고정(기본 None=yaml 값).",
    )
    parser.add_argument(
        "--shrinkage-gate", action="store_true", dest="shrinkage_gate",
        help="Step 8 전용 — scen_gates를 ShrinkageHardConcreteGate로(공유+시나리오별 delta). "
             "그룹 계층 게이팅(--specific-group-ids-json 등)과는 동시 사용 불가.",
    )
    parser.add_argument(
        "--lambda-shrink", type=float, default=0.0, dest="lambda_shrink",
        help="Step 8 전용 — --shrinkage-gate의 delta 정칙화 강도(기본 0.0).",
    )
    parser.add_argument(
        "--l0-norm-constant", type=int, default=None, dest="l0_norm_constant",
        help="Step 8 전용 — L0 페널티 정규화 상수를 n_scenarios 대신 이 값으로 고정.",
    )
    parser.add_argument(
        "--scen-gate-direction-only", action="store_true", dest="scen_gate_direction_only",
        help="Step 8 전용 — scen_gates(+scen_kernel_gates) 뱅크 폭을 n_scenarios 대신 "
             "방향 수(2)로 축소.",
    )
    parser.add_argument(
        "--scenario-onehot-input", action="store_true", dest="scenario_onehot_input",
        help="Step 8 전용 — cap_head 입력에 실제 zone/level 원-핫을 추가(게이트와 무관한 "
             "별도 경로).",
    )
    parser.add_argument(
        "--warmstart-branch-epoch", type=int, default=None, dest="warmstart_branch_epoch",
        help="Step 8 전용 — 웜스타트 후 분기 커리큘럼의 분기 시점 T(에폭).",
    )
    parser.add_argument(
        "--device", default=None, dest="p1_device",
        help="Step 8/9 전용 — cuda/cpu/auto(기본 None=phase1_trainer_v2.py/"
             "test_phase1_checkpoint.py 자체 기본값 auto).",
    )
    parser.add_argument(
        "--include-stat-leak", action="store_true", dest="include_stat_leak",
        help="Step 6~9 전용 — SOH_EXCLUDE_STAT_LEAK=1 자동 주입을 끄고 N_HI=66(stat_q_abs/"
             "stat_energy_seg 포함)으로 그룹구성·커널생성·학습·평가한다. 기본(미지정)은 "
             "N_HI=64(v0~v5 체크포인트 계보와 동일). 66으로 돌리려면 --kernel-features-pkl/"
             "--interaction-json도 66-HI 기준으로 새로 만든 파일을 같이 지정해야 한다 — "
             "64-HI용 기본 파일(shared_hi_mask가 64 길이)을 그대로 쓰면 shape 불일치로 "
             "죽는다(docs/260903_RESULTS.md §5-3).",
    )
    parser.add_argument(
        "--p1-tag", default="p1v4_full", dest="p1_tag",
        help="Step 8 phase1_trainer_v2.py의 --tag (run 디렉터리 이름에 들어감, 기본: p1v4_full). "
             "--synergy-tag/--kernel-tag를 안 주면 이 값에서 자동 파생됨.",
    )
    parser.add_argument(
        "--rep-cells", nargs="+", default=None, dest="rep_cells",
        help="Step 9 평가 시 용량곡선 비교 플랏을 그릴 셀 ID(들) (미지정 시 데이터셋별 5개 자동 선정)",
    )
    parser.add_argument(
        "--export-for-visualize", action="store_true", dest="export_for_visualize",
        help="Step 9 평가 결과를 visualize_results.py가 읽을 수 있는 "
             "metrics/predictions/routing 파일로 run_dir에 추가 저장",
    )
    parser.add_argument(
        "--seg-axis", default=None, metavar="AXIS",
        help="세그멘테이션 축 (Step 4~8에 전달). 예: qfrac, q_frac_wide, q_frac_ref",
    )
    parser.add_argument(
        "--axis-config", default=None, metavar="JSON",
        help="축 파라미터 JSON (Step 4~8에 전달). 예: '{\"n1\": 0.4, \"n2\": 0.2, \"n_samples\": 4}'",
    )
    # 단축 인자 (PowerShell JSON 우회) — Step 4~8에 그대로 전달
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
    parser.add_argument("--offset-amp", type=float, default=None, dest="offset_amp",
                        help="q_frac_ref 센서 offset 오차 최대진폭, A 단위(기본 0=비활성). "
                             "전류 크기와 무관하게 사이클 소요시간에 비례하는 절대오차를 추가한다 "
                             "(common/scenario/q_frac_ref.py 모듈 docstring '센서 offset 오차' 절, "
                             "--axis-config 대체)")
    parser.add_argument("--data-dir", default=None, dest="p1_data_dir",
                        help="Step 6~8(시너지 그룹/커널 HI/phase1_trainer_v2.py) 공통 cycle pkl "
                             "경로 오버라이드. 이 세 스크립트 다 --axis-config만으로 데이터 경로를 "
                             "자동 계산하지 않고 항상 자기 자신의 기본 경로(정식 q_frac_ref 캐논 "
                             "설정)로 fallback하므로, n2 범위 모드·calibration·assign=none/"
                             "mid_nmid 등 캐논이 아닌 축 설정으로 돌리려면 **반드시** 이 옵션과 "
                             "--seg-data-dir을 함께 줘야 한다(안 주면 축 설정과 무관하게 조용히 "
                             "캐논 데이터로 돌아감).")
    parser.add_argument("--seg-data-dir", default=None, dest="p1_seg_data_dir",
                        help="Step 6~8 공통 seg pkl 경로 오버라이드. --data-dir와 항상 같이 줄 것 "
                             "(위 설명 참고).")
    parser.add_argument("--exclude-cv", action="store_true", dest="exclude_cv",
                        help="충전 세그먼트 HI에서 CC→CV 전환 이후 구간 제외 (Step 4 전달; "
                             "Step 6~9는 이 플래그가 없어 미전달, 경고만 출력). "
                             "Step 4는 결과를 '_ccOnly' 접미사 경로에 저장한다.")
    parser.add_argument("--skip-shape", action="store_true", dest="skip_shape",
                        help="전처리 필터7(형상 이상치 제거) 비활성화 (Step 2/4 전달; "
                             "Step 6~9는 이 플래그가 없어 미전달, 경고만 출력). "
                             "Step 2는 _4_data_hi/clean_noshape/에 저장하고, Step 4는 그 데이터로 "
                             "'_noshape' 접미사 경로에 추출한다.")
    # m/k 오버라이드 (Step 8에 그대로 전달 — phase1_trainer_v2.py --charge-m/--discharge-m/--scen-k)
    parser.add_argument("--charge-m",    type=int, default=None,
                        help="충전 probe 상위 m개 (yaml charge_probe_m 오버라이드, Step 8 전달)")
    parser.add_argument("--discharge-m", type=int, default=None,
                        help="방전 probe 상위 m개 (yaml discharge_probe_m 오버라이드, Step 8 전달)")
    parser.add_argument("--scen-k",      type=int, default=None,
                        help="시나리오별 scen HI 수 (yaml scen_k_count 오버라이드, Step 8 전달)")
    parser.add_argument("--lambda-l0-override", type=float, default=None, dest="lambda_l0_override",
                        help="loss.lambda_l0을 이 값으로 강제 고정(lambda_l0_auto/yaml 값 무시, "
                             "phase1_trainer_v2.py --lambda-l0-override 그대로 전달, Step 8 전용)")
    parser.add_argument("--regression-model", default=None, dest="regression_model",
                        choices=["mlp", "transformer", "i_transformer", "resnet_tab", "ft_transformer"],
                        help="Phase1 cap_head 종류(기본 미지정 시 phase1_trainer_v2.py 자체 기본값 "
                             "'mlp' 사용, Step 8/9 전달)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="재현성 시드 — 모델 초기화 torch/numpy/random RNG (Step 8 전달, 기본 42)")
    parser.add_argument("--split-seed",  type=int, default=None,
                        help="train/val/test 셀 분할 시드 (Step 6~8 전달, 기본 42)")
    args = parser.parse_args()

    # 단축 인자 → args.axis_config(JSON) 로 합침. 이후 기존 --axis-config 전달 로직이
    # 모든 하위 스텝(4~8)에 올바른 JSON을 넘긴다. subprocess는 shell 없이 인자를 그대로
    # 전달하므로 PowerShell 따옴표 벗김 문제가 발생하지 않는다.
    if (args.n1 is not None or args.n2 is not None or args.n_samples is not None
            or args.ref_lag is not None or args.noise_amp is not None
            or args.noise_mode is not None or args.noise_period_cycles is not None
            or args.min_pts is not None or args.n2_start is not None
            or args.n2_end is not None or args.n2_step is not None
            or args.calibration_period is not None or args.calibration_mode is not None
            or args.calibration_jitter is not None or args.offset_amp is not None):
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
        if args.offset_amp is not None: _quick["offset_amp"] = args.offset_amp
        args.axis_config = _json.dumps(_quick)

    to_step = args.to_step if args.to_step is not None else len(STEPS)

    if not (1 <= args.from_step <= len(STEPS)):
        parser.error(f"from_step 은 1~{len(STEPS)} 사이여야 합니다.")
    if not (1 <= to_step <= len(STEPS)):
        parser.error(f"--to-step 은 1~{len(STEPS)} 사이여야 합니다.")
    if args.from_step > to_step:
        parser.error("from_step 이 to_step 보다 클 수 없습니다.")

    selected = [s for s in STEPS if args.from_step <= s[0] <= to_step]

    # 시너지/커널 태그 — 둘 다 --p1-tag에서 자동 파생(미지정 시). 실제 파일 경로는
    # 100% 이 태그로만 결정되므로(_synergy_out_path/_kernel_out_paths), Step 6~7을 이번에
    # 안 돌려도 이전에 같은 태그로 만들어둔 결과물이 있으면 Step 8이 그대로 찾아 쓴다.
    synergy_tag = args.synergy_tag or f"{args.p1_tag}_groups"
    kernel_tag = args.kernel_tag or f"{args.p1_tag}_kernel"
    synergy_out = _synergy_out_path(synergy_tag)
    kernel_pkl_out, kernel_redundancy_out = _kernel_out_paths(kernel_tag)

    # 미리보기용 1회 해석(아래 print 요약에만 씀) — Step 6~7이 이번 실행에 포함돼 있으면
    # 이 시점엔 아직 파일이 없어 fallback 값이 찍힐 수 있다. 실제로 Step 8에 전달되는 값은
    # Step 8 블록에서 _resolve_kernel_paths()를 다시 불러 그 시점 기준으로 새로 해석한다
    # (2026-09-18 버그수정: 예전엔 이 1회 해석값을 Step 8에도 그대로 썼다가, Step 6~7을
    # 같은 실행에서 막 돌려 파일이 생겼는데도 Step 8이 옛 fallback 값을 계속 쓰는 문제가
    # 있었다 — 실제로 한 번 발생 확인함).
    resolved_kernel_pkl, resolved_combined_redundancy = _resolve_kernel_paths(
        args, kernel_pkl_out, kernel_redundancy_out, P1V4_KERNEL_FEATURES_PKL,
    )

    print("\n" + "="*60)
    print("  LFP SOH Prediction — 전체 파이프라인")
    print("="*60)
    print(f"  스텝 범위   : {args.from_step} → {to_step}")
    print(f"  병렬 워커   : {args.workers}  (데이터 스텝 전용)")
    if any(s[0] in (6, 7) for s in selected):
        print(f"  synergy-tag : {synergy_tag}  (Step 6 출력 -> {synergy_out.name})")
        print(f"  kernel-tag  : {kernel_tag}  (Step 7 출력 -> {kernel_pkl_out.name})")
        print(f"  max-group-size: {args.max_group_size}  (Step 6)")
    if any(s[0] == 8 for s in selected):
        print(f"  Phase1 설정 : {args.phase1_model_config}  (Step 8, phase1_trainer_v2.py)")
        print(f"  Phase1 tag  : {args.p1_tag}")
        print(f"  kernel-pkl  : {resolved_kernel_pkl or '(미사용)'}"
              f"{'  [자동: Step 6~7 결과]' if args.kernel_features_pkl is None and kernel_pkl_out.exists() else ''}")
        print(f"  combined-redundancy : {resolved_combined_redundancy or '(미사용)'}")
        print(f"  interaction : {args.interaction_json or '(미사용)'}")
        if args.specific_group_ids_json:
            print(f"  group-ids   : {args.specific_group_ids_json}  (v5 계보)")
        if args.shrinkage_gate:
            print(f"  shrinkage-gate: True  (lambda-shrink={args.lambda_shrink})")
        if args.l0_norm_constant is not None:
            print(f"  l0-norm-constant: {args.l0_norm_constant}")
        if args.scen_gate_direction_only:
            print(f"  scen-gate-direction-only: True")
        if args.scenario_onehot_input:
            print(f"  scenario-onehot-input: True")
        if args.warmstart_branch_epoch is not None:
            print(f"  warmstart-branch-epoch: {args.warmstart_branch_epoch}")
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
    if args.lambda_l0_override is not None:
        print(f"  lambda-l0   : {args.lambda_l0_override} (고정, auto/yaml 무시)")
    if args.force_extract:
        print(f"  force-extract: True  (Step4 캐시 무시)")
    print(f"  실행 스텝   :")
    for n, name, _, _, _ in selected:
        print(f"    Step {n}  {name}")
    print("="*60)

    total_t0 = time.time()
    failed: list[int] = []

    # Phase 1 run_dir 핸드오프 (Step 8 → Step 9)
    p1_run_dir: Path | None = None
    snapshot: set[Path] = set()
    run_src: str | None = None
    _split_seed = args.split_seed if args.split_seed is not None else 42

    for num, name, script, extra, use_workers in selected:
        step_extra = list(extra)

        # ── 축 정보 주입 (Step 4~8 — 시너지 그룹(6)/커널 HI(7) 생성도 학습(8)과 같은 "실제
        #    데이터가 뭔지"에 의존하므로 축 설정을 여기서부터 같이 받는다) ──────────────
        if num in (4, 5, 6, 7, 8):
            if args.seg_axis:
                step_extra += ["--seg-axis", args.seg_axis]
            if args.axis_config:
                step_extra += ["--axis-config", args.axis_config]

        # ── 강제 재추출 옵션 주입 (Step 4만 — 기본은 캐시 재사용, 2026-08-15) ──
        if num == 4 and args.force_extract:
            step_extra += ["--force"]

        # ── CV 제외 옵션 주입 (Step 4=추출만 — '_ccOnly' 경로 인지 필요).
        #    Step 6~9는 --exclude-cv 플래그 자체가 없어 제외 — 아래 Step 8 전용
        #    블록에서 경고만 출력한다. ──
        if num == 4 and args.exclude_cv:
            step_extra += ["--exclude-cv"]

        # ── shape filter 비활성화 옵션 주입 (Step 2=전처리 자체가 필터7 스킵,
        #    Step 4=추출 — '_noshape' 경로 인지 필요).
        #    Step 6~9는 --skip-shape 플래그가 없어 제외(아래 Step 8 전용 블록에서 경고) ──
        if num in (2, 4) and args.skip_shape:
            step_extra += ["--skip-shape"]

        # ── Step 6(HI 시너지 그룹 구성, build_synergy_groups.py) 전용 ────────────
        if num == 6:
            step_extra += ["--split-seed", str(_split_seed)]
            step_extra += ["--max-group-size", str(args.max_group_size)]
            step_extra += ["--redundancy-threshold", str(args.synergy_redundancy_threshold)]
            step_extra += ["--min-partial-corr", str(args.min_partial_corr)]
            step_extra += ["--prefilter-top-m", str(args.prefilter_top_m)]
            if args.global_dedup:
                step_extra += ["--global-dedup"]
            step_extra += ["--tag", synergy_tag]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]

        # ── Step 7(커널 HI 피처 생성, build_kernel_group_features.py) 전용 ───────
        if num == 7:
            _synergy_in = args.kernel_synergy_groups_json or str(synergy_out)
            if not Path(_synergy_in).exists():
                print(f"\n  [경고] Step 7 입력 시너지 그룹 파일이 없습니다: {_synergy_in} "
                      "(Step 6을 이번 선택 범위에 포함시키거나, --kernel-synergy-groups-json "
                      "으로 기존 파일을 지정하세요).")
            step_extra += ["--synergy-groups-json", _synergy_in]
            step_extra += ["--split-seed", str(_split_seed)]
            step_extra += ["--alpha", str(args.kernel_alpha)]
            if args.kernel_gamma is not None:
                step_extra += ["--gamma", str(args.kernel_gamma)]
            step_extra += ["--n-components", str(args.kernel_n_components)]
            step_extra += ["--redundancy-threshold", str(args.kernel_redundancy_threshold)]
            if args.kernel_max_features is not None:
                step_extra += ["--max-features", str(args.kernel_max_features)]
            if args.min_raw_partial_corr is not None:
                step_extra += ["--min-raw-partial-corr", str(args.min_raw_partial_corr)]
            step_extra += ["--tag", kernel_tag]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]

        # ── m/k/시드/아키텍처 오버라이드 주입 (Step 8=Phase1 학습) ────────────────
        if num == 8:
            if args.charge_m is not None:
                step_extra += ["--charge-m", str(args.charge_m)]
            if args.discharge_m is not None:
                step_extra += ["--discharge-m", str(args.discharge_m)]
            if args.scen_k is not None:
                step_extra += ["--scen-k", str(args.scen_k)]
            if args.lambda_l0_override is not None:
                step_extra += ["--lambda-l0-override", str(args.lambda_l0_override)]
            if args.regression_model is not None:
                step_extra += ["--regression-model", args.regression_model]
            # phase1_trainer_v2.py는 --seed/--split-seed가 required=True라 항상 값을
            # 넘겨야 한다 — 미지정 시 42로 채운다.
            _seed = args.seed if args.seed is not None else 42
            if args.seed is None or args.split_seed is None:
                print(f"\n  [안내] Step 8(phase1_trainer_v2.py)는 --seed/--split-seed가 "
                      f"필수 인자라 미지정 값을 기본 42로 채웁니다 "
                      f"(seed={_seed}, split-seed={_split_seed}).")
            step_extra += ["--seed", str(_seed), "--split-seed", str(_split_seed)]

            step_extra += ["--tag", args.p1_tag]
            # 여기서 다시 해석한다(맨 위 미리보기 값을 그대로 쓰지 않음) — Step 6~7을
            # 이번 실행에 포함시켰다면 지금쯤 파일이 실제로 생겨있어야 정상이다.
            _kernel_pkl_now, _combined_redundancy_now = _resolve_kernel_paths(
                args, kernel_pkl_out, kernel_redundancy_out, P1V4_KERNEL_FEATURES_PKL,
            )
            if _kernel_pkl_now != resolved_kernel_pkl or _combined_redundancy_now != resolved_combined_redundancy:
                print(f"\n  [안내] Step 6~7 결과가 방금 반영됨 — kernel-pkl: {_kernel_pkl_now}"
                      f"{f', combined-redundancy: {_combined_redundancy_now}' if _combined_redundancy_now else ''}")
            resolved_kernel_pkl, resolved_combined_redundancy = _kernel_pkl_now, _combined_redundancy_now
            if resolved_kernel_pkl:
                step_extra += ["--kernel-features-pkl", resolved_kernel_pkl]
            if resolved_combined_redundancy:
                step_extra += ["--combined-redundancy-json", resolved_combined_redundancy]
            if args.interaction_json:
                step_extra += ["--interaction-json", args.interaction_json]
            if args.specific_group_ids_json:
                step_extra += ["--specific-group-ids-json", args.specific_group_ids_json]
            if args.train_cycle_frac is not None:
                step_extra += ["--train-cycle-frac", str(args.train_cycle_frac)]
            if args.beta_min is not None:
                step_extra += ["--beta-min", str(args.beta_min)]
            if args.p1_max_epochs is not None:
                step_extra += ["--max-epochs", str(args.p1_max_epochs)]
            if args.p1_patience is not None:
                step_extra += ["--patience", str(args.p1_patience)]
            if args.p1_batch_size is not None:
                step_extra += ["--batch-size", str(args.p1_batch_size)]
            if args.l0_warmup_epochs_override is not None:
                step_extra += ["--l0-warmup-epochs-override", str(args.l0_warmup_epochs_override)]
            if args.shrinkage_gate:
                step_extra += ["--shrinkage-gate"]
                if args.lambda_shrink:
                    step_extra += ["--lambda-shrink", str(args.lambda_shrink)]
            if args.l0_norm_constant is not None:
                step_extra += ["--l0-norm-constant", str(args.l0_norm_constant)]
            if args.scen_gate_direction_only:
                step_extra += ["--scen-gate-direction-only"]
            if args.scenario_onehot_input:
                step_extra += ["--scenario-onehot-input"]
            if args.warmstart_branch_epoch is not None:
                step_extra += ["--warmstart-branch-epoch", str(args.warmstart_branch_epoch)]
            if args.p1_device is not None:
                step_extra += ["--device", args.p1_device]

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
                      "--data-dir/--seg-data-dir을 안 줬습니다 — Step 8이 이 축 설정을 반영한 "
                      "scenario_spec.json은 만들면서, 실제 pkl 데이터는 phase1_trainer_v2.py의 "
                      "기본 캐논 경로에서 그대로 읽어버립니다(spec-데이터 불일치, 조용히 틀린 "
                      "결과). Step 4로 미리 추출한 경로를 --data-dir/--seg-data-dir로 명시하세요.")

            if args.exclude_cv or args.skip_shape:
                print("\n  [경고] phase1_trainer_v2.py는 --exclude-cv/--skip-shape 옵션이 "
                      "없습니다(train_scr.py 전용 플래그) — Step 8에는 전달하지 않습니다. "
                      "해당 변형 데이터로 Phase 1을 학습하려면 --data-dir/--seg-data-dir을 "
                      "phase1_trainer_v2.py에 직접 지정하는 별도 실행이 필요합니다.")
            if args.include_stat_leak:
                print("\n  [안내] --include-stat-leak 지정 — SOH_EXCLUDE_STAT_LEAK을 설정하지 "
                      "않습니다(N_HI=66, stat_q_abs/stat_energy_seg 포함). --kernel-features-pkl/"
                      "--interaction-json이 66-HI 기준 파일이 아니면 shape 불일치로 실패합니다.")
            elif os.environ.get("SOH_EXCLUDE_STAT_LEAK") != "1":
                print("\n  [안내] SOH_EXCLUDE_STAT_LEAK=1 을 Step 6~9 하위 프로세스 환경에 "
                      "자동 설정합니다(v0~v5 체크포인트 계보는 전부 N_HI=64 기준 — 이 값이 "
                      "없으면 66으로 계산돼 shape 불일치가 납니다). 66으로 돌리려면 "
                      "--include-stat-leak을 지정하세요.")

            # Phase 1 학습 전 스냅샷 (phase1_trainer_v2.py는 p1v2_runs/ 전용 디렉터리 사용)
            snapshot = _snapshot_p1v2_run_dirs()

        # ── Step 9 전용: 평가할 run_dir + v4/v5 재구성에 필요한 인자 주입 ──────
        if num == 9:
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
            if args.regression_model is not None:
                step_extra += ["--regression-model", args.regression_model]
            if args.rep_cells:
                step_extra += ["--rep-cells", *args.rep_cells]
            if args.export_for_visualize:
                step_extra += ["--export-for-visualize"]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]
            if args.p1_device is not None:
                step_extra += ["--device", args.p1_device]

        # v0~v5 체크포인트 계보는 N_HI=64(SOH_EXCLUDE_STAT_LEAK=1) 기준으로 통일돼 있어야
        # 하므로, 시너지 그룹(6)/커널 HI(7)/학습(8)/평가(9) 전부 동일 값을 준다(전부
        # build_datasets를 거쳐 N_HI가 그 값에 따라 64/66으로 갈리는 스크립트들).
        # --include-stat-leak을 주면 이 자동주입 자체를 건너뛰어 N_HI=66으로 계산되게 한다
        # (2026-09-06, docs/260903_RESULTS.md §5-3의 "실질적으로 유일한 코드 수정").
        _config_flag = "--model-config" if num in (6, 7, 8) else None
        _step_model_config = args.phase1_model_config if num in (6, 7, 8) else None
        if num in (6, 7, 8, 9) and not args.include_stat_leak:
            _extra_env = {"SOH_EXCLUDE_STAT_LEAK": "1"}
        else:
            _extra_env = None
        ok = run_step(num, name, script, step_extra, use_workers, args.workers,
                       _step_model_config, config_flag=_config_flag, extra_env=_extra_env)

        # ── Phase 1 후: 신규 run dir 기록 (phase1_trainer_v2.py → p1v2_runs/) ──────
        if num == 8:
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

        # ── Step 9 완료 후: 시나리오별 raw/kernel HI 게이트 선택 매트릭스 자동 생성
        # (2026-09-08) — plot_hi_selection_matrix.py는 파이프라인 번호가 없는 부가
        # 스텝이라 run_step()의 --workers/--model-config 규약과 안 맞아 여기서 직접
        # subprocess로 호출한다. 실패해도(예: gates/regression_HIs.json 없는 구버전
        # run) 파이프라인 전체를 막지 않는다 — 순수 시각화 부산물이라 결과 자체와는
        # 무관.
        if num == 9 and ok and run_src:
            plot_script = ROOT / "5_model" / "experiments" / "phase1_lab" / "plot_hi_selection_matrix.py"
            print(f"\n  → HI 선택 매트릭스 플랏 생성: {run_src}")
            subprocess.run(
                [sys.executable, str(plot_script), "--run-dir", str(run_src)],
                cwd=str(ROOT),
            )

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    if failed:
        print(f"  완료 (실패 스텝: {failed})  총 {_fmt_time(total_elapsed)}")
    else:
        print(f"  전체 완료  총 {_fmt_time(total_elapsed)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
