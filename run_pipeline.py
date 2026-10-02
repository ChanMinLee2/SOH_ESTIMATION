"""
run_pipeline.py

LFP SOH Prediction 전체 파이프라인 실행기.
데이터 전처리(Step 1~4)부터 모델 학습/평가(Step 5~9)까지 지원.

2026-09-21 리팩토링: CLI 파라미터를 "자주 바꾸는 것"과 "거의 안 바꾸는 것"으로 나눠
`parameters.py`로 옮겼다. `--include-stat-leak`/`--exclude-dqdv-leak` 두 불리언
플래그는 `--n-hi {63,64,66}` 단일 선택으로 통합했다.

2026-09-29: 이 오케스트레이터 자신의 CLI는 스텝 선택(위치 인자 `from_step`,
`--to-step`)만 남기고 나머지 전부(ACTIVE_*/FIXED_* 둘 다) 제거했다 — 이 파일은
"실행 파라미터를 커맨드라인으로 받는 곳"이 아니라 "parameters.py 값을 각 스텝에
전달하는 오케스트레이터"이기만 하면 되는데, 그동안 ACTIVE_*는 CLI로도 계속
오버라이드 가능하게 남겨둬서 이 파일 자체가 parameters.py 말고 또 하나의 파라미터
입력 경로가 되어 있었다. 실험값을 바꾸려면 이제 parameters.py를 직접 고칠 것 — 한
번만 다르게 돌려보고 싶으면(예: seed만 다르게) 이 스크립트를 거치지 말고 해당
스텝 스크립트(예: `8_train/train.py --seed 123 ...`)를 직접 실행하면 된다(그쪽은
자기 CLI를 그대로 유지 — 이 정리는 run_pipeline.py 한 파일에만 적용).

2026-08-15: Step 4(HI 추출)가 예전엔 항상 `--force`로 캐시를 무시하고 재추출했다
(코드/파라미터를 바꾸고 전체 파이프라인을 처음부터 돌릴 때 낡은 캐시를 실수로 쓰는 걸
막기 위함). 하지만 `python run_pipeline.py 4 --to-step 4 ...`처럼 캐시만 미리
만들어두려는 실행에서도 매번 강제 재추출이 되는 게 비효율적이라, 기본값을
"캐시 있으면 재사용"으로 바꾸고 강제 재추출은 `--force-extract`로 명시할 때만
하도록 뒤집었다 — `hi_correlation.py` 직접 실행과 동일한 기본 동작이 됐다.

2026-09-03: SCR Phase 1 학습을 `train.py`(v0~v5 게이트 안정화 계보)로 교체하고,
기본 레시피를 v4로 맞췄다. 구 시나리오 분류기 스텝과 구 SCR Phase 2 스텝은
파이프라인에서 완전히 제거했다 — train.py가 probe게이트+시나리오게이트+cap_head를
전부 포함한 단일 통합 모델을 한 번에 학습한다.

2026-09-18: synergy.py/kernel.py/interaction.py(상호작용 검정→시너지 그룹→커널 HI
생성)를 학습 앞단 스텝으로 신규 편입 — 파이프라인 밖에서 손으로 순서대로 돌려야 했던
산출물 체인을 한 번에 같은 axis-config/data-dir/split-seed로 연결한다.

2026-09-24: `5_model/`(코드 없이 과거 run 아카이브만 남아있던 폴더)을
`legacy_results/`로 이름 변경하면서 생긴 번호 공백을 없애려고, HI 세그먼트
시각화(`hi_segment_viz.py`)를 HI 상관 분석과 같은 Step 4로 합치고(둘 다
`4_hi_analysis/` 소속 — 이제 스텝 하나에 스크립트 두 개), 그 뒤 상호작용 검정→시너지
그룹→커널 HI→학습→평가를 Step 5~9로 한 칸씩 당겼다(구 6~10 → 신 5~9). 폴더 이름도
`5_interaction/`~`9_eval/`로 동시에 개명해 "폴더 번호 = 스텝 번호" 원칙을 유지한다.

사용(실험값을 바꾸려면 parameters.py를 먼저 수정):
  python run_pipeline.py                          # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2                        # Step 2부터 재실행
  python run_pipeline.py 8                        # 학습+평가만 (Step 8~9, v4 기본 — Step 5~7 산출물 재사용)
  python run_pipeline.py 8 --to-step 8            # 학습만(평가 제외)
  python run_pipeline.py 9                        # 평가만(직전 Phase 1 run 자동 탐색)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import parameters as P

# Windows 콘솔이 cp949일 때(특히 파이프/리다이렉트로 stdout이 콘솔이 아니게 되는 경우,
# 예: `| Tee-Object -FilePath ...`) em-dash 등 특수문자 print가 UnicodeEncodeError로
# 죽는 문제 방지(train.py/lambda_sweep.py와 동일 패턴).
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent

# train.py는 "{MMDD_HHMM}_p1v2_{tag}_seed{seed}" 형식으로 저장한다.
# 2026-09-23: 5_model/ 전체를 run_pipeline.py 스텝 순서에 맞춰 6_interaction/~10_eval/ +
# model_lib/(공유 인프라)로 재분배했다 — 새 run은 전부 model_lib/results/ 밑에 쌓인다.
# 기존 run 데이터(noscen/scen/HI63/64/66 등, 대용량)는 legacy_results/experiments/
# phase1_lab/results/에 그대로 남겨뒀다(재배치 리스크 회피 — 2026-09-21 결정과 동일
# 원칙). 2026-09-24: 그 폴더 자체를 5_model/ → legacy_results/로 개명(더는 어떤 스텝
# 번호와도 대응하지 않는 순수 아카이브임을 이름으로 명확히 함).
P1V2_RUNS_DIR = ROOT / "model_lib" / "results" / "p1v2_runs"
LEGACY_P1V2_RUNS_DIR = ROOT / "legacy_results" / "experiments" / "phase1_lab" / "results" / "p1v2_runs"

# v4의 실제 학습 레시피(docs/260827_RESULTS.md "v4 정식 학습 결과" 절 그대로) — 다른
# 버전(v0/v2/v3)으로 돌리고 싶으면 parameters.py: ACTIVE_KERNEL_FEATURES_PKL/
# ACTIVE_INTERACTION_JSON을 덮어쓰면 된다. Step 5~7(상호작용 검정/시너지 그룹/커널 HI
# 재생성)을 선택 범위에서 빼면(예: `python run_pipeline.py 8`) 이 고정 경로들이 그대로
# Step 8의 기본 interaction-json/kernel-features-pkl로 쓰인다. 2026-10-02: train.py
# 자신도 CLI 제거로 이 fallback 로직을 똑같이 알아야 해서 parameters.py:
# FIXED_LEGACY_V4_KERNEL_FEATURES_PKL/FIXED_LEGACY_V4_INTERACTION_JSON으로 이전(단일
# 소스화) — 여기선 별칭만 유지.
P1V4_KERNEL_FEATURES_PKL = P.FIXED_LEGACY_V4_KERNEL_FEATURES_PKL
P1V4_INTERACTION_JSON = P.FIXED_LEGACY_V4_INTERACTION_JSON

# (번호, 이름, 스크립트 경로, 기본 추가 인자, --workers 지원 여부)
# 2026-09-24: legacy_results/ 개명으로 생긴 번호 공백을 없애려고 HI 세그먼트 시각화를
# HI 상관 분석과 같은 Step 4로 합치고(스텝 하나에 스크립트 두 개 — 아래 for 루프의
# num==4 분기가 두 엔트리 모두에 적용됨), 그 뒤 스텝들을 5~9로 한 칸씩 당겼다. 따라서
# STEPS의 "번호"는 더는 1..len(STEPS)와 일치하지 않는다 — 총 스텝 수는 N_STEPS(아래,
# =max 번호)를 써야 한다.
STEPS = [
    (1, "데이터 변환",             "1_convert/convert_unified.py",    ["--dataset", P.ACTIVE_DATASET], True),
    (2, "이상 사이클 제거",        "2_preprocess/preprocess.py",       [],                   True),
    (3, "무결성 검사",             "3_integrity/check_integrity.py",   [],                   True),
    # hi_correlation.py는 2026-09-29부로 CLI를 전혀 안 받는다(parameters.py 직접
    # 참조) — use_workers=False로 --workers도 안 붙인다. 같은 날 hi_segment_viz.py
    # (세그먼트별 HI 추이/오버레이 플롯, 예전엔 별도 Step4 엔트리로 자체 CLI를 갖고
    # 독립적으로 load_or_extract를 다시 호출했었음)를 4_hi_analysis/plot.py로 합치고
    # hi_correlation.py::main()이 끝에 자동으로 호출하도록 바꿔서, Step4는 이제
    # 엔트리 하나로 추출+상관분석+전체 플롯을 한 프로세스 안에서 전부 처리한다.
    (4, "HI 상관 분석",            "4_hi_analysis/hi_correlation.py",  [],                   False),
    (5, "HI-시나리오 상호작용 검정", "5_interaction/interaction.py", [], False),
    (6, "HI 시너지 그룹 구성",     "6_synergy/synergy.py",        [], False),
    (7, "커널 HI 피처 생성",       "7_kernel/kernel.py", [], False),
    (8, "SCR Phase 1 학습(v4)",    "8_train/train.py",     [], False),
    (9, "Phase 1 평가",           "9_eval/test.py", [], False),
]
N_STEPS = max(s[0] for s in STEPS)  # len(STEPS)=10이지만 Step4가 두 엔트리를 공유해 실제 최대 번호는 9


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m}분 {s}초" if m else f"{s}초"


def _latest_p1v2_run_dir() -> Path | None:
    """p1v2_runs/ 중 가장 최근에 수정된 디렉터리 — Step 9을 --run-dir 없이 단독
    실행했는데 run_dir(p1-tag+seed 폴더)에 체크포인트가 없을 때의 최후 fallback.
    새 위치(P1V2_RUNS_DIR)와 재배치 이전 기존 run이 남아있는 구 위치
    (LEGACY_P1V2_RUNS_DIR) 둘 다 뒤져서 더 최근 것을 고른다(2026-09-23 재배치 직후
    당분간은 구 위치에 더 최근 run이 있을 수 있음)."""
    all_dirs = [
        d for base in (P1V2_RUNS_DIR, LEGACY_P1V2_RUNS_DIR) if base.exists()
        for d in base.iterdir() if d.is_dir()
    ]
    if not all_dirs:
        return None
    return max(all_dirs, key=lambda d: d.stat().st_mtime)


def _run_dir_for(run_ts: str, p1_tag: str, seed: int) -> Path:
    return P1V2_RUNS_DIR / f"{run_ts}_p1v2_{p1_tag}_seed{seed}"


def _interaction_out_path(run_dir: Path, tag: str) -> Path:
    return run_dir / f"hi_scenario_interaction_{tag}.json"


def _resolve_interaction_path(
    interaction_json: str | None, interaction_out: Path, default_json: str,
) -> str | None:
    """Step 8(학습)/9(평가)용 interaction-json 최종 해석 — kernel-features-pkl과 동일한
    3단 우선순위: 1) parameters.py: ACTIVE_INTERACTION_JSON이 명시적으로 주어지면(빈
    문자열 포함) 그 값 그대로(빈 문자열이면 아예 전달 안 함, v0/v2/v3 재현용) 2) Step 5
    (자동 경로)을 이번 실행에서 방금 만들었거나 이전에 만들어둔 파일이 있으면 그걸
    3) 그것도 없으면 default_json(v4 정식 고정 경로)로 최종 fallback."""
    if interaction_json is not None:
        return interaction_json or None
    if interaction_out.exists():
        return str(interaction_out)
    return default_json


def _synergy_out_path(run_dir: Path, tag: str) -> Path:
    return run_dir / f"synergy_groups_{tag}.json"


def _kernel_out_paths(run_dir: Path, tag: str) -> tuple[Path, Path]:
    """(kernel pkl, combined_redundancy json) 경로."""
    base = run_dir / f"kernel_group_features_{tag}"
    return Path(f"{base}.pkl"), Path(f"{base}_combined_redundancy.json")


def _resolve_kernel_paths(
    kernel_features_pkl: str | None, combined_redundancy_json: str | None,
    kernel_pkl_out: Path, kernel_redundancy_out: Path, default_pkl: str,
) -> tuple[str, str | None]:
    """Step 8용 kernel-features-pkl/combined-redundancy-json 최종 해석 — parameters.py
    값이 명시적으로 주어지면 최우선, 없으면 자동 경로가 실존하면 그걸, 그것도 없으면
    default_pkl(v4 정식 고정 경로)로 최종 fallback한다. 파일 존재 여부를 매번 새로
    검사한다(Step 6~7을 같은 실행 안에서 막 돌렸을 때도 정확히 잡히도록)."""
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
    extra_env: dict | None = None,
) -> bool:
    cmd = [sys.executable, str(ROOT / script)] + extra_args
    if use_workers:
        cmd += ["--workers", str(workers)]

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
    # 2026-09-29: 이 오케스트레이터의 CLI는 스텝 선택(아래 두 인자)만 남긴다 —
    # 그 외 모든 실행 파라미터는 parameters.py: ACTIVE_*/FIXED_*에서만 읽는다
    # (모듈 docstring 2026-09-29 항목 참고). 값을 바꾸려면 parameters.py를 고칠 것.
    parser = argparse.ArgumentParser(
        description="LFP SOH 파이프라인 실행기 (데이터 전처리 → Phase 1 통합 학습/평가)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="스텝 목록:\n" + "\n".join(
            f"  {n}  {name}" for n, name, _, _, _ in STEPS
        ),
    )
    parser.add_argument(
        "from_step", nargs="?", type=int, default=P.ACTIVE_FROM_STEP, metavar="FROM_STEP",
        help=f"시작 스텝 번호 (기본: {P.ACTIVE_FROM_STEP}, 범위: 1~{N_STEPS})",
    )
    parser.add_argument(
        "--to-step", type=int, default=None, metavar="TO_STEP",
        help=f"종료 스텝 번호 포함 (미지정 시 끝까지, 범위: 1~{N_STEPS})",
    )
    args = parser.parse_args()

    # parameters.py에서 그대로 읽는 실행 파라미터 — 전부 CLI로 오버라이드 불가.
    # 한 번만 다르게 돌려보고 싶으면 해당 스텝 스크립트(예: 8_train/train.py)를
    # 직접 실행할 것 — 그쪽은 자기 CLI를 그대로 유지한다.
    workers = min(P.ACTIVE_WORKERS, os.cpu_count() or 1)
    force_extract = P.ACTIVE_FORCE_EXTRACT
    kernel_features_pkl = P.ACTIVE_KERNEL_FEATURES_PKL
    interaction_json = P.ACTIVE_INTERACTION_JSON
    combined_redundancy_json = P.ACTIVE_COMBINED_REDUNDANCY_JSON
    max_group_size = P.ACTIVE_MAX_GROUP_SIZE
    synergy_redundancy_threshold = P.ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD
    hi_cost_weighted_l0 = P.ACTIVE_HI_COST_WEIGHTED_L0
    n_hi = P.ACTIVE_N_HI
    p1_tag = P.ACTIVE_P1_TAG
    axis_config = json.dumps(P.ACTIVE_AXIS_CONFIG)
    lambda_l0_override = P.ACTIVE_LAMBDA_L0_OVERRIDE
    seed = P.ACTIVE_SEED

    to_step = args.to_step if args.to_step is not None else N_STEPS

    if not (1 <= args.from_step <= N_STEPS):
        parser.error(f"from_step 은 1~{N_STEPS} 사이여야 합니다.")
    if not (1 <= to_step <= N_STEPS):
        parser.error(f"--to-step 은 1~{N_STEPS} 사이여야 합니다.")
    if args.from_step > to_step:
        parser.error("from_step 이 to_step 보다 클 수 없습니다.")

    selected = [s for s in STEPS if args.from_step <= s[0] <= to_step]

    # Step 5~9이 전부 공유하는 실험 폴더 — seed는 원래 Step 8(학습) 전용 값이지만,
    # 폴더명에 필요해서 여기서 한 번만 해석한다(train.py는 --seed/--split-seed가
    # 필수라 미지정 시 42로 채우는 것과 동일 규칙). run_ts도 여기서 한 번만 찍어서 Step 5~8
    # 내내 같은 타임스탬프를 쓴다.
    _seed = seed if seed is not None else P.FIXED_DEFAULT_SEED
    run_ts = datetime.now().strftime("%m%d_%H%M")
    run_dir = _run_dir_for(run_ts, p1_tag, _seed)
    if any(s[0] in (5, 6, 7, 8, 9) for s in selected):
        run_dir.mkdir(parents=True, exist_ok=True)

    # 상호작용/시너지/커널 태그 — 전부 p1_tag에서 자동 파생(FIXED_INTERACTION_TAG 등이
    # None이므로 항상 이 경로). 실제 파일 경로는 100% 이 태그 + run_dir로 결정되므로,
    # Step 5~7을 이번에 안 돌려도 이전에 같은 p1-tag+seed로 만들어둔 결과물이 run_dir에
    # 있으면 Step 8가 그대로 찾아 쓴다.
    interaction_tag = P.FIXED_INTERACTION_TAG or f"{p1_tag}_interaction"
    synergy_tag = P.FIXED_SYNERGY_TAG or f"{p1_tag}_groups"
    kernel_tag = P.FIXED_KERNEL_TAG or f"{p1_tag}_kernel"
    interaction_out = _interaction_out_path(run_dir, interaction_tag)
    synergy_out = _synergy_out_path(run_dir, synergy_tag)
    kernel_pkl_out, kernel_redundancy_out = _kernel_out_paths(run_dir, kernel_tag)

    # 미리보기용 1회 해석(아래 print 요약에만 씀) — 실제로 Step 8에 전달되는 값은 Step 8
    # 블록에서 다시 해석한다(Step 5~7을 같은 실행에서 막 돌려 파일이 생긴 경우를 반영하기 위함).
    resolved_kernel_pkl, resolved_combined_redundancy = _resolve_kernel_paths(
        kernel_features_pkl, combined_redundancy_json, kernel_pkl_out, kernel_redundancy_out,
        P1V4_KERNEL_FEATURES_PKL,
    )
    resolved_interaction = _resolve_interaction_path(
        interaction_json, interaction_out, P1V4_INTERACTION_JSON,
    )

    print("\n" + "="*60)
    print("  LFP SOH Prediction — 전체 파이프라인")
    print("="*60)
    print(f"  스텝 범위   : {args.from_step} → {to_step}")
    print(f"  병렬 워커   : {workers}  (데이터 스텝 전용)")
    if any(s[0] in (5, 6, 7, 8, 9) for s in selected):
        print(f"  실험 폴더   : {run_dir}")
    if any(s[0] == 5 for s in selected):
        print(f"  interaction-tag: {interaction_tag}  (Step 5 출력 -> {interaction_out.name})")
    if any(s[0] in (6, 7) for s in selected):
        print(f"  synergy-tag : {synergy_tag}  (Step 6 출력 -> {synergy_out.name})")
        print(f"  kernel-tag  : {kernel_tag}  (Step 7 출력 -> {kernel_pkl_out.name})")
        print(f"  max-group-size: {max_group_size}  (Step 6)")
    if any(s[0] == 8 for s in selected):
        print(f"  Phase1 설정 : parameters.py: P1_MODEL_CONFIG  (Step 8, train.py)")
        print(f"  Phase1 tag  : {p1_tag}")
        print(f"  n-hi        : {n_hi}")
        print(f"  kernel-pkl  : {resolved_kernel_pkl or '(미사용)'}"
              f"{'  [자동: Step 6~7 결과]' if kernel_features_pkl is None and kernel_pkl_out.exists() else ''}")
        print(f"  combined-redundancy : {resolved_combined_redundancy or '(미사용)'}")
        print(f"  interaction : {resolved_interaction or '(미사용)'}"
              f"{'  [자동: Step 5 결과]' if interaction_json is None and interaction_out.exists() else ''}")
        print(f"  hi-cost-weighted-l0: {hi_cost_weighted_l0}")
        if lambda_l0_override is not None:
            print(f"  lambda-l0   : {lambda_l0_override} (고정)")
    if axis_config:
        print(f"  axis-config : {axis_config}")
    if force_extract:
        print(f"  force-extract: True  (Step4 캐시 무시)")
    print(f"  실행 스텝   :")
    for n, name, _, _, _ in selected:
        print(f"    Step {n}  {name}")
    print("="*60)

    total_t0 = time.time()
    failed: list[int] = []

    # Phase 1 run_dir 핸드오프 (Step 8 → Step 9) — run_dir이 위에서 이미 확정돼 있으므로
    # 스냅샷-diff 없이 바로 쓴다. Step 8가 이번 실행에 없으면(예: `run_pipeline.py 9`만
    # 단독 실행) run_dir에 체크포인트가 실제로 있는지 확인해서, 없으면 최후 fallback한다.
    p1_run_dir: Path | None = run_dir if any(s[0] == 8 for s in selected) else None
    if p1_run_dir is None and (run_dir / "checkpoints").exists():
        p1_run_dir = run_dir
    run_src: str | None = None

    for num, name, script, extra, use_workers in selected:
        step_extra = list(extra)

        # (2026-10-02: Step 4~8의 축 정보 CLI 주입 블록을 여기서 완전히 제거했다 —
        #    train.py(Step 8)가 마지막까지 --seg-axis/--axis-config를 CLI로 받던
        #    스텝이었는데 이번 정리로 parameters.py: FIXED_SEG_AXIS/ACTIVE_AXIS_CONFIG를
        #    직접 읽게 바뀌어서, 이제 축 정보를 CLI로 받는 파이프라인 스텝이 하나도
        #    없다. axis_config 변수 자체는 위 요약 print에서만 계속 쓰인다.)

        # ── Step 5(HI-시나리오 상호작용 검정, interaction.py) 전용 — 2026-09-30부로
        #    --out-dir(run_pipeline.py가 여러 스텝이 공유하는 실험 폴더를 계산해서
        #    넘겨주는 값, parameters.py로 복원 불가)만 남고 나머지(split-seed/alpha/
        #    min-effect-size/tag/data-dir/seg-data-dir)는 전부 interaction.py 자신이
        #    parameters.py에서 직접 읽는다 ──────────────────────────────────────
        if num == 5:
            step_extra += ["--out-dir", str(run_dir)]

        # ── Step 6(HI 시너지 그룹 구성, synergy.py) 전용 — 2026-10-01부로 --out-dir만
        #    남고 나머지(seg-axis/axis-config/split-seed/max-group-size/
        #    redundancy-threshold/min-partial-corr/prefilter-top-m/global-dedup/
        #    tag/data-dir/seg-data-dir)는 전부 synergy.py 자신이 parameters.py에서
        #    직접 읽는다(interaction.py 2026-09-30 정리와 동일 원칙) ──────────────
        if num == 6:
            step_extra += ["--out-dir", str(run_dir)]

        # ── Step 7(커널 HI 피처 생성, kernel.py) 전용 — 2026-10-02부로 --out-dir만
        #    남고 나머지(seg-axis/axis-config/data-dir/seg-data-dir/split-seed/
        #    synergy-groups-json/alpha/gamma/n-components/redundancy-threshold/
        #    max-features/min-raw-partial-corr/combined-redundancy-threshold/tag)는
        #    전부 kernel.py 자신이 parameters.py에서 직접 읽는다(synergy.py 2026-10-01
        #    정리와 동일 원칙) — synergy-groups-json은 kernel.py가 synergy.py와 동일한
        #    태그 파생 규칙으로 같은 run_dir에서 자동으로 찾으므로 경로를 안 넘겨도 된다.
        if num == 7:
            if not synergy_out.exists():
                print(f"\n  [경고] Step 7 입력 시너지 그룹 파일이 없습니다: {synergy_out} "
                      "(Step 6을 이번 선택 범위에 포함시키세요, 또는 "
                      "FIXED_KERNEL_SYNERGY_GROUPS_JSON으로 다른 경로를 지정하세요).")
            step_extra += ["--out-dir", str(run_dir)]

        # ── Step 8(SCR Phase 1 학습, train.py) 전용 — 2026-10-02부로 --output-dir만
        #    남고 나머지(seg-axis/axis-config/data-dir/seg-data-dir/charge-m/
        #    discharge-m/scen-k/seed/split-seed/train-cycle-frac/beta-min/device/
        #    max-epochs/patience/batch-size/tag/lambda-l0-override/
        #    l0-warmup-epochs-override/l0-norm-constant/hi-cost-weighted-l0/
        #    val-rmse-epsilon/kernel-features-pkl/combined-redundancy-json/
        #    interaction-json)는 전부 train.py 자신이 parameters.py에서 직접
        #    읽는다(synergy.py/kernel.py 정리와 동일 원칙) — kernel-features-pkl/
        #    combined-redundancy-json/interaction-json은 train.py가 kernel.py의
        #    synergy-groups-json 자동탐색과 동일한 3단 우선순위 패턴으로 같은
        #    --output-dir 안에서 직접 찾는다(train.py 모듈 docstring 참고) ──────
        if num == 8:
            step_extra += ["--output-dir", str(run_dir)]

            # Step 9(test.py)는 아직 --kernel-features-pkl/--combined-redundancy-json/
            # --interaction-json을 CLI로 받으므로, 그쪽에 넘겨줄 resolved_* 값은 여기서
            # 계속 갱신해둔다(train.py 자신은 이제 이 값을 CLI로 안 받고 독립적으로
            # 똑같이 재계산 — 맨 위 미리보기 값을 그대로 쓰지 않는 이유는 Step 5~7을
            # 이번 실행에 포함시켰다면 지금쯤 파일이 실제로 생겨있어야 정상이기 때문).
            _kernel_pkl_now, _combined_redundancy_now = _resolve_kernel_paths(
                kernel_features_pkl, combined_redundancy_json, kernel_pkl_out, kernel_redundancy_out,
                P1V4_KERNEL_FEATURES_PKL,
            )
            if _kernel_pkl_now != resolved_kernel_pkl or _combined_redundancy_now != resolved_combined_redundancy:
                print(f"\n  [안내] Step 6~7 결과가 방금 반영됨 — kernel-pkl: {_kernel_pkl_now}"
                      f"{f', combined-redundancy: {_combined_redundancy_now}' if _combined_redundancy_now else ''}")
            resolved_kernel_pkl, resolved_combined_redundancy = _kernel_pkl_now, _combined_redundancy_now
            _interaction_now = _resolve_interaction_path(
                interaction_json, interaction_out, P1V4_INTERACTION_JSON,
            )
            if _interaction_now != resolved_interaction:
                print(f"\n  [안내] Step 5 결과가 방금 반영됨 — interaction-json: {_interaction_now}")
            resolved_interaction = _interaction_now

            if n_hi != 66:
                print(f"\n  [안내] SOH_EXCLUDE_STAT_LEAK/SOH_EXCLUDE_DQDV_LEAK을 Step 5~9 "
                      f"하위 프로세스 환경에 명시 주입합니다(N_HI={n_hi}). N_HI가 64가 "
                      "아니면 kernel-features-pkl/interaction-json도 그 N_HI 기준으로 "
                      "새로 만든 파일이 아니면 shape 불일치로 실패합니다.")
            else:
                print(f"\n  [안내] N_HI=66(전부 포함) — SOH_EXCLUDE_STAT_LEAK=0/"
                      "SOH_EXCLUDE_DQDV_LEAK=0을 하위 프로세스 환경에 명시 주입합니다. "
                      "kernel-features-pkl/interaction-json이 66-HI 기준 파일이 아니면 "
                      "shape 불일치로 실패합니다.")

        # ── Step 9(평가, test.py) 전용 — 2026-10-02부로 --run-dir만 남고 나머지
        #    (checkpoint/interaction-json/kernel-features-pkl/
        #    combined-redundancy-json/rep-cells/data-dir/seg-data-dir/device)는
        #    전부 test.py 자신이 parameters.py에서 직접 읽는다(train.py 정리와
        #    동일 원칙) — interaction/kernel/combined-redundancy는 CLI로 안 받아도
        #    그 run 자신의 p1v2_summary.json에 train.py가 이미 기록해둔 값을 test.py가
        #    그대로 읽으므로(parameters.py: ACTIVE_*가 명시돼 있으면 그게 최우선)
        #    run_pipeline.py가 resolved_* 값을 넘겨줄 필요가 없어졌다 ──────────
        if num == 9:
            run_src = str(p1_run_dir) if p1_run_dir else None
            if run_src is None:
                latest = _latest_p1v2_run_dir()
                run_src = str(latest) if latest else None
            if run_src:
                step_extra += ["--run-dir", run_src]
                print(f"\n  → run-dir (평가 대상): {run_src}")
            else:
                print("\n  [경고] Phase 1 run 디렉터리를 찾을 수 없습니다. "
                      "--run-dir을 직접 지정하려면 9_eval/test.py를 따로 실행하세요.")

        # v0~v4 체크포인트 계보는 N_HI를 상호작용 검정(5)/시너지 그룹(6)/커널 HI(7)/
        # 학습(8)/평가(9) 전부 동일하게 맞춰야 한다(전부 build_datasets를 거쳐 N_HI가
        # 이 값에 따라 갈리는 스크립트들). run_step()의 env = os.environ.copy()가 호출
        # 셸의 기존 환경변수를 그대로 물려받으므로, 매번 두 값을 명시적으로 "0"/"1"로
        # 못박아 호출 셸 상태와 무관하게 만든다(2026-09-20 버그 수정 — 예전엔 "필요할
        # 때만 추가"라 호출 셸에 이미 SOH_EXCLUDE_STAT_LEAK=1이 켜져 있으면 --n-hi 66을
        # 줘도 못 지워서 조용히 무시됐음).
        # 2026-09-27: Step 5~7의 --model-config(프리셋 yaml) 요구사항 폐기 — interaction.py/
        # synergy.py/kernel.py 전부 train.py와 동일하게 parameters.py: P1_MODEL_CONFIG를
        # 직접 참조하도록 바뀌어서, 파이프라인 전 스텝이 yaml을 아예 거치지 않는다
        # (단일 소스 원칙 완성).
        _extra_env = None
        if num in (5, 6, 7, 8, 9):
            _exclude_stat_leak, _exclude_dqdv_leak = P.N_HI_TO_ENV[n_hi]
            _extra_env = {
                "SOH_EXCLUDE_STAT_LEAK": _exclude_stat_leak,
                "SOH_EXCLUDE_DQDV_LEAK": _exclude_dqdv_leak,
            }
        ok = run_step(num, name, script, step_extra, use_workers, workers,
                       extra_env=_extra_env)

        if num == 8:
            if ok:
                print(f"  → Phase 1 run dir: {p1_run_dir}")
            else:
                print(f"  [경고] Step 8 실패 — run dir({p1_run_dir})에 체크포인트가 없을 수 있습니다.")

        if not ok:
            failed.append(num)
            if not _ask_continue(num):
                print("  파이프라인 중단.")
                sys.exit(1)

        # ── Step 9 완료 후: 시나리오별 raw HI 게이트 선택 매트릭스 자동 생성 —
        # 순수 시각화 부산물이라 실패해도 파이프라인 전체를 막지 않는다. ──
        if num == 9 and ok and run_src:
            plot_script = ROOT / "9_eval" / "plot_hi_selection_matrix.py"
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
