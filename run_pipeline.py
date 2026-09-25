"""
run_pipeline.py

LFP SOH Prediction 전체 파이프라인 실행기.
데이터 전처리(Step 1~4)부터 모델 학습/평가(Step 5~9)까지 지원.

2026-09-21 리팩토링: CLI 파라미터를 "자주 바꾸는 것"과 "거의 안 바꾸는 것"으로 나눠
`parameters.py`로 옮겼다. 자주 바꾸는 파라미터(ACTIVE_*)는 여전히 CLI 플래그로 노출되고
기본값만 parameters.py를 참조한다. 거의 안 바꾸는 파라미터(FIXED_*)는 CLI 플래그 자체를
없애고 parameters.py 값을 그대로 하위 스크립트에 전달한다 — 바꾸려면 parameters.py를
직접 수정할 것. `--include-stat-leak`/`--exclude-dqdv-leak` 두 불리언 플래그는
`--n-hi {63,64,66}` 단일 선택으로 통합했다.

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

사용:
  python run_pipeline.py                          # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2                        # Step 2부터 재실행
  python run_pipeline.py 8                        # 학습+평가만 (Step 8~9, v4 기본 — Step 5~7 산출물 재사용)
  python run_pipeline.py 8 --to-step 8            # 학습만(평가 제외)
  python run_pipeline.py 9                        # 평가만(직전 Phase 1 run 자동 탐색)
  python run_pipeline.py 3 --workers 8
  python run_pipeline.py 8 --seed 0 --split-seed 0 --p1-tag p1v4_seed0
  python run_pipeline.py 4 --to-step 4 --force-extract --axis-config '{...}'  # 캐시 무시하고 강제 재추출
  # 상호작용 검정→시너지 그룹→커널 HI→학습→평가를 max-group-size=2로 전부 새로 만들어서 실행:
  python run_pipeline.py 5 --max-group-size 2 --p1-tag p1v4_scen_g2 --axis-config '{...}' --data-dir ... --seg-data-dir ...
  # HI63(q_abs/energy/dqdv_area 전부 제외)으로 돌리기:
  python run_pipeline.py 5 --n-hi 63 --p1-tag p1v4_scen_hi63 --axis-config '{...}' --data-dir ... --seg-data-dir ...
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
# 버전(v0/v2/v3)으로 돌리고 싶으면 --kernel-features-pkl/--interaction-json을 CLI로
# 덮어쓰면 된다. Step 5~7(상호작용 검정/시너지 그룹/커널 HI 재생성)을 선택 범위에서 빼면
# (예: `python run_pipeline.py 8`) 이 고정 경로들이 그대로 Step 8의 기본
# --interaction-json/--kernel-features-pkl로 쓰인다.
P1V4_KERNEL_FEATURES_PKL = ("legacy_results/experiments/phase1_lab/results/"
                             "kernel_group_features_k25_full_N2_kernel_v3.pkl")
P1V4_INTERACTION_JSON = ("legacy_results/experiments/phase1_lab/results/"
                          "hi_scenario_interaction_k25_full_N2.json")

# (번호, 이름, 스크립트 경로, 기본 추가 인자, --workers 지원 여부)
# 2026-09-24: legacy_results/ 개명으로 생긴 번호 공백을 없애려고 HI 세그먼트 시각화를
# HI 상관 분석과 같은 Step 4로 합치고(스텝 하나에 스크립트 두 개 — 아래 for 루프의
# num==4 분기가 두 엔트리 모두에 적용됨), 그 뒤 스텝들을 5~9로 한 칸씩 당겼다. 따라서
# STEPS의 "번호"는 더는 1..len(STEPS)와 일치하지 않는다 — 총 스텝 수는 N_STEPS(아래,
# =max 번호)를 써야 한다.
STEPS = [
    (1, "데이터 변환",             "1_convert/convert_unified.py",    ["--dataset", "all"], True),
    (2, "이상 사이클 제거",        "2_preprocess/preprocess.py",       [],                   True),
    (3, "무결성 검사",             "3_integrity/check_integrity.py",   [],                   True),
    (4, "HI 상관 분석",            "4_hi_analysis/hi_correlation.py",  [],                   True),
    (4, "HI 세그먼트 시각화",      "4_hi_analysis/hi_segment_viz.py",  [],                   True),
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


def _resolve_interaction_path(args, interaction_out: Path, default_json: str) -> str | None:
    """Step 8(학습)/9(평가)용 interaction-json 최종 해석 — kernel-features-pkl과 동일한
    3단 우선순위: 1) --interaction-json이 명시적으로 주어지면(빈 문자열 포함) 그 값 그대로
    (빈 문자열이면 아예 전달 안 함, v0/v2/v3 재현용) 2) Step 5(자동 경로)을 이번 실행에서
    방금 만들었거나 이전에 만들어둔 파일이 있으면 그걸 3) 그것도 없으면 default_json(v4
    정식 고정 경로)로 최종 fallback."""
    if args.interaction_json is not None:
        return args.interaction_json or None
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
    args, kernel_pkl_out: Path, kernel_redundancy_out: Path, default_pkl: str,
) -> tuple[str, str | None]:
    """Step 8용 kernel-features-pkl/combined-redundancy-json 최종 해석 — 명시적 CLI 값이
    최우선, 없으면 자동 경로가 실존하면 그걸, 그것도 없으면 default_pkl(v4 정식 고정 경로)로
    최종 fallback한다. 파일 존재 여부를 매번 새로 검사한다(Step 6~7을 같은 실행 안에서
    막 돌렸을 때도 정확히 잡히도록)."""
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
        "from_step", nargs="?", type=int, default=P.ACTIVE_FROM_STEP, metavar="FROM_STEP",
        help=f"시작 스텝 번호 (기본: {P.ACTIVE_FROM_STEP}, 범위: 1~{N_STEPS})",
    )
    parser.add_argument(
        "--to-step", type=int, default=None, metavar="TO_STEP",
        help=f"종료 스텝 번호 포함 (미지정 시 끝까지, 범위: 1~{N_STEPS})",
    )
    parser.add_argument(
        "--workers", type=int, default=min(P.ACTIVE_WORKERS, os.cpu_count() or 1),
        help=f"데이터 스텝(1~4)에 전달할 병렬 프로세스 수 (기본: {P.ACTIVE_WORKERS})",
    )
    parser.add_argument(
        "--force-extract", action="store_true", dest="force_extract",
        default=P.ACTIVE_FORCE_EXTRACT,
        help="Step 4(HI 추출) 캐시를 무시하고 강제 재추출. 기본은 캐시가 있으면 재사용. "
             "코드/파라미터를 바꾼 뒤나, random/random_grid처럼 캐시 파일명에 axis_config "
             "값이 안 들어가는 축의 파라미터만 바꿨을 때는 이 플래그를 꼭 같이 줘야 한다.",
    )
    parser.add_argument(
        "--kernel-features-pkl", default=P.ACTIVE_KERNEL_FEATURES_PKL,
        dest="kernel_features_pkl",
        help="Step 8 전용 — 커널 특징 pkl 경로. 미지정 시: Step 6~7을 이번에 돌렸거나(또는 "
             "이전에 돌려서) 자동 경로에 파일이 있으면 그걸 쓰고, 없으면 "
             f"v4 정식 kernel_v3 파일({P1V4_KERNEL_FEATURES_PKL})로 최종 fallback한다. "
             "빈 문자열('')을 명시하면 아예 전달하지 않음(v0 재현 등).",
    )
    parser.add_argument(
        "--interaction-json", default=P.ACTIVE_INTERACTION_JSON,
        dest="interaction_json",
        help="Step 8/9(학습/평가) 전용 — HI x 시나리오 상호작용 JSON 경로. 미지정 시: "
             "Step 5을 이번에 돌렸거나(또는 이전에 돌려서) 자동 경로에 파일이 있으면 그걸 "
             "쓰고, 없으면 v4 정식 고정 경로로 최종 fallback한다. 학습(8)과 평가(9) "
             "양쪽에 동일하게 전달된다. 빈 문자열('')을 명시하면 아예 전달하지 않음.",
    )
    parser.add_argument(
        "--combined-redundancy-json", default=P.ACTIVE_COMBINED_REDUNDANCY_JSON,
        dest="combined_redundancy_json",
        help="Step 8 전용 — kernel.py의 결합(raw+kernel) 다중공선성 "
             "배제 산출물(*_combined_redundancy.json, v4 요구사항2). 미지정이고 Step 6~7 "
             "결과물(자동 경로)이 존재하면 그걸 자동으로 쓴다.",
    )
    parser.add_argument(
        "--max-group-size", type=int, default=P.ACTIVE_MAX_GROUP_SIZE, dest="max_group_size",
        help=f"Step 6 전용 — 시너지 그룹 최대 크기(그룹당 raw HI 개수 상한, 기본 "
             f"{P.ACTIVE_MAX_GROUP_SIZE}). 줄이면(2~3) 그룹당 RBF 커널 입력 차원이 줄어, "
             "시나리오당 샘플이 적을 때(scen처럼 6분할) 커널 과적합 위험을 낮출 수 있다는 "
             "가설을 검증하는 용도.",
    )
    parser.add_argument(
        "--synergy-redundancy-threshold", type=float,
        default=P.ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD, dest="synergy_redundancy_threshold",
        help="Step 6 전용 — synergy.py --redundancy-threshold(그룹 내부 raw HI "
             f"다중공선성 배제 기준, 기본 {P.ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=P.ACTIVE_MAX_EPOCHS, dest="p1_max_epochs",
        help="Step 8 전용 — 학습 에폭 상한(기본 None=yaml training.epochs).",
    )
    parser.add_argument(
        "--patience", type=int, default=P.ACTIVE_PATIENCE, dest="p1_patience",
        help="Step 8 전용 — 조기종료 patience(기본 None=train.py 자체 기본 60).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=P.ACTIVE_BATCH_SIZE, dest="p1_batch_size",
        help="Step 8 전용 — 배치 크기 오버라이드(기본 None=yaml training.batch_size).",
    )
    parser.add_argument(
        "--hi-cost-weighted-l0", action="store_true", dest="hi_cost_weighted_l0",
        default=P.ACTIVE_HI_COST_WEIGHTED_L0,
        help="Step 8 전용 — L0 페널티에 HI 카테고리 비용(CATEGORY_COSTS)을 가중치로 곱한다. "
             "기본(미지정)은 균일 비용 1.0.",
    )
    parser.add_argument(
        "--n-hi", type=int, default=P.ACTIVE_N_HI, dest="n_hi", choices=P.N_HI_CHOICES,
        help=f"Step 5~9 전용 — raw HI 개수(기본 {P.ACTIVE_N_HI}). 63=stat_q_abs/"
             "stat_energy_seg/diff_dqdv_area 전부 제외, 64(기본)=q_abs/energy만 제외, "
             "66=전부 포함. SOH_EXCLUDE_STAT_LEAK/SOH_EXCLUDE_DQDV_LEAK 환경변수를 이 값에 "
             "맞춰 하위 프로세스에 자동 주입한다(docs/MODEL_FLOW.md §13 N_HI 토글 참고). "
             "64가 아닌 값으로 돌리려면 --kernel-features-pkl/--interaction-json도 그 N_HI "
             "기준으로 새로 만든 파일을 같이 지정해야 한다 — shape 불일치 방지를 위해 "
             "Step 5부터 다시 생성하는 걸 권장.",
    )
    parser.add_argument(
        "--p1-tag", default=P.ACTIVE_P1_TAG, dest="p1_tag",
        help=f"Step 8 train.py의 --tag (run 디렉터리 이름에 들어감, 기본: "
             f"{P.ACTIVE_P1_TAG}). 상호작용/시너지/커널 태그를 안 주면 이 값에서 자동 파생됨.",
    )
    parser.add_argument(
        "--rep-cells", nargs="+", default=P.ACTIVE_REP_CELLS, dest="rep_cells",
        help="Step 9 평가 시 용량곡선 비교 플랏을 그릴 셀 ID(들) (미지정 시 데이터셋별 5개 자동 선정)",
    )
    parser.add_argument(
        "--axis-config", default=json.dumps(P.ACTIVE_AXIS_CONFIG), metavar="JSON",
        help="축 파라미터 JSON (Step 4~8에 전달, 기본값은 parameters.py: ACTIVE_AXIS_CONFIG). "
             "부분 수정이 아니라 통째로 교체된다 — 일부만 바꾸고 싶어도 전체 딕셔너리를 "
             "다시 써서 넘길 것(예: '{\"n1\": 0.4, \"n2\": 0.2, \"n_samples\": 4}').",
    )
    parser.add_argument("--data-dir", default=P.ACTIVE_DATA_DIR, dest="p1_data_dir",
                        help="Step 5~8(상호작용 검정/시너지 그룹/커널 HI/train.py) 공통 "
                             "cycle pkl 경로 오버라이드. 이 스크립트들은 --axis-config만으로 데이터 "
                             "경로를 자동 계산하지 않고 항상 자기 자신의 기본 경로(정식 q_frac_ref "
                             "캐논 설정)로 fallback하므로, 캐논이 아닌 축 설정으로 돌리려면 "
                             "**반드시** 이 옵션과 --seg-data-dir을 함께 줘야 한다.")
    parser.add_argument("--seg-data-dir", default=P.ACTIVE_SEG_DATA_DIR, dest="p1_seg_data_dir",
                        help="Step 5~8 공통 seg pkl 경로 오버라이드. --data-dir와 항상 같이 줄 것.")
    parser.add_argument("--lambda-l0-override", type=float, default=P.ACTIVE_LAMBDA_L0_OVERRIDE,
                        dest="lambda_l0_override",
                        help=f"loss.lambda_l0을 이 값으로 강제 고정(기본 "
                             f"{P.ACTIVE_LAMBDA_L0_OVERRIDE}, train.py "
                             "--lambda-l0-override 그대로 전달, Step 8 전용). None으로 끄면 "
                             "yaml의 lambda_l0_auto 로직으로 되돌아간다(레거시, 비권장).")
    parser.add_argument("--regression-model", default=P.ACTIVE_REGRESSION_MODEL, dest="regression_model",
                        choices=["mlp", "transformer", "i_transformer", "resnet_tab", "ft_transformer"],
                        help="Phase1 cap_head 종류(기본 미지정 시 train.py 자체 기본값 "
                             "'mlp' 사용, Step 8/9 전달)")
    parser.add_argument("--seed",        type=int, default=P.ACTIVE_SEED,
                        help="재현성 시드 — 모델 초기화 torch/numpy/random RNG (Step 8 전달, 기본 42)")
    parser.add_argument("--split-seed",  type=int, default=P.ACTIVE_SPLIT_SEED,
                        help="train/val/test 셀 분할 시드 (Step 5~8 전달, 기본 42)")
    args = parser.parse_args()

    to_step = args.to_step if args.to_step is not None else N_STEPS

    if not (1 <= args.from_step <= N_STEPS):
        parser.error(f"from_step 은 1~{N_STEPS} 사이여야 합니다.")
    if not (1 <= to_step <= N_STEPS):
        parser.error(f"--to-step 은 1~{N_STEPS} 사이여야 합니다.")
    if args.from_step > to_step:
        parser.error("from_step 이 to_step 보다 클 수 없습니다.")

    selected = [s for s in STEPS if args.from_step <= s[0] <= to_step]

    # Step 5~9이 전부 공유하는 실험 폴더 — --seed는 원래 Step 8(학습) 전용 CLI였지만,
    # 폴더명에 필요해서 여기서 한 번만 해석한다(train.py는 --seed/--split-seed가
    # 필수라 미지정 시 42로 채우는 것과 동일 규칙). run_ts도 여기서 한 번만 찍어서 Step 5~8
    # 내내 같은 타임스탬프를 쓴다.
    _seed = args.seed if args.seed is not None else 42
    run_ts = datetime.now().strftime("%m%d_%H%M")
    run_dir = _run_dir_for(run_ts, args.p1_tag, _seed)
    if any(s[0] in (5, 6, 7, 8, 9) for s in selected):
        run_dir.mkdir(parents=True, exist_ok=True)

    # 상호작용/시너지/커널 태그 — 전부 --p1-tag에서 자동 파생(FIXED_INTERACTION_TAG 등이
    # None이므로 항상 이 경로). 실제 파일 경로는 100% 이 태그 + run_dir로 결정되므로,
    # Step 5~7을 이번에 안 돌려도 이전에 같은 p1-tag+seed로 만들어둔 결과물이 run_dir에
    # 있으면 Step 8가 그대로 찾아 쓴다.
    interaction_tag = P.FIXED_INTERACTION_TAG or f"{args.p1_tag}_interaction"
    synergy_tag = P.FIXED_SYNERGY_TAG or f"{args.p1_tag}_groups"
    kernel_tag = P.FIXED_KERNEL_TAG or f"{args.p1_tag}_kernel"
    interaction_out = _interaction_out_path(run_dir, interaction_tag)
    synergy_out = _synergy_out_path(run_dir, synergy_tag)
    kernel_pkl_out, kernel_redundancy_out = _kernel_out_paths(run_dir, kernel_tag)

    # 미리보기용 1회 해석(아래 print 요약에만 씀) — 실제로 Step 8에 전달되는 값은 Step 8
    # 블록에서 다시 해석한다(Step 5~7을 같은 실행에서 막 돌려 파일이 생긴 경우를 반영하기 위함).
    resolved_kernel_pkl, resolved_combined_redundancy = _resolve_kernel_paths(
        args, kernel_pkl_out, kernel_redundancy_out, P1V4_KERNEL_FEATURES_PKL,
    )
    resolved_interaction = _resolve_interaction_path(args, interaction_out, P1V4_INTERACTION_JSON)

    print("\n" + "="*60)
    print("  LFP SOH Prediction — 전체 파이프라인")
    print("="*60)
    print(f"  스텝 범위   : {args.from_step} → {to_step}")
    print(f"  병렬 워커   : {args.workers}  (데이터 스텝 전용)")
    if any(s[0] in (5, 6, 7, 8, 9) for s in selected):
        print(f"  실험 폴더   : {run_dir}")
    if any(s[0] == 5 for s in selected):
        print(f"  interaction-tag: {interaction_tag}  (Step 5 출력 -> {interaction_out.name})")
    if any(s[0] in (6, 7) for s in selected):
        print(f"  synergy-tag : {synergy_tag}  (Step 6 출력 -> {synergy_out.name})")
        print(f"  kernel-tag  : {kernel_tag}  (Step 7 출력 -> {kernel_pkl_out.name})")
        print(f"  max-group-size: {args.max_group_size}  (Step 6)")
    if any(s[0] == 8 for s in selected):
        print(f"  Phase1 설정 : {P.FIXED_PHASE1_MODEL_CONFIG}  (Step 8, train.py)")
        print(f"  Phase1 tag  : {args.p1_tag}")
        print(f"  n-hi        : {args.n_hi}")
        print(f"  kernel-pkl  : {resolved_kernel_pkl or '(미사용)'}"
              f"{'  [자동: Step 6~7 결과]' if args.kernel_features_pkl is None and kernel_pkl_out.exists() else ''}")
        print(f"  combined-redundancy : {resolved_combined_redundancy or '(미사용)'}")
        print(f"  interaction : {resolved_interaction or '(미사용)'}"
              f"{'  [자동: Step 5 결과]' if args.interaction_json is None and interaction_out.exists() else ''}")
        print(f"  hi-cost-weighted-l0: {args.hi_cost_weighted_l0}")
        if args.lambda_l0_override is not None:
            print(f"  lambda-l0   : {args.lambda_l0_override} (고정)")
    if args.axis_config:
        print(f"  axis-config : {args.axis_config}")
    if args.force_extract:
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
    _split_seed = args.split_seed if args.split_seed is not None else 42

    for num, name, script, extra, use_workers in selected:
        step_extra = list(extra)

        # ── 축 정보 주입 (Step 4~8 — 상호작용 검정(5)/시너지 그룹(6)/커널 HI(7) 생성도
        #    학습(8)과 같은 "실제 데이터가 뭔지"에 의존하므로 축 설정을 여기서부터
        #    같이 받는다) ─────────────────────────────────────────────────────
        if num in (4, 5, 6, 7, 8):
            step_extra += ["--seg-axis", P.FIXED_SEG_AXIS]
            if args.axis_config:
                step_extra += ["--axis-config", args.axis_config]

        # ── 강제 재추출 옵션 주입 (Step 4만 — 기본은 캐시 재사용, HI 상관 분석/HI 세그먼트
        #    시각화 둘 다 Step 4라 두 엔트리 모두에 적용됨) ──
        if num == 4 and args.force_extract:
            step_extra += ["--force"]

        # ── Step 5(HI-시나리오 상호작용 검정, interaction.py) 전용 ──
        if num == 5:
            step_extra += ["--split-seed", str(_split_seed)]
            step_extra += ["--alpha", str(P.FIXED_INTERACTION_ALPHA)]
            step_extra += ["--min-effect-size", str(P.FIXED_INTERACTION_MIN_EFFECT_SIZE)]
            step_extra += ["--tag", interaction_tag]
            step_extra += ["--out-dir", str(run_dir)]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]

        # ── Step 6(HI 시너지 그룹 구성, synergy.py) 전용 ────────────
        if num == 6:
            step_extra += ["--split-seed", str(_split_seed)]
            step_extra += ["--max-group-size", str(args.max_group_size)]
            step_extra += ["--redundancy-threshold", str(args.synergy_redundancy_threshold)]
            step_extra += ["--min-partial-corr", str(P.FIXED_MIN_PARTIAL_CORR)]
            step_extra += ["--prefilter-top-m", str(P.FIXED_PREFILTER_TOP_M)]
            if P.FIXED_GLOBAL_DEDUP:
                step_extra += ["--global-dedup"]
            step_extra += ["--tag", synergy_tag]
            step_extra += ["--out-dir", str(run_dir)]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]

        # ── Step 7(커널 HI 피처 생성, kernel.py) 전용 ───────
        if num == 7:
            _synergy_in = P.FIXED_KERNEL_SYNERGY_GROUPS_JSON or str(synergy_out)
            if not Path(_synergy_in).exists():
                print(f"\n  [경고] Step 7 입력 시너지 그룹 파일이 없습니다: {_synergy_in} "
                      "(Step 6을 이번 선택 범위에 포함시키세요).")
            step_extra += ["--synergy-groups-json", _synergy_in]
            step_extra += ["--split-seed", str(_split_seed)]
            step_extra += ["--alpha", str(P.FIXED_KERNEL_ALPHA)]
            if P.FIXED_KERNEL_GAMMA is not None:
                step_extra += ["--gamma", str(P.FIXED_KERNEL_GAMMA)]
            step_extra += ["--n-components", str(P.FIXED_KERNEL_N_COMPONENTS)]
            step_extra += ["--redundancy-threshold", str(P.FIXED_KERNEL_REDUNDANCY_THRESHOLD)]
            if P.FIXED_KERNEL_MAX_FEATURES is not None:
                step_extra += ["--max-features", str(P.FIXED_KERNEL_MAX_FEATURES)]
            if P.FIXED_MIN_RAW_PARTIAL_CORR is not None:
                step_extra += ["--min-raw-partial-corr", str(P.FIXED_MIN_RAW_PARTIAL_CORR)]
            step_extra += ["--combined-redundancy-threshold", str(P.FIXED_COMBINED_REDUNDANCY_THRESHOLD)]
            step_extra += ["--tag", kernel_tag]
            step_extra += ["--out-dir", str(run_dir)]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]

        # ── 학습 파라미터 주입 (Step 8=Phase1 학습) ────────────────
        if num == 8:
            if args.lambda_l0_override is not None:
                step_extra += ["--lambda-l0-override", str(args.lambda_l0_override)]
            if args.regression_model is not None:
                step_extra += ["--regression-model", args.regression_model]
            # train.py는 --seed/--split-seed가 required=True라 항상 값을
            # 넘겨야 한다 — 미지정 시 42로 채운다(_seed는 run_dir 이름을 정할 때 이미
            # 동일 규칙으로 계산해뒀다 — 여기서 다시 계산하면 값이 어긋날 위험이 있어 재사용).
            if args.seed is None or args.split_seed is None:
                print(f"\n  [안내] Step 8(train.py)는 --seed/--split-seed가 "
                      f"필수 인자라 미지정 값을 기본 42로 채웁니다 "
                      f"(seed={_seed}, split-seed={_split_seed}).")
            step_extra += ["--seed", str(_seed), "--split-seed", str(_split_seed)]

            step_extra += ["--tag", args.p1_tag]
            step_extra += ["--output-dir", str(run_dir)]
            # 여기서 다시 해석한다(맨 위 미리보기 값을 그대로 쓰지 않음) — Step 5~7을
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
            _interaction_now = _resolve_interaction_path(args, interaction_out, P1V4_INTERACTION_JSON)
            if _interaction_now != resolved_interaction:
                print(f"\n  [안내] Step 5 결과가 방금 반영됨 — interaction-json: {_interaction_now}")
            resolved_interaction = _interaction_now
            if resolved_interaction:
                step_extra += ["--interaction-json", resolved_interaction]
            if P.FIXED_TRAIN_CYCLE_FRAC is not None:
                step_extra += ["--train-cycle-frac", str(P.FIXED_TRAIN_CYCLE_FRAC)]
            if P.FIXED_BETA_MIN is not None:
                step_extra += ["--beta-min", str(P.FIXED_BETA_MIN)]
            if args.p1_max_epochs is not None:
                step_extra += ["--max-epochs", str(args.p1_max_epochs)]
            if args.p1_patience is not None:
                step_extra += ["--patience", str(args.p1_patience)]
            if args.p1_batch_size is not None:
                step_extra += ["--batch-size", str(args.p1_batch_size)]
            if P.FIXED_L0_WARMUP_EPOCHS_OVERRIDE is not None:
                step_extra += ["--l0-warmup-epochs-override", str(P.FIXED_L0_WARMUP_EPOCHS_OVERRIDE)]
            if P.FIXED_L0_NORM_CONSTANT is not None:
                step_extra += ["--l0-norm-constant", str(P.FIXED_L0_NORM_CONSTANT)]
            if args.hi_cost_weighted_l0:
                step_extra += ["--hi-cost-weighted-l0"]
            if P.FIXED_DEVICE is not None:
                step_extra += ["--device", P.FIXED_DEVICE]

            # train.py는 --axis-config만으로 데이터 경로를 자동 계산하지
            # 않는다 — --data-dir/--seg-data-dir을 안 주면 자기 자신의 기본값(정식
            # q_frac_ref 캐논 경로)으로 항상 fallback한다. n2 범위 모드처럼 캐논이 아닌
            # 축 설정을 쓰면서 이걸 빠뜨리면, scenario_spec.json은 그 설정을 반영해
            # 만들어지는데 실제로 로드되는 pkl은 캐논 데이터라는 "spec과 데이터 불일치"가
            # 조용히 발생한다 — 반드시 명시적으로 확인.
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]
            _non_canon = args.axis_config != json.dumps(P.ACTIVE_AXIS_CONFIG)
            if _non_canon and not (args.p1_data_dir and args.p1_seg_data_dir):
                print("\n  [경고] --axis-config가 정식(캐논) q_frac_ref 설정과 다른데 "
                      "--data-dir/--seg-data-dir을 안 줬습니다 — Step 8가 이 축 설정을 반영한 "
                      "scenario_spec.json은 만들면서, 실제 pkl 데이터는 train.py의 "
                      "기본 캐논 경로에서 그대로 읽어버립니다(spec-데이터 불일치, 조용히 틀린 "
                      "결과). Step 4로 미리 추출한 경로를 --data-dir/--seg-data-dir로 명시하세요.")

            if args.n_hi != 66:
                print(f"\n  [안내] SOH_EXCLUDE_STAT_LEAK/SOH_EXCLUDE_DQDV_LEAK을 Step 5~9 "
                      f"하위 프로세스 환경에 명시 주입합니다(N_HI={args.n_hi}). N_HI가 64가 "
                      "아니면 --kernel-features-pkl/--interaction-json도 그 N_HI 기준으로 "
                      "새로 만든 파일이 아니면 shape 불일치로 실패합니다.")
            else:
                print(f"\n  [안내] N_HI=66(전부 포함) — SOH_EXCLUDE_STAT_LEAK=0/"
                      "SOH_EXCLUDE_DQDV_LEAK=0을 하위 프로세스 환경에 명시 주입합니다. "
                      "--kernel-features-pkl/--interaction-json이 66-HI 기준 파일이 아니면 "
                      "shape 불일치로 실패합니다.")

        # ── Step 9 전용: 평가할 run_dir + v4 재구성에 필요한 인자 주입 ──────
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
            # v4는 interaction_json이 p1v2_summary.json에 자동 기록되지 않으므로 학습 때와
            # 동일한 값을 평가에도 다시 넘겨야 한다 — Step 8가 이번 실행에 포함돼 있었다면
            # 그때 갱신된 resolved_interaction을 그대로 쓴다.
            if resolved_interaction:
                step_extra += ["--interaction-json", resolved_interaction]
            if resolved_kernel_pkl:
                step_extra += ["--kernel-features-pkl", resolved_kernel_pkl]
            if resolved_combined_redundancy:
                step_extra += ["--combined-redundancy-json", resolved_combined_redundancy]
            if args.rep_cells:
                step_extra += ["--rep-cells", *args.rep_cells]
            if args.p1_data_dir:
                step_extra += ["--data-dir", args.p1_data_dir]
            if args.p1_seg_data_dir:
                step_extra += ["--seg-data-dir", args.p1_seg_data_dir]
            if P.FIXED_DEVICE is not None:
                step_extra += ["--device", P.FIXED_DEVICE]

        # v0~v4 체크포인트 계보는 N_HI를 상호작용 검정(5)/시너지 그룹(6)/커널 HI(7)/
        # 학습(8)/평가(9) 전부 동일하게 맞춰야 한다(전부 build_datasets를 거쳐 N_HI가
        # 이 값에 따라 갈리는 스크립트들). run_step()의 env = os.environ.copy()가 호출
        # 셸의 기존 환경변수를 그대로 물려받으므로, 매번 두 값을 명시적으로 "0"/"1"로
        # 못박아 호출 셸 상태와 무관하게 만든다(2026-09-20 버그 수정 — 예전엔 "필요할
        # 때만 추가"라 호출 셸에 이미 SOH_EXCLUDE_STAT_LEAK=1이 켜져 있으면 --n-hi 66을
        # 줘도 못 지워서 조용히 무시됐음).
        # Step 8는 2026-09-23부터 제외 — train.py가 --model-config 없이
        # 실행되면 자기 자신의 기본값(parameters.py: P1_MODEL_CONFIG, fixed.yaml+
        # main_qfref_S.yaml 병합과 100% 동일한 값)을 쓰도록 바뀌어서, 여기서 굳이
        # yaml 경로를 넘겨줄 필요가 없어졌다(단일 소스 원칙 — Step 8는 이제 yaml을
        # 아예 거치지 않는다). Step 5~7은 아직 --model-config가 필수(required=True)라
        # 그대로 유지.
        _config_flag = "--model-config" if num in (5, 6, 7) else None
        _step_model_config = P.FIXED_PHASE1_MODEL_CONFIG if num in (5, 6, 7) else None
        _extra_env = None
        if num in (5, 6, 7, 8, 9):
            _exclude_stat_leak, _exclude_dqdv_leak = P.N_HI_TO_ENV[args.n_hi]
            _extra_env = {
                "SOH_EXCLUDE_STAT_LEAK": _exclude_stat_leak,
                "SOH_EXCLUDE_DQDV_LEAK": _exclude_dqdv_leak,
            }
        ok = run_step(num, name, script, step_extra, use_workers, args.workers,
                       _step_model_config, config_flag=_config_flag, extra_env=_extra_env)

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
