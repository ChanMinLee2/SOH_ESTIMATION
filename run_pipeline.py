"""
run_pipeline.py

LFP SOH Prediction 전체 파이프라인 실행기.
데이터 전처리(Step 1~5)부터 모델 학습/평가(Step 6~9)까지 지원.

2026-07-30: Step 7(분류기)/Step 8(Phase 2) 순서를 재배치했다 — 분류기는 Phase 1
게이트만 있으면 학습되므로(회귀와 완전 분리) Phase 2보다 먼저 돌리고, Phase 2가
--with-raw-cnn으로 그 분류기의 CNN을 바로 재사용할 수 있게 했다(REGRESSION_UPGRADE.md
§8/§10). Step 6=Phase1, Step 7=분류기, Step 8=Phase2, Step 9=평가.

사용:
  python run_pipeline.py                          # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2                        # Step 2부터 재실행
  python run_pipeline.py 6                        # 학습/평가만 (Step 6~9)
  python run_pipeline.py 6 --to-step 8            # Phase1→분류기→Phase2 학습만(평가 제외)
  python run_pipeline.py 6 --to-step 8 --with-raw-cnn  # 위와 동일 + Phase2에 raw CNN 융합(방안 a, 단일 패스)
  python run_pipeline.py 9                         # 평가만(직전 Phase2 run의 best.pt 자동 탐색)
  python run_pipeline.py 3 --workers 8
  python run_pipeline.py --model-config 5_model/config/scr.yaml
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = ROOT / "_5_data_model_scr"
_RUN_DIR_RE = re.compile(r"_p(\d+)_")  # train_scr.py 명명 규칙: {MMDD_HHMM}_p{phase}_...

# (번호, 이름, 스크립트 경로, 기본 추가 인자, --workers 지원 여부)
#
# 2026-07-30 재배치: 분류기(구 Step 8)를 Phase 2(구 Step 7) 앞으로 옮겼다.
# train_classifier.py는 원래도 Phase 1 게이트(run_dir/gates/*)만 있으면 동작했고
# Phase 2 가중치는 전혀 안 썼다(회귀와 완전 분리 설계) — 다만 체크포인트 저장 위치가
# "그 run_dir"이었을 뿐이라 관례상 Phase 2 뒤에 뒀었다. Phase 2가 with_raw_cnn으로
# 그 분류기의 CNN을 재사용(REGRESSION_UPGRADE.md §8 방안 a)하려면 분류기가 먼저
# 학습돼 있어야 하므로, Phase 2를 두 번 돌리지 않도록 순서를 Phase1→분류기→Phase2로
# 바꿨다. 기존 run(분류기가 Phase 2 폴더 안에 저장됨)은 test_scr.py가 레거시 위치를
# 우선 탐색해 그대로 호환된다(_resolve_classifier_ckpt 참조).
STEPS = [
    (1, "데이터 변환",          "1_convert/convert_unified.py",    ["--dataset", "all"], True),
    (2, "이상 사이클 제거",     "2_preprocess/preprocess.py",       [],                   True),
    (3, "무결성 검사",          "3_integrity/check_integrity.py",   [],                   True),
    (4, "HI 상관 분석",         "4_hi_analysis/hi_correlation.py",  ["--force"],          True),
    (5, "HI 세그먼트 시각화",   "4_hi_analysis/hi_segment_viz.py",  [],                   True),
    (6, "SCR Phase 1 학습",     "5_model/train_scr.py",             ["--phase", "1"],     False),
    (7, "시나리오 분류기 학습", "5_model/train_classifier.py",       [],                   False),
    (8, "SCR Phase 2 학습",     "5_model/train_scr.py",             ["--phase", "2"],     False),
    (9, "SCR 평가",             "5_model/test_scr.py",              [],                   False),
]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m}분 {s}초" if m else f"{s}초"


def _snapshot_run_dirs() -> set[Path]:
    """현재 MODEL_OUTPUT_DIR 내 디렉터리 집합 스냅샷."""
    if not MODEL_OUTPUT_DIR.exists():
        return set()
    return {d for d in MODEL_OUTPUT_DIR.iterdir() if d.is_dir()}


def _find_new_run_dir(before: set[Path], phase: int | None = None) -> Path | None:
    """스냅샷 이후 새로 생긴 run 디렉터리 반환. 없으면 가장 최신 run 디렉터리.

    train_scr.py가 만드는 run 디렉터리 이름 패턴('_p{phase}_' 포함)에 맞는 것만
    후보로 삼는다 — 이 필터가 없으면 파이프라인 실행 도중 우연히 mtime이 더 최근으로
    갱신된 무관한 폴더(사용자가 수동으로 정리/이동해 만든 폴더 등)를 잘못 골라
    --gates-from/--run-dir/--checkpoint 에 엉뚱한 경로가 들어갈 수 있다.
    phase 지정 시 해당 phase(_p{phase}_)만 후보로 제한.
    """
    if not MODEL_OUTPUT_DIR.exists():
        return None
    after = {d for d in MODEL_OUTPUT_DIR.iterdir() if d.is_dir() and _RUN_DIR_RE.search(d.name)}
    if phase is not None:
        after = {d for d in after if f"_p{phase}_" in d.name}
    new = after - before
    if new:
        return max(new, key=lambda d: d.stat().st_mtime)
    # fallback: 가장 최근 수정 run 디렉터리
    all_dirs = sorted(after, key=lambda d: d.stat().st_mtime)
    return all_dirs[-1] if all_dirs else None


def _find_checkpoint(run_dir: Path | None) -> str | None:
    """run 디렉터리에서 best.pt 경로 반환."""
    if run_dir is None:
        return None
    ckpt = run_dir / "checkpoints" / "best.pt"
    return str(ckpt) if ckpt.exists() else None


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
    model_config: str,
) -> bool:
    cmd = [sys.executable, str(ROOT / script)] + extra_args
    if use_workers:
        cmd += ["--workers", str(workers)]
    elif "--config" not in extra_args:
        cmd += ["--config", model_config]

    print(f"\n{'='*60}")
    print(f"  Step {num}  {name}")
    print(f"  $ {' '.join(str(a) for a in cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
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
        description="LFP SOH 파이프라인 실행기 (데이터 전처리 → 모델 학습/평가)",
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
        "--skip-classifier", action="store_true",
        help="Step 7(시나리오 분류기 학습) 건너뛰기 — test_scr.py가 분류기 체크포인트 "
             "없으면 자동으로 oracle 모드만 평가하므로 안전함(2026-08-12). "
             "hard/soft 라우팅 비교가 필요 없을 때 학습 시간 절약용.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="데이터 스텝(1~5)에 전달할 병렬 프로세스 수 (기본: 8)",
    )
    parser.add_argument(
        "--model-config", default="5_model/config/scr.yaml",
        help="모델 학습/평가 설정 파일 (기본: 5_model/config/scr.yaml)",
    )
    parser.add_argument(
        "--seg-axis", default=None, metavar="AXIS",
        help="세그멘테이션 축 (Step 4~6, 8에 전달 — 분류기(Step 7)는 run_dir의 "
             "scenario_spec.json에서 축을 읽으므로 이 옵션이 필요 없음). 예: qfrac, q_frac_wide",
    )
    parser.add_argument(
        "--axis-config", default=None, metavar="JSON",
        help="축 파라미터 JSON (Step 4~6, 8에 전달). 예: '{\"n1\": 0.4, \"n2\": 0.2, \"n_samples\": 4}'",
    )
    # 단축 인자 (PowerShell JSON 우회) — Step 4~6, 8에 그대로 전달
    parser.add_argument("--n1",        type=float, default=None,
                        help="q_frac_wide 구간 크기 (--axis-config 대체, PowerShell 호환)")
    parser.add_argument("--n2",        type=float, default=None,
                        help="q_frac_wide 세그먼트 길이 (--axis-config 대체)")
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
    parser.add_argument("--exclude-cv", action="store_true", dest="exclude_cv",
                        help="충전 세그먼트 HI에서 CC→CV 전환 이후 구간 제외 (Step 4/6/7/8 전달). "
                             "Step 4는 결과를 '_ccOnly' 접미사 경로에 저장하고, "
                             "Step 6/7/8은 동일 접미사로 해당 데이터를 찾는다.")
    parser.add_argument("--skip-shape", action="store_true", dest="skip_shape",
                        help="전처리 필터7(형상 이상치 제거) 비활성화 (Step 2/4/6/7/8 전달). "
                             "Step 2는 _4_data_hi/clean_noshape/에 저장하고, Step 4는 그 데이터로 "
                             "'_noshape' 접미사 경로에 추출하며, Step 6/7/8은 동일 접미사로 "
                             "해당 데이터를 찾는다.")
    # m/k 오버라이드 (Step 6, 8에 그대로 전달 — train_scr.py --charge-m/--discharge-m/--scen-k.
    # 분류기(Step 7)는 이 CLI 옵션이 없고 yaml classifier.charge_probe_m 등을 직접 읽음)
    parser.add_argument("--charge-m",    type=int, default=None,
                        help="충전 probe 상위 m개 (yaml charge_probe_m 오버라이드, Step 6/8 전달)")
    parser.add_argument("--discharge-m", type=int, default=None,
                        help="방전 probe 상위 m개 (yaml discharge_probe_m 오버라이드, Step 6/8 전달)")
    parser.add_argument("--scen-k",      type=int, default=None,
                        help="시나리오별 scen HI 수 (yaml scen_k_count 오버라이드, Step 6/8 전달)")
    parser.add_argument("--seed",        type=int, default=None,
                        help="재현성 시드 — 모델 초기화 torch/numpy/random RNG (yaml training.seed 오버라이드, Step 6/8 전달)")
    parser.add_argument("--split-seed",  type=int, default=None,
                        help="train/val/test 셀 분할 시드 (yaml data.split_seed 오버라이드, Step 6/8 전달)")
    parser.add_argument("--with-raw-cnn", action="store_true", dest="with_raw_cnn",
                        help="Phase 2(Step 8)에 회귀 헤드 raw CNN 융합 적용 (REGRESSION_UPGRADE.md "
                             "§5/§8/§10). Step 7에서 학습된 분류기의 RawCNN을 자동으로 얼려서 "
                             "재사용한다(방안 a) — Step 6→9를 한 번에 돌리면 Phase 2를 두 번 학습할 "
                             "필요 없이 단일 패스로 완료됨. yaml model.regression_model 등 다른 "
                             "model 설정은 --model-config로 지정한 파일 그대로 사용됨")
    args = parser.parse_args()

    # 단축 인자 → args.axis_config(JSON) 로 합침. 이후 기존 --axis-config 전달 로직이
    # 모든 하위 스텝(4~7)에 올바른 JSON을 넘긴다. subprocess는 shell 없이 인자를 그대로
    # 전달하므로 PowerShell 따옴표 벗김 문제가 발생하지 않는다.
    if (args.n1 is not None or args.n2 is not None or args.n_samples is not None
            or args.ref_lag is not None or args.noise_amp is not None
            or args.noise_mode is not None or args.noise_period_cycles is not None
            or args.min_pts is not None):
        import json as _json
        _quick: dict = {}
        if args.n1        is not None: _quick["n1"]        = args.n1
        if args.n2        is not None: _quick["n2"]        = args.n2
        if args.n_samples is not None: _quick["n_samples"] = args.n_samples
        if args.ref_lag   is not None: _quick["ref_lag"]   = args.ref_lag
        if args.noise_amp is not None: _quick["noise_amp"] = args.noise_amp
        if args.noise_mode is not None: _quick["noise_mode"] = args.noise_mode
        if args.noise_period_cycles is not None: _quick["noise_period_cycles"] = args.noise_period_cycles
        if args.min_pts is not None: _quick["min_pts"] = args.min_pts
        args.axis_config = _json.dumps(_quick)

    to_step = args.to_step if args.to_step is not None else len(STEPS)

    if not (1 <= args.from_step <= len(STEPS)):
        parser.error(f"from_step 은 1~{len(STEPS)} 사이여야 합니다.")
    if not (1 <= to_step <= len(STEPS)):
        parser.error(f"--to-step 은 1~{len(STEPS)} 사이여야 합니다.")
    if args.from_step > to_step:
        parser.error("from_step 이 to_step 보다 클 수 없습니다.")

    selected = [s for s in STEPS if args.from_step <= s[0] <= to_step]
    if args.skip_classifier:
        selected = [s for s in selected if s[0] != 7]

    print("\n" + "="*60)
    print("  LFP SOH Prediction — 전체 파이프라인")
    print("="*60)
    print(f"  스텝 범위   : {args.from_step} → {to_step}"
          + ("  (Step 7 분류기 스킵)" if args.skip_classifier else ""))
    print(f"  병렬 워커   : {args.workers}  (데이터 스텝 전용)")
    print(f"  모델 설정   : {args.model_config}")
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
    print(f"  실행 스텝   :")
    for n, name, _, _, _ in selected:
        print(f"    Step {n}  {name}")
    print("="*60)

    total_t0 = time.time()
    failed: list[int] = []

    # Phase 간 디렉터리 핸드오프
    p1_run_dir: Path | None = None
    p2_run_dir: Path | None = None
    snapshot: set[Path] = set()

    for num, name, script, extra, use_workers in selected:
        step_extra = list(extra)

        # ── 축 정보 주입 (Step 4~6, 8 — 분류기(7)는 run_dir의 scenario_spec.json에서 축을 읽음) ──
        if num in (4, 5, 6, 8):
            if args.seg_axis:
                step_extra += ["--seg-axis", args.seg_axis]
            if args.axis_config:
                step_extra += ["--axis-config", args.axis_config]

        # ── CV 제외 옵션 주입 (Step 4=추출, 6=Phase1, 7=분류기, 8=Phase2 — 모두 '_ccOnly' 경로 인지 필요) ──
        if num in (4, 6, 7, 8) and args.exclude_cv:
            step_extra += ["--exclude-cv"]

        # ── shape filter 비활성화 옵션 주입 (Step 2=전처리 자체가 필터7 스킵,
        #    Step 4=추출, 6=Phase1, 7=분류기, 8=Phase2 — 모두 '_noshape' 경로 인지 필요) ──
        if num in (2, 4, 6, 7, 8) and args.skip_shape:
            step_extra += ["--skip-shape"]

        # ── m/k 오버라이드 주입 (Step 6=Phase1, 8=Phase2 — 분류기(7)는 이 CLI 옵션이 없음) ──
        if num in (6, 8):
            if args.charge_m is not None:
                step_extra += ["--charge-m", str(args.charge_m)]
            if args.discharge_m is not None:
                step_extra += ["--discharge-m", str(args.discharge_m)]
            if args.scen_k is not None:
                step_extra += ["--scen-k", str(args.scen_k)]
            if args.seed is not None:
                step_extra += ["--seed", str(args.seed)]
            if args.split_seed is not None:
                step_extra += ["--split-seed", str(args.split_seed)]

        # ── Phase 1 전: 스냅샷 ──────────────────────────────────────────────
        if num == 6:
            snapshot = _snapshot_run_dirs()

        # ── 분류기 학습 전: Phase 1 run_dir 주입 (2026-07-30 재배치 — 분류기는 Phase 1
        #    게이트만 있으면 되므로 Phase 2보다 먼저 학습해, Phase 2가 with_raw_cnn으로
        #    이 분류기의 CNN을 그 자리에서 재사용할 수 있게 한다) ──────────────────────
        if num == 7:
            clf_run = str(p1_run_dir) if p1_run_dir else None
            if clf_run is None:
                latest = _find_new_run_dir(set(), phase=1)
                clf_run = str(latest) if latest else None
            if clf_run:
                step_extra += ["--run-dir", str(clf_run)]
                print(f"\n  → run-dir (분류기): {clf_run}")
            else:
                print("\n  [경고] Phase 1 run 디렉터리를 찾을 수 없습니다. "
                      "--run-dir 직접 지정 권장.")

        # ── Phase 2 전: 스냅샷 + gates-from 주입 (+ --with-raw-cnn 자동 연결) ──────
        if num == 8:
            snapshot = _snapshot_run_dirs()
            gates_src = str(p1_run_dir) if p1_run_dir else None
            if gates_src is None:
                latest = _find_new_run_dir(set(), phase=1)  # Phase 1 run 중 최신
                gates_src = str(latest) if latest else None
            if gates_src:
                step_extra += ["--gates-from", gates_src]
                print(f"\n  → gates-from: {gates_src}")
            else:
                print("\n  [경고] Phase 1 run 디렉터리를 찾을 수 없습니다.")

            if args.with_raw_cnn:
                step_extra += ["--with-raw-cnn"]
                if gates_src:
                    _clf_ckpt = Path(gates_src) / "classifier" / "clf_best.pt"
                    if not _clf_ckpt.is_absolute():
                        _clf_ckpt = ROOT / _clf_ckpt
                    if _clf_ckpt.exists():
                        step_extra += ["--raw-cnn-pretrained-from", str(_clf_ckpt)]
                        print(f"  → raw-cnn-pretrained-from: {_clf_ckpt}")
                    else:
                        print(f"\n  [경고] --with-raw-cnn 지정했지만 분류기 체크포인트가 없습니다"
                              f"({_clf_ckpt}) — RawCNN을 랜덤 초기화해 Phase2와 함께 학습합니다(방안 b).")

        # ── 평가 전: checkpoint 주입 ─────────────────────────────────────────
        if num == 9:
            ckpt = _find_checkpoint(p2_run_dir)
            if ckpt is None:
                # fallback: 가장 최신 Phase 2 run의 best.pt
                latest = _find_new_run_dir(set(), phase=2)
                ckpt = _find_checkpoint(latest)
            if ckpt:
                step_extra += ["--checkpoint", ckpt]
                print(f"\n  → checkpoint: {ckpt}")
            else:
                print("\n  [경고] 체크포인트를 찾을 수 없습니다. --checkpoint 직접 지정 권장.")

        ok = run_step(num, name, script, step_extra, use_workers, args.workers, args.model_config)

        # ── Phase 1 후: 신규 run dir 기록 ────────────────────────────────────
        if num == 6:
            p1_run_dir = _find_new_run_dir(snapshot, phase=1)
            if p1_run_dir:
                print(f"  → Phase 1 run dir: {p1_run_dir}")
            else:
                print("  [경고] Phase 1 run 디렉터리를 감지하지 못했습니다.")

        # ── Phase 2 후: 신규 run dir 기록 ────────────────────────────────────
        if num == 8:
            p2_run_dir = _find_new_run_dir(snapshot, phase=2)
            if p2_run_dir:
                print(f"  → Phase 2 run dir: {p2_run_dir}")
            else:
                print("  [경고] Phase 2 run 디렉터리를 감지하지 못했습니다.")

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
