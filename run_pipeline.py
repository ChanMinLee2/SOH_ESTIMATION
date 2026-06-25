"""
run_pipeline.py

LFP SOH Prediction 전체 데이터 파이프라인 실행기.
스텝 번호를 지정하면 해당 스텝부터 실행.

사용:
  python run_pipeline.py           # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2         # Step 2부터 재실행
  python run_pipeline.py 3 --workers 8
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (번호, 이름, 스크립트 경로, 추가 인자)
STEPS = [
    (1, "데이터 변환",        "1_convert/convert_unified.py",     ["--dataset", "all"]),
    (2, "이상 사이클 제거",   "2_preprocess/preprocess.py",        []),
    (3, "무결성 검사",        "3_integrity/check_integrity.py",    []),
    (4, "HI 상관 분석",       "4_hi_analysis/hi_correlation.py",   ["--force"]),
    (5, "HI 세그먼트 시각화", "4_hi_analysis/hi_segment_viz.py",   []),
]


def _fmt_time(sec: float) -> str:
    m, s = int(sec) // 60, int(sec) % 60
    return f"{m}분 {s}초" if m else f"{s}초"


def run_step(num: int, name: str, script: str, extra_args: list, workers: int) -> bool:
    cmd = [sys.executable, str(ROOT / script)] + extra_args + ["--workers", str(workers)]

    print(f"\n{'='*60}")
    print(f"  Step {num}  {name}")
    print(f"  $ {' '.join(str(a) for a in cmd)}")
    print(f"{'='*60}")

    t0     = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAIL] Step {num} 실패  (exit={result.returncode}, {_fmt_time(elapsed)})")
        return False

    print(f"\n  [OK] Step {num} 완료  ({_fmt_time(elapsed)})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="LFP SOH 데이터 파이프라인 실행기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="스텝 목록:\n" + "\n".join(f"  {n}  {name}" for n, name, _, _ in STEPS),
    )
    parser.add_argument(
        "from_step", nargs="?", type=int, default=1, metavar="FROM_STEP",
        help=f"시작 스텝 번호 (기본: 1, 범위: 1~{len(STEPS)})",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="각 스텝에 전달할 병렬 프로세스 수 (기본: 8)",
    )
    args = parser.parse_args()

    if not (1 <= args.from_step <= len(STEPS)):
        parser.error(f"from_step 은 1~{len(STEPS)} 사이여야 합니다.")

    selected = [s for s in STEPS if s[0] >= args.from_step]

    print("\n" + "="*60)
    print("  LFP SOH Prediction — 데이터 파이프라인")
    print("="*60)
    print(f"  시작 스텝  : {args.from_step}")
    print(f"  병렬 워커  : {args.workers}")
    print(f"  실행 스텝  :")
    for n, name, _, _ in selected:
        print(f"    Step {n}  {name}")
    print("="*60)

    total_t0 = time.time()
    failed: list[int] = []

    for num, name, script, extra in selected:
        ok = run_step(num, name, script, extra, args.workers)
        if not ok:
            failed.append(num)
            try:
                ans = input(f"\n  Step {num} 실패. 계속 진행하시겠습니까? [y/N]: ").strip().lower()
            except EOFError:
                ans = "n"
            if ans != "y":
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
