"""
run_pipeline.py

LFP SOH Prediction 전체 파이프라인 실행기.
데이터 전처리(Step 1~5)부터 모델 학습/평가(Step 6~8)까지 지원.

사용:
  python run_pipeline.py                          # 전체 파이프라인 (Step 1부터)
  python run_pipeline.py 2                        # Step 2부터 재실행
  python run_pipeline.py 6                        # 학습/평가만 (Step 6~8)
  python run_pipeline.py 6 --to-step 7           # Phase 1~2 학습만
  python run_pipeline.py 8 --checkpoint path/to/best.pt  # 평가만
  python run_pipeline.py 3 --workers 8
  python run_pipeline.py --model-config 5_model/config/scr.yaml
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_OUTPUT_DIR = ROOT / "_5_data_model_scr"

# (번호, 이름, 스크립트 경로, 기본 추가 인자, --workers 지원 여부)
STEPS = [
    (1, "데이터 변환",          "1_convert/convert_unified.py",    ["--dataset", "all"], True),
    (2, "이상 사이클 제거",     "2_preprocess/preprocess.py",       [],                   True),
    (3, "무결성 검사",          "3_integrity/check_integrity.py",   [],                   True),
    (4, "HI 상관 분석",         "4_hi_analysis/hi_correlation.py",  ["--force"],          True),
    (5, "HI 세그먼트 시각화",   "4_hi_analysis/hi_segment_viz.py",  [],                   True),
    (6, "SCR Phase 1 학습",     "5_model/train_scr.py",             ["--phase", "1"],     False),
    (7, "SCR Phase 2 학습",     "5_model/train_scr.py",             ["--phase", "2"],     False),
    (8, "시나리오 분류기 학습", "5_model/train_classifier.py",       [],                   False),
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


def _find_new_run_dir(before: set[Path]) -> Path | None:
    """스냅샷 이후 새로 생긴 run 디렉터리 반환. 없으면 가장 최신 dir."""
    if not MODEL_OUTPUT_DIR.exists():
        return None
    after = {d for d in MODEL_OUTPUT_DIR.iterdir() if d.is_dir()}
    new = after - before
    if new:
        return max(new, key=lambda d: d.stat().st_mtime)
    # fallback: 가장 최근 수정 디렉터리
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
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="데이터 스텝(1~5)에 전달할 병렬 프로세스 수 (기본: 8)",
    )
    parser.add_argument(
        "--model-config", default="5_model/config/scr.yaml",
        help="모델 학습/평가 설정 파일 (기본: 5_model/config/scr.yaml)",
    )
    parser.add_argument(
        "--gates-from", default=None, metavar="DIR",
        help="Phase 2 학습 시 gates 디렉터리 직접 지정 (미지정 시 Phase 1 출력 자동 탐색)",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="SCR 평가 체크포인트 직접 지정 (미지정 시 Phase 2 출력 자동 탐색)",
    )
    args = parser.parse_args()

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
    print(f"  모델 설정   : {args.model_config}")
    if args.gates_from:
        print(f"  gates-from  : {args.gates_from}")
    if args.checkpoint:
        print(f"  checkpoint  : {args.checkpoint}")
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

        # ── Phase 1 전: 스냅샷 ──────────────────────────────────────────────
        if num == 6:
            snapshot = _snapshot_run_dirs()

        # ── Phase 2 전: 스냅샷 + gates-from 주입 ────────────────────────────
        if num == 7:
            snapshot = _snapshot_run_dirs()
            gates_src = args.gates_from or (str(p1_run_dir) if p1_run_dir else None)
            if gates_src is None:
                latest = _find_new_run_dir(set())  # 전체 중 최신
                gates_src = str(latest) if latest else None
            if gates_src:
                step_extra += ["--gates-from", gates_src]
                print(f"\n  → gates-from: {gates_src}")
            else:
                print("\n  [경고] Phase 1 run 디렉터리를 찾을 수 없습니다.")

        # ── 분류기 학습 전: Phase 2 run_dir 주입 ────────────────────────────
        if num == 8:
            clf_run = p2_run_dir or _find_new_run_dir(set())
            if clf_run:
                step_extra += ["--run-dir", str(clf_run)]
                print(f"\n  → run-dir (분류기): {clf_run}")
            else:
                print("\n  [경고] Phase 2 run 디렉터리를 찾을 수 없습니다. "
                      "--run-dir 직접 지정 권장.")

        # ── 평가 전: checkpoint 주입 ─────────────────────────────────────────
        if num == 9:
            ckpt = args.checkpoint or _find_checkpoint(p2_run_dir)
            if ckpt is None:
                # fallback: 가장 최신 run의 best.pt
                latest = _find_new_run_dir(set())
                ckpt = _find_checkpoint(latest)
            if ckpt:
                step_extra += ["--checkpoint", ckpt]
                print(f"\n  → checkpoint: {ckpt}")
            else:
                print("\n  [경고] 체크포인트를 찾을 수 없습니다. --checkpoint 직접 지정 권장.")

        ok = run_step(num, name, script, step_extra, use_workers, args.workers, args.model_config)

        # ── Phase 1 후: 신규 run dir 기록 ────────────────────────────────────
        if num == 6:
            p1_run_dir = _find_new_run_dir(snapshot)
            if p1_run_dir:
                print(f"  → Phase 1 run dir: {p1_run_dir}")
            else:
                print("  [경고] Phase 1 run 디렉터리를 감지하지 못했습니다.")

        # ── Phase 2 후: 신규 run dir 기록 ────────────────────────────────────
        if num == 7:
            p2_run_dir = _find_new_run_dir(snapshot)
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
