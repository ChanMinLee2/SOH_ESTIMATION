"""
plot_all_mit_cells.py

_1_data_unified/<DATASET> 폴더의 모든 셀에 대해 plot_cell_cycles.py 를 한 번씩 실행.
각 셀마다 cell/cell_cycles_<dataset>_<cell>.png 가 생성된다.

각 PNG 에는 방전·충전 사이클이 함께 그려진다.

2026-09-08: --dataset 인자 추가(mit/hust/calce/tju) — 파일명은 원래 이름(mit 전용
스크립트로 시작) 그대로 유지하지만 기본값이 "mit"이라 기존 호출(인자 없이 실행)은
100% 그대로 동작한다.

사용:
  python 4_hi_analysis/plot_all_mit_cells.py                          # MIT, 순차
  python 4_hi_analysis/plot_all_mit_cells.py --workers 8              # MIT, 8개 병렬
  python 4_hi_analysis/plot_all_mit_cells.py --dataset calce --workers 8
  python 4_hi_analysis/plot_all_mit_cells.py --dataset tju --workers 0  # CPU 코어 수만큼
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIRS = {
    "mit":   PROJECT_ROOT / "_1_data_unified" / "MIT",
    "hust":  PROJECT_ROOT / "_1_data_unified" / "HUST",
    "calce": PROJECT_ROOT / "_1_data_unified" / "CALCE",
    "tju":   PROJECT_ROOT / "_1_data_unified" / "TJU",
}
PLOT_SCRIPT = Path(__file__).resolve().parent / "plot_cell_cycles.py"


def run_one(dataset: str, cell: str):
    """단일 셀 플롯. (cell, 성공여부, 에러메시지) 반환."""
    result = subprocess.run(
        [sys.executable, str(PLOT_SCRIPT),
         "--dataset", dataset, "--cell", cell],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return cell, False, err
    return cell, True, None


def main():
    parser = argparse.ArgumentParser(description="한 데이터셋의 모든 셀 일괄 시각화")
    parser.add_argument("--dataset", default="mit", choices=list(_DATASET_DIRS))
    parser.add_argument("--workers", type=int, default=1,
                        help="동시 실행 프로세스 수 (1=순차, 0=CPU 코어 수, 기본: 1)")
    args = parser.parse_args()

    data_dir = _DATASET_DIRS[args.dataset]
    cells = sorted(p.stem for p in data_dir.glob("*.pkl"))
    if not cells:
        raise FileNotFoundError(f"PKL 파일 없음: {data_dir}")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    n = len(cells)
    print(f"총 {n}개 {args.dataset.upper()} 셀 시각화 시작 (workers={workers})\n")

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, args.dataset, cell): cell for cell in cells}
        for future in as_completed(futures):
            cell, ok, err = future.result()
            done += 1
            if ok:
                print(f"[{done}/{n}] ✓ {cell}", flush=True)
            else:
                failed.append(cell)
                print(f"[{done}/{n}] ✗ {cell}: {err}", flush=True)

    print(f"\n완료: {n - len(failed)}/{n} 성공")
    if failed:
        print(f"실패 셀: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
