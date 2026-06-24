"""
plot_all_mit_cells.py

data_unified/MIT 폴더의 모든 셀에 대해 plot_cell_cycles.py 를 한 번씩 실행.
각 셀마다 cell/cell_cycles_mit_<cell>.png 가 생성된다.

각 PNG 에는 방전·충전 사이클이 함께 그려진다.

사용:
  python 4_hi_analysis/plot_all_mit_cells.py                 # 순차 (workers=1)
  python 4_hi_analysis/plot_all_mit_cells.py --workers 8     # 8개 병렬
  python 4_hi_analysis/plot_all_mit_cells.py --workers 0     # CPU 코어 수만큼 병렬
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MIT_DIR = PROJECT_ROOT / "data_postprocess" / "MIT"
MIT_DIR = PROJECT_ROOT / "data_unified" / "MIT"
PLOT_SCRIPT = Path(__file__).resolve().parent / "plot_cell_cycles.py"


def run_one(cell: str):
    """단일 셀 플롯. (cell, 성공여부, 에러메시지) 반환."""
    result = subprocess.run(
        [sys.executable, str(PLOT_SCRIPT),
         "--dataset", "mit", "--cell", cell],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        return cell, False, err
    return cell, True, None


def main():
    parser = argparse.ArgumentParser(description="모든 MIT 셀 일괄 시각화")
    parser.add_argument("--workers", type=int, default=1,
                        help="동시 실행 프로세스 수 (1=순차, 0=CPU 코어 수, 기본: 1)")
    args = parser.parse_args()

    cells = sorted(p.stem for p in MIT_DIR.glob("*.pkl"))
    if not cells:
        raise FileNotFoundError(f"PKL 파일 없음: {MIT_DIR}")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    n = len(cells)
    print(f"총 {n}개 MIT 셀 시각화 시작 (workers={workers})\n")

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, cell): cell for cell in cells}
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
