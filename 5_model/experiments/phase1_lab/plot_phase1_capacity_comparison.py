"""
5_model/experiments/phase1_lab/plot_phase1_capacity_comparison.py

phase1_lab 체크포인트 여러 개(v0/v2/v3/v4 등)를 놓고, 지정한 셀 하나의 시나리오별 용량곡선
복원을 나란히 비교한다. 이제는 삭제된 5_model/plot_scenario_verification_comparison.py와
같은 패턴 — 5_model/visualize_results.py의 기존 함수(RunBundle/_load_bundles/
_check_cross_axis/_plot_capacity_curve_comparison)를 무수정으로 재사용한다(중복 구현 금지).

전제: 각 run_dir은 test_phase1_checkpoint.py를 --export-for-visualize로 먼저 실행해서
metrics/predictions/routing 세 파일을 채워둔 상태여야 한다(RunBundle.__init__이 이 셋을
무조건 읽음) — checkpoints/config.yaml/scenario_spec.json/gates는 phase1_trainer_v2.py가
이미 저장해둔 걸 그대로 쓴다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_phase1_capacity_comparison.py \
      --run-dirs <v0_run> <v2_run> <v3_run> <v4_run> \
      --labels v0 v2 v3 v4 \
      --rep-cells b1c0 \
      --out-dir _5_data_model_scr/l0_auto_p1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from visualize_results import (  # noqa: E402  (경로 설정 뒤 import, 중복 구현 금지)
    _check_cross_axis,
    _load_bundles,
    _plot_capacity_curve_comparison,
)

_ANY_CELL_RE = re.compile(r".+")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="phase1_lab run 여러 개의 셀별 용량곡선 비교")
    p.add_argument("--run-dirs", nargs="+", required=True, dest="run_dirs")
    p.add_argument("--labels", nargs="+", default=None,
                   help="--run-dirs와 같은 개수. 미지정 시 run_dir 폴더명을 라벨로 씀")
    p.add_argument("--rep-cells", nargs="+", required=True, dest="rep_cells",
                   help="비교할 셀 ID(들) — 모든 run의 test split에 공통으로 있어야 함")
    p.add_argument("--out-dir", required=True, dest="out_dir",
                   help="capacity_curve_verification_<cell>.png를 저장할 폴더")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bundles = _load_bundles(args.run_dirs, args.labels)
    cross_axis = _check_cross_axis(bundles)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    for cell in args.rep_cells:
        missing = [b.label for b in bundles if not any(r["cell_id"] == cell for r in b.pred_rows)]
        if missing:
            print(f"[plot] 셀 '{cell}'이 다음 run의 test split에 없습니다 — 건너뜀: {missing}")
            continue
        _plot_capacity_curve_comparison(
            bundles, cell, out_dir / f"capacity_curve_verification_{cell}.png",
            cross_axis=cross_axis,
        )


if __name__ == "__main__":
    main()
