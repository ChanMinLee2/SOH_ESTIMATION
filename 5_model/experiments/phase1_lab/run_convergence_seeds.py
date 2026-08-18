"""
5_model/experiments/phase1_lab/run_convergence_seeds.py

Phase1 랭킹 시드-수렴성 검증 — 1단계(오케스트레이션).

기존 5_model/train_scr.py를 수정하지 않고, 그대로 subprocess로 여러 seed에 대해
반복 호출한다(--phase 1). 각 실행이 만든 run_dir을 stdout에서 파싱해 모아
manifest json으로 저장한다 — analyze_convergence.py가 이 manifest를 읽어
Jaccard/Kendall/선택빈도를 계산한다.

기존 코드 무변경 원칙: train_scr.py/run_pipeline.py는 전혀 건드리지 않고
"여러 번 실행 + 결과 수집"만 이 파일이 담당한다.

사용 예:
  python 5_model/experiments/phase1_lab/run_convergence_seeds.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --scen-k 25 \
      --seeds 42 0 123 7 2024 \
      --tag k25_convergence
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_RUN_DIR_RE = re.compile(r"\[train\] run dir:\s*(.+)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 시드-수렴성 검증용 다중 seed 실행 오케스트레이터")
    p.add_argument("--model-config", required=True, help="5_model/train_scr.py --config 그대로 전달")
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True, help="JSON 문자열")
    p.add_argument("--charge-m", type=int, default=None)
    p.add_argument("--discharge-m", type=int, default=None)
    p.add_argument("--scen-k", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="예: --seeds 42 0 123 7 2024  (split-seed도 동일 값 사용)")
    p.add_argument("--tag", required=True, help="manifest 파일명 구분용 태그")
    p.add_argument("--dry-run", action="store_true", help="명령어만 출력하고 실행하지 않음")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "tag": args.tag,
        "model_config": args.model_config,
        "seg_axis": args.seg_axis,
        "axis_config": json.loads(args.axis_config),
        "scen_k": args.scen_k,
        "seeds": args.seeds,
        "runs": {},  # seed -> run_dir
    }

    for seed in args.seeds:
        cmd = [
            sys.executable, str(PROJECT_ROOT / "5_model" / "train_scr.py"),
            "--phase", "1",
            "--config", args.model_config,
            "--seg-axis", args.seg_axis,
            "--axis-config", args.axis_config,
            "--seed", str(seed),
            "--split-seed", str(seed),
        ]
        if args.charge_m is not None:
            cmd += ["--charge-m", str(args.charge_m)]
        if args.discharge_m is not None:
            cmd += ["--discharge-m", str(args.discharge_m)]
        if args.scen_k is not None:
            cmd += ["--scen-k", str(args.scen_k)]

        print(f"\n{'='*70}\n[convergence] seed={seed}\n$ {' '.join(cmd)}\n{'='*70}")
        if args.dry_run:
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        elapsed = time.time() - t0
        print(result.stdout[-3000:])  # 마지막 부분만 (전체 로그는 각 run_dir에 이미 남음)
        if result.returncode != 0:
            print(f"[convergence] seed={seed} 실패 (exit={result.returncode}):\n{result.stderr[-2000:]}")
            continue

        m = _RUN_DIR_RE.search(result.stdout)
        if not m:
            print(f"[convergence] seed={seed}: run dir를 stdout에서 못 찾음 — 수동 확인 필요")
            continue
        run_dir = m.group(1).strip()
        manifest["runs"][str(seed)] = run_dir
        print(f"[convergence] seed={seed} 완료 ({elapsed:.0f}s) → {run_dir}")

    out_path = RESULTS_DIR / f"convergence_manifest_{args.tag}.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[convergence] manifest 저장: {out_path}")
    print(f"[convergence] 다음 단계: python 5_model/experiments/phase1_lab/analyze_convergence.py --manifest {out_path}")


if __name__ == "__main__":
    main()
