"""
5_model/experiments/phase1_lab/run_convergence_seeds.py

Phase1 랭킹 시드-수렴성 검증 — 1단계(오케스트레이션).

기존 5_model/train_scr.py(Stage0 baseline)나 이 랩의 phase1_trainer_v2.py
(Stage1+2 적용본)를 수정하지 않고, 그대로 subprocess로 여러 seed에 대해
반복 호출한다. 각 실행이 만든 run_dir을 stdout에서 파싱해 모아 manifest json으로
저장한다 — analyze_convergence.py가 이 manifest를 읽어 Jaccard/Kendall/선택빈도를
계산하고, materialize_ensemble_gates.py가 같은 manifest로 앙상블 gates를 만든다.

병렬화: --parallel N (기본 1=순차). ThreadPoolExecutor로 N개 subprocess를 동시에
띄운다 — 학습 자체는 별도 프로세스라 GIL 영향 없음. **GPU 1개로 학습하는 경우
--parallel을 2 이상으로 올리면 CUDA 메모리 경합/OOM 위험이 있으니, GPU 환경에서는
기본값(1) 유지를 권장**(CPU 전용이거나 GPU 메모리가 넉넉하면 올려도 됨).

기존 코드 무변경 원칙: train_scr.py/phase1_trainer_v2.py/run_pipeline.py는 전혀
건드리지 않고 "여러 번 실행 + 결과 수집"만 이 파일이 담당한다.

사용 예:
  # Stage0 baseline (기존 train_scr.py 그대로)
  python 5_model/experiments/phase1_lab/run_convergence_seeds.py \
      --trainer baseline \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --scen-k 25 --seeds 42 0 123 7 2024 --tag k25_baseline --parallel 1

  # Stage1+2 (phase1_trainer_v2.py — 체크포인트기준 변경 + temperature annealing)
  python 5_model/experiments/phase1_lab/run_convergence_seeds.py \
      --trainer v2 --beta-min 0.1 \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --scen-k 25 --seeds 42 0 123 7 2024 --tag k25_stage12 --parallel 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LAB_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LAB_DIR / "results"

sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
from utils.tqdm_utils import tqdm, write as tqdm_write  # noqa: E402

# train_scr.py(baseline)/phase1_trainer_v2.py(v2) 각각의 실제 run 출력 위치.
# run_dir 발견을 stdout 파싱이 아니라 이 디렉터리들의 스냅샷 비교로 하므로
# (아래 _run_one_seed 참고) 이 경로들이 실제 저장 위치와 일치해야 한다.
_BASELINE_OUTPUT_ROOT = PROJECT_ROOT / "_5_data_model_scr"
_V2_OUTPUT_ROOT = LAB_DIR / "results" / "p1v2_runs"


def _snapshot_dirs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.iterdir() if p.is_dir()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 시드-수렴성 검증용 다중 seed 실행 오케스트레이터")
    p.add_argument("--trainer", choices=["baseline", "v2"], default="baseline",
                   help="baseline=기존 train_scr.py --phase 1 그대로 / v2=phase1_trainer_v2.py(Stage1+2 적용본)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True, help="JSON 문자열")
    p.add_argument("--charge-m", type=int, default=None)
    p.add_argument("--discharge-m", type=int, default=None)
    p.add_argument("--scen-k", type=int, default=None)
    p.add_argument("--data-dir", default=None, help="--trainer v2 전용 (cfg에 없으면 필수)")
    p.add_argument("--seg-data-dir", default=None, help="--trainer v2 전용 (cfg에 없으면 필수)")
    p.add_argument("--beta-min", type=float, default=0.1, help="--trainer v2 전용")
    p.add_argument("--max-epochs", type=int, default=None, help="--trainer v2 전용 (phase1_trainer_v2.py --max-epochs)")
    p.add_argument("--patience", type=int, default=None, help="--trainer v2 전용 (phase1_trainer_v2.py --patience)")
    p.add_argument("--batch-size", type=int, default=None, help="--trainer v2 전용 (phase1_trainer_v2.py --batch-size)")
    p.add_argument("--seeds", type=int, nargs="+", required=True,
                   help="예: --seeds 42 0 123 7 2024  (split-seed도 동일 값 사용)")
    p.add_argument("--tag", required=True, help="manifest 파일명 구분용 태그")
    p.add_argument("--parallel", type=int, default=1,
                   help="동시 실행 subprocess 수 (기본 1=순차, GPU 학습이면 1 권장)")
    p.add_argument("--dry-run", action="store_true", help="명령어만 출력하고 실행하지 않음")
    return p.parse_args()


def _build_cmd(seed: int, args: argparse.Namespace) -> list[str]:
    if args.trainer == "baseline":
        cmd = [
            sys.executable, str(PROJECT_ROOT / "5_model" / "train_scr.py"),
            "--phase", "1",
            "--config", args.model_config,
            "--seg-axis", args.seg_axis,
            "--axis-config", args.axis_config,
            "--seed", str(seed),
            "--split-seed", str(seed),
        ]
    else:  # v2
        cmd = [
            sys.executable, str(LAB_DIR / "phase1_trainer_v2.py"),
            "--model-config", args.model_config,
            "--seg-axis", args.seg_axis,
            "--axis-config", args.axis_config,
            "--seed", str(seed),
            "--split-seed", str(seed),
            "--beta-min", str(args.beta_min),
            "--tag", args.tag,
        ]
        if args.data_dir is not None:
            cmd += ["--data-dir", args.data_dir]
        if args.seg_data_dir is not None:
            cmd += ["--seg-data-dir", args.seg_data_dir]
        if args.max_epochs is not None:
            cmd += ["--max-epochs", str(args.max_epochs)]
        if args.patience is not None:
            cmd += ["--patience", str(args.patience)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
    if args.charge_m is not None:
        cmd += ["--charge-m", str(args.charge_m)]
    if args.discharge_m is not None:
        cmd += ["--discharge-m", str(args.discharge_m)]
    if args.scen_k is not None:
        cmd += ["--scen-k", str(args.scen_k)]
    return cmd


def _run_one_seed(seed: int, args: argparse.Namespace) -> dict:
    """capture_output을 쓰지 않고 자식 프로세스가 터미널에 직접 쓰게 둔다 —
    train_scr.py/phase1_trainer_v2.py 내부의 tqdm(에폭별 진행률/ETA)이 그대로
    실시간으로 보인다(캡처하면 프로세스가 끝나야 한꺼번에 출력되어 tqdm이 무의미해짐).
    run_dir은 stdout 파싱 대신 디렉터리 스냅샷 비교로 찾는다."""
    cmd = _build_cmd(seed, args)

    # PYTHONIOENCODING: 콘솔 코드페이지(한국어 Windows cp949) UnicodeEncodeError 방지.
    # capture_output을 안 쓰는 지금은 자식이 이 터미널에 직접 쓰므로 보통 불필요하지만,
    # 혹시 자식 내부에서 별도로 파일/파이프에 쓰는 경로가 있을 수 있어 방어적으로 유지.
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    root = _V2_OUTPUT_ROOT if args.trainer == "v2" else _BASELINE_OUTPUT_ROOT
    before = _snapshot_dirs(root)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=child_env)
    elapsed = time.time() - t0

    out = {"seed": seed, "cmd": cmd, "elapsed": elapsed, "ok": result.returncode == 0, "run_dir": None}
    if not out["ok"]:
        return out

    new_dirs = _snapshot_dirs(root) - before
    if args.trainer == "v2":
        marker = f"_{args.tag}_seed{seed}"
        new_dirs = {d for d in new_dirs if marker in d.name}
    else:
        # train_scr.py 디렉터리명엔 seed가 안 박히므로 Phase1 패턴(_p1_)으로만 거름
        # (run_pipeline.py._RUN_DIR_RE와 동일 관례) — 동시 실행(--parallel>1) 시
        # 여러 seed가 동시에 새 폴더를 만들면 구분이 안 되니 baseline은 --parallel 1 권장.
        new_dirs = {d for d in new_dirs if "_p1_" in d.name}
    if new_dirs:
        out["run_dir"] = str(max(new_dirs, key=lambda p: p.stat().st_mtime))
    return out


def main() -> None:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for seed in args.seeds:
            print(f"$ {' '.join(_build_cmd(seed, args))}")
        return

    manifest: dict = {
        "tag": args.tag, "trainer": args.trainer,
        "model_config": args.model_config, "seg_axis": args.seg_axis,
        "axis_config": json.loads(args.axis_config), "scen_k": args.scen_k,
        "seeds": args.seeds, "runs": {},
    }

    n_workers = max(1, args.parallel)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_one_seed, seed, args): seed for seed in args.seeds}
        with tqdm(total=len(futs), desc=f"[convergence:{args.trainer}] seeds", unit="run") as pbar:
            for fut in as_completed(futs):
                seed = futs[fut]
                r = fut.result()
                results.append(r)
                if r["ok"] and r["run_dir"]:
                    tqdm_write(f"  seed={seed:>6}  OK  ({r['elapsed']:.0f}s)  -> {r['run_dir']}")
                elif r["ok"]:
                    tqdm_write(f"  seed={seed:>6}  완료했지만 run dir을 못 찾음 "
                               f"({r['elapsed']:.0f}s) — 결과 디렉터리 수동 확인 필요")
                else:
                    tqdm_write(f"  seed={seed:>6}  실패(exit != 0) — 위 실시간 출력의 에러 메시지 참고")
                pbar.update(1)

    for r in sorted(results, key=lambda x: x["seed"]):
        if r["ok"] and r["run_dir"]:
            manifest["runs"][str(r["seed"])] = r["run_dir"]

    n_ok = len(manifest["runs"])
    print(f"\n[convergence] {n_ok}/{len(args.seeds)}개 seed 성공")

    out_path = RESULTS_DIR / f"convergence_manifest_{args.tag}.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[convergence] manifest 저장: {out_path}")
    print(f"[convergence] 다음 단계: python 5_model/experiments/phase1_lab/analyze_convergence.py --manifest {out_path}")


if __name__ == "__main__":
    main()
