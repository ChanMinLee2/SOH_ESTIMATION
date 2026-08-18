"""
5_model/experiments/phase1_lab/run_head_comparison.py

Phase1 cap_head 아키텍처 sanity check — "게이트 선택이 최종 회귀 헤드와 무관해야
한다"는 기존 design intent(train_scr.py는 Phase1을 항상 MLP로 강제)가 실제로
맞는지, 같은 seed에서 헤드 종류만 바꿔가며 게이트 랭킹이 얼마나 달라지는지 본다.

run_convergence_seeds.py(시드 축)와 같은 패턴이지만 이건 "헤드 종류" 축이다 —
seed/split-seed는 전부 고정하고 --regression-model만 바꿔가며
phase1_trainer_v2.py(--regression-model 오버라이드 지원)를 반복 호출한다.
결과 manifest는 analyze_convergence.py가 그대로 먹는다(그 스크립트는 manifest의
키가 "시드"인지 "헤드이름"인지 상관하지 않고 그룹 라벨로만 씀 — 코드 재사용).

전체 Stage0-6 로드맵(seed 5개 x 트레이너 2종)에 헤드 5종을 곱하면 5배로
불어나므로, 이 스크립트는 의도적으로 "seed 1개 x 헤드 N개"의 값싼 사전 점검
용도다 — 여기서 차이가 크게 나면 그때 더 넓게(여러 seed) 재검증한다.

사용 예:
  python 5_model/experiments/phase1_lab/run_head_comparison.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --scen-k 25 --seed 42 --heads mlp resnet_tab transformer \
      --max-epochs 300 --patience 60 --beta-min 0.1 --tag head_sanity_k25

  # 완료 후:
  python 5_model/experiments/phase1_lab/analyze_convergence.py \
      --manifest 5_model/experiments/phase1_lab/results/convergence_manifest_head_sanity_k25.json \
      --k-values 5 15 25
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

_P1V2_RUNS_DIR = RESULTS_DIR / "p1v2_runs"  # phase1_trainer_v2.py의 실제 출력 위치와 일치시킴

_VALID_HEADS = ["mlp", "transformer", "i_transformer", "resnet_tab", "ft_transformer"]


def _snapshot_p1v2_dirs() -> set[Path]:
    if not _P1V2_RUNS_DIR.exists():
        return set()
    return {p for p in _P1V2_RUNS_DIR.iterdir() if p.is_dir()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 cap_head 종류별 게이트 랭킹 sanity check (단일 seed)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True)
    p.add_argument("--charge-m", type=int, default=None)
    p.add_argument("--discharge-m", type=int, default=None)
    p.add_argument("--scen-k", type=int, default=None)
    p.add_argument("--data-dir", default=None, help="cfg에 없으면 필수 (cycle pkl 경로)")
    p.add_argument("--seg-data-dir", default=None, help="cfg에 없으면 필수 (seg pkl 경로)")
    p.add_argument("--seed", type=int, required=True, help="모든 헤드에 공통으로 쓸 고정 seed")
    p.add_argument("--split-seed", type=int, default=None, help="기본: --seed와 동일")
    p.add_argument("--heads", nargs="+", default=["mlp", "resnet_tab", "transformer"],
                   choices=_VALID_HEADS, help="비교할 cap_head 종류 목록 (기본: mlp/resnet_tab/transformer)")
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--parallel", type=int, default=1,
                   help="동시 실행 수 (기본 1=순차, GPU 1개면 1 권장)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _build_cmd(head: str, args: argparse.Namespace) -> list[str]:
    split_seed = args.split_seed if args.split_seed is not None else args.seed
    cmd = [
        sys.executable, str(LAB_DIR / "phase1_trainer_v2.py"),
        "--model-config", args.model_config,
        "--seg-axis", args.seg_axis,
        "--axis-config", args.axis_config,
        "--seed", str(args.seed),
        "--split-seed", str(split_seed),
        "--regression-model", head,
        "--beta-min", str(args.beta_min),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--tag", f"{args.tag}_{head}",
    ]
    if args.data_dir is not None:
        cmd += ["--data-dir", args.data_dir]
    if args.seg_data_dir is not None:
        cmd += ["--seg-data-dir", args.seg_data_dir]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.charge_m is not None:
        cmd += ["--charge-m", str(args.charge_m)]
    if args.discharge_m is not None:
        cmd += ["--discharge-m", str(args.discharge_m)]
    if args.scen_k is not None:
        cmd += ["--scen-k", str(args.scen_k)]
    return cmd


def _run_one_head(head: str, args: argparse.Namespace) -> dict:
    """capture_output을 안 쓰고 자식 프로세스가 터미널에 직접 쓰게 둔다 —
    phase1_trainer_v2.py 내부의 trange(에폭별 진행률/ETA)가 그대로 실시간으로 보인다
    (캡처하면 프로세스가 끝나야 한꺼번에 출력되어 tqdm 의미가 없어짐).
    run_dir은 stdout 파싱 대신 디렉터리 스냅샷 비교로 찾는다(--parallel>1일 때도
    태그+헤드+seed로 유일하게 매칭되므로 안전)."""
    cmd = _build_cmd(head, args)
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    before = _snapshot_p1v2_dirs()
    marker = f"_{args.tag}_{head}_seed{args.seed}"
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=child_env)
    elapsed = time.time() - t0

    out = {"head": head, "cmd": cmd, "elapsed": elapsed, "ok": result.returncode == 0, "run_dir": None}
    if not out["ok"]:
        return out

    new_dirs = [d for d in (_snapshot_p1v2_dirs() - before) if marker in d.name]
    if new_dirs:
        out["run_dir"] = str(max(new_dirs, key=lambda p: p.stat().st_mtime))
    return out


def main() -> None:
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for head in args.heads:
            print(f"$ {' '.join(_build_cmd(head, args))}")
        return

    manifest: dict = {
        "tag": args.tag, "axis": "regression_model_comparison",
        "seed": args.seed, "split_seed": args.split_seed or args.seed,
        "heads": args.heads, "runs": {},
    }

    n_workers = max(1, args.parallel)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_one_head, head, args): head for head in args.heads}
        with tqdm(total=len(futs), desc="[head-comparison] heads", unit="run") as pbar:
            for fut in as_completed(futs):
                head = futs[fut]
                r = fut.result()
                results.append(r)
                if r["ok"] and r["run_dir"]:
                    tqdm_write(f"  head={head:<14}  OK  ({r['elapsed']:.0f}s)  -> {r['run_dir']}")
                elif r["ok"]:
                    tqdm_write(f"  head={head:<14}  완료했지만 run_dir을 못 찾음 "
                               f"({r['elapsed']:.0f}s) — results/p1v2_runs/ 수동 확인 필요")
                else:
                    tqdm_write(f"  head={head:<14}  실패(exit != 0) — 위 실시간 출력의 에러 메시지 참고")
                pbar.update(1)

    for r in sorted(results, key=lambda x: args.heads.index(x["head"])):
        if r["ok"] and r["run_dir"]:
            manifest["runs"][r["head"]] = r["run_dir"]

    n_ok = len(manifest["runs"])
    print(f"\n[head-comparison] {n_ok}/{len(args.heads)}개 헤드 성공")

    out_path = RESULTS_DIR / f"convergence_manifest_{args.tag}.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[head-comparison] manifest 저장: {out_path}")
    print("[head-comparison] 다음 단계 (헤드간 Jaccard/Kendall 비교, analyze_convergence.py 재사용):")
    print(f"  python 5_model/experiments/phase1_lab/analyze_convergence.py --manifest {out_path} --k-values 5 15 25")
    print("  (주의: 이 리포트의 'seed' 표기는 실제로는 헤드 이름입니다 — 스크립트가 라벨을 그대로 재사용할 뿐)")


if __name__ == "__main__":
    main()
