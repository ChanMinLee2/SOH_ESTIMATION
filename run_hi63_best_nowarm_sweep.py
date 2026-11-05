"""run_hi63_best_nowarm_sweep.py — 261102 스윕 최종 설정으로 scen vs noscen을 nowarm에서 비교.

최종 설정(docs/261102_RESULTS.md): lambda_l0=2.37e-4, n2=20%/N=2, transformer M, union probe,
FIXED_L0_NORM_CONSTANT=6, MIT+HUST, warmstart 없음(ACTIVE_WARMSTART_BRANCH_EPOCH=None).

  1) scen   + nowarm : 같은 설정의 기존 run = 1029_0347
  2) noscen + nowarm : 같은 설정의 기존 run = 1029_1119

패치 로직은 run_hi63_union_scen_tr_sweep.py::patch_params를 재사용하고 noscen이면 assign 한 줄만 덮어쓴다.

실행: python run_hi63_best_nowarm_sweep.py           # 둘 다
      python run_hi63_best_nowarm_sweep.py scen      # 하나만 (scen | noscen)
로그: model_lib/results/p1v2_runs/_hi63_best_nowarm_logs/<tag>.log
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import run_hi63_union_scen_tr_sweep as base

ROOT = base.ROOT
PARAMS_PATH = base.PARAMS_PATH
PYTHON = base.PYTHON
LOG_DIR = ROOT / "model_lib" / "results" / "p1v2_runs" / "_hi63_best_nowarm_logs"

BEST = dict(base.BASE)   # lambda_l0=0.000237, n2=0.20, size=m, datasets=[MIT, HUST]
EXPERIMENTS = {
    "scen": "hi63_scen_trm_union_nowarm_l0-2e-4_n2-20",
    "noscen": "hi63_noscen_trm_union_nowarm_l0-2e-4_n2-20",
}


def patch_params(text: str, mode: str) -> str:
    text = base.patch_params(text, {**BEST, "tag": EXPERIMENTS[mode]})   # warmstart는 여기서 None으로 고정됨
    if mode == "noscen":
        text = base._sub(text, r'^(\s*)# "assign": "none",.*$',
                         r'\1"assign": "none",           # 시나리오 라벨링: noscen (best_nowarm_sweep)', '"assign"')
    return text


def main() -> None:
    modes = sys.argv[1:] or list(EXPERIMENTS)
    unknown = [m for m in modes if m not in EXPERIMENTS]
    assert not unknown, f"알 수 없는 모드 {unknown} — scen | noscen"
    original = PARAMS_PATH.read_text(encoding="utf-8")
    for m in modes:   # 실행 전에 패치가 전부 들어맞는지 먼저 확인
        patch_params(original, m)

    print(f"{len(modes)}개 실험 (warmstart=None, lambda_l0={BEST['lambda_l0']}, n2={BEST['n2']}, "
          f"size={BEST['size']}, datasets={BEST['datasets']})")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for i, m in enumerate(modes, 1):
            tag = EXPERIMENTS[m]
            print(f"\n{'=' * 70}\n[{i}/{len(modes)}] {tag}  (from_step=5)\n{'=' * 70}")
            PARAMS_PATH.write_text(patch_params(original, m), encoding="utf-8")

            log_path = LOG_DIR / f"{tag}.log"
            t0 = time.time()
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    [PYTHON, str(ROOT / "run_pipeline.py"), "5"],
                    cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=env,
                )
            ok = proc.returncode == 0
            print(f"  -> {'OK' if ok else f'FAIL(exit={proc.returncode})'}, "
                  f"{(time.time() - t0) / 60:.1f}분, 로그: {log_path}")
            if not ok:
                print(f"  !! {tag} 실패 — 중단. 로그를 먼저 확인할 것: {log_path}")
                sys.exit(1)
    finally:
        PARAMS_PATH.write_text(original, encoding="utf-8")
        print("\nparameters.py 원복 완료(실행 전 상태로 되돌림).")


if __name__ == "__main__":
    main()
