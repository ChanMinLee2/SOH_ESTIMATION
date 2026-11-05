"""run_hi63_union_nowarm_sweep.py — probe_union_into_scen=True, scen/noscen x mlp/transformer 2x2 스윕.

docs/261020_RESULTS.md #5의 가족 C(probe 회귀 포함 + warmstart 없음, `1023_2204`/`1024_0347`/
`1024_0643`/`1024_1522`)와 같은 조건에서, probe_x를 이어붙이는 대신 회귀 raw HI 블록을
x·OR(z_probe, z_scen)·redundancy_mask 한 덩어리(N_HI칸)로 넣는 union 모드만 바꿔 비교한다.

공통: ACTIVE_N_HI=63, ACTIVE_WARMSTART_BRANCH_EPOCH=None, FIXED_L0_NORM_CONSTANT=6,
drop_probe_from_regression=False, probe_union_into_scen=True, from_step=5(Step4 재추출 불필요).
바뀌는 축: assign(scen/noscen) x regression_model(mlp/transformer).

실행: python run_hi63_union_nowarm_sweep.py
로그: model_lib/results/p1v2_runs/_hi63_union_nowarm_sweep_logs/<tag>.log
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "parameters.py"
PYTHON = r"C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe"
LOG_DIR = ROOT / "model_lib" / "results" / "p1v2_runs" / "_hi63_union_nowarm_sweep_logs"

# (tag, assign, regression_model)  assign: None=scen(position_bin 기본) | "none"=noscen
EXPERIMENTS: list[tuple[str, str | None, str]] = [
    ("hi63_scen_mlp_union_nowarm", None, "mlp"),
    ("hi63_noscen_mlp_union_nowarm", "none", "mlp"),
    ("hi63_scen_transformer_union_nowarm", None, "transformer"),
    ("hi63_noscen_transformer_union_nowarm", "none", "transformer"),
]


def _sub(text: str, pattern: str, repl: str, name: str, flags: int = re.M) -> str:
    text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    assert n == 1, f"{name} 라인을 못 찾음"
    return text


def patch_params(text: str, tag: str, assign: str | None, regression_model: str) -> str:
    text = _sub(text, r"^ACTIVE_FROM_STEP = \d+", "ACTIVE_FROM_STEP = 5", "ACTIVE_FROM_STEP")
    text = _sub(text, r"^ACTIVE_FORCE_EXTRACT = (True|False)", "ACTIVE_FORCE_EXTRACT = False", "ACTIVE_FORCE_EXTRACT")
    text = _sub(text, r"^ACTIVE_WARMSTART_BRANCH_EPOCH = \S+", "ACTIVE_WARMSTART_BRANCH_EPOCH = None",
                "ACTIVE_WARMSTART_BRANCH_EPOCH")
    text = _sub(text, r"^ACTIVE_SOC_REF_TO_REGRESSION = (True|False)", "ACTIVE_SOC_REF_TO_REGRESSION = False",
                "ACTIVE_SOC_REF_TO_REGRESSION")
    text = _sub(text, r"^ACTIVE_N_HI = \d+", "ACTIVE_N_HI = 63", "ACTIVE_N_HI")
    text = _sub(text, r'^ACTIVE_P1_TAG = ".*?"', f'ACTIVE_P1_TAG = "{tag}"', "ACTIVE_P1_TAG")
    text = _sub(text, r"^FIXED_L0_NORM_CONSTANT = (None|\d+)", "FIXED_L0_NORM_CONSTANT = 6", "FIXED_L0_NORM_CONSTANT")

    if assign == "none":
        repl = r'\1"assign": "none",           # 시나리오 라벨링: noscen (hi63_union_nowarm_sweep)'
    else:
        repl = r'\1# "assign": "none",           # 시나리오 라벨링: position_bin 기본값 사용 (hi63_union_nowarm_sweep, scen)'
    text = _sub(text, r'^(\s*)#?\s*"assign": "[a-z_]+",.*$', repl, '"assign"')

    text = _sub(text, r'("regression_model": )"[a-z_]+"(,)', rf'\g<1>"{regression_model}"\g<2>', '"regression_model"', 0)
    text = _sub(text, r'("drop_probe_from_regression": )(True|False)(,)', r"\g<1>False\g<3>",
                '"drop_probe_from_regression"', 0)
    text = _sub(text, r'("probe_union_into_scen": )(True|False)(,)', r"\g<1>True\g<3>",
                '"probe_union_into_scen"', 0)
    return text


def main() -> None:
    original = PARAMS_PATH.read_text(encoding="utf-8")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for i, (tag, assign, regression_model) in enumerate(EXPERIMENTS, 1):
            label = assign or "position_bin(scen)"
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(EXPERIMENTS)}] {tag}  (assign={label}, regression_model={regression_model}, "
                  f"N_HI=63, warmstart=None, probe_union_into_scen=True)")
            print("=" * 70)

            PARAMS_PATH.write_text(patch_params(original, tag, assign, regression_model), encoding="utf-8")

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
                print(f"  !! {tag} 실패 — 스윕 중단. 로그를 먼저 확인할 것: {log_path}")
                sys.exit(1)
    finally:
        PARAMS_PATH.write_text(original, encoding="utf-8")
        print("\nparameters.py 원복 완료(스윕 시작 전 상태로 되돌림).")


if __name__ == "__main__":
    main()
