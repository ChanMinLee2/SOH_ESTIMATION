"""run_socref_sweep.py — ACTIVE_SOC_REF_TO_REGRESSION 2-way(scen/noscen) 스윕 드라이버.

n2=20%/n_samples=2(캐노니컬 축) 고정, soc_ref_to_regression=True 고정, scen(assign
기본값=position_bin) vs noscen(assign="none")만 바꿔 순차 실행한다. 그 외 설정은
1015_1618_p1v2_baseline_no_probe_x_for_reg_seed42 베이스라인과 동일(N_HI=64,
drop_probe_from_regression=True, regression_model=mlp, lambda_l0 override 0.000237,
warmstart_branch_epoch=None — 2026-10-20 확인: n2sweep_noscen_n2-10_ns4 run과 n2/
n_samples 축만 다르고 나머지 전부 일치).

**둘 다 Step4 재추출 필요**: 캐노니컬 n2=20% 축의 scen/noscen 디렉터리 둘 다
q_ref 컬럼이 추가되기 전(2026-09) 캐시라 q_ref가 없다(2026-10-20 확인) — 그래서
이 스윕은 두 실험 모두 from_step=4 + force_extract=True로 돌린다(n2_sweep/reghead_sweep과
달리 캐시 재사용 불가). 실험당 Step4(~2시간 추정) + 학습(~8시간, 이전 run들 기준)이라
전체 스윕이 상당히 오래 걸린다.

실행: python run_socref_sweep.py
로그: model_lib/results/p1v2_runs/_socref_sweep_logs/<tag>.log
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
LOG_DIR = ROOT / "model_lib" / "results" / "p1v2_runs" / "_socref_sweep_logs"

# (tag, assign)  assign: None=scen(position_bin, 기본값 유지) | "none"=noscen
EXPERIMENTS: list[tuple[str, str | None]] = [
    ("socref_scen", None),
    ("socref_noscen", "none"),
]


def patch_params(text: str, tag: str, assign: str | None) -> str:
    text, n = re.subn(r"^ACTIVE_FROM_STEP = \d+", "ACTIVE_FROM_STEP = 4", text, count=1, flags=re.M)
    assert n == 1, "ACTIVE_FROM_STEP 라인을 못 찾음"

    text, n = re.subn(
        r"^ACTIVE_FORCE_EXTRACT = (True|False)",
        "ACTIVE_FORCE_EXTRACT = True",
        text, count=1, flags=re.M,
    )
    assert n == 1, "ACTIVE_FORCE_EXTRACT 라인을 못 찾음"

    text, n = re.subn(
        r"^ACTIVE_WARMSTART_BRANCH_EPOCH = .+",
        "ACTIVE_WARMSTART_BRANCH_EPOCH = None",
        text, count=1, flags=re.M,
    )
    assert n == 1, "ACTIVE_WARMSTART_BRANCH_EPOCH 라인을 못 찾음"

    text, n = re.subn(
        r"^ACTIVE_SOC_REF_TO_REGRESSION = (True|False)",
        "ACTIVE_SOC_REF_TO_REGRESSION = True",
        text, count=1, flags=re.M,
    )
    assert n == 1, "ACTIVE_SOC_REF_TO_REGRESSION 라인을 못 찾음"

    text, n = re.subn(r'^ACTIVE_P1_TAG = ".*"', f'ACTIVE_P1_TAG = "{tag}"', text, count=1, flags=re.M)
    assert n == 1, "ACTIVE_P1_TAG 라인을 못 찾음"

    assign_re = re.compile(r'^(\s*)#?\s*"assign": "[a-z_]+",.*$', re.M)
    if assign == "none":
        replacement = r'\1"assign": "none",           # 시나리오 라벨링: noscen (socref_sweep)'
    else:
        replacement = r'\1# "assign": "none",           # 시나리오 라벨링: position_bin 기본값 사용 (socref_sweep, scen)'
    text, n = assign_re.subn(replacement, text, count=1)
    assert n == 1, '"assign" 라인을 못 찾음'

    return text


def main() -> None:
    original = PARAMS_PATH.read_text(encoding="utf-8")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for i, (tag, assign) in enumerate(EXPERIMENTS, 1):
            label = assign or "position_bin(scen)"
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(EXPERIMENTS)}] {tag}  (assign={label}, n2=0.20, n_samples=2, "
                  f"soc_ref_to_regression=True)")
            print("=" * 70)

            patched = patch_params(original, tag, assign)
            PARAMS_PATH.write_text(patched, encoding="utf-8")

            log_path = LOG_DIR / f"{tag}.log"
            t0 = time.time()
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    [PYTHON, str(ROOT / "run_pipeline.py"), "4"],
                    cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=env,
                )
            elapsed = time.time() - t0
            ok = proc.returncode == 0
            print(f"  -> {'OK' if ok else f'FAIL(exit={proc.returncode})'}, "
                  f"{elapsed / 60:.1f}분, 로그: {log_path}")
            if not ok:
                print(f"  !! {tag} 실패 — 스윕 중단. 로그를 먼저 확인할 것: {log_path}")
                sys.exit(1)
    finally:
        PARAMS_PATH.write_text(original, encoding="utf-8")
        print("\nparameters.py 원복 완료(스윕 시작 전 상태로 되돌림).")


if __name__ == "__main__":
    main()
