"""run_hi63_union_scen_tr_sweep.py — 1029_0347(scen + transformer + union + no warmstart) 기준 하이퍼파라미터 스윕.

기준 run: model_lib/results/p1v2_runs/1029_0347_p1v2_hi63_scen_transformer_union_nowarm_seed42
  N_HI=63, assign=position_bin(scen), regression_model=transformer, probe_union_into_scen=True,
  drop_probe_from_regression=False, warmstart=None, FIXED_L0_NORM_CONSTANT=6,
  lambda_l0=0.000237, n2=20%/N=2, transformer M(d_head=128, n_layers=2, n_heads=4, d_ff=512).

한 번에 한 축만 바꾼다(나머지는 기준값). 기준과 똑같은 조합(n2=20%, size M)은 1029_0347을
그대로 기준으로 쓰므로 다시 돌리지 않는다(INCLUDE_BASELINE=True면 포함).

  그룹 0) 단일 데이터셋   : MIT, HUST, TJU 각각 (나머지는 기준값)          (3 run)
          TJU(NCM)는 FIXED_DATASET_GROUP="ncm"으로 `<축 폴더>_ncm`에 추출된 데이터를 쓴다.
          기존 _ncm 데이터는 2026-09-05 추출본이라 seg에 q_ref가 없는 옛 스키마 —
          스윕에서 처음 만나는 NCM run 하나만 Step 4 강제 재추출(FORCE_EXTRACT=True)한다.
  그룹 1) lambda_l0      : 1e-2, 1e-3, 1e-4                              (3 run)
  그룹 2) n2             : 10%, 30%  (N = 존 커버리지 100% 최소 개수)     (2 run)
          존 폭 = n1(0.35)이므로 N = ceil(n1 / n2) → 10%:4, 20%:2, 30%:2.
          n2-30%_N-2 데이터는 아직 없어 Step 4부터 추출한다(있으면 캐시 재사용).
  그룹 3) transformer 크기: S, L  (M = 기준)                              (2 run)
  그룹 4) (주석 처리) 1~3 중 최고 조합으로 MIT / HUST / TJU 단일 데이터셋 학습

실행: python run_hi63_union_scen_tr_sweep.py            # 그룹 0,1,2,3 전부
      python run_hi63_union_scen_tr_sweep.py 2 3        # 일부 그룹만
로그: model_lib/results/p1v2_runs/_hi63_union_scen_tr_sweep_logs/<tag>.log
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "parameters.py"
PYTHON = r"C:\Users\ksshin\.conda\envs\LFP_SOH_ESTIMATION\python.exe"
LOG_DIR = ROOT / "model_lib" / "results" / "p1v2_runs" / "_hi63_union_scen_tr_sweep_logs"

INCLUDE_BASELINE = False

N1 = 0.35
BASE = {
    "lambda_l0": 0.000237,
    "n2": 0.20,
    "size": "m",
    "datasets": ["MIT", "HUST"],
}
NCM_DATASETS = {"TJU", "CALCE"}
SINGLE_DATASETS = ("MIT", "HUST", "TJU")   # CALCE는 셀 13개라 6:2:2 분할 시 test 2~3셀 — 필요하면 추가
TR_SIZES = {
    # d_head는 transformer cap_head의 d_model로 쓰인다(cap_heads.py::build_cap_head).
    "s": {"d_head": 64, "tr_n_layers": 1, "tr_n_heads": 4, "tr_d_ff": 256},
    "m": {"d_head": 128, "tr_n_layers": 2, "tr_n_heads": 4, "tr_d_ff": 512},
    "l": {"d_head": 256, "tr_n_layers": 4, "tr_n_heads": 8, "tr_d_ff": 1024},
}


def n_samples_for(n2: float) -> int:
    """존(폭 n1) 하나를 길이 n2 세그먼트로 빈틈없이 덮는 최소 개수."""
    return math.ceil(round(N1 / n2, 6))


def _tag(cfg: dict) -> str:
    lam = f"{cfg['lambda_l0']:.0e}".replace("e-0", "e-")
    n2 = int(round(cfg["n2"] * 100))
    ds = "" if cfg["datasets"] == BASE["datasets"] else "_" + "-".join(d.lower() for d in cfg["datasets"])
    return f"hi63_scen_tr{cfg['size']}_union_nowarm_l0-{lam}_n2-{n2}{ds}"


def build_experiments(groups: list[int]) -> list[dict]:
    exps: list[dict] = []
    if INCLUDE_BASELINE:
        exps.append(dict(BASE))
    if 0 in groups:
        for ds in SINGLE_DATASETS:
            exps.append({**BASE, "datasets": [ds]})
    if 1 in groups:
        for lam in (1e-2, 1e-3, 1e-4):
            exps.append({**BASE, "lambda_l0": lam})
    if 2 in groups:
        for n2 in (0.10, 0.30):
            exps.append({**BASE, "n2": n2})
    if 3 in groups:
        for size in ("s", "l"):
            exps.append({**BASE, "size": size})
    # if 4 in groups:
    #     # 1~3 결과를 보고 BEST를 손으로 채운 뒤 주석 해제.
    #     # TJU(NCM) 데이터 경로/재추출은 그룹 0과 같은 방식으로 자동 처리된다.
    #     BEST = {**BASE, "lambda_l0": ..., "n2": ..., "size": ...}
    #     for ds in ("MIT", "HUST", "TJU"):
    #         exps.append({**BEST, "datasets": [ds]})
    for e in exps:
        e["tag"] = _tag(e)
    return exps


def _sub(text: str, pattern: str, repl: str, name: str, flags: int = re.M) -> str:
    text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    assert n == 1, f"{name} 라인을 못 찾음"
    return text


def data_group(cfg: dict) -> str:
    ncm = [d in NCM_DATASETS for d in cfg["datasets"]]
    if any(ncm) and not all(ncm):
        raise ValueError(f"LFP/NCM 혼합 학습은 지원 안 함: {cfg['datasets']}")
    return "ncm" if all(ncm) else "lfp"


def from_step_for(cfg: dict) -> int:
    # n2가 바뀌거나 NCM이면 Step 4(추출/캐시 확인)부터
    return 5 if cfg["n2"] == BASE["n2"] and data_group(cfg) == "lfp" else 4


def patch_params(text: str, cfg: dict) -> str:
    from_step = from_step_for(cfg)
    group = data_group(cfg)
    size = TR_SIZES[cfg["size"]]

    text = _sub(text, r"^ACTIVE_FROM_STEP = \d+", f"ACTIVE_FROM_STEP = {from_step}", "ACTIVE_FROM_STEP")
    text = _sub(text, r"^ACTIVE_FORCE_EXTRACT = (True|False)",
                f"ACTIVE_FORCE_EXTRACT = {bool(cfg.get('force_extract', False))}", "ACTIVE_FORCE_EXTRACT")
    text = _sub(text, r'^FIXED_DATASET_GROUP = "[a-z]+"', f'FIXED_DATASET_GROUP = "{group}"', "FIXED_DATASET_GROUP")
    if group != "lfp":
        # Step 4는 lfp가 아닌 그룹을 `<축 폴더>_<group>`에 저장한다(4_hi_analysis/logics.py
        # resolve_cache_and_axis_dir) — Step 5~9도 그 폴더를 읽게 맞춘다.
        text = _sub(text, r'^(_ACTIVE_AXIS_DIR = DATA_4_HI_ROOT / "q_frac_ref" / )_qfref_axis_tag\(ACTIVE_AXIS_CONFIG\)$',
                    rf'\1(_qfref_axis_tag(ACTIVE_AXIS_CONFIG) + "_{group}")', "_ACTIVE_AXIS_DIR")
    text = _sub(text, r"^ACTIVE_WARMSTART_BRANCH_EPOCH = \S+", "ACTIVE_WARMSTART_BRANCH_EPOCH = None",
                "ACTIVE_WARMSTART_BRANCH_EPOCH")
    text = _sub(text, r"^ACTIVE_SOC_REF_TO_REGRESSION = (True|False)", "ACTIVE_SOC_REF_TO_REGRESSION = False",
                "ACTIVE_SOC_REF_TO_REGRESSION")
    text = _sub(text, r"^ACTIVE_LAMBDA_L0_OVERRIDE = \S+", f"ACTIVE_LAMBDA_L0_OVERRIDE = {cfg['lambda_l0']!r}",
                "ACTIVE_LAMBDA_L0_OVERRIDE")
    text = _sub(text, r"^ACTIVE_N_HI = \d+", "ACTIVE_N_HI = 63", "ACTIVE_N_HI")
    text = _sub(text, r'^ACTIVE_P1_TAG = ".*?"', f'ACTIVE_P1_TAG = "{cfg["tag"]}"', "ACTIVE_P1_TAG")
    text = _sub(text, r"^FIXED_L0_NORM_CONSTANT = (None|\d+)", "FIXED_L0_NORM_CONSTANT = 6", "FIXED_L0_NORM_CONSTANT")
    text = _sub(text, r"^FIXED_CANONICAL_DATASETS = \[.*?\]",
                f"FIXED_CANONICAL_DATASETS = {json.dumps(cfg['datasets'])}", "FIXED_CANONICAL_DATASETS")

    # 축: scen(position_bin) + n2/N
    text = _sub(text, r'^(\s*)"n2": [0-9.]+,', rf'\g<1>"n2": {cfg["n2"]:.2f},', '"n2"')
    text = _sub(text, r'^(\s*)"n_samples": \d+,', rf'\g<1>"n_samples": {n_samples_for(cfg["n2"])},', '"n_samples"')
    text = _sub(text, r'^(\s*)#?\s*"assign": "[a-z_]+",.*$',
                r'\1# "assign": "none",           # 시나리오 라벨링: position_bin 기본값 사용 (union_scen_tr_sweep, scen)',
                '"assign"')

    # 모델: transformer + union, 크기
    text = _sub(text, r'("regression_model": )"[a-z_]+"(,)', r'\g<1>"transformer"\g<2>', '"regression_model"', 0)
    text = _sub(text, r'("drop_probe_from_regression": )(True|False)(,)', r"\g<1>False\g<3>",
                '"drop_probe_from_regression"', 0)
    text = _sub(text, r'("probe_union_into_scen": )(True|False)(,)', r"\g<1>True\g<3>",
                '"probe_union_into_scen"', 0)
    for key, val in size.items():
        text = _sub(text, rf'("{key}": )\d+(,)', rf"\g<1>{val}\g<2>", f'"{key}"', 0)
    return text


def main() -> None:
    groups = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3]
    exps = build_experiments(groups)
    original = PARAMS_PATH.read_text(encoding="utf-8")
    for e in exps:   # 실행 전에 패치가 전부 들어맞는지 먼저 확인
        patch_params(original, e)

    first_ncm = next((e for e in exps if data_group(e) == "ncm"), None)
    if first_ncm is not None:
        first_ncm["force_extract"] = True   # 옛 스키마 _ncm 데이터 갱신(1회)
    print(f"그룹 {groups}: {len(exps)}개 실험")
    for e in exps:
        print(f"  - {e['tag']}  (N={n_samples_for(e['n2'])}, size={TR_SIZES[e['size']]})")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for i, e in enumerate(exps, 1):
            from_step = from_step_for(e)
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(exps)}] {e['tag']}  (lambda_l0={e['lambda_l0']}, n2={e['n2']}, "
                  f"N={n_samples_for(e['n2'])}, size={e['size']}, datasets={e['datasets']}, from_step={from_step}, "
                  f"force_extract={bool(e.get('force_extract', False))})")
            print("=" * 70)

            PARAMS_PATH.write_text(patch_params(original, e), encoding="utf-8")

            log_path = LOG_DIR / f"{e['tag']}.log"
            t0 = time.time()
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            with open(log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    [PYTHON, str(ROOT / "run_pipeline.py"), str(from_step)],
                    cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=env,
                )
            ok = proc.returncode == 0
            print(f"  -> {'OK' if ok else f'FAIL(exit={proc.returncode})'}, "
                  f"{(time.time() - t0) / 60:.1f}분, 로그: {log_path}")
            if not ok:
                print(f"  !! {e['tag']} 실패 — 스윕 중단. 로그를 먼저 확인할 것: {log_path}")
                sys.exit(1)
    finally:
        PARAMS_PATH.write_text(original, encoding="utf-8")
        print("\nparameters.py 원복 완료(스윕 시작 전 상태로 되돌림).")


if __name__ == "__main__":
    main()
