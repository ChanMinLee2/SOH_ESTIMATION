"""run_reghead_sweep.py — regression_model(회귀 헤드) 3-way 스윕 드라이버.

n2 sweep에서 고정축으로 쓴 n2=20%/n_samples=2(캐노니컬 축, assign=position_bin 기본)를
그대로 두고, `P1_MODEL_CONFIG["model"]["regression_model"]`만 transformer/
i_transformer(inverted transformer)/resnet_tab 3가지로 바꿔 순차 실행한다. 다른 설정은
`1016_1915_p1v2_warmstart_100_no_probe_x_seed42`(현재 확정된 최고 설정 — N_HI=64,
drop_probe_from_regression=True, warmstart_branch_epoch=100: noscen 대비 oracle
+0.46pp/hard +0.45pp로 scen이 명확히 이김, 2026-10-21 확인)와 동일하게 맞춘다.
warmstart_branch_epoch=200은 l0_fully_ramped_epoch(~150, warmup 50+ramp 100) 이후라
게이트 포화 후 분기로 oracle R²가 -3.4pp 붕괴하는 게 확인됐으므로(`1011_1035`) 반드시
150 이전 값(100)을 써야 한다.

축이 캐노니컬 그대로라 Step4 추출 캐시가 이미 있음 — n2_sweep과 달리 from_step=5로
충분하다(재추출 없음, Step4의 ~2시간을 아낀다).

**drop_probe_from_regression=True, 3개 전부 동일하게 적용(2026-10-21)**: `cap_heads.py`가
transformer/i_transformer도 지원하도록 확장돼(TransformerHead는 probe 토큰 자체를 안
만들고, ITransformerHead는 n_feat만 N_HI만큼 줄이는 식 — 스모크 테스트로 파라미터 수
감소까지 확인함), 이제 3개 실험 전부 베이스라인(`1015_1618_...`)과 "회귀 헤드 종류"만
다르고 나머지는 완전히 동일한 조건으로 비교 가능하다. ft_transformer는 probe+scen을
하나의 연속 sparse-mask 블록으로 구조에 박아둬서 여전히 미지원(이번 스윕 대상 아님).

실행: python run_reghead_sweep.py
로그: model_lib/results/p1v2_runs/_reghead_sweep_logs/<tag>.log
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
LOG_DIR = ROOT / "model_lib" / "results" / "p1v2_runs" / "_reghead_sweep_logs"

# (tag, regression_model, drop_probe_from_regression) — 전부 True(베이스라인과 동일 조건)
EXPERIMENTS: list[tuple[str, str, bool]] = [
    ("reghead_transformer", "transformer", True),
    ("reghead_itransformer", "i_transformer", True),
    ("reghead_resnettab", "resnet_tab", True),
]


def patch_params(text: str, tag: str, regression_model: str, drop_probe: bool) -> str:
    text, n = re.subn(r"^ACTIVE_FROM_STEP = \d+", "ACTIVE_FROM_STEP = 5", text, count=1, flags=re.M)
    assert n == 1, "ACTIVE_FROM_STEP 라인을 못 찾음"

    # socref_sweep 등 이전 스윕이 남겨둔 상태(ACTIVE_FORCE_EXTRACT=True/
    # ACTIVE_SOC_REF_TO_REGRESSION=True)가 그대로 섞여 들어가지 않도록 이 스윕 전용
    # 값으로 명시적으로 고정한다 — 캐노니컬 축은 이미 추출돼 있어 force 불필요하고,
    # 회귀 헤드 비교에 soc_ref가 끼면 1015_1618 베이스라인과 더 이상 비교가 안 된다.
    text, n = re.subn(
        r"^ACTIVE_FORCE_EXTRACT = (True|False)",
        "ACTIVE_FORCE_EXTRACT = False",
        text, count=1, flags=re.M,
    )
    assert n == 1, "ACTIVE_FORCE_EXTRACT 라인을 못 찾음"

    text, n = re.subn(
        r"^ACTIVE_SOC_REF_TO_REGRESSION = (True|False)",
        "ACTIVE_SOC_REF_TO_REGRESSION = False",
        text, count=1, flags=re.M,
    )
    assert n == 1, "ACTIVE_SOC_REF_TO_REGRESSION 라인을 못 찾음"

    # 2026-10-21: 1016_1915에서 T=100이 noscen을 명확히 이기는 걸 확인(oracle +0.46pp) —
    # 회귀 헤드 비교도 이 확정 레시피 기준으로 맞춘다. None(웜스타트 없음)으로 돌리면
    # 1015_1618(구버전 최고 설정)과 비교하는 셈이 돼 더 이상 "현재 최고 설정" 기준이 아니다.
    text, n = re.subn(
        r"^ACTIVE_WARMSTART_BRANCH_EPOCH = .+",
        "ACTIVE_WARMSTART_BRANCH_EPOCH = 100",
        text, count=1, flags=re.M,
    )
    assert n == 1, "ACTIVE_WARMSTART_BRANCH_EPOCH 라인을 못 찾음"

    text, n = re.subn(r'^ACTIVE_P1_TAG = ".*"', f'ACTIVE_P1_TAG = "{tag}"', text, count=1, flags=re.M)
    assert n == 1, "ACTIVE_P1_TAG 라인을 못 찾음"

    text, n = re.subn(
        r'("regression_model": )"[a-z_]+"(,)',
        rf'\g<1>"{regression_model}"\g<2>',
        text, count=1,
    )
    assert n == 1, '"regression_model" 라인을 못 찾음'

    text, n = re.subn(
        r'("drop_probe_from_regression": )(True|False)(,)',
        rf"\g<1>{drop_probe}\g<3>",
        text, count=1,
    )
    assert n == 1, '"drop_probe_from_regression" 라인을 못 찾음'

    return text


def main() -> None:
    original = PARAMS_PATH.read_text(encoding="utf-8")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for i, (tag, regression_model, drop_probe) in enumerate(EXPERIMENTS, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(EXPERIMENTS)}] {tag}  (regression_model={regression_model}, "
                  f"drop_probe_from_regression={drop_probe})")
            print("=" * 70)

            patched = patch_params(original, tag, regression_model, drop_probe)
            PARAMS_PATH.write_text(patched, encoding="utf-8")

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
