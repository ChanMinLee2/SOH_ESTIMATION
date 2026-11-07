"""run_ablation_kernelroute_ckptrule.py — 2026-11-05 두 수정의 어블레이션(기준 run: scen 1029_0347, noscen 1029_1119).

  수정 A : hard/soft 평가에서 커널 HI 블록을 예측 시나리오로 다시 고름(정답 시나리오 누수 제거)
           — scr_evaluator.py / 9_eval/test.py, 스위치 FIXED_EVAL_ROUTED_KERNEL
  수정 B : 체크포인트 선택 규칙 legacy → tiebreak(sat은 val ε 이내 동률에서만, best_val은 최솟값 유지)
           — 8_train/train.py, 스위치 ACTIVE_CKPT_SELECT_RULE

단계(각 모드 scen/noscen마다):
  eval  : 기준 run 폴더를 _ablation_1105/<mode>/ 아래 두 벌로 복사(체크포인트·설정·커널 산출물만, 원본은
          건드리지 않음)해 같은 체크포인트를 현재 코드로 두 번 평가한다. 재학습 없음.
            evalA_off : FIXED_EVAL_ROUTED_KERNEL=False  (수정 전 평가 재현 — 원본 metrics와 같아야 정상)
            evalA_on  : FIXED_EVAL_ROUTED_KERNEL=True   (수정 A)
  train : 기준 run과 같은 설정 + ACTIVE_CKPT_SELECT_RULE="tiebreak"로 재학습(Step 5~9, 평가는 수정 A 적용).
          학습 궤적은 seed 고정으로 같고 선택 epoch·조기종료만 바뀐다. 로그 재생 예측:
            scen   : legacy ep 374 → tiebreak ep 179 선택, ep 239 종료
            noscen : legacy와 같은 ep 189 선택, ep 249 종료(변화 없음 — 대조군)
  report: 모드별 원본 / A off / A on / A+B 의 test hard·oracle 지표, A off↔on 사이 hard 예측이 바뀐 행 수,
          그리고 단계별 scen − noscen hard RMSE 차이를 출력.
          noscen은 방향당 시나리오가 1개라 hard = oracle(오분류 없음)이므로 A도 변화가 없어야 정상.

실행: python run_ablation_kernelroute_ckptrule.py                     # 두 모드 × eval → train → report
      python run_ablation_kernelroute_ckptrule.py scen eval report    # 모드·단계 골라서(생략한 쪽은 전부)
로그: model_lib/results/p1v2_runs/_ablation_1105/<mode>/<단계>.log
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import run_hi63_union_scen_tr_sweep as base

ROOT = base.ROOT
PARAMS_PATH = base.PARAMS_PATH
PYTHON = base.PYTHON
RUNS_DIR = ROOT / "model_lib" / "results" / "p1v2_runs"
ABL_DIR = RUNS_DIR / "_ablation_1105"

BEST = dict(base.BASE)   # lambda_l0=0.000237, n2=0.20, size=m, datasets=[MIT, HUST] — 두 기준 run과 동일
MODES = {
    "scen": {
        "baseline": RUNS_DIR / "1029_0347_p1v2_hi63_scen_transformer_union_nowarm_seed42",
        "baseline_tag": "hi63_scen_transformer_union_nowarm",
        "train_tag": "hi63_scen_trm_union_nowarm_tiebreak_l0-2e-4_n2-20",
    },
    "noscen": {
        "baseline": RUNS_DIR / "1029_1119_p1v2_hi63_noscen_transformer_union_nowarm_seed42",
        "baseline_tag": "hi63_noscen_transformer_union_nowarm",
        "train_tag": "hi63_noscen_trm_union_nowarm_tiebreak_l0-2e-4_n2-20",
    },
}
STAGES = ("eval", "train", "report")
EVAL_COPIES = {"evalA_off": False, "evalA_on": True}
# 평가 결과물(predictions/figures/metrics/routing)은 복사하지 않는다 — test.py가 새로 만든다
COPY_IGNORE = shutil.ignore_patterns("predictions", "figures", "metrics", "routing")


def patch_params(text: str, mode: str, tag: str, routed_kernel: bool = True, select_rule: str = "legacy") -> str:
    text = base.patch_params(text, {**BEST, "tag": tag})
    if mode == "noscen":
        text = base._sub(text, r'^(\s*)# "assign": "none",.*$',
                         r'\1"assign": "none",           # 시나리오 라벨링: noscen (ablation_1105)', '"assign"')
    text = base._sub(text, r"^FIXED_EVAL_ROUTED_KERNEL = (True|False)",
                     f"FIXED_EVAL_ROUTED_KERNEL = {routed_kernel}", "FIXED_EVAL_ROUTED_KERNEL")
    text = base._sub(text, r'^ACTIVE_CKPT_SELECT_RULE = "[a-z]+"',
                     f'ACTIVE_CKPT_SELECT_RULE = "{select_rule}"', "ACTIVE_CKPT_SELECT_RULE")
    return text


def _n_hi_env(run_dir) -> dict[str, str]:
    """test.py를 run_pipeline.py 없이 직접 띄울 때도 그 run의 N_HI(leak 제외 범위) env를 맞춘다 —
    hi_schema가 import 시점에 env로 N_HI를 고정하는데, 기본값(66)이면 63-HI 체크포인트와 shape가
    어긋나 load_state_dict가 실패한다(9_eval/scenario_attribution.py와 같은 방식: config.yaml 기준)."""
    import yaml
    data = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["data"]
    return {"SOH_EXCLUDE_STAT_LEAK": "1" if data.get("exclude_stat_leak") else "0",
            "SOH_EXCLUDE_DQDV_LEAK": "1" if data.get("exclude_dqdv_leak") else "0"}


def _run(cmd: list[str], log_path, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env or {})
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, env=env)
    ok = proc.returncode == 0
    print(f"  -> {'OK' if ok else f'FAIL(exit={proc.returncode})'}, {(time.time() - t0) / 60:.1f}분, 로그: {log_path}")
    if not ok:
        sys.exit(1)


def stage_eval(original: str, mode: str) -> None:
    m = MODES[mode]
    for name, routed in EVAL_COPIES.items():
        dst = ABL_DIR / mode / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(m["baseline"], dst, ignore=COPY_IGNORE)
        print(f"\n[{mode}/eval] {name}: FIXED_EVAL_ROUTED_KERNEL={routed}  ({dst})")
        PARAMS_PATH.write_text(patch_params(original, mode, m["baseline_tag"], routed), encoding="utf-8")
        _run([PYTHON, str(ROOT / "9_eval" / "test.py"), "--run-dir", str(dst)], ABL_DIR / mode / f"{name}.log",
             _n_hi_env(dst))


def stage_train(original: str, mode: str) -> None:
    tag = MODES[mode]["train_tag"]
    print(f"\n[{mode}/train] {tag}  (ACTIVE_CKPT_SELECT_RULE=tiebreak, from_step=5)")
    PARAMS_PATH.write_text(patch_params(original, mode, tag, True, "tiebreak"), encoding="utf-8")
    _run([PYTHON, str(ROOT / "run_pipeline.py"), "5"], ABL_DIR / mode / "train_tiebreak.log")


def _metrics(run_dir) -> dict | None:
    f = run_dir / "metrics" / "metrics.json" if run_dir else None
    return json.loads(f.read_text(encoding="utf-8"))["test"] if f and f.exists() else None


def _rows(mode: str) -> list[tuple[str, object]]:
    m = MODES[mode]
    trained = sorted(RUNS_DIR.glob(f"*_p1v2_{m['train_tag']}_seed*"))
    return [("원본", m["baseline"]), ("A off (재현)", ABL_DIR / mode / "evalA_off"),
            ("A on", ABL_DIR / mode / "evalA_on"), ("A+B (tiebreak)", trained[-1] if trained else None)]


def stage_report(modes: list[str]) -> None:
    import numpy as np
    import pandas as pd
    import torch

    hard_rmse: dict[str, dict[str, float]] = {}
    for mode in modes:
        print(f"\n=== {mode} ===")
        print(f"{'단계':<18}{'hard RMSE':>11}{'hard R²':>9}{'hard acc':>10}{'oracle RMSE':>13}  ckpt epoch")
        for label, d in _rows(mode):
            t = _metrics(d)
            if t is None:
                print(f"{label:<18}  (결과 없음)")
                continue
            h, o = t["hard"], t["oracle"]
            hard_rmse.setdefault(label, {})[mode] = h["capacity"]["rmse"]
            acc = h["classification"].get("accuracy")   # noscen은 정답 레짐 라벨이 없어 키 자체가 없음
            ck = d / "checkpoints" / "best_by_saturation.pt"
            ep = torch.load(ck, map_location="cpu").get("epoch", "") if ck.exists() else ""
            print(f"{label:<18}{h['capacity']['rmse']:>11.5f}{h['capacity']['r2']:>9.4f}"
                  f"{(f'{acc:.4%}' if acc is not None else '-'):>10}{o['capacity']['rmse']:>13.5f}  {ep}")

        # A off ↔ on: 바뀐 행 = 오분류 행이어야 한다(정분류 행은 입력이 완전히 같음)
        p_off = ABL_DIR / mode / "evalA_off" / "predictions" / "test_predictions_hard.csv"
        p_on = ABL_DIR / mode / "evalA_on" / "predictions" / "test_predictions_hard.csv"
        if p_off.exists() and p_on.exists():
            a, b = pd.read_csv(p_off), pd.read_csv(p_on)
            assert len(a) == len(b) and (a.soh_true.values == b.soh_true.values).all(), "두 평가의 행 순서가 다름"
            changed = ~np.isclose(a.soh_pred.values, b.soh_pred.values, rtol=0, atol=1e-7)
            n = int(changed.sum())
            rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
            print(f"[A] hard 예측이 바뀐 행: {n} / {len(a)} ({n / len(a):.4%})")
            if n:
                print(f"    그 행들의 RMSE: off {rmse(a.soh_pred[changed] - a.soh_true[changed]):.5f} → "
                      f"on {rmse(b.soh_pred[changed] - b.soh_true[changed]):.5f}")

        if mode == "scen":
            _check_trajectory(mode)

    # noscen은 A·B 모두 영향이 없는 구조(hard = oracle, 로그 재생상 선택 epoch 동일)라 안 돌렸으면
    # 원본 값으로 대신한다
    noscen_orig = _metrics(MODES["noscen"]["baseline"])
    print("\n=== scen − noscen (test hard RMSE, 음수 = scen이 더 좋음) ===")
    for label, v in hard_rmse.items():
        if "scen" not in v:
            continue
        ns, note = v.get("noscen"), ""
        if ns is None and noscen_orig is not None:
            ns, note = noscen_orig["hard"]["capacity"]["rmse"], "  (noscen 원본 사용)"
        if ns is not None:
            print(f"{label:<18}{v['scen']:>10.5f} − {ns:.5f} = {v['scen'] - ns:+.5f}{note}")


def _check_trajectory(mode: str) -> None:
    """tiebreak 재학습 로그가 기준 run 로그와 종료 epoch까지 같은지 확인 — 선택 규칙은 학습 궤적을
    바꾸지 않으므로 같아야 정상이고, 다르면 10/29 이후 코드 변경이 학습에 영향을 준 것이다."""
    import pandas as pd

    trained = _rows(mode)[-1][1]
    if trained is None:
        return
    new_logs = sorted((trained / "logs").glob("*.csv"))
    old_logs = sorted((MODES[mode]["baseline"] / "logs").glob("*.csv"))
    if not new_logs or not old_logs:
        return
    a, b = pd.read_csv(old_logs[0]), pd.read_csv(new_logs[0])
    n = min(len(a), len(b))
    cols = ["tr_rmse", "val_rmse", "gate_saturation"]
    diff = (a[cols].iloc[:n] - b[cols].iloc[:n]).abs().max(axis=1)
    first_bad = diff[diff > 1e-6]
    if first_bad.empty:
        print(f"[궤적] 기준 run과 tiebreak 재학습 로그가 epoch 1~{n} 동안 동일 → 재현 OK")
    else:
        print(f"[궤적] epoch {int(b.epoch.iloc[first_bad.index[0]])}부터 기준 run과 로그가 다름 "
              f"(최대 차이 {first_bad.max():.2e}) — 선택 규칙 외 코드 변경 영향 의심")


def main() -> None:
    args = sys.argv[1:]
    unknown = [a for a in args if a not in MODES and a not in STAGES]
    assert not unknown, f"알 수 없는 인자 {unknown} — 모드 {list(MODES)} / 단계 {list(STAGES)}"
    modes = [a for a in args if a in MODES] or list(MODES)
    stages = [a for a in args if a in STAGES] or list(STAGES)
    for mode in modes:
        assert MODES[mode]["baseline"].exists(), f"기준 run 없음: {MODES[mode]['baseline']}"
        (ABL_DIR / mode).mkdir(parents=True, exist_ok=True)

    original = PARAMS_PATH.read_text(encoding="utf-8")
    for mode in modes:   # 실행 전에 패치가 전부 들어맞는지 먼저 확인
        patch_params(original, mode, MODES[mode]["train_tag"], True, "tiebreak")
    try:
        for mode in modes:
            if "eval" in stages:
                stage_eval(original, mode)
            if "train" in stages:
                stage_train(original, mode)
    finally:
        PARAMS_PATH.write_text(original, encoding="utf-8")
        print("\nparameters.py 원복 완료(실행 전 상태로 되돌림).")
    if "report" in stages:
        stage_report(modes)


if __name__ == "__main__":
    main()
