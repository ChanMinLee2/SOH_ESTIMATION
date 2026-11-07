"""run_sohbin_weight.py — 2026-11-06 SOH 구간 가중 MSE 실험(기준: A+B run 1105_1827, scen).

목적: 오차 분해에서 가장 큰 성분(셀 안의 느린 추세 0.0116)과 SOH < 0.95 구간 과대예측
(+0.002~0.006)을 손실 가중으로 줄일 수 있는지 확인. 구조·HI·나머지 설정은 기준과 동일
(수정 A=FIXED_EVAL_ROUTED_KERNEL True, 수정 B=ACTIVE_CKPT_SELECT_RULE "tiebreak").

  구간 경계 : ACTIVE_SOH_BIN_WEIGHT_EDGES = (0.80, 0.85, 0.90, 0.95)  → 5개 구간
  가중치    : (1/구간 샘플 수)^power, train 평균 가중치 1로 정규화(8_train/train.py::_soh_bin_weights)
  변형      : p10 = power 1.0(완전 역빈도), p05 = power 0.5(완화)

val/체크포인트 선택은 가중 없는 val RMSE 그대로라, 가중치는 학습 신호만 바꾼다.

실행: python run_sohbin_weight.py              # p10 → p05 학습 후 report
      python run_sohbin_weight.py p10 report   # 변형·단계 골라서(생략한 쪽은 전부)
로그: model_lib/results/p1v2_runs/_sohbin_1106/<변형>.log
"""

from __future__ import annotations

import json
import sys

import run_ablation_kernelroute_ckptrule as abl

base = abl.base
ROOT, PARAMS_PATH, PYTHON, RUNS_DIR = abl.ROOT, abl.PARAMS_PATH, abl.PYTHON, abl.RUNS_DIR
OUT_DIR = RUNS_DIR / "_sohbin_1106"

EDGES = (0.80, 0.85, 0.90, 0.95)
BASELINE = RUNS_DIR / "1105_1827_p1v2_hi63_scen_trm_union_nowarm_tiebreak_l0-2e-4_n2-20_seed42"
VARIANTS = {
    "p10": {"power": 1.0, "tag": "hi63_scen_trm_union_nowarm_tiebreak_sohbin-p10_l0-2e-4_n2-20"},
    "p05": {"power": 0.5, "tag": "hi63_scen_trm_union_nowarm_tiebreak_sohbin-p05_l0-2e-4_n2-20"},
}
STAGES = ("train", "report")


def patch_params(text: str, tag: str, power: float) -> str:
    text = abl.patch_params(text, "scen", tag, True, "tiebreak")
    text = base._sub(text, r"^ACTIVE_SOH_BIN_WEIGHT_EDGES = .*?(?=\s+#)",
                     f"ACTIVE_SOH_BIN_WEIGHT_EDGES = {EDGES}", "ACTIVE_SOH_BIN_WEIGHT_EDGES")
    text = base._sub(text, r"^FIXED_SOH_BIN_WEIGHT_POWER = [0-9.]+",
                     f"FIXED_SOH_BIN_WEIGHT_POWER = {power}", "FIXED_SOH_BIN_WEIGHT_POWER")
    return text


def stage_train(original: str, name: str) -> None:
    v = VARIANTS[name]
    print(f"\n[{name}/train] {v['tag']}  (edges={EDGES}, power={v['power']}, from_step=5)")
    PARAMS_PATH.write_text(patch_params(original, v["tag"], v["power"]), encoding="utf-8")
    abl._run([PYTHON, str(ROOT / "run_pipeline.py"), "5"], OUT_DIR / f"{name}.log")


def _trained(tag: str):
    runs = sorted(RUNS_DIR.glob(f"*_p1v2_{tag}_seed*"))
    return runs[-1] if runs else None


def stage_report(names: list[str]) -> None:
    import numpy as np
    import pandas as pd
    import torch

    rows = [("기준 A+B", BASELINE)] + [(f"{n} (power {VARIANTS[n]['power']})", _trained(VARIANTS[n]["tag"]))
                                       for n in names]
    bins = [-np.inf, *EDGES, np.inf]
    bin_labels = [f"<{EDGES[0]:.2f}"] + [f"{a:.2f}-{b:.2f}" for a, b in zip(EDGES, EDGES[1:])] + [f"≥{EDGES[-1]:.2f}"]
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))

    # 변형 선택은 val RMSE(선택된 체크포인트 epoch의 가중 없는 val RMSE) 기준 — test는 선택 후 보고만 한다
    print(f"\n{'run':<20}{'val RMSE':>10}{'hard RMSE':>11}{'hard R²':>9}{'hard acc':>10}{'oracle RMSE':>13}  ckpt epoch")
    per_bin = {}
    for label, d in rows:
        f = d / "metrics" / "metrics.json" if d else None
        if f is None or not f.exists():
            print(f"{label:<20}  (결과 없음)")
            continue
        t = json.loads(f.read_text(encoding="utf-8"))["test"]
        h, o = t["hard"], t["oracle"]
        ck = d / "checkpoints" / "best_by_saturation.pt"
        ep = torch.load(ck, map_location="cpu").get("epoch", "") if ck.exists() else ""
        log = pd.read_csv(d / "logs" / "train_log_v2.csv")
        val = float(log.val_rmse[log.is_selected == 1].iloc[-1]) if (log.is_selected == 1).any() else float("nan")
        print(f"{label:<20}{val:>10.5f}{h['capacity']['rmse']:>11.5f}{h['capacity']['r2']:>9.4f}"
              f"{h['classification']['accuracy']:>10.4%}{o['capacity']['rmse']:>13.5f}  {ep}")

        p = pd.read_csv(d / "predictions" / "test_predictions_hard.csv")
        e = p.soh_pred - p.soh_true
        b = pd.cut(p.soh_true, bins, right=False, labels=bin_labels)
        per_bin[label] = {lab: (rmse(e[b == lab]), float(e[b == lab].mean()), float((b == lab).mean()))
                          for lab in bin_labels}
        per_bin[label]["충전"] = (rmse(e[p.seg_name.str.startswith("chg")]), float(e[p.seg_name.str.startswith("chg")].mean()), np.nan)
        per_bin[label]["방전"] = (rmse(e[p.seg_name.str.startswith("dis")]), float(e[p.seg_name.str.startswith("dis")].mean()), np.nan)

    if not per_bin:
        return
    print("\n=== SOH 구간·방향별 test hard RMSE / 평균 오차(+ = 높게 예측) ===")
    labels = list(per_bin)
    print(f"{'구간':<12}{'비중':>7}" + "".join(f"{l:>24}" for l in labels))
    for k in [*bin_labels, "충전", "방전"]:
        share = per_bin[labels[0]][k][2]
        cells = "".join(f"{per_bin[l][k][0]:>13.5f} / {per_bin[l][k][1]:+.4f}" for l in labels)
        print(f"{k:<12}{(f'{share:.1%}' if share == share else ''):>7}{cells}")


def main() -> None:
    args = sys.argv[1:]
    unknown = [a for a in args if a not in VARIANTS and a not in STAGES]
    assert not unknown, f"알 수 없는 인자 {unknown} — 변형 {list(VARIANTS)} / 단계 {list(STAGES)}"
    names = [a for a in args if a in VARIANTS] or list(VARIANTS)
    stages = [a for a in args if a in STAGES] or list(STAGES)
    assert BASELINE.exists(), f"기준 run 없음: {BASELINE}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    original = PARAMS_PATH.read_text(encoding="utf-8")
    for n in names:   # 실행 전에 패치가 전부 들어맞는지 먼저 확인
        patch_params(original, VARIANTS[n]["tag"], VARIANTS[n]["power"])
    try:
        if "train" in stages:
            for n in names:
                stage_train(original, n)
    finally:
        PARAMS_PATH.write_text(original, encoding="utf-8")
        print("\nparameters.py 원복 완료(실행 전 상태로 되돌림).")
    if "report" in stages:
        stage_report(names)


if __name__ == "__main__":
    main()
