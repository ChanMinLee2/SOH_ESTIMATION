"""
lambda_shrink 스윕(0.01/0.05/0.2/1.0, rawonly 조건) val R²/RMSE를 로그스케일 x축으로 플랏.

주의: 이 4개 run은 SOH_EXCLUDE_STAT_LEAK=1 누락으로 N_HI=66 (baseline rawonly/
noscen_zonetile은 N_HI=64) — baseline과 절대값 직접 비교 불가, 참고선으로만 표시.
4개끼리는 조건 동일(N_HI=66)이라 내부 비교는 유효.

Run: python 4_hi_analysis/shrinkage_lambda_sweep_plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _f
        break
matplotlib.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "5_model" / "experiments" / "phase1_lab" / "results" / "p1v2_runs"
OUT_DIR = PROJECT_ROOT / "4_hi_analysis" / "outputs" / "shrinkage_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    0.01: "0913_1015_p1v2_p1v4_rawonly_shrink0p01_seed42",
    0.05: "0913_1257_p1v2_p1v4_rawonly_shrink0p05_seed42",
    0.2:  "0913_1516_p1v2_p1v4_rawonly_shrink0p2_seed42",
    1.0:  "0913_1724_p1v2_p1v4_rawonly_shrink1p0_seed42",
}

# 참고선 (N_HI=64, 직접 비교 불가 — 시각적 참고용)
BASELINE_RAWONLY_R2 = 0.91157
BASELINE_NOSCEN_R2 = 0.94815
BASELINE_RAWONLY_RMSE = 0.020713
BASELINE_NOSCEN_RMSE = 0.015861


def load_point(run_dir_name: str):
    d = RUNS_DIR / run_dir_name
    summary = json.loads((d / "p1v2_summary.json").read_text(encoding="utf-8"))
    sel_ep = summary["selected_epoch"]
    log_path = d / "logs" / "train_log_v2.csv"
    header = None
    row = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if header is None:
            header = parts
            continue
        if int(parts[0]) == sel_ep:
            row = parts
    assert row is not None, f"epoch {sel_ep} not found in {log_path}"
    rec = dict(zip(header, row))
    return float(rec["val_r2"]), float(rec["val_rmse"]), sel_ep


def main():
    lambdas = sorted(RUNS.keys())
    r2s, rmses, eps = [], [], []
    for lam in lambdas:
        r2, rmse, ep = load_point(RUNS[lam])
        r2s.append(r2); rmses.append(rmse); eps.append(ep)
        print(f"[shrink_plot] lambda_shrink={lam}: selected_epoch={ep}, val_r2={r2:.4f}, val_rmse={rmse:.6f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(lambdas, r2s, marker="o", color="#2E5C8A", lw=2, ms=7, label="shrinkage sweep (N_HI=66)")
    ax1.axhline(BASELINE_RAWONLY_R2, color="#B33", ls="--", lw=1.2, label=f"rawonly baseline {BASELINE_RAWONLY_R2:.3f} (N_HI=64, 참고)")
    ax1.axhline(BASELINE_NOSCEN_R2, color="#3A8", ls="--", lw=1.2, label=f"noscen baseline {BASELINE_NOSCEN_R2:.3f} (N_HI=64, 참고)")
    ax1.set_xscale("log")
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels([str(l) for l in lambdas])
    ax1.set_xlabel("lambda_shrink")
    ax1.set_ylabel("val R²")
    ax1.set_title("val R² vs lambda_shrink")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(lambdas, rmses, marker="o", color="#2E5C8A", lw=2, ms=7, label="shrinkage sweep (N_HI=66)")
    ax2.axhline(BASELINE_RAWONLY_RMSE, color="#B33", ls="--", lw=1.2, label=f"rawonly baseline {BASELINE_RAWONLY_RMSE:.4f} (참고)")
    ax2.axhline(BASELINE_NOSCEN_RMSE, color="#3A8", ls="--", lw=1.2, label=f"noscen baseline {BASELINE_NOSCEN_RMSE:.4f} (참고)")
    ax2.set_xscale("log")
    ax2.set_xticks(lambdas)
    ax2.set_xticklabels([str(l) for l in lambdas])
    ax2.set_xlabel("lambda_shrink")
    ax2.set_ylabel("val RMSE")
    ax2.set_title("val RMSE vs lambda_shrink")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("lambda_shrink 스윕 (rawonly, N_HI=66 — baseline과 직접비교 불가, 참고선만)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "lambda_shrink_sweep.png"
    fig.savefig(out, dpi=200)
    print(f"[shrink_plot] saved: {out}")


if __name__ == "__main__":
    main()
