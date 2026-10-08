"""
model_lib/results/comparison/compare_runs.py — p1v2_runs 두 run을 비교하는 범용 보고용 스크립트.

2026-10-08: scratchpad의 1회성 임시 스크립트들(final_oracle_hard_compare.py,
final_capacity_curve_v2.py, 그리고 그 둘을 1007_1328 vs 1008_0144용으로 복제한
compare_1007_vs_1008_*.py)을 하나로 합치고 일반화했다 — run 경로/라벨을 하드코딩하지
않고 CLI 인자로 받으며, 실행 시각(MMDD_HHMM)으로 출력 폴더를 자동 생성한다(실행마다
새 비교 케이스 폴더가 생겨 섞이지 않음, p1v2_runs의 run 폴더 네이밍 관례와 동일).

실행:
    python model_lib/results/comparison/compare_runs.py <run_a> <run_b> \
        [--label-a LABEL] [--label-b LABEL] [--cells b1c0,1-7]

<run_a>/<run_b>는 전체 경로 또는 model_lib/results/p1v2_runs/ 밑의 폴더명만 줘도 된다.

출력: model_lib/results/comparison/{MMDD_HHMM}/ 밑에
    perf_compare.png                — oracle/hard RMSE·R² 2x2 막대비교
    curve_{cell}_{mode}.png         — 셀별(기본 b1c0, 1-7) oracle/hard 시나리오별 용량 예측 추이
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[3]
P1V2_RUNS_DIR = ROOT / "model_lib" / "results" / "p1v2_runs"
COMPARISON_DIR = ROOT / "model_lib" / "results" / "comparison"

SEG_COLORS = {
    "chg_lo": "#1f77b4", "chg_mid": "#2ca02c", "chg_hi": "#9467bd",
    "dis_hi": "#d62728", "dis_mid": "#ff7f0e", "dis_lo": "#8c564b",
}
COLOR_A = "#9e9e9e"
COLOR_B = "#1f5fa8"


def _resolve_run_dir(run: str) -> Path:
    p = Path(run)
    if p.exists():
        return p
    p = P1V2_RUNS_DIR / run
    if p.exists():
        return p
    raise FileNotFoundError(f"run 디렉터리를 못 찾음: {run} (전체경로도, "
                             f"{P1V2_RUNS_DIR}/{run}도 없음)")


def _run_source_desc(run_dir: Path) -> str:
    """p1v2_summary.json에서 kernel/combined-redundancy/interaction 출처를 읽어
    캡션용 한 줄로 요약 — run마다 다중공선성 제거가 실제 적용됐는지 한눈에 보려고."""
    summary_path = run_dir / "p1v2_summary.json"
    if not summary_path.exists():
        return f"{run_dir.name}: p1v2_summary.json 없음"
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    redund = d.get("combined_redundancy_json")
    kernel = d.get("kernel_features_pkl") or ""
    is_legacy = "legacy_results" in str(kernel)
    tag = "legacy fallback(다중공선성 미적용)" if (not redund or is_legacy) else "정식(다중공선성 적용)"
    return f"{run_dir.name}: {tag}"


def _load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))["test"]


def _bar_labels(ax, bars, fmt="{:.4f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")


def plot_perf(run_a: Path, run_b: Path, label_a: str, label_b: str, out_dir: Path) -> Path:
    before = _load_metrics(run_a)
    after = _load_metrics(run_b)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, mode in enumerate(("oracle", "hard")):
        b_cap, a_cap = before[mode]["capacity"], after[mode]["capacity"]

        ax_rmse = axes[0, col]
        bars = ax_rmse.bar([label_a, label_b], [b_cap["rmse"], a_cap["rmse"]],
                            color=[COLOR_A, COLOR_B], width=0.55)
        _bar_labels(ax_rmse, bars)
        ax_rmse.set_title(f"{mode} — RMSE (낮을수록 좋음)", fontsize=11)
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.spines[["top", "right"]].set_visible(False)

        ax_r2 = axes[1, col]
        bars = ax_r2.bar([label_a, label_b], [b_cap["r2"], a_cap["r2"]],
                          color=[COLOR_A, COLOR_B], width=0.55)
        _bar_labels(ax_r2, bars)
        ax_r2.set_title(f"{mode} — R² (높을수록 좋음)", fontsize=11)
        ax_r2.set_ylabel("R²")
        ax_r2.set_ylim(0, 1.0)
        ax_r2.spines[["top", "right"]].set_visible(False)

    caveat = _run_source_desc(run_a) + "  |  " + _run_source_desc(run_b)
    fig.suptitle(f"{label_a} vs {label_b} — oracle·hard 성능 비교\n(test split)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, caveat, ha="center", va="top", fontsize=8.3, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out_path = out_dir / "perf_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    for mode in ("oracle", "hard"):
        print(mode, label_a, before[mode]["capacity"], label_b, after[mode]["capacity"])
    return out_path


def plot_curve_one(run_a: Path, run_b: Path, label_a: str, label_b: str,
                    cell: str, mode: str, fname: str, out_dir: Path) -> Path | None:
    before = pd.read_csv(run_a / "predictions" / fname)
    before = before[before["cell_id"] == cell]
    after = pd.read_csv(run_b / "predictions" / fname)
    after = after[after["cell_id"] == cell]
    if len(before) == 0 or len(after) == 0:
        print(f"skip {cell}/{mode}: 데이터 없음(a={len(before)}, b={len(after)})")
        return None

    ymin = min(before["cap_true_Ah"].min(), before["cap_pred_Ah"].min(),
               after["cap_true_Ah"].min(), after["cap_pred_Ah"].min()) * 0.98
    ymax = max(before["cap_true_Ah"].max(), before["cap_pred_Ah"].max(),
               after["cap_true_Ah"].max(), after["cap_pred_Ah"].max()) * 1.02

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, data, title in ((ax1, before, label_a), (ax2, after, label_b)):
        true_by_cycle = data.groupby("cycle")["cap_true_Ah"].mean().sort_index()
        ax.plot(true_by_cycle.index, true_by_cycle.values, color="black",
                linewidth=2.0, label="실측", zorder=10)
        for seg_name, color in SEG_COLORS.items():
            seg_df = data[data["seg_name"] == seg_name]
            if len(seg_df) == 0:
                continue
            pred_by_cycle = seg_df.groupby("cycle")["cap_pred_Ah"].mean().sort_index()
            ax.plot(pred_by_cycle.index, pred_by_cycle.values, color=color,
                    linewidth=0.9, alpha=0.75, label=seg_name)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Cycle")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2, loc="lower left")

    ax1.set_ylabel("Capacity (Ah)")
    fig.suptitle(f"셀 \"{cell}\" — 시나리오별 용량 예측 추이({mode} 모드), {label_a} vs {label_b}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / f"curve_{cell}_{mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="p1v2_runs 두 run 성능/예측곡선 비교")
    ap.add_argument("run_a", help="전체경로 또는 p1v2_runs/ 밑 폴더명")
    ap.add_argument("run_b", help="전체경로 또는 p1v2_runs/ 밑 폴더명")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--cells", default="b1c0,1-7", help="쉼표구분 cell_id 목록")
    args = ap.parse_args()

    run_a = _resolve_run_dir(args.run_a)
    run_b = _resolve_run_dir(args.run_b)
    label_a = args.label_a or run_a.name
    label_b = args.label_b or run_b.name
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    out_dir = COMPARISON_DIR / datetime.now().strftime("%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare_runs] 출력 폴더: {out_dir}")

    plot_perf(run_a, run_b, label_a, label_b, out_dir)

    modes = {"oracle": "test_predictions.csv", "hard": "test_predictions_hard.csv"}
    for cell in cells:
        for mode, fname in modes.items():
            plot_curve_one(run_a, run_b, label_a, label_b, cell, mode, fname, out_dir)


if __name__ == "__main__":
    main()
