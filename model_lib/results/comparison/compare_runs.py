"""
model_lib/results/comparison/compare_runs.py — p1v2_runs 두 run을 비교하는 범용 보고용 스크립트.

2026-10-08: scratchpad의 1회성 임시 스크립트들(final_oracle_hard_compare.py,
final_capacity_curve_v2.py, 그리고 그 둘을 1007_1328 vs 1008_0144용으로 복제한
compare_1007_vs_1008_*.py)을 하나로 합치고 일반화했다 — run 경로/라벨을 하드코딩하지
않고 CLI 인자로 받으며, 실행 시각(MMDD_HHMM)으로 출력 폴더를 자동 생성한다(실행마다
새 비교 케이스 폴더가 생겨 섞이지 않음, p1v2_runs의 run 폴더 네이밍 관례와 동일).

2026-11-04: run을 2개 이상(N개) 받도록 일반화 — 패밀리별 4-run(scen/noscen x MLP/Transformer)
비교용. 2-run 호출(--label-a/--label-b)은 그대로 동작한다.

실행:
    python model_lib/results/comparison/compare_runs.py <run_1> <run_2> [<run_3> ...] \
        [--labels L1,L2,...] [--label-a LABEL] [--label-b LABEL] [--cells b1c0,1-7] [--tag NAME]

각 run은 전체 경로 또는 model_lib/results/p1v2_runs/ 밑의 폴더명만 줘도 된다.

출력: model_lib/results/comparison/{MMDD_HHMM}[_{tag}]/ 밑에
    perf_compare.png                — oracle/hard RMSE·R² 2x2 막대비교
    perf_compare.csv                — 위 막대의 수치(+ 충/방전 RMSE, 데이터셋별 RMSE)
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
    # noscen(assign="none") 축은 seg_name이 "chg"/"dis" 2개뿐 — 6존 전용 팔레트와
    # 겹치지 않는 색을 따로 둔다(2026-10-19, 안 그러면 noscen run의 곡선이 전혀
    # 안 그려지고 실측선만 남는다 — SEG_COLORS 키로만 필터링했었기 때문).
    "chg": "#17becf", "dis": "#e377c2",
}
COLOR_A = "#9e9e9e"
COLOR_B = "#1f5fa8"
RUN_COLORS = [COLOR_A, COLOR_B, "#c98a2b", "#3a7d44", "#8e5ba8", "#b0413e"]


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


def plot_perf(runs: list[Path], labels: list[str], out_dir: Path) -> Path:
    mets = [_load_metrics(r) for r in runs]
    colors = RUN_COLORS[:len(runs)] if len(runs) <= len(RUN_COLORS) else None

    fig, axes = plt.subplots(2, 2, figsize=(max(11, 2.6 * len(runs) + 2), 9))
    for col, mode in enumerate(("oracle", "hard")):
        caps = [m[mode]["capacity"] for m in mets]

        ax_rmse = axes[0, col]
        bars = ax_rmse.bar(labels, [c["rmse"] for c in caps], color=colors, width=0.55)
        _bar_labels(ax_rmse, bars)
        ax_rmse.set_title(f"{mode} — RMSE (낮을수록 좋음)", fontsize=11)
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.spines[["top", "right"]].set_visible(False)

        ax_r2 = axes[1, col]
        bars = ax_r2.bar(labels, [c["r2"] for c in caps], color=colors, width=0.55)
        _bar_labels(ax_r2, bars)
        ax_r2.set_title(f"{mode} — R² (높을수록 좋음)", fontsize=11)
        ax_r2.set_ylabel("R²")
        ax_r2.set_ylim(0, 1.0)
        ax_r2.spines[["top", "right"]].set_visible(False)
        if len(runs) > 2:
            for ax in (ax_rmse, ax_r2):
                ax.tick_params(axis="x", labelsize=9, rotation=15)

    caveat = "  |  ".join(_run_source_desc(r) for r in runs)
    fig.suptitle(f"{' vs '.join(labels)} — oracle·hard 성능 비교\n(test split)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, caveat, ha="center", va="top", fontsize=8.3, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out_path = out_dir / "perf_compare.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    rows = []
    for run, label, m in zip(runs, labels, mets):
        for mode in ("oracle", "hard"):
            cap, brk = m[mode]["capacity"], m[mode].get("breakdown", {})
            row = {"label": label, "run": run.name, "mode": mode,
                   "rmse": cap["rmse"], "mae": cap["mae"], "r2": cap["r2"], "mape": cap["mape"],
                   "acc": m[mode].get("classification", {}).get("accuracy")}
            for k in ("charge", "discharge"):
                if k in brk:
                    row[f"rmse_{k}"] = brk[k]["rmse"]
            for k, v in brk.items():
                if k.startswith("dataset_"):
                    row[f"rmse_{k[8:]}"] = v["rmse"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "perf_compare.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return out_path


def plot_curve_one(runs: list[Path], labels: list[str],
                   cell: str, mode: str, fname: str, out_dir: Path) -> Path | None:
    datas = []
    for run in runs:
        df = pd.read_csv(run / "predictions" / fname)
        datas.append(df[df["cell_id"] == cell])
    if any(len(d) == 0 for d in datas):
        print(f"skip {cell}/{mode}: 데이터 없음({', '.join(f'{l}={len(d)}' for l, d in zip(labels, datas))})")
        return None

    ymin = min(min(d["cap_true_Ah"].min(), d["cap_pred_Ah"].min()) for d in datas) * 0.98
    ymax = max(max(d["cap_true_Ah"].max(), d["cap_pred_Ah"].max()) for d in datas) * 1.02

    n = len(runs)
    ncol = 2 if n == 4 else n
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.5 * nrow), sharey=True, squeeze=False)
    for ax, data, title in zip(axes.flat, datas, labels):
        true_by_cycle = data.groupby("cycle")["cap_true_Ah"].mean().sort_index()
        ax.plot(true_by_cycle.index, true_by_cycle.values, color="black",
                linewidth=2.0, label="실측", zorder=10)
        for seg_name in sorted(data["seg_name"].unique()):
            color = SEG_COLORS.get(seg_name, "#777777")
            seg_df = data[data["seg_name"] == seg_name]
            if len(seg_df) == 0:
                continue
            pred_by_cycle = seg_df.groupby("cycle")["cap_pred_Ah"].mean().sort_index()
            ax.plot(pred_by_cycle.index, pred_by_cycle.values, color=color,
                    linewidth=0.9, alpha=0.75, label=seg_name)
        rmse_pct = ((data["soh_pred"] - data["soh_true"]) ** 2).mean() ** 0.5 * 100
        ax.set_title(f"{title}  (셀 SOH RMSE {rmse_pct:.2f}%)", fontsize=11.5)
        ax.set_xlabel("Cycle")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Capacity (Ah)")

    fig.suptitle(f"셀 \"{cell}\" — 시나리오별 용량 예측 추이({mode} 모드), {' vs '.join(labels)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / f"curve_{cell}_{mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="p1v2_runs 2개 이상 run 성능/예측곡선 비교")
    ap.add_argument("runs", nargs="+", help="전체경로 또는 p1v2_runs/ 밑 폴더명 (2개 이상)")
    ap.add_argument("--labels", default=None, help="쉼표구분 라벨 (run 순서대로)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--cells", default="b1c0,1-7", help="쉼표구분 cell_id 목록")
    ap.add_argument("--tag", default=None, help="출력 폴더 이름 뒤에 붙일 꼬리표")
    args = ap.parse_args()
    if len(args.runs) < 2:
        ap.error("run을 2개 이상 줘야 한다")

    runs = [_resolve_run_dir(r) for r in args.runs]
    if args.labels:
        labels = [x.strip() for x in args.labels.split(",")]
        if len(labels) != len(runs):
            ap.error(f"--labels 개수({len(labels)})가 run 개수({len(runs)})와 다르다")
    else:
        labels = [r.name for r in runs]
        if args.label_a:
            labels[0] = args.label_a
        if args.label_b:
            labels[1] = args.label_b
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    out_name = datetime.now().strftime("%m%d_%H%M") + (f"_{args.tag}" if args.tag else "")
    out_dir = COMPARISON_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare_runs] 출력 폴더: {out_dir}")

    plot_perf(runs, labels, out_dir)

    modes = {"oracle": "test_predictions.csv", "hard": "test_predictions_hard.csv"}
    for cell in cells:
        for mode, fname in modes.items():
            plot_curve_one(runs, labels, cell, mode, fname, out_dir)


if __name__ == "__main__":
    main()
