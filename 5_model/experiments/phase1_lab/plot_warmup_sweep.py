"""
5_model/experiments/phase1_lab/plot_warmup_sweep.py

docs/260825_RESULTS.md의 lambda_l0_warmup_epochs 스위핑(50/150/250) 결과를 비교한다.
lambda_sweep.py의 산출물(summary.csv, lambda 축)과 달리 이건 "같은 lambda_l0에서
warmup만 다르게 줬을 때 학습 곡선 자체가 다르게 가는가"를 보는 것이므로, x축이
lambda가 아니라 **epoch**이다 — 각 run의 train_log_v2.csv를 통째로 읽어 겹쳐 그린다.

세 run 모두 lambda_l0=0.000237로 고정, warmup만 50/150/250(ramp=100 불변)이라
전제한다(--l0-warmup-epochs-override 미지정 시 기존 동작=50).

사용 예:
  python 5_model/experiments/phase1_lab/plot_warmup_sweep.py \
      --run w50=5_model/experiments/phase1_lab/results/p1v2_runs/0824_1855_p1v2_lsweep_l0p000237_seed42 \
      --run w150=5_model/experiments/phase1_lab/results/p1v2_runs/0825_1819_p1v2_wsweep_w150_l0p000237_seed42 \
      --run w250=5_model/experiments/phase1_lab/results/p1v2_runs/0825_2105_p1v2_wsweep_w250_l0p000237_seed42 \
      --out-dir 5_model/experiments/phase1_lab/results/lambda_sweep/0826_warmup_compare
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="warmup epoch 스위핑(50/150/250) 학습곡선 비교")
    p.add_argument("--run", action="append", required=True, dest="runs",
                   help="label=run_dir 형식으로 여러 번 지정(예: --run w50=<경로> --run w150=<경로>). "
                        "run_dir 밑의 logs/train_log_v2.csv를 읽는다.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ylim-skip-epochs", type=int, default=15,
                   help="y축 범위를 계산할 때 앞의 이 개수만큼 epoch은 무시한다(기본 15). "
                        "학습 초반(LR warmup 10epoch 근처)은 val_r2가 크게 음수·val_rmse가 "
                        "크게 튀어서, 이걸 포함해 자동 스케일하면 수렴 이후 구간이 눌려서 "
                        "안 보인다 — 곡선 자체는 처음(epoch 0)부터 그대로 그리되, y축 범위만 "
                        "이 이후 데이터 기준으로 잡는다.")
    return p.parse_args()


def _load_log(run_dir: Path) -> list[dict]:
    log_path = run_dir / "logs" / "train_log_v2.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} 없음 — run_dir 경로 확인 필요")
    return list(csv.DictReader(log_path.open(encoding="utf-8")))


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return

    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    runs = {}
    for item in args.runs:
        label, _, path_str = item.partition("=")
        if not path_str:
            raise SystemExit(f"--run 형식 오류(label=path 필요): {item}")
        runs[label] = _load_log(Path(path_str))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {"w50": "#2166ac", "w150": "#b2182b", "w250": "#1a9850"}
    default_colors = list(colors.values())

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    metrics = [("val_r2", "val R2"), ("val_rmse", "val RMSE"), ("gate_saturation", "포화도")]

    skip = args.ylim_skip_epochs
    for ax, (col, title) in zip(axes, metrics):
        ylim_vals = []
        for i, (label, rows) in enumerate(runs.items()):
            epochs = [int(r["epoch"]) for r in rows]
            ys = [float(r[col]) for r in rows]
            color = colors.get(label, default_colors[i % len(default_colors)])
            ax.plot(epochs, ys, "-", color=color, label=label, linewidth=1.3, alpha=0.85)
            sel = [r for r in rows if r.get("is_selected") in ("1", "True")]
            if sel:
                sel_epoch = int(sel[-1]["epoch"])
                sel_y = float(sel[-1][col])
                ax.scatter([sel_epoch], [sel_y], color=color, s=60, zorder=5,
                           marker="*", edgecolors="black", linewidths=0.5)
            ylim_vals += [y for e, y in zip(epochs, ys) if e >= skip]
        if ylim_vals:
            lo, hi = min(ylim_vals), max(ylim_vals)
            pad = (hi - lo) * 0.08 or abs(hi) * 0.05 or 0.01
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs epoch (★=선택된 체크포인트, y축은 epoch≥{skip} 기준)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "warmup_sweep_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    print("\n[plot] 선택된 체크포인트 요약:")
    print(f"{'label':>8} {'sel_epoch':>10} {'val_r2':>8} {'val_rmse':>9} {'sat':>8}")
    for label, rows in runs.items():
        sel = [r for r in rows if r.get("is_selected") in ("1", "True")]
        r = sel[-1] if sel else rows[-1]
        print(f"{label:>8} {r['epoch']:>10} {float(r['val_r2']):>8.4f} "
              f"{float(r['val_rmse']):>9.5f} {float(r['gate_saturation']):>8.4f}")


if __name__ == "__main__":
    main()
