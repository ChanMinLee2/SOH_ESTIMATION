"""
5_model/experiments/phase1_lab/plot_combined_redundancy.py

build_kernel_group_features.py의 3차(결합 raw+kernel, 시나리오별) 다중공선성 배제
(kernel_group_features_*_combined_redundancy.json, 2026-09-18 신규, v4 요구사항2)로
실제로 얼마나 많은 HI가 제거됐는지 시나리오별로 보여주는 진단 플랏.

raw HI(64개 카탈로그, 모든 시나리오 공통 폭)와 커널 HI(시나리오마다 폭이 다름 —
n_total_checked - 64)를 따로 집계해서 나란히 그린다. --input에 여러 tag의
_combined_redundancy.json을 같이 주면(예: noscen용 하나 + 6-시나리오 scen용 하나)
패널을 나란히 배치해 "라벨 조건(noscen/scen)에 따라 배제 강도가 달라지는지"를 한
그림에서 비교할 수 있다 — phase1_trainer_v2.py와 분리된 독립 스크립트, 저장된
산출물만 읽고 학습은 안 한다.

사용 예(noscen vs scen 비교):
  python 5_model/experiments/phase1_lab/plot_combined_redundancy.py \
      --input 5_model/experiments/phase1_lab/results/kernel_group_features_k25_noscen_zone_N2_v2fix_combined_redundancy.json \
              5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v2fix_combined_redundancy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

N_RAW_TOTAL = 64  # raw HI 카탈로그 폭 -- 모든 시나리오/조건에 공통(고정)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="결합(raw+kernel) 다중공선성 배제로 시나리오별 몇 개의 HI가 "
                     "제거됐는지 시각화(raw/kernel 분리 집계, 여러 조건 나란히 비교 가능)"
    )
    p.add_argument("--input", nargs="+", required=True,
                   help="build_kernel_group_features.py의 *_combined_redundancy.json 1개 이상 "
                        "(여러 개 주면 조건별 패널을 나란히 배치)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--out-name", default="combined_redundancy_removed.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    in_paths = [Path(p) for p in args.input]
    out_dir = Path(args.out_dir) if args.out_dir else in_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for in_path in in_paths:
        data = json.loads(in_path.read_text(encoding="utf-8"))
        datasets.append((in_path, data))

    n_cond = len(datasets)
    fig, axes = plt.subplots(1, n_cond, figsize=(max(6, 1.3 * max(
        len(d["by_scenario"]) for _, d in datasets)) * n_cond, 5.2), squeeze=False)
    axes = axes[0]

    for ax, (in_path, data) in zip(axes, datasets):
        by_scen = data["by_scenario"]
        scen_names = list(by_scen.keys())
        n_removed_raw = [len(v["removed_raw_idx"]) for v in by_scen.values()]
        n_removed_kernel = [len(v["removed_kernel_names"]) for v in by_scen.values()]
        n_kernel_total = [max(v["n_total_checked"] - N_RAW_TOTAL, 0) for v in by_scen.values()]

        x = np.arange(len(scen_names))
        w = 0.38
        b1 = ax.bar(x - w / 2, n_removed_raw, width=w, color="#2166ac", alpha=0.85,
                    label=f"raw HI 제거(전체 {N_RAW_TOTAL}개 중)")
        b2 = ax.bar(x + w / 2, n_removed_kernel, width=w, color="#b2182b", alpha=0.85,
                    label="kernel HI 제거(시나리오별 전체 개수 중)")
        for i, (r, k, kt) in enumerate(zip(n_removed_raw, n_removed_kernel, n_kernel_total)):
            ax.text(i - w / 2, r + 0.3, f"{r}\n({r / N_RAW_TOTAL * 100:.0f}%)",
                    ha="center", va="bottom", fontsize=7.5)
            pct = f"{k / kt * 100:.0f}%" if kt > 0 else "-"
            ax.text(i + w / 2, k + 0.3, f"{k}/{kt}\n({pct})",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(scen_names, fontsize=9, rotation=15 if len(scen_names) > 3 else 0)
        ax.set_ylabel("제거된 HI 개수")
        ax.set_ylim(0, max(n_removed_raw + n_removed_kernel, default=1) * 1.35 + 1)
        ax.set_title(f"{data.get('tag', in_path.stem)}\n"
                     f"(threshold=|r|>={data.get('threshold', 0.95)}, "
                     f"{len(scen_names)}개 시나리오)", fontsize=9.5)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("결합(raw+kernel) 다중공선성 배제로 제거된 HI 개수 — 시나리오별",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] 저장: {out_path}")

    for in_path, data in datasets:
        by_scen = data["by_scenario"]
        total_raw = sum(len(v["removed_raw_idx"]) for v in by_scen.values())
        total_kernel = sum(len(v["removed_kernel_names"]) for v in by_scen.values())
        print(f"[plot] {data.get('tag', in_path.stem)}: raw {total_raw}개/kernel {total_kernel}개 "
              f"제거(총 {len(by_scen)}개 시나리오 합산)")


if __name__ == "__main__":
    main()
