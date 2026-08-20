"""
5_model/experiments/phase1_lab/plot_synergy_groups.py

build_synergy_groups.py의 산출물(synergy_groups_<tag>.json)을 시각화한다.
기존 5_model 코드는 건드리지 않고, train_scr.py의 gate_probs.png 플랏과 같은 스타일
(matplotlib Agg, 시나리오별 서브플롯)을 재사용한다.

그림 2장을 만든다:
  1) 그룹 크기 분포 — 시나리오마다 "크기 1/2/3/4 그룹이 몇 개씩 있는가" 묶음 막대그래프.
     Stage4의 클러스터 개수(39~55/64)와 나란히 비교하기 좋음 — 막대가 4쪽으로 쏠릴수록
     그 시나리오는 중복(다중공선성)이 많다는 뜻.
  2) 그룹 스코어 프로파일 — 시나리오마다 가장 큰(크기 4) 그룹 상위 3개를 골라, 시드의
     개별 상관계수(1번째 막대)에서 시작해 편상관계수(2~4번째 막대)가 어떻게 이어지는지
     "폭포식" 막대로 표시. 시너지가 실제로 유의미한 크기인지(막대가 min-partial-corr
     근처에서 그냥 억지로 붙은 건 아닌지) 감으로 확인하기 좋음.

사용 예:
  python 5_model/experiments/phase1_lab/plot_synergy_groups.py \
      --input 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_test2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="synergy_groups_*.json 시각화")
    p.add_argument("--input", required=True, help="build_synergy_groups.py가 만든 JSON 경로")
    p.add_argument("--out-dir", default=None, help="PNG 저장 위치 (기본: JSON과 같은 results/ 폴더)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib 미설치 — 종료")
        return

    # 한글 라벨이 기본 폰트(DejaVu Sans)에 없어서 깨지는 문제 방지 — Windows 기본
    # 한글 폰트로 명시 지정. 못 찾으면(다른 OS 등) 조용히 기본값 유지.
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    in_path = Path(args.input)
    report = json.loads(in_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = report["tag"]

    seg_ids = sorted(
        int(k.split("_")[1]) for k in report if k.endswith("_groups")
    )
    seg_names = [report[f"seg_{s}_seg_name"] for s in seg_ids]
    max_size = report.get("max_group_size", 4)

    # ------------------------------------------------------------------
    # 그림 1 — 그룹 크기 분포 (시나리오별 묶음 막대그래프)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    size_range = list(range(1, max_size + 1))
    width = 0.8 / len(size_range)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(size_range)))

    for j, sz in enumerate(size_range):
        counts = []
        for s in seg_ids:
            groups = report[f"seg_{s}_groups"]
            counts.append(sum(1 for g in groups if len(g) == sz))
        xpos = np.arange(len(seg_ids)) + (j - (len(size_range) - 1) / 2) * width
        ax1.bar(xpos, counts, width=width, label=f"크기 {sz}", color=colors[j])

    ax1.set_xticks(range(len(seg_ids)))
    ax1.set_xticklabels(seg_names)
    ax1.set_ylabel("그룹 개수")
    ax1.set_title(f"시나리오별 시너지 그룹 크기 분포 — {tag}\n"
                   f"(막대가 오른쪽/큰 크기로 쏠릴수록 다중공선성·중복이 많다는 뜻)",
                   fontsize=11)
    ax1.legend(title="그룹 크기", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    out1 = out_dir / f"synergy_groups_sizes_{tag}.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[plot] 저장: {out1}")

    # ------------------------------------------------------------------
    # 그림 2 — 시나리오별 상위 3개(크기 최대) 그룹의 스코어 폭포 그래프
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    min_pc = report.get("min_partial_corr", 0.02)

    for ax, s, seg_name in zip(axes, seg_ids, seg_names):
        groups = report[f"seg_{s}_groups"]
        names = report[f"seg_{s}_group_names"]
        scores = report[f"seg_{s}_group_scores"]

        # 크기 큰 순 상위 3개 그룹만
        order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))[:3]

        bar_x, bar_h, bar_c, tick_labels = [], [], [], []
        x0 = 0
        for gi in order:
            for m, (nm, sc) in enumerate(zip(names[gi], scores[gi])):
                bar_x.append(x0)
                bar_h.append(sc)
                bar_c.append("steelblue" if m == 0 else ("seagreen" if abs(sc) >= min_pc else "lightgray"))
                short = nm.rsplit("_" + seg_name, 1)[0]
                tick_labels.append(short[:14])
                x0 += 1
            x0 += 1  # 그룹 사이 간격

        ax.bar(bar_x, bar_h, color=bar_c)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(min_pc, color="red", linestyle=":", linewidth=0.8, label=f"min_partial_corr={min_pc}")
        ax.axhline(-min_pc, color="red", linestyle=":", linewidth=0.8)
        ax.set_xticks(bar_x)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_title(f"{seg_name} — 최대 그룹 top-3", fontsize=10)
        ax.set_ylabel("상관계수 → 편상관계수")
        ax.legend(fontsize=6)

    fig2.suptitle(f"그룹 성장 과정 — 첫 막대(파랑)=시드 개별 상관계수, 이후(초록)=편상관계수 — {tag}",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = out_dir / f"synergy_groups_waterfall_{tag}.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] 저장: {out2}")


if __name__ == "__main__":
    main()
