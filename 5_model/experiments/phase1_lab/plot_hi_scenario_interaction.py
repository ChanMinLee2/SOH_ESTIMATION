"""
5_model/experiments/phase1_lab/plot_hi_scenario_interaction.py

test_hi_scenario_interaction.py 산출물을 시각화한다. HI 64개 각각 "시나리오 간
상관계수가 얼마나 갈리는가"(std_r_across_scenarios)를 막대그래프로 보여주고,
--min-effect-size(v4에서 shared_gate/scen_gates를 가르는 실제 기준선)를 참고선으로
같이 그린다. phase1_trainer_v2.py와 완전히 분리된 독립 스크립트 — 통계검정 결과를
읽기만 하고 학습은 안 한다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_hi_scenario_interaction.py \
      --input 5_model/experiments/phase1_lab/results/hi_scenario_interaction_k25_full_N2.json
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HI x 시나리오 상호작용 검정 결과 시각화")
    p.add_argument("--input", required=True, help="test_hi_scenario_interaction.py 산출물 경로")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


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

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    per_hi = data["per_hi"]
    threshold = data.get("min_effect_size", 0.1)

    items = sorted(per_hi.items(), key=lambda kv: kv[1]["std_r_across_scenarios"])
    names = [k for k, _ in items]
    stds = [v["std_r_across_scenarios"] for _, v in items]
    sig = [v["significant"] for _, v in items]
    colors = ["#e67e22" if s else "#2166ac" for s in sig]

    fig, ax = plt.subplots(figsize=(10, max(8, len(names) * 0.22)))
    y_pos = range(len(names))
    ax.barh(y_pos, stds, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"v4 shared/specific 경계(min_effect_size={threshold})")
    ax.set_xlabel("시나리오 간 상관계수 표준편차 (std_r_across_scenarios)")
    ax.set_title(
        f"HI x 시나리오 상호작용 — {data['n_significant']}/{data['n_hi']}개가 유의(주황=scen_gates 유지, "
        f"파랑=shared_gate 통합)\n(참고: p-value 기준 유의는 {data.get('n_p_significant_raw', '?')}/"
        f"{data['n_hi']} — 대표본이라 과다검출, 판정엔 효과크기만 사용)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "hi_scenario_interaction.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")
    print(f"[plot] shared_gate 대상(HI 불변): {data['n_hi'] - data['n_significant']}개 / "
          f"scen_gates 유지(HI 특이): {data['n_significant']}개")


if __name__ == "__main__":
    main()
