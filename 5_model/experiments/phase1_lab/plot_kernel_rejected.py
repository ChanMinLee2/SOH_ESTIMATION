"""
5_model/experiments/phase1_lab/plot_kernel_rejected.py

build_kernel_group_features.py가 만든 "_rejected.json"(어떤 raw HI 조합으로 만든 커널
후보가 왜 최종 목록에서 빠졌는지)을 시각화한다. 탈락 사유 3가지:
  - raw_conditioned_filter: 자기 그룹 raw 멤버로 이미 설명됨(--min-raw-partial-corr 미만)
  - kernel_kernel_dedup: 이미 채택된 다른 커널과 |corr|>=redundancy_threshold(중복)
  - max_features_quota: 다중공선성은 통과했지만 --max-features 쿼터 초과

시나리오별 패널로, 탈락한 커널 후보 하나당 막대 하나(그 커널을 만든 raw HI 이름들을
라벨로, train_r2를 막대 길이로, 탈락 사유를 색으로) — 탈락이 없는 시나리오는 "전부 통과"로
표시. phase1_trainer_v2.py와 분리된 독립 스크립트 — 저장된 산출물만 읽고 학습은 안 한다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_kernel_rejected.py \
      --input 5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v2_rejected.json
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

_REASON_LABEL = {
    "raw_conditioned_filter": "raw로 이미 설명됨",
    "kernel_kernel_dedup": "다른 커널과 중복",
    "max_features_quota": "쿼터 초과(다중공선성은 통과)",
}
_REASON_COLOR = {
    "raw_conditioned_filter": "#2166ac",
    "kernel_kernel_dedup": "#b2182b",
    "max_features_quota": "#e67e22",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="커널 후보 탈락(어떤 HI 조합이 왜 빠졌는지) 시각화")
    p.add_argument("--input", required=True, help="build_kernel_group_features.py의 *_rejected.json")
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
    rejected = data["rejected"]

    if not rejected:
        print(f"[plot] {in_path}: 탈락 항목이 0개입니다 — 그릴 게 없습니다"
              "(다중공선성 배제가 이미 완벽히 걸러졌다는 뜻일 수 있음).")
        return

    scenarios = sorted({r["scenario"] for r in rejected})
    n_scen = len(scenarios)
    fig, axes = plt.subplots(1, n_scen, figsize=(6 * n_scen, max(4, max(
        len([r for r in rejected if r["scenario"] == sc]) for sc in scenarios) * 0.4)))
    if n_scen == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        items = sorted((r for r in rejected if r["scenario"] == sc), key=lambda r: r["train_r2"])
        labels = ["+".join(r["member_names"]) for r in items]
        r2s = [r["train_r2"] for r in items]
        colors = [_REASON_COLOR.get(r["reason"], "#888888") for r in items]
        y_pos = range(len(items))
        ax.barh(y_pos, r2s, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("train R²(탈락 전 fit 값)")
        ax.set_title(f"{sc} — 탈락 {len(items)}개", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _REASON_COLOR.values()]
    fig.legend(handles, [_REASON_LABEL[k] for k in _REASON_COLOR], loc="upper center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"커널 후보 탈락 내역 — {data.get('tag', '')} (전체 {data['n_rejected']}개)",
                 fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout()

    out_path = out_dir / "kernel_rejected.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    by_reason = {}
    for r in rejected:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    print("[plot] 사유별 탈락 개수: " + ", ".join(f"{_REASON_LABEL.get(k, k)}={v}" for k, v in by_reason.items()))


if __name__ == "__main__":
    main()
