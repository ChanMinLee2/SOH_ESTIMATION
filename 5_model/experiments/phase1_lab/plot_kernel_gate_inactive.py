"""
5_model/experiments/phase1_lab/plot_kernel_gate_inactive.py

build_kernel_group_features.py의 "_rejected.json"(구성 단계에서 탈락한 조합)과는 다른
질문을 본다 — 최종 커널 목록에는 들어갔지만, 실제 학습(phase1_trainer_v2.py) 후 게이트가
"이 커널은 안 쓴다"(gate_prob<0.9)고 판정한 커널이 어떤 raw HI 조합으로 만들어졌는지
보여준다. 즉 "구성 단계 탈락"이 아니라 "학습 후 선택 탈락".

입력 2개를 대조한다:
  1) --gates: 학습된 run의 gates/regression_kernel_HIs.json(시나리오별 커널 gate_prob)
  2) --kernel-pkl: 그 학습에 쓰인 kernel_group_features_*.pkl(커널 이름 -> raw HI 구성 매핑)

phase1_trainer_v2.py와 분리된 독립 스크립트 — 저장된 산출물만 읽고 학습은 안 한다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_kernel_gate_inactive.py \
      --gates "5_model/experiments/phase1_lab/results/p1v2_runs/0827_1636_p1v2_p1v3_full_seed42/gates/regression_kernel_HIs.json" \
      --kernel-pkl 5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v3.pkl \
      --min-active-prob 0.9
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="학습 후 게이트가 끈 커널 HI — raw 구성과 함께 시각화")
    p.add_argument("--gates", required=True, help="학습된 run의 gates/regression_kernel_HIs.json")
    p.add_argument("--kernel-pkl", required=True, dest="kernel_pkl",
                    help="그 학습에 쓰인 kernel_group_features_*.pkl(커널명->raw 구성 매핑용)")
    p.add_argument("--min-active-prob", type=float, default=0.9, dest="min_active_prob",
                    help="이 값 미만이면 '꺼짐'으로 분류(기본 0.9, 이 문서 전체의 활성 기준과 동일)")
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

    gates_path = Path(args.gates)
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    artifact = pickle.load(open(args.kernel_pkl, "rb"))
    member_names_by_kernel = {f["name"]: f["member_names"] for f in artifact["features"]}

    n_scen = 0
    while f"seg_{n_scen}_names" in gates:
        n_scen += 1

    rows = []  # (scenario, kernel_name, member_names, prob)
    for s in range(n_scen):
        seg_name = gates.get(f"seg_{s}_seg_name", f"seg_{s}")
        names = gates[f"seg_{s}_names"]
        probs = gates[f"seg_{s}_probs"]
        for name, prob in zip(names, probs):
            members = member_names_by_kernel.get(name)
            if members is None:
                continue  # 이 시나리오 소속이 아닌 커널(다른 시나리오 커널도 게이트 폭에 다 나열되는 구조 대비)
            rows.append((seg_name, name, members, prob))

    inactive = [r for r in rows if r[3] < args.min_active_prob]
    if not inactive:
        print(f"[plot] 활성 문턱({args.min_active_prob}) 미만인 커널이 0개입니다 — 전부 켜져 있습니다.")
        return

    scenarios = sorted({r[0] for r in inactive})
    n_s = len(scenarios)
    fig, axes = plt.subplots(1, n_s, figsize=(6 * n_s, max(4, max(
        len([r for r in inactive if r[0] == sc]) for sc in scenarios) * 0.45)))
    if n_s == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        items = sorted((r for r in inactive if r[0] == sc), key=lambda r: r[3])
        labels = ["+".join(r[2]) for r in items]
        probs = [r[3] for r in items]
        colors = ["#b2182b" if p < 0.1 else "#e67e22" for p in probs]
        y_pos = range(len(items))
        ax.barh(y_pos, probs, color=colors, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.axvline(args.min_active_prob, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("gate_prob(학습 후)")
        ax.set_title(f"{sc} — 꺼진 커널 {len(items)}개", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#b2182b", "#e67e22")]
    fig.legend(handles, ["확실히 꺼짐(<0.1)", f"애매(0.1~{args.min_active_prob})"],
               loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"학습 후 게이트가 끈 커널 HI — {gates_path.parent.parent.name} "
                 f"(전체 {len(rows)}개 중 {len(inactive)}개 비활성)",
                 fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout()

    out_dir = Path(args.out_dir) if args.out_dir else gates_path.parent
    out_path = out_dir / "kernel_gate_inactive.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")
    print(f"[plot] 전체 커널 {len(rows)}개 중 비활성 {len(inactive)}개 "
          f"({100*len(inactive)/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
