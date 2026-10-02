"""
7_kernel/plot.py

kernel.py(Step 7)의 산출물(kernel_group_features_{tag}_combined_redundancy.json)을
시각화한다. 2026-09-21 커밋(893cb97)에서 구 5_model/experiments/phase1_lab/ 통째로
삭제되며 같이 없어졌던 plot_combined_redundancy.py를 복원 — 3차(결합 raw+kernel,
시나리오별) 다중공선성 배제로 실제로 몇 개의 HI가 제거됐는지 raw/kernel을 나눠
시나리오별 막대그래프로 보여준다.

원본은 --input에 여러 tag의 json을 같이 줘서(noscen용/scen용 등) 조건별 비교 패널을
나란히 배치하는 기능이 있었는데, 이번 세션의 단일-실행-단일-결과 관례
(parameters.py 단일 소스, tag는 ACTIVE_P1_TAG에서 자동 파생, --out-dir만 CLI 예외)에
맞춰 1개 tag만 그리도록 단순화했다 — 다른 조건과 비교하고 싶으면 그림을 따로
생성해서 나란히 보면 된다.

raw HI 카탈로그 폭(원본은 N_RAW_TOTAL=64로 하드코딩)은 하드코딩하지 않는다 — HI
카탈로그가 세션 도중에도 바뀔 수 있다는 걸 6_synergy/plot.py 작업에서 실측으로
확인했다(2026-10-01, cluster_structure 플롯이 저장 시점과 로드 시점의 카탈로그가
달라 KeyError로 죽었던 사례). 대신 같은 tag의 kernel_group_features_{tag}.pkl에
저장된 "그 시나리오의 최종 커널 HI 개수"를 n_total_checked에서 빼서 시나리오별로
직접 역산한다(모든 시나리오에서 일치해야 정상 — 다르면 경고만 출력하고 계속 진행).

synergy.py/kernel.py와 동일하게 --out-dir(선택, kernel.py 실행 때 쓴 값과 동일하게
넘기면 됨 — 기본은 RESULTS_DIR)로 산출물 폴더를 찾는다.

실행(kernel.py가 먼저 실행돼 pkl + _combined_redundancy.json이 있어야 함):
    python 7_kernel/plot.py
    python 7_kernel/plot.py --out-dir <run_pipeline.py가 쓴 run_dir>
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import parameters as P

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "model_lib" / "results"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="kernel_group_features_{tag}_combined_redundancy.json 시각화"
    )
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="kernel.py가 산출물을 저장한 폴더(기본: results/) — "
                        "run_pipeline.py로 돌렸다면 그때 쓴 run_dir을 그대로 넘기면 된다.")
    return p.parse_args()


def _plot_combined_redundancy(data: dict, kernel_count_by_scenario: dict[str, int],
                               tag: str, out_path: Path) -> None:
    """시나리오별 raw/kernel 제거 개수 막대그래프 1장."""
    by_scen = data["by_scenario"]
    scen_names = list(by_scen.keys())

    raw_total_by_scen = {
        seg_name: v["n_total_checked"] - kernel_count_by_scenario.get(seg_name, 0)
        for seg_name, v in by_scen.items()
    }
    if len(set(raw_total_by_scen.values())) > 1:
        print(f"[plot_kernel] 경고: 시나리오별로 역산한 raw HI 카탈로그 폭이 다름 "
              f"{raw_total_by_scen} — pkl/json 태그가 서로 다른 실행 결과이거나 "
              "카탈로그가 시나리오마다 다를 수 있음. 그래도 시나리오별 값을 각자 써서 그린다.")

    n_removed_raw = [len(v["removed_raw_idx"]) for v in by_scen.values()]
    n_removed_kernel = [len(v["removed_kernel_names"]) for v in by_scen.values()]
    n_raw_total = [raw_total_by_scen[s] for s in scen_names]
    n_kernel_total = [kernel_count_by_scenario.get(s, 0) for s in scen_names]

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(scen_names)), 5.2))
    x = np.arange(len(scen_names))
    w = 0.38
    ax.bar(x - w / 2, n_removed_raw, width=w, color="#2166ac", alpha=0.85, label="raw HI 제거")
    ax.bar(x + w / 2, n_removed_kernel, width=w, color="#b2182b", alpha=0.85, label="kernel HI 제거")
    for i, (r, k, rt, kt) in enumerate(zip(n_removed_raw, n_removed_kernel, n_raw_total, n_kernel_total)):
        rpct = f"{r / rt * 100:.0f}%" if rt > 0 else "-"
        kpct = f"{k / kt * 100:.0f}%" if kt > 0 else "-"
        ax.text(i - w / 2, r + 0.3, f"{r}/{rt}\n({rpct})", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + w / 2, k + 0.3, f"{k}/{kt}\n({kpct})", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scen_names, fontsize=9, rotation=15 if len(scen_names) > 3 else 0)
    ax.set_ylabel("제거된 HI 개수")
    ax.set_ylim(0, max(n_removed_raw + n_removed_kernel, default=1) * 1.35 + 1)
    ax.set_title(f"결합(raw+kernel) 다중공선성 배제로 제거된 HI 개수 — {tag}\n"
                 f"(threshold=|r|>={data.get('threshold', 0.95)}, {len(scen_names)}개 시나리오)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    tag = P.FIXED_KERNEL_TAG or f"{P.ACTIVE_P1_TAG}_kernel"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR

    json_path = out_dir / f"kernel_group_features_{tag}_combined_redundancy.json"
    pkl_path = out_dir / f"kernel_group_features_{tag}.pkl"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    with open(pkl_path, "rb") as fh:
        artifact = pickle.load(fh)

    kernel_count_by_scenario: dict[str, int] = {}
    for f in artifact["features"]:
        kernel_count_by_scenario[f["scenario"]] = kernel_count_by_scenario.get(f["scenario"], 0) + 1

    by_scen = data["by_scenario"]
    total_raw = sum(len(v["removed_raw_idx"]) for v in by_scen.values())
    total_kernel = sum(len(v["removed_kernel_names"]) for v in by_scen.values())
    print(f"[plot_kernel] tag={tag}, 시나리오 {len(by_scen)}개: raw {total_raw}개/"
          f"kernel {total_kernel}개 제거(결합 다중공선성 배제)")

    out_path = out_dir / f"kernel_combined_redundancy_removed_{tag}.png"
    _plot_combined_redundancy(data, kernel_count_by_scenario, tag, out_path)
    print(f"[plot_kernel] 저장: {out_path}")


if __name__ == "__main__":
    main()
