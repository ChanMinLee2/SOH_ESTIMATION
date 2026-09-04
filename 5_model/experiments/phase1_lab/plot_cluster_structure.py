"""
5_model/experiments/phase1_lab/plot_cluster_structure.py

v3.1(build_synergy_groups.py --global-dedup, 연결요소/Union-Find 방식) 그룹 구조를
직관적으로 보여주는 시각화. 시나리오마다 두 패널로 된 그림 하나:

  좌(히트맵): HI x HI raw correlation 행렬 — HI를 최종 그룹 순서로 재정렬해서 그린다.
    대각선 근처의 밝은 블록 = 그룹 내부(다중공선성으로 서로 뭉친 HI들), 블록 밖(비대각) =
    그룹 간 — v3.1이라면 |corr|>=redundancy_threshold인 비대각 칸이 전혀 없어야 한다(0건
    실측 완료, docs 참고).

  우(네트워크 그래프): HI 하나 = 노드 하나, 그룹별로 같은 색 + 같은 자리에 소집(그룹마다
    작은 원형으로 배치, 그룹들은 큰 원 위에 분산 배치). 간선은 |corr|>=--min-edge-corr인
    쌍에만 그리고, 색은 상관계수 부호/크기(발산 컬러맵), |corr|>=redundancy_threshold(=이
    관계가 실제로 두 HI를 같은 그룹으로 묶은 이유)는 굵은 실선, 그 미만(약하지만 참고할
    만한 관계)은 가는 점선으로 구분한다. 노드 크기는 그룹 성장에 실제로 참여한 survivor는
    크게, 사후 편입된(다중공선성 때문에 제거되어 대표에 귀속된) attached는 작게 그려
    "이 그룹에서 얼마나 제거됐는지"를 한눈에 보여준다.

phase1_trainer_v2.py와 분리된 독립 스크립트 — groups json과 raw 데이터만 읽고 학습은
안 한다.

사용 예(--seg-axis/--axis-config/--data-dir/--seg-data-dir은 표준 조합이면 생략 가능 —
기본값 자동 적용):
  python 5_model/experiments/phase1_lab/plot_cluster_structure.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --split-seed 42 \
      --groups-json 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_v3.json \
      --scenarios dis_hi dis_lo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from build_synergy_groups import _load_all_scenarios  # noqa: E402

# 루트는 data_directories.py의 DATA_4_HI_ROOT_STR에서 가져온다(build_synergy_groups.py/
# lambda_sweep.py와 동일 이유 — PC마다 실제 드라이브가 다름).
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

# seg-axis/axis-config도 이 세션 전체에서 한 번도 안 바뀐 고정값 — 위 데이터 경로와 세트로
# 묶인 값이라(다른 조합이면 데이터 경로도 같이 바뀌어야 함) 다른 조합을 쓰려면 셋 다
# 함께 오버라이드해야 한다.
DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v3.1 그룹 구조 시각화 — 정렬 히트맵 + 상관관계 네트워크")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR)
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--groups-json", required=True, dest="groups_json",
                    help="build_synergy_groups.py 산출물(v3.1 --global-dedup 권장 — "
                         "group_attached 필드가 있어야 attached 노드를 표시)")
    p.add_argument("--scenarios", nargs="+", default=None,
                    help="그릴 시나리오 이름(기본: json에 있는 전부)")
    p.add_argument("--redundancy-threshold", type=float, default=0.9, dest="redundancy_threshold",
                    help="이 값 이상인 간선을 굵은 실선(=다중공선성 배제를 유발한 관계)으로 표시")
    p.add_argument("--min-edge-corr", type=float, default=0.5, dest="min_edge_corr",
                    help="네트워크 그래프에 그릴 간선의 최소 |corr| (너무 낮으면 64개 노드가 "
                         "선으로 뒤덮여 못 알아봄 — 기본 0.5)")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _group_order_and_colors(groups: list[list[int]], attached: list[list[int]], n_hi: int):
    """그룹 순서대로(json에 저장된 순서 = seed 중요도순) HI 인덱스를 나열한 order와,
    각 HI가 속한 그룹 인덱스(color_of), survivor 여부(is_survivor)를 반환."""
    order: list[int] = []
    color_of: dict[int, int] = {}
    is_survivor: dict[int, bool] = {}
    boundaries: list[int] = []  # 그룹 경계(누적 크기) — 히트맵 구분선용
    for gi, (members, att) in enumerate(zip(groups, attached)):
        for m in members:
            order.append(m)
            color_of[m] = gi
            is_survivor[m] = True
        for a in att:
            order.append(a)
            color_of[a] = gi
            is_survivor[a] = False
        boundaries.append(len(order))
    return order, color_of, is_survivor, boundaries


def _plot_one_scenario(
    ax_heat, ax_net, seg_name: str, raw_corr: np.ndarray, names: list[str],
    groups: list[list[int]], attached: list[list[int]],
    redundancy_threshold: float, min_edge_corr: float,
):
    import matplotlib.pyplot as plt
    import networkx as nx

    n_hi = raw_corr.shape[0]
    order, color_of, is_survivor, boundaries = _group_order_and_colors(groups, attached, n_hi)
    n_groups = len(groups)
    n_survivor = sum(is_survivor.values())
    n_attached = n_hi - n_survivor

    # ---- 좌: 정렬 히트맵 ----
    reordered = raw_corr[np.ix_(order, order)]
    im = ax_heat.imshow(reordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for b in boundaries[:-1]:
        ax_heat.axhline(b - 0.5, color="black", linewidth=0.6, alpha=0.6)
        ax_heat.axvline(b - 0.5, color="black", linewidth=0.6, alpha=0.6)
    ax_heat.set_xticks([]); ax_heat.set_yticks([])
    ax_heat.set_title(
        f"{seg_name} — HI x HI raw corr (그룹 {n_groups}개, 경계선=그룹 구분)\n"
        f"survivor {n_survivor}개(성장 참여) + attached {n_attached}개(다중공선성으로 제거·귀속)",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="raw corr")

    # ---- 우: 네트워크 그래프 ----
    G = nx.Graph()
    for i in range(n_hi):
        G.add_node(i)

    # 그룹별 소집 배치: 그룹은 큰 원 위에, 그룹 내 HI는 그 앵커 주변 작은 원 위에
    pos: dict[int, tuple[float, float]] = {}
    for gi, (members, att) in enumerate(zip(groups, attached)):
        theta_g = 2 * np.pi * gi / max(n_groups, 1)
        R = 10.0
        anchor = np.array([R * np.cos(theta_g), R * np.sin(theta_g)])
        local = members + att
        r_local = 0.55 + 0.12 * len(local)
        for k, m in enumerate(local):
            theta_l = 2 * np.pi * k / max(len(local), 1)
            pos[m] = tuple(anchor + r_local * np.array([np.cos(theta_l), np.sin(theta_l)]))

    strong_edges, weak_edges = [], []
    for i in range(n_hi):
        for j in range(i + 1, n_hi):
            c = raw_corr[i, j]
            if abs(c) >= redundancy_threshold:
                strong_edges.append((i, j, c))
            elif abs(c) >= min_edge_corr:
                weak_edges.append((i, j, c))

    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=-1, vmax=1)

    for i, j, c in weak_edges:
        x0, y0 = pos[i]; x1, y1 = pos[j]
        ax_net.plot([x0, x1], [y0, y1], color=cmap(norm(c)), linewidth=0.8,
                    linestyle="--", alpha=0.45, zorder=1)
    for i, j, c in strong_edges:
        x0, y0 = pos[i]; x1, y1 = pos[j]
        ax_net.plot([x0, x1], [y0, y1], color=cmap(norm(c)), linewidth=1.8,
                    linestyle="-", alpha=0.85, zorder=2)

    group_cmap = plt.get_cmap("tab20")
    xs_s, ys_s, c_s = [], [], []
    xs_a, ys_a, c_a = [], [], []
    for i in range(n_hi):
        x, y = pos[i]
        col = group_cmap(color_of[i] % 20)
        if is_survivor[i]:
            xs_s.append(x); ys_s.append(y); c_s.append(col)
        else:
            xs_a.append(x); ys_a.append(y); c_a.append(col)
    ax_net.scatter(xs_s, ys_s, c=c_s, s=90, edgecolors="black", linewidths=0.8,
                    zorder=3, label=f"survivor({n_survivor})")
    ax_net.scatter(xs_a, ys_a, c=c_a, s=28, edgecolors="black", linewidths=0.4,
                    zorder=3, label=f"attached({n_attached})", marker="s")

    ax_net.set_xticks([]); ax_net.set_yticks([])
    ax_net.set_aspect("equal")
    ax_net.set_title(
        f"{seg_name} — 상관관계 네트워크(같은 색=같은 최종 그룹)\n"
        f"굵은 실선=|corr|>={redundancy_threshold}(다중공선성으로 제거된 관계), "
        f"점선=|corr|in[{min_edge_corr},{redundancy_threshold})(참고용)",
        fontsize=9,
    )
    ax_net.legend(fontsize=7, loc="upper right", markerscale=1.0)


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return
    try:
        import networkx  # noqa: F401
    except ImportError:
        print("[plot] networkx 미설치 - 종료 (pip install networkx)")
        return
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    x_all, y_all, scen_idx_all, spec, names_by_seg = _load_all_scenarios(args)
    groups_data = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.groups_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.groups_json).stem.replace("synergy_groups_", "")

    for s, seg_name in enumerate(spec.scenario_names):
        key = f"seg_{s}_groups"
        if key not in groups_data:
            continue
        if args.scenarios and seg_name not in args.scenarios:
            continue

        sel = scen_idx_all == s
        x_s = x_all[sel]
        raw_corr = np.nan_to_num(np.corrcoef(x_s, rowvar=False), nan=0.0)
        names = names_by_seg[s]
        groups = groups_data[key]
        attached = groups_data.get(f"seg_{s}_group_attached", [[] for _ in groups])

        fig, (ax_heat, ax_net) = plt.subplots(1, 2, figsize=(15, 7))
        _plot_one_scenario(
            ax_heat, ax_net, seg_name, raw_corr, names, groups, attached,
            args.redundancy_threshold, args.min_edge_corr,
        )
        fig.tight_layout()
        out_path = out_dir / f"cluster_structure_{tag}_{seg_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] 저장: {out_path}")


if __name__ == "__main__":
    main()
