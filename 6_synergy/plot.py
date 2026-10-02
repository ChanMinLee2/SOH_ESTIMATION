"""
6_synergy/plot.py

synergy.py(Step 6)의 산출물(synergy_groups_{tag}.json)을 시각화한다. 2026-09-21에
구 5_model/experiments/phase1_lab/plot_synergy_groups.py가 삭제됐는데, 그 기능이
필요해 복원 — 다만 경로/태그 해석은 이번 리팩토링 관례(parameters.py 단일 소스,
--out-dir만 예외로 CLI 유지, 5_interaction/plot.py와 동일 패턴)에 맞춰 다시 짰다.

그림 3종을 만든다(JSON만 읽는 1)/2)와, raw 세그먼트 데이터를 다시 로드해야 하는 3)으로
나뉜다 — 3)이 전체 실행 시간의 대부분을 차지한다):
  1) 그룹 크기 분포 — 시나리오마다 "크기 1/2/3/4 그룹이 몇 개씩 있는가" 묶음 막대그래프.
     막대가 큰 크기 쪽으로 쏠릴수록 그 시나리오는 HI끼리 시너지(또는 중복)가 많다는 뜻.
  2) 그룹 스코어 폭포 그래프 — 시나리오마다 가장 큰 그룹 상위 3개를 골라, 시드의 단순
     상관계수(1번째 막대)에서 편상관계수(2번째~, 그룹에 추가된 순서)가 어떻게 이어지는지
     표시 — 시너지가 min_partial_corr 근처에서 억지로 붙은 건 아닌지 감으로 확인.
  3) 그룹 구조(시나리오마다 1장, 좌/우 2패널) — 좌: HI x HI raw correlation 히트맵을 최종
     그룹 순서로 재정렬(대각선 블록=그룹 내부, 블록 밖에 |corr|>=redundancy_threshold가
     있으면 다중공선성 배제가 새는 중). 우: 같은 그룹=같은 색으로 소집한 상관관계
     네트워크 그래프(굵은 실선=다중공선성으로 제거된 관계, 점선=참고용 약한 관계,
     큰 원=그룹 성장에 참여한 survivor, 작은 사각형=다중공선성으로 제거돼 대표에 귀속된
     attached). synergy_groups_{tag}.json엔 raw correlation 행렬 자체가 없어서(멤버
     인덱스/스코어만 저장) interaction.py 소유 _load_all_scenarios로 세그먼트 데이터를
     다시 로드해 그 자리에서 재계산한다(중복 구현 금지 원칙, synergy.py main()과 동일
     SimpleNamespace 패턴). networkx가 optional dependency라 없으면 이 그림만 건너뛴다.

synergy.py가 저장하는 JSON은 output 폴더가 두 갈래다: run_pipeline.py가 호출하면
여러 스텝이 공유하는 run_dir, 단독 실행이면 RESULTS_DIR(model_lib/results) — 이
스크립트도 동일하게 --out-dir(선택, synergy.py와 같은 값을 넘기면 됨)로 그 경로를
받는다(synergy.py 자체의 --out-dir 예외 사유와 동일, 모듈 docstring 참고).

실행(synergy.py가 먼저 실행돼 synergy_groups_{tag}.json이 있어야 함):
    python 6_synergy/plot.py
    python 6_synergy/plot.py --out-dir <run_pipeline.py가 쓴 run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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

# _load_all_scenarios는 interaction.py(Step 5) 소유 — synergy.py main()과 동일한 이유로
# 가져다 쓴다(중복 구현 금지, 2026-09-30 의존 방향 정리 참고).
sys.path.insert(0, str(PROJECT_ROOT / "5_interaction"))
from interaction import _load_all_scenarios  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="synergy_groups_{tag}.json 시각화")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="synergy.py가 산출물을 저장한 폴더(기본: results/) — "
                        "run_pipeline.py로 돌렸다면 그때 쓴 run_dir을 그대로 넘기면 된다.")
    return p.parse_args()


def _plot_group_size_distribution(report: dict, seg_ids: list[int], seg_names: list[str],
                                   tag: str, out_path: Path) -> None:
    """시나리오별 그룹 크기(1~max_group_size) 분포 묶음 막대그래프."""
    max_size = report.get("max_group_size", 4)
    size_range = list(range(1, max_size + 1))
    width = 0.8 / len(size_range)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(size_range)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, sz in enumerate(size_range):
        counts = [sum(1 for g in report[f"seg_{s}_groups"] if len(g) == sz) for s in seg_ids]
        xpos = np.arange(len(seg_ids)) + (j - (len(size_range) - 1) / 2) * width
        ax.bar(xpos, counts, width=width, label=f"크기 {sz}", color=colors[j])

    ax.set_xticks(range(len(seg_ids)))
    ax.set_xticklabels(seg_names)
    ax.set_ylabel("그룹 개수")
    ax.set_title(f"시나리오별 시너지 그룹 크기 분포 — {tag}\n"
                 f"(막대가 오른쪽/큰 크기로 쏠릴수록 다중공선성·시너지가 많다는 뜻)",
                 fontsize=11)
    ax.legend(title="그룹 크기", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_group_score_waterfall(report: dict, seg_ids: list[int], seg_names: list[str],
                                 tag: str, out_path: Path) -> None:
    """시나리오별 최대 그룹 top-3의 멤버별 스코어(시드=단순상관, 이후=편상관) 폭포 막대."""
    min_pc = report.get("min_partial_corr", 0.02)
    n = len(seg_ids)
    n_cols = min(3, n) or 1
    n_rows = -(-n // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, s, seg_name in zip(axes, seg_ids, seg_names):
        groups = report[f"seg_{s}_groups"]
        names = report[f"seg_{s}_group_names"]
        scores = report[f"seg_{s}_group_scores"]
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

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"그룹 성장 과정 — 첫 막대(파랑)=시드 개별 상관계수, 이후(초록/회색)=편상관계수 — {tag}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _group_order_and_colors(groups: list[list[int]], attached: list[list[int]], n_hi: int):
    """그룹 순서대로(json 저장 순서 = seed 중요도순) HI 인덱스를 나열한 order와,
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


def _plot_cluster_structure_scenario(
    ax_heat, ax_net, seg_name: str, raw_corr: np.ndarray,
    groups: list[list[int]], attached: list[list[int]],
    redundancy_threshold: float, min_edge_corr: float,
) -> None:
    """시나리오 하나의 그룹 구조 — 좌(정렬 히트맵) + 우(상관관계 네트워크)."""
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


def _plot_cluster_structure(report: dict, seg_ids: list[int], seg_names: list[str],
                             tag: str, out_dir: Path) -> None:
    """시나리오별 그룹 구조(정렬 히트맵 + 네트워크) — raw correlation 재계산이 필요해
    세그먼트 데이터를 다시 로드한다(JSON엔 멤버 인덱스/스코어만 있고 행렬 자체는 없음)."""
    redundancy_threshold = P.ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD
    min_edge_corr = P.FIXED_SYNERGY_PLOT_MIN_EDGE_CORR

    split_seed = P.ACTIVE_SPLIT_SEED if P.ACTIVE_SPLIT_SEED is not None else P.FIXED_DEFAULT_SEED
    _loader_args = SimpleNamespace(
        data_dir=P.FIXED_CANONICAL_DATA_DIR, seg_data_dir=P.FIXED_CANONICAL_SEG_DATA_DIR,
        datasets=P.FIXED_CANONICAL_DATASETS, split_seed=split_seed,
        axis_config=json.dumps(P.ACTIVE_AXIS_CONFIG), seg_axis=P.FIXED_SEG_AXIS,
    )
    print("[plot_synergy] 그룹 구조 플롯 — raw correlation 재계산을 위해 세그먼트 데이터 로드 중...")
    x_all, _y_all, scen_idx_all, _spec, names_by_seg, _cell_ids = _load_all_scenarios(_loader_args)

    for s, seg_name in zip(seg_ids, seg_names):
        group_names = report[f"seg_{s}_group_names"]
        attached_names = report.get(f"seg_{s}_group_attached_names", [[] for _ in group_names])

        # JSON의 멤버는 정수 인덱스로 저장돼 있지만, HI 카탈로그 순서가 저장 시점과
        # 지금 사이에 바뀔 수 있어(피처 추가/재정렬 — 실측으로 확인됨, 2026-10-01) 그
        # 정수 인덱스를 그대로 재사용하면 엉뚱한 HI를 가리키게 된다. 그래서 이름으로
        # 다시 매핑: 지금 로드된 names_by_seg[s]에서 "이 시나리오의 그룹에 실제로 등장한
        # HI 이름들"만 뽑아 로컬 인덱스를 새로 부여하고, raw_corr도 그 열만 골라서 계산한다
        # (전체 카탈로그가 아니라 이 그룹 구성에 쓰인 HI만 — 신규로 추가된 미분류 HI는 이
        # 그림에서 다룰 대상이 아니므로 제외).
        current_idx = {nm: i for i, nm in enumerate(names_by_seg[s])}
        used_names: list[str] = []
        for gn in group_names:
            used_names.extend(gn)
        for an in attached_names:
            used_names.extend(an)
        missing = [nm for nm in used_names if nm not in current_idx]
        if missing:
            print(f"[plot_synergy] {seg_name}: 저장된 그룹의 HI {len(missing)}개가 지금 카탈로그에 "
                  f"없음(이름 변경/삭제로 추정) — 이 시나리오는 건너뜀: {missing[:5]}...")
            continue
        col_idx = [current_idx[nm] for nm in used_names]
        local_idx = {nm: i for i, nm in enumerate(used_names)}

        sel = scen_idx_all == s
        x_sub = x_all[sel][:, col_idx]
        raw_corr = np.nan_to_num(np.corrcoef(x_sub, rowvar=False), nan=0.0)

        groups = [[local_idx[nm] for nm in gn] for gn in group_names]
        attached = [[local_idx[nm] for nm in an] for an in attached_names]

        fig, (ax_heat, ax_net) = plt.subplots(1, 2, figsize=(15, 7))
        _plot_cluster_structure_scenario(
            ax_heat, ax_net, seg_name, raw_corr, groups, attached,
            redundancy_threshold, min_edge_corr,
        )
        fig.tight_layout()
        out_path = out_dir / f"cluster_structure_{tag}_{seg_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot_synergy] 저장: {out_path}")


def main() -> None:
    args = _parse_args()
    tag = P.FIXED_SYNERGY_TAG or f"{P.ACTIVE_P1_TAG}_groups"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    json_path = out_dir / f"synergy_groups_{tag}.json"
    report = json.loads(json_path.read_text(encoding="utf-8"))

    seg_ids = sorted(int(k.split("_")[1]) for k in report if k.endswith("_groups"))
    seg_names = [report[f"seg_{s}_seg_name"] for s in seg_ids]
    print(f"[plot_synergy] tag={tag}, 시나리오 {len(seg_ids)}개: {seg_names}")

    out1 = out_dir / f"synergy_groups_sizes_{tag}.png"
    _plot_group_size_distribution(report, seg_ids, seg_names, tag, out1)
    print(f"[plot_synergy] 저장: {out1}")

    out2 = out_dir / f"synergy_groups_waterfall_{tag}.png"
    _plot_group_score_waterfall(report, seg_ids, seg_names, tag, out2)
    print(f"[plot_synergy] 저장: {out2}")

    try:
        import networkx  # noqa: F401
    except ImportError:
        print("[plot_synergy] networkx 미설치 — 그룹 구조(히트맵+네트워크) 그림은 건너뜀 "
              "(pip install networkx)")
    else:
        _plot_cluster_structure(report, seg_ids, seg_names, tag, out_dir)


if __name__ == "__main__":
    main()
