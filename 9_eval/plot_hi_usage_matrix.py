"""
9_eval/plot_hi_usage_matrix.py

특정 run에서 "분류기(probe_mlp)가 쓰는 HI"와 "회귀기(scen head)가 쓰는 HI"를 나란히
비교하는 매트릭스 2장을 만든다 — plot_hi_selection_matrix.py(회귀 raw HI만, 시나리오
간 비교)를 보완: 그 스크립트는 분류기를 아예 안 다루고 커널 HI도 "시나리오 간
비교가 구조적으로 무의미하다"는 이유로 제외했는데, 이 스크립트가 보려는 건
"시나리오 간" 비교가 아니라 "같은 HI를 분류기 vs 회귀기가 얼마나 다르게 쓰는가"라서
그 제약이 적용되지 않는다(kernel_chg_mid_g2처럼 같은 커널을 두 헤드가 같은 시나리오
안에서 각자 얼마나 켜는지 비교하는 것은 의미 있음).

입력(학습이 이미 저장해 둔 산출물만 읽음, 재학습/체크포인트 로드 없음):
  <run-dir>/gates/regression_HIs.json          raw HI, 회귀기(시나리오 6개 열)
  <run-dir>/gates/classification_HIs.json      raw HI, 분류기(charge/discharge 2개 열 —
                                                probe_gate가 방향 단위라 시나리오 단위가 아님)
  <run-dir>/gates/regression_kernel_HIs.json   커널 HI, 회귀기(own-scenario)
  <run-dir>/gates/classification_kernel_HIs.json 커널 HI, 분류기(own-scenario, 2026-10-07~)
  뒤 둘 중 하나라도 없으면(구 run) 커널 패널은 건너뛴다.

출력(기본 <run-dir>/gates/):
  hi_usage_matrix_raw.png     raw HI — 회귀(6열) | 분류(2열), 같은 행 순서로 나란히.
                               행 라벨에 "(공유)"+이탤릭 표시 = 그 run이 실제로 쓴
                               interaction_json 기준 shared_gate 대상 HI(p1v2_summary.json의
                               interaction_json 경로를 읽어 자동 판정, 2026-10-08 추가).
  hi_usage_matrix_kernel.png  커널 HI — 회귀 vs 분류 2열 히트맵, 시나리오별로 묶어 행 배치
                               (커널 HI는 shared_gate 대상이 아니라 공유/전용 구분 없음)

사용 예:
    python 9_eval/plot_hi_usage_matrix.py --run-dir model_lib/results/p1v2_runs/1008_0144_p1v2_refact_seed42
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

_CATEGORY_COLORS = {
    "stat":  "#1f77b4",
    "diff":  "#ff7f0e",
    "lfp":   "#2ca02c",
    "morph": "#9467bd",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="분류기 vs 회귀기 HI 사용 매트릭스(raw+kernel)")
    p.add_argument("--run-dir", required=True, dest="run_dir",
                    help="results/p1v2_runs/<run> 디렉터리(gates/*.json이 있는 곳)")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="이 값 이상이면 '선택됨' 마커(●) 표시 (기본 0.5 — gate_prob의 "
                         "자연스러운 on/off 경계. plot_hi_selection_matrix.py의 0.9보다 낮은 "
                         "이유: 여기 목적은 '거의 확실히 쓴다' 판정이 아니라 '꺼짐과 구분되는 "
                         "켜짐'을 보는 것)")
    p.add_argument("--out-dir", default=None, dest="out_dir", help="기본: <run-dir>/gates/")
    return p.parse_args()


def _strip_seg_suffix(name: str, seg_name: str) -> str:
    suffix = f"_{seg_name}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _category_of(base_name: str) -> str:
    return base_name.split("_", 1)[0] if "_" in base_name else base_name


def _strip_known_suffix(names: list[str], seg_names: list[str]) -> list[str]:
    """classification_HIs.json은 charge/discharge 구분과 무관하게 하나의 고정
    참조 시나리오 접미사(예: 전부 "_dis_hi")로 HI 이름을 저장한다 — direction_names[0]
    ("charge"/"discharge")로는 접미사가 안 맞아 못 벗겨지므로, regression 쪽 seg_names
    후보 중 실제로 맞는 접미사를 찾아 공통으로 적용한다."""
    for seg in seg_names:
        suf = f"_{seg}"
        if names and names[0].endswith(suf):
            return [n[: -len(suf)] if n.endswith(suf) else n for n in names]
    return names


def _load_scen_matrix(json_path: Path):
    """seg_s_ranked/names/probs/seg_name 스키마(regression_HIs.json,
    regression_kernel_HIs.json, classification_kernel_HIs.json 공통) →
    (mat[n_hi, n_scen], hi_base_names, seg_names)."""
    import numpy as np
    d = json.loads(json_path.read_text(encoding="utf-8"))
    n_scen = 0
    while f"seg_{n_scen}_names" in d:
        n_scen += 1
    seg_names = [d[f"seg_{s}_seg_name"] for s in range(n_scen)]
    idx_to_name = dict(zip(d["seg_0_ranked"], d["seg_0_names"]))
    n_hi = len(idx_to_name)
    mat = np.zeros((n_hi, n_scen), dtype=float)
    for s in range(n_scen):
        idx_to_prob = dict(zip(d[f"seg_{s}_ranked"], d[f"seg_{s}_probs"]))
        for idx in range(n_hi):
            mat[idx, s] = idx_to_prob[idx]
    hi_names = [idx_to_name[i] for i in range(n_hi)]
    return mat, hi_names, seg_names


def _load_kernel_scen_lists(json_path: Path):
    """커널 HI는 raw HI와 달리 시나리오마다 로컬 카탈로그 폭(K_s)이 서로 달라서
    (own-scenario 제한, 공유 글로벌 인덱스 없음) _load_scen_matrix처럼 N_HI 고정폭
    행렬로 못 만든다 — 시나리오별 {이름: gate_prob} dict만 돌려주고, 시나리오 간
    정렬/매칭은 호출자가 이름 기준으로 직접 한다."""
    d = json.loads(json_path.read_text(encoding="utf-8"))
    n_scen = 0
    while f"seg_{n_scen}_names" in d:
        n_scen += 1
    seg_names = [d[f"seg_{s}_seg_name"] for s in range(n_scen)]
    by_scen = {s: dict(zip(d[f"seg_{s}_names"], d[f"seg_{s}_probs"])) for s in range(n_scen)}
    return by_scen, seg_names


def _load_shared_flags(run_dir: Path) -> dict[str, bool] | None:
    """해당 run이 학습 시 실제로 쓴 interaction_json(p1v2_summary.json에 기록된 경로)을
    읽어, 각 raw HI 개념(concept, 시나리오 접미사 제거된 이름)이 shared_gate로 갔는지
    (True=공유, interaction.py가 significant=False로 판정해 시나리오 간 공유) 아니면
    scen_gates/probe_gate로 전용 학습됐는지(False) 돌려준다 — train.py::t3_build_tensor_masks의
    shared_hi_mask 계산과 동일 로직(`not per_hi[c]["significant"]`). 파일을 못 찾으면
    None(그 run은 shared/specific 구분 없이 플랏 — 구 run이거나 interaction_json 미적용)."""
    summary_path = run_dir / "p1v2_summary.json"
    if not summary_path.exists():
        return None
    interaction_json = json.loads(summary_path.read_text(encoding="utf-8")).get("interaction_json")
    if not interaction_json:
        return None
    p = Path(interaction_json)
    if not p.exists():
        p = Path(__file__).resolve().parents[1] / interaction_json
    if not p.exists():
        print(f"[plot] interaction_json을 못 찾아 shared/specific 구분 생략: {interaction_json}")
        return None
    per_hi = json.loads(p.read_text(encoding="utf-8"))["per_hi"]
    return {c: not info.get("significant", False) for c, info in per_hi.items()}


def _load_probe_matrix(json_path: Path):
    """classification_HIs.json(charge_*/discharge_* 스키마, probe_gate는 방향 단위라
    시나리오가 아니라 charge/discharge 2열) → (mat[n_hi, 2], hi_base_names, ["charge","discharge"])."""
    import numpy as np
    d = json.loads(json_path.read_text(encoding="utf-8"))
    ch_idx_to_name = dict(zip(d["charge_ranked"], d["charge_names"]))
    dis_idx_to_name = dict(zip(d["discharge_ranked"], d["discharge_names"]))
    n_hi = len(ch_idx_to_name)
    mat = np.zeros((n_hi, 2), dtype=float)
    ch_idx_to_prob = dict(zip(d["charge_ranked"], d["charge_probs"]))
    dis_idx_to_prob = dict(zip(d["discharge_ranked"], d["discharge_probs"]))
    for idx in range(n_hi):
        mat[idx, 0] = ch_idx_to_prob[idx]
        mat[idx, 1] = dis_idx_to_prob[idx]
    hi_names = [ch_idx_to_name[i] for i in range(n_hi)]
    return mat, hi_names, ["charge", "discharge"]


def _draw_heatmap(ax, mat, row_labels, col_labels, threshold, title, row_colors=None,
                   row_shared=None):
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    n_rows = len(row_labels)
    ax.set_yticks(range(n_rows))
    fontsize = 7.2 if n_rows <= 70 else max(4.0, 700 / n_rows / 10)
    ax.set_yticklabels(row_labels, fontsize=fontsize)
    if row_colors is not None:
        for tick, c in zip(ax.get_yticklabels(), row_colors):
            tick.set_color(c)
    if row_shared is not None:
        # 공유(shared_gate) HI는 이탤릭으로 — 색(카테고리)과는 별개 채널이라 겹쳐 써도 됨.
        for tick, is_shared in zip(ax.get_yticklabels(), row_shared):
            if is_shared:
                tick.set_fontstyle("italic")
    for i in range(n_rows):
        for j in range(len(col_labels)):
            if mat[i, j] >= threshold:
                color = "white" if mat[i, j] < 0.55 else "black"
                ax.text(j, i, "●", ha="center", va="center", fontsize=6, color=color)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def _plot_raw(run_dir: Path, gates_dir: Path, out_dir: Path, threshold: float, run_name: str) -> None:
    import matplotlib.pyplot as plt

    reg_path = gates_dir / "regression_HIs.json"
    cls_path = gates_dir / "classification_HIs.json"
    if not (reg_path.exists() and cls_path.exists()):
        print(f"[plot] raw HI 패널 생략 — {reg_path.name}/{cls_path.name} 둘 다 있어야 함")
        return

    reg_mat, reg_names, seg_names = _load_scen_matrix(reg_path)
    reg_base = [_strip_seg_suffix(n, seg_names[0]) for n in reg_names]
    cls_mat, cls_names, dir_names = _load_probe_matrix(cls_path)
    cls_base = _strip_known_suffix(cls_names, seg_names)
    shared_flags = _load_shared_flags(run_dir)

    # 회귀기 순서(시나리오 평균 gate_prob 내림차순)를 기준 행 순서로 쓰고,
    # 분류기 쪽은 이름으로 매칭해 같은 행에 재배열 — 둘 다 같은 N_HI 카탈로그를 공유하므로
    # 이름 기준 매칭이 항상 1:1로 성립한다.
    import numpy as np
    order = np.argsort(-reg_mat.mean(axis=1))
    reg_mat, reg_base = reg_mat[order], [reg_base[i] for i in order]
    name_to_cls_row = {n: cls_mat[i] for i, n in enumerate(cls_base)}
    cls_mat_aligned = np.stack([name_to_cls_row[n] for n in reg_base])
    row_colors = [_CATEGORY_COLORS.get(_category_of(n), "#000000") for n in reg_base]
    row_shared = ([shared_flags.get(n, False) for n in reg_base]
                  if shared_flags is not None else None)
    row_labels = ([f"{n}  (공유)" if row_shared[i] else n for i, n in enumerate(reg_base)]
                  if row_shared is not None else reg_base)

    n_hi = len(reg_base)
    fig_h = max(6.0, n_hi * 0.16)
    fig, (ax_reg, ax_cls) = plt.subplots(1, 2, figsize=(10.3, fig_h),
                                          gridspec_kw={"width_ratios": [3, 1.1]})
    im = _draw_heatmap(ax_reg, reg_mat, row_labels, seg_names, threshold,
                        f"회귀기(scenario, N={n_hi})", row_colors, row_shared)
    ax_cls.set_yticks([])
    _draw_heatmap(ax_cls, cls_mat_aligned, [""] * n_hi, dir_names, threshold,
                  "분류기(direction)")

    cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                               markersize=8, label=cat)
                   for cat, c in _CATEGORY_COLORS.items()]
    marker_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                                markersize=6, label=f"선택됨(gate_prob≥{threshold})")
    fig.legend(handles=cat_handles + [marker_handle], loc="upper center",
               ncol=len(cat_handles) + 1, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"raw HI 사용 매트릭스 — 회귀기 vs 분류기 — {run_name}",
                 fontsize=12, fontweight="bold", y=1.07)
    if row_shared is not None:
        fig.text(0.5, 0.955, "이탤릭 + \"(공유)\" = shared_gate(interaction.py 비유의 "
                 "HI → 시나리오 간 게이트 공유, 굵게/직립 = 시나리오 전용 scen_gates)",
                 ha="center", va="top", fontsize=7.5, color="#555")
    fig.colorbar(im, ax=[ax_reg, ax_cls], fraction=0.025, pad=0.02, label="gate_prob")
    out_path = out_dir / "hi_usage_matrix_raw.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    print(f"\n[요약] raw HI 선택 개수(gate_prob≥{threshold}, N={n_hi}):")
    for s, sname in enumerate(seg_names):
        print(f"  회귀/{sname:<8} {int((reg_mat[:, s] >= threshold).sum()):>3}/{n_hi}")
    for j, dname in enumerate(dir_names):
        print(f"  분류/{dname:<8} {int((cls_mat_aligned[:, j] >= threshold).sum()):>3}/{n_hi}")


def _plot_kernel(gates_dir: Path, out_dir: Path, threshold: float, run_name: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    reg_path = gates_dir / "regression_kernel_HIs.json"
    cls_path = gates_dir / "classification_kernel_HIs.json"
    if not (reg_path.exists() and cls_path.exists()):
        print(f"[plot] kernel HI 패널 생략 — {reg_path.name}/{cls_path.name} 둘 다 있어야 함"
              f"(분류기 커널 확장은 2026-10-07 이후 run부터 존재)")
        return

    reg_by_scen, seg_names = _load_kernel_scen_lists(reg_path)
    cls_by_scen, cls_seg_names = _load_kernel_scen_lists(cls_path)
    assert seg_names == cls_seg_names, "회귀/분류 커널 시나리오 이름 순서가 다름 — run이 섞인 듯"

    # 커널 HI는 own-scenario라 각 (행=HI)가 자기 시나리오에만 존재한다(다른 시나리오엔
    # 아예 없음, raw HI처럼 "다른 열은 0"이 아니라 애초에 그 열 자체가 없음) —
    # 시나리오별로 모아 "그 시나리오 안에서" 회귀/분류 gate_prob 2열만 비교한다.
    rows, labels, row_colors, seg_dividers = [], [], [], []
    cmap_scen = plt.cm.tab10(np.linspace(0, 1, len(seg_names)))
    for s, sname in enumerate(seg_names):
        reg_d, cls_d = reg_by_scen[s], cls_by_scen[s]
        names_sorted = sorted(reg_d.keys(), key=lambda n: -reg_d[n])
        for n in names_sorted:
            rows.append([reg_d[n], cls_d.get(n, 0.0)])
            labels.append(n)
            row_colors.append(cmap_scen[s])
        seg_dividers.append(len(rows))

    mat = np.array(rows)
    n_k = len(labels)
    fig_h = max(6.0, n_k * 0.16)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, fig_h))
    im = _draw_heatmap(ax, mat, labels, ["회귀", "분류"], threshold,
                        f"커널 HI — 회귀기 vs 분류기 (N={n_k}, own-scenario)", row_colors)
    for d in seg_dividers[:-1]:
        ax.axhline(d - 0.5, color="white", linewidth=1.2)

    scen_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap_scen[s],
                                markersize=8, label=sname)
                    for s, sname in enumerate(seg_names)]
    marker_handle = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                                markersize=6, label=f"선택됨(gate_prob≥{threshold})")
    fig.legend(handles=scen_handles + [marker_handle], loc="upper center",
               ncol=min(len(scen_handles) + 1, 7), fontsize=7.5, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"kernel HI 사용 매트릭스 — 회귀기 vs 분류기 — {run_name}",
                 fontsize=12, fontweight="bold", y=1.1)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="gate_prob")
    out_path = out_dir / "hi_usage_matrix_kernel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    n_reg_on = int((mat[:, 0] >= threshold).sum())
    n_cls_on = int((mat[:, 1] >= threshold).sum())
    print(f"\n[요약] kernel HI 선택 개수(gate_prob≥{threshold}, N={n_k}):")
    print(f"  회귀 {n_reg_on}/{n_k}   분류 {n_cls_on}/{n_k}")


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return
    for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = _font
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    run_dir = Path(args.run_dir)
    gates_dir = run_dir / "gates"
    out_dir = Path(args.out_dir) if args.out_dir else gates_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_raw(run_dir, gates_dir, out_dir, args.threshold, run_dir.name)
    _plot_kernel(gates_dir, out_dir, args.threshold, run_dir.name)


if __name__ == "__main__":
    main()
