"""
plot_cell_cycles.py

특정 셀의 전체 사이클을 한 plot에 시각화.
  위: 사이클별 용량 (열화 곡선)
  아래: 전체 사이클 방전 / 충전 V vs q_frac 오버레이 (좌: 방전, 우: 충전)
        색상: 초기 사이클(초록) → 말기 사이클(빨강)

제거 후보 표시:
  2_preprocess/outputs/shape_outlier_report.csv 가 있으면, 해당 셀의 형상 이상
  사이클(제거 후보)을 오버레이 위에 굵은 검은선으로 강조한다.
  dev(전압 RMSE) 스케일이 phase마다 달라(충전≈0.1, 방전≈0.3+) 임계를 분리함.
  또한 국소 돌출 지표(max_dev)는 RMSE와 별도 임계를 둔다 — 임계를 충분히 높게
  잡으면 초기 "초반부 국소 돌출" 사이클만 잡고 말기 열화(곡선 전반의 완만한
  이동)는 제외된다.
  편차와 무관한 절대 지표도 함께 사용:
    v_span < 임계   → 곡선 전체가 평탄한 사이클
    frac_high > 임계 → 3.6V 부근에 q_frac 상당 구간 머무는 "3.6 유지"형
  임계값은 --z-thresh / --dev-thresh-{discharge,charge}
          / --maxdev-thresh-{discharge,charge}
          / --flat-thresh-{discharge,charge}
          / --fhigh-thresh-{discharge,charge} 로 조정.

  수동 지정:
    지표 임계로 못 잡거나 반대로 과하게 잡히는 개별 사이클은
    2_preprocess/manual_outliers.csv 에 (dataset,cell_id,phase,cycle) 로 적으면
    임계와 무관하게 항상 제거 후보로 강제된다. (예: b1c23 charge 1003 만 제거)

사용:
  python 4_hi_analysis/plot_cell_cycles.py --cell b1c0
  python 4_hi_analysis/plot_cell_cycles.py --dataset hust --cell 1-1
  python 4_hi_analysis/plot_cell_cycles.py --cell b1c1 --dev-thresh-charge 0.05
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MIT_DIR  = PROJECT_ROOT / "data_postprocess" / "MIT"
# HUST_DIR = PROJECT_ROOT / "data_postprocess" / "HUST"
MIT_DIR  = PROJECT_ROOT / "data_unified" / "MIT"
HUST_DIR = PROJECT_ROOT / "data_unified" / "HUST"
STEP_DIR = Path(__file__).resolve().parent
SHAPE_CSV = PROJECT_ROOT / "2_preprocess" / "outputs" / "shape_outlier_report.csv"
MANUAL_CSV = PROJECT_ROOT / "2_preprocess" / "manual_outliers.csv"

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def load_cell(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["meta"], raw["cycles"]


def compute_qfrac(phase_df):
    if len(phase_df) < 10:
        return None, None
    t  = phase_df["time_s"].values.astype(float)
    v  = phase_df["voltage_V"].values.astype(float)
    i  = np.abs(phase_df["current_A"].values.astype(float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q_cum = np.cumsum(i * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None, None
    return q_cum / q_tot, v


def load_flagged_cycles(dataset: str, cell_id: str, z_thresh: float,
                        dev_thresh: dict, maxdev_thresh: dict,
                        flat_thresh: dict, fhigh_thresh: dict):
    """진단 CSV에서 제거 후보 사이클을 phase별 집합으로 반환.

    조건 (넷 중 하나라도 충족):
      - 전체 형상 붕괴 : z     > z_thresh AND dev     > dev_thresh[phase]
      - 국소 돌출/글리치: z_max > z_thresh AND max_dev > maxdev_thresh[phase]
      - 전체 평탄선     : v_span    < flat_thresh[phase]    (값>0 일 때만)
      - 3.6V 체류형     : frac_high > fhigh_thresh[phase]   (값>0 일 때만)

    dev(RMSE)와 max_dev(국소 최대편차)는 스케일이 다르므로 임계를 분리한다.
    특히 max_dev 임계를 RMSE 임계와 분리해 충분히 높게 두면,
    초기 "초반부 국소 돌출" 사이클만 잡고 말기 열화(곡선 전반의 완만한 이동)는
    제외할 수 있다. (예: 충전 타겟 max_dev≈0.32 통과, 말기 무리≈0.18 제외)

    v_span / frac_high 는 편차와 무관한 절대 지표:
      v_span    작음 → 곡선 전체가 한 전압에 평탄 (보통 dev/max_dev 로도 잡힘)
      frac_high 큼   → 3.6V 부근에 q_frac 상당 구간 머무는 "3.6 유지"형
                       (v_span 은 정상이라 다른 지표로는 못 잡음)
    임계<=0 이면 해당 검출을 끈다.
    Returns: {"discharge": set(...), "charge": set(...)}
    """
    empty = {"discharge": set(), "charge": set()}
    if not SHAPE_CSV.exists():
        return empty, False
    rep = pd.read_csv(SHAPE_CSV)
    has_max   = "z_max" in rep.columns and "max_dev" in rep.columns
    has_span  = "v_span" in rep.columns
    has_fhigh = "frac_high" in rep.columns
    base = rep[(rep["dataset"].str.lower() == dataset.lower())
               & (rep["cell_id"] == cell_id)]
    flagged = dict(empty)
    for phase in ("discharge", "charge"):
        ph = base[base["phase"] == phase]
        cond = (ph["z"] > z_thresh) & (ph["dev"] > dev_thresh[phase])
        if has_max:
            cond = cond | ((ph["z_max"] > z_thresh)
                           & (ph["max_dev"] > maxdev_thresh[phase]))
        if has_span and flat_thresh[phase] > 0:
            cond = cond | (ph["v_span"] < flat_thresh[phase])
        if has_fhigh and fhigh_thresh[phase] > 0:
            cond = cond | (ph["frac_high"] > fhigh_thresh[phase])
        flagged[phase] = set(ph[cond]["cycle"].astype(int))
    return flagged, True


def load_manual_flags(dataset: str, cell_id: str):
    """수동 지정 제거 목록(manual_outliers.csv)을 phase별 집합으로 반환.

    지표 임계로는 못 잡거나 반대로 과하게 잡히는 케이스를, 사람이 직접
    (dataset, cell_id, phase, cycle) 로 지정해 강제로 제거 후보에 넣는다.
    임계와 무관하게 항상 반영된다.
    """
    manual = {"discharge": set(), "charge": set()}
    if not MANUAL_CSV.exists():
        return manual
    # 사람이 손으로 편집하는 파일 — note 에 쉼표가 섞여도 깨지지 않도록
    # 앞 4개 컬럼(dataset,cell_id,phase,cycle)만 분리하고 나머지는 note 로 흡수.
    with open(MANUAL_CSV, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line or ln == 0:        # 빈 줄 / 헤더 건너뜀
                continue
            parts = line.split(",", 4)
            if len(parts) < 4:
                continue
            ds, cid, phase, cyc = (p.strip() for p in parts[:4])
            if ds.lower() != dataset.lower() or cid != str(cell_id):
                continue
            if phase not in manual:
                continue
            try:
                manual[phase].add(int(cyc))
            except ValueError:
                continue
    return manual


def plot_overlay(ax, df, cycles, phase, cmap, norm, flagged=None):
    """주어진 phase 의 전체 사이클 V-q_frac 오버레이를 ax 에 그린다.

    flagged: 제거 후보 사이클 집합 — 굵은 검은선으로 강조.
    """
    flagged = flagged or set()
    rank_of = {cyc: rank for rank, cyc in enumerate(cycles)}

    # phase로 1회 필터 후 groupby — 사이클마다 전체 df 스캔 방지
    groups = df[df["phase"] == phase].groupby("cycle")

    n_flag = 0
    flag_curves = []
    for cyc, cyc_df in groups:
        q_frac, v = compute_qfrac(cyc_df)
        if q_frac is None:
            continue
        if cyc in flagged:
            flag_curves.append((q_frac, v))   # 강조는 나중에 위에 덧그림
            continue
        ax.plot(q_frac, v, color=cmap(norm(rank_of.get(cyc, 0))), lw=0.6, alpha=0.6)

    # 제거 후보 사이클을 맨 위에 굵게 강조
    for q_frac, v in flag_curves:
        n_flag += 1
        ax.plot(q_frac, v, color="black", lw=1.8, alpha=0.95, zorder=5,
                label="제거 후보" if n_flag == 1 else None)

    phase_label = "방전" if phase == "discharge" else "충전"
    title = f"전체 사이클 {phase_label} V-q_frac 오버레이"
    if flagged:
        title += f"  (제거 후보 {n_flag}건)"
    ax.set_xlabel("q_frac (누적 용량 비율)", fontsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    if n_flag:
        ax.legend(fontsize=8, loc="best")


def main():
    parser = argparse.ArgumentParser(description="셀 전체 사이클 시각화")
    parser.add_argument("--dataset", default="mit", choices=["mit", "hust"])
    parser.add_argument("--cell",    default="b1c0")
    parser.add_argument("--z-thresh",   type=float, default=6.0,
                        help="제거 후보 z 임계값 (기본: 6.0)")
    parser.add_argument("--dev-thresh-discharge", type=float, default=0.25,
                        help="방전 dev(전압 RMSE) 임계값 (기본: 0.25)")
    parser.add_argument("--dev-thresh-charge",    type=float, default=0.07,
                        help="충전 dev(전압 RMSE) 임계값 (기본: 0.07)")
    parser.add_argument("--maxdev-thresh-discharge", type=float, default=0.40,
                        help="방전 max_dev(국소 최대편차) 임계값 (기본: 0.40)")
    parser.add_argument("--maxdev-thresh-charge",    type=float, default=0.25,
                        help="충전 max_dev(국소 최대편차) 임계값 (기본: 0.25)")
    parser.add_argument("--flat-thresh-discharge", type=float, default=0.0,
                        help="방전 v_span(전압 변동폭) 하한 — 미만이면 평탄선으로 제거 "
                             "(0=끔, 기본: 0.0)")
    parser.add_argument("--flat-thresh-charge",    type=float, default=0.0,
                        help="충전 v_span(전압 변동폭) 하한 — 미만이면 평탄선으로 제거 "
                             "(0=끔, 기본: 0.0; 전체평탄은 보통 max_dev로 이미 잡힘)")
    parser.add_argument("--fhigh-thresh-discharge", type=float, default=0.0,
                        help="방전 frac_high 상한 — 초과면 고전압 체류로 제거 (0=끔)")
    parser.add_argument("--fhigh-thresh-charge",    type=float, default=0.0,
                        help="충전 frac_high 상한 — 초과면 '3.6 유지'형으로 제거 "
                             "(0=끔, 기본: 0.0). frac_high 는 노화에 따라 매끄럽게 커져 "
                             "말기 밴드를 통째로 잡으므로 기본 비활성; 개별 사이클은 수동 목록 사용")
    args = parser.parse_args()

    dev_thresh = {"discharge": args.dev_thresh_discharge,
                  "charge":    args.dev_thresh_charge}
    maxdev_thresh = {"discharge": args.maxdev_thresh_discharge,
                     "charge":    args.maxdev_thresh_charge}
    flat_thresh = {"discharge": args.flat_thresh_discharge,
                   "charge":    args.flat_thresh_charge}
    fhigh_thresh = {"discharge": args.fhigh_thresh_discharge,
                    "charge":    args.fhigh_thresh_charge}

    data_dir = MIT_DIR if args.dataset == "mit" else HUST_DIR
    pkl_path = data_dir / f"{args.cell}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 파일 없음: {pkl_path}")

    meta, df = load_cell(pkl_path)
    cell_id  = meta.get("cell_id", args.cell)
    cycles   = sorted(df["cycle"].unique())
    n_cycles = len(cycles)

    # 제거 후보 사이클 로드 (CSV)
    flagged, has_csv = load_flagged_cycles(args.dataset, cell_id, args.z_thresh,
                                           dev_thresh, maxdev_thresh,
                                           flat_thresh, fhigh_thresh)
    # 수동 지정 목록을 합집합으로 추가 (임계와 무관하게 강제 제거)
    manual = load_manual_flags(args.dataset, cell_id)
    n_manual = len(manual["discharge"]) + len(manual["charge"])
    for phase in ("discharge", "charge"):
        flagged[phase] = flagged[phase] | manual[phase]
    flagged_any = flagged["discharge"] | flagged["charge"]

    # 사이클별 용량 (방전 기준)
    dis_all = df[df["phase"] == "discharge"]
    cap_by_cyc = dis_all.groupby("cycle")["capacity_Ah"].first().reindex(cycles)

    # 컬러맵: 초기=초록, 말기=빨강
    cmap  = cm.RdYlGn_r
    norm  = mcolors.Normalize(vmin=0, vmax=n_cycles - 1)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    ax_cap = fig.add_subplot(gs[0, :])          # 위: 용량 곡선 (전체 폭)
    ax_dis = fig.add_subplot(gs[1, 0])          # 아래 좌: 방전
    ax_chg = fig.add_subplot(gs[1, 1])          # 아래 우: 충전
    fig.suptitle(f"Cell: {cell_id}  |  전체 {n_cycles} 사이클", fontsize=12, fontweight="bold")

    # ── 위: 용량 열화 곡선 ───────────────────────────────────────────────────
    colors_cap = [cmap(norm(i)) for i in range(n_cycles)]
    ax_cap.scatter(cycles, cap_by_cyc.values, c=colors_cap, s=8, zorder=2)
    ax_cap.plot(cycles, cap_by_cyc.values, color="gray", lw=0.6, alpha=0.5, zorder=1)
    # 제거 후보를 용량 곡선 위에 빨간 X로 표시
    if flagged_any:
        cap_map = cap_by_cyc.to_dict()
        fx = sorted(c for c in flagged_any if c in cap_map and not np.isnan(cap_map[c]))
        if fx:
            ax_cap.scatter(fx, [cap_map[c] for c in fx],
                           color="black", s=40, marker="x", lw=1.2, zorder=4,
                           label=f"제거 후보 ({len(fx)}건)")
            ax_cap.legend(fontsize=8, loc="best")
    ax_cap.set_ylabel("Capacity (Ah)", fontsize=10)
    ax_cap.set_xlabel("Cycle", fontsize=10)
    ax_cap.set_title("사이클별 방전 용량", fontsize=10)
    ax_cap.grid(True, alpha=0.3)
    ax_cap.tick_params(labelsize=9)

    # ── 아래: 전체 사이클 V-q_frac 오버레이 (방전 / 충전) ───────────────────
    plot_overlay(ax_dis, df, cycles, "discharge", cmap, norm, flagged["discharge"])
    plot_overlay(ax_chg, df, cycles, "charge",    cmap, norm, flagged["charge"])

    # 컬러바
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_cap, ax_dis, ax_chg], fraction=0.02, pad=0.01)
    cbar.set_label("Cycle rank (초기→말기)", fontsize=9)
    cbar.set_ticks([0, n_cycles - 1])
    cbar.set_ticklabels([str(cycles[0]), str(cycles[-1])])

    out_dir = STEP_DIR / "cell"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"cell_cycles_{args.dataset}_{args.cell}.png"  # 방전+충전 함께
    plt.savefig(out, dpi=150, bbox_inches="tight")
    msg = f"저장: {out}"
    if has_csv:
        msg += (f"  | 제거 후보 방전={len(flagged['discharge'])} "
                f"충전={len(flagged['charge'])} "
                f"(z>{args.z_thresh}, "
                f"dev_dis>{dev_thresh['discharge']}, dev_chg>{dev_thresh['charge']}, "
                f"maxdev_dis>{maxdev_thresh['discharge']}, "
                f"maxdev_chg>{maxdev_thresh['charge']}, "
                f"flat_chg<{flat_thresh['charge']}, "
                f"fhigh_chg>{fhigh_thresh['charge']}, "
                f"수동={n_manual})")
    else:
        msg += "  | (CSV 없음 — 제거 후보 표시 생략)"
    print(msg)


if __name__ == "__main__":
    main()
