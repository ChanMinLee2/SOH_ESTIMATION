"""
plot_cycle_segments.py

특정 셀·사이클의 충전 → 방전 세그먼트를 3행 단일 plot으로 시각화.
  위  : Voltage vs time_s  (세그먼트 밴드 + charge/discharge 구분)
  중간: Current vs time_s
  아래: 전 사이클 방전 용량 열화 곡선 (선택 사이클 강조 표시)

사용:
  python 4_hi_analysis/plot_cycle_segments.py
  python 4_hi_analysis/plot_cycle_segments.py --cell b1c0 --cycle 2
  python 4_hi_analysis/plot_cycle_segments.py --dataset hust --cell 1-1 --cycle 5
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIT_DIR  = PROJECT_ROOT / "_1_data_unified" / "MIT"
HUST_DIR = PROJECT_ROOT / "_1_data_unified" / "HUST"
STEP_DIR = Path(__file__).resolve().parent

for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = _font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

SEG_BOUNDS = [0.0, 0.4, 0.7, 1.0]

CHG_SEGS = [
    ("#f9e79f", "Chg SoC 0~30%"),
    ("#a9dfbf", "Chg SoC 30~60%"),
    ("#aed6f1", "Chg SoC 60~100%"),
]

DIS_SEGS = [
    ("#aed6f1", "Dis SoC 60~100%"),
    ("#a9dfbf", "Dis SoC 30~60%"),
    ("#f9e79f", "Dis SoC 0~30%"),
]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_cell(pkl_path: Path):
    """전체 DataFrame + meta 반환."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    return raw["meta"], raw["cycles"]


def compute_seg_times(phase_df):
    """q_frac 경계(0.4, 0.7)에 해당하는 time_s 값 반환.
    Returns: (t 배열, q_tot_Ah, [t_bound_0.4, t_bound_0.7]) or None."""
    if len(phase_df) < 10:
        return None
    t  = phase_df["time_s"].values.astype(float)
    i  = np.abs(phase_df["current_A"].values.astype(float))
    dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
    q_cum = np.cumsum(i * dt) / 3600.0
    q_tot = float(q_cum[-1])
    if q_tot < 0.05:
        return None
    q_frac = q_cum / q_tot

    boundary_times = []
    for b in SEG_BOUNDS[1:-1]:  # 0.4, 0.7
        idx = int(np.clip(np.searchsorted(q_frac, b), 0, len(t) - 1))
        boundary_times.append(float(t[idx]))

    return t, q_tot, boundary_times


def draw_seg_bands(ax, t, boundary_times, seg_defs, label_top=False):
    """세그먼트 배경 밴드 + 경계선. boundary_times: [t_0.4, t_0.7]."""
    t_bounds = [t[0]] + boundary_times + [t[-1]]
    for i, (color, label) in enumerate(seg_defs):
        t0, t1 = t_bounds[i], t_bounds[i + 1]
        ax.axvspan(t0, t1, color=color, alpha=0.28, zorder=0)
        if label_top:
            tc = (t0 + t1) / 2
            ax.text(tc, 0.98, label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5,
                    color="#444444", fontweight="bold", linespacing=1.3)
    for tb in boundary_times:
        ax.axvline(tb, color="#999999", lw=1.0, ls="--", zorder=1)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="충전+방전 세그먼트 + 용량 열화 3행 plot 시각화")
    parser.add_argument("--dataset", default="mit", choices=["mit", "hust"])
    parser.add_argument("--cell",  default="b1c0")
    parser.add_argument("--cycle", type=int, default=2)
    args = parser.parse_args()

    data_dir = MIT_DIR if args.dataset == "mit" else HUST_DIR
    pkl_path = data_dir / f"{args.cell}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 파일 없음: {pkl_path}")

    meta, df_all = load_cell(pkl_path)
    cell_id = meta.get("cell_id", args.cell)

    cyc_df = df_all[df_all["cycle"] == args.cycle]
    if len(cyc_df) == 0:
        available = sorted(df_all["cycle"].unique())
        raise ValueError(f"cycle {args.cycle} 없음. 사용 가능: {available[:10]}...")

    # ── 전 사이클 용량 시리즈 ────────────────────────────────────────────────
    dis_all  = df_all[df_all["phase"] == "discharge"]
    cap_ser  = dis_all.groupby("cycle")["capacity_Ah"].first().dropna().sort_index()
    all_cycs = cap_ser.index.to_numpy()
    all_caps = cap_ser.values

    # 선택 사이클 용량
    sel_cap = float(cap_ser[args.cycle]) if args.cycle in cap_ser.index else np.nan

    chg_df = cyc_df[cyc_df["phase"] == "charge"]
    dis_df = cyc_df[cyc_df["phase"] == "discharge"]

    chg_info = compute_seg_times(chg_df)
    dis_info = compute_seg_times(dis_df)

    # ── Figure (3행) ─────────────────────────────────────────────────────────
    fig, (ax_v, ax_i, ax_c) = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1, 1.2], "hspace": 0.42},
        constrained_layout=False,
    )
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.08, right=0.97,
                        hspace=0.42)
    fig.suptitle(
        f"Cell: {cell_id}  |  Cycle: {args.cycle}  "
        f"{'|  Q=' + f'{sel_cap:.4f} Ah' if np.isfinite(sel_cap) else ''}",
        fontsize=12, fontweight="bold",
    )

    # ── 세그먼트 밴드 + 데이터 곡선 (V, I) ──────────────────────────────────
    for ax, label_top in [(ax_v, True), (ax_i, False)]:
        if chg_info is not None:
            t_chg, _, bt_chg = chg_info
            draw_seg_bands(ax, t_chg, bt_chg, CHG_SEGS, label_top)
        if dis_info is not None:
            t_dis, _, bt_dis = dis_info
            draw_seg_bands(ax, t_dis, bt_dis, DIS_SEGS, label_top)

        if chg_info is not None and dis_info is not None:
            t_sep = float(chg_info[0][-1])
            ax.axvline(t_sep, color="#c0392b", lw=1.8, zorder=3)

    if chg_info is not None:
        t_chg, q_tot_chg, _ = chg_info
        ax_v.scatter(t_chg, chg_df["voltage_V"].values, color="#2980b9",
                     s=2, label=f"Charge  Q={q_tot_chg:.4f} Ah")
        ax_i.scatter(t_chg, chg_df["current_A"].values, color="#2980b9", s=2)

    if dis_info is not None:
        t_dis, q_tot_dis, _ = dis_info
        ax_v.scatter(t_dis, dis_df["voltage_V"].values, color="#2c3e50",
                     s=2, label=f"Discharge  Q={q_tot_dis:.4f} Ah")
        ax_i.scatter(t_dis, dis_df["current_A"].values, color="#2c3e50", s=2)

    if chg_info is not None and dis_info is not None:
        t_chg_mid = (chg_info[0][0] + chg_info[0][-1]) / 2
        t_dis_mid = (dis_info[0][0] + dis_info[0][-1]) / 2
        for ax in (ax_v, ax_i):
            ax.text(t_chg_mid, 0.04, "← Charge →",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#2980b9", alpha=0.8)
            ax.text(t_dis_mid, 0.04, "← Discharge →",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9, color="#2c3e50", alpha=0.8)

    ax_i.axhline(0, color="black", lw=0.8, ls=":", zorder=1)

    ax_v.set_ylabel("Voltage (V)", fontsize=10)
    ax_v.legend(loc="lower right", fontsize=9)
    ax_v.grid(True, alpha=0.3)
    ax_v.tick_params(labelbottom=False, labelsize=9)

    ax_i.set_ylabel("Current (A)", fontsize=10)
    ax_i.set_xlabel("Time (s)", fontsize=10)
    ax_i.grid(True, alpha=0.3)
    ax_i.tick_params(labelsize=9)

    # ── 용량 열화 곡선 (ax_c) ────────────────────────────────────────────────
    n_cycs = len(all_cycs)
    norm   = plt.Normalize(vmin=0, vmax=max(n_cycs - 1, 1))
    cmap   = plt.cm.RdYlGn_r

    ax_c.scatter(all_cycs, all_caps,
                 c=[cmap(norm(i)) for i in range(n_cycs)],
                 s=4, zorder=2, alpha=0.7)
    ax_c.plot(all_cycs, all_caps, color="lightgray", lw=0.6, zorder=1)

    # 선택 사이클 강조
    if np.isfinite(sel_cap):
        ax_c.scatter([args.cycle], [sel_cap],
                     color="#e74c3c", s=120, zorder=5,
                     marker="*", label=f"Cycle {args.cycle}  ({sel_cap:.4f} Ah)")
        ax_c.axvline(args.cycle, color="#e74c3c", lw=1.2, ls="--", alpha=0.7, zorder=4)
        ax_c.annotate(
            f"  cycle {args.cycle}\n  {sel_cap:.4f} Ah",
            xy=(args.cycle, sel_cap),
            xytext=(args.cycle + max(n_cycs * 0.03, 5), sel_cap),
            fontsize=8.5, color="#c0392b", va="center",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8),
        )

    # SOH 참고선 (초기 용량 기준 80%)
    if len(all_caps) > 0:
        cap_init = float(all_caps[0])
        eol_cap  = cap_init * 0.80
        ax_c.axhline(eol_cap, color="#999999", lw=0.8, ls=":",
                     label=f"80% SOH ({eol_cap:.3f} Ah)")

    ax_c.set_xlabel("Cycle", fontsize=10)
    ax_c.set_ylabel("Discharge Capacity (Ah)", fontsize=10)
    ax_c.set_title("방전 용량 열화 곡선 (전 사이클)", fontsize=9, pad=4)
    ax_c.legend(fontsize=8.5, loc="upper right")
    ax_c.grid(True, alpha=0.3)
    ax_c.tick_params(labelsize=9)

    # 컬러바 (cycle 진행)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_c, fraction=0.018, pad=0.01)
    cbar.set_label("Cycle rank", fontsize=8)
    if n_cycs > 1:
        cbar.set_ticks([0, n_cycs - 1])
        cbar.set_ticklabels([str(all_cycs[0]), str(all_cycs[-1])])

    # 세그먼트 범례 (하단 중앙)
    patches = [
        mpatches.Patch(color="#aed6f1", alpha=0.6, label="SoC 60~100% (고전압)"),
        mpatches.Patch(color="#a9dfbf", alpha=0.6, label="SoC 30~60% (플래토)"),
        mpatches.Patch(color="#f9e79f", alpha=0.6, label="SoC 0~30% (저전압)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.01))

    out_dir = STEP_DIR / "segment"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"segment_{args.dataset}_{args.cell}_cycle{args.cycle}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
