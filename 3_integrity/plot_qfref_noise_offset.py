"""
plot_qfref_noise_offset.py — q_frac_ref 센서 offset 오차(offset_amp) 도입 전/후 비교 플랏.

기존 plot_qfref_noise.py(2026-08-10, 이후 정리됨)와 같은 느낌으로, 조건별로 (ref_lag,
offset_amp)를 따로 줘서 QFracRefSegmenter._normalizer()가 실제로 반환하는 레퍼런스
값을 비교한다 — 실제 추출된 두 데이터 디렉터리(예: lag-0/offset 없음 vs
lag-1/offA-5mA)와 정확히 같은 설정 조합을 재현할 수 있도록 ref_lag도 조건마다
바꿀 수 있다. offset 오차는 "사이클 소요시간(T_cycle)에 비례하는 절대오차"라 원본
전류/시간 배열이 필요하므로(common/scenario/q_frac_ref.py 모듈 docstring "센서 offset
오차" 절 참고), 이 스크립트는 _4_data_hi/clean의 원본 v/i/t를 직접 읽어 dt를 계산한다
(q_frac_wide 캐시의 cycle pkl에는 dt가 없음 — HI만 남아있음).

사용:
  python 3_integrity/plot_qfref_noise_offset.py
  python 3_integrity/plot_qfref_noise_offset.py --conditions "기존:0:0,신규:1:0.005"
  python 3_integrity/plot_qfref_noise_offset.py --ref-lag 1 --offset-amps 0,0.005   # 구버전 호환(ref_lag 공통 고정)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for _font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _font_name
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.use_mathtext"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from common.scenario.q_frac_ref import QFracRefSegmenter  # noqa: E402
from data_directories import DATA_4_HI_ROOT  # noqa: E402

CLEAN_DIR = DATA_4_HI_ROOT / "clean"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

EXAMPLE_CELLS = [
    ("MIT", "b1c0"), ("MIT", "b2c0"), ("MIT", "b3c0"),
    ("HUST", "1-1"), ("HUST", "1-2"), ("HUST", "1-3"),
]

_PHASE_POS = 0.01
_PHASE_NEG = -0.01

_COLORS = {0.0: "#4C72B0", 0.005: "#C44E52"}
_FALLBACK = ["#55A868", "#DD8452", "#8172B2", "#937860"]


def _load_discharge_per_cycle(cell_pkl: Path) -> list[tuple[int, np.ndarray]]:
    """(cycle, dt) 목록 + q_cum(누적 방전량)을 사이클 오름차순으로 반환.

    _4_data_hi/clean 스키마는 phase 컬럼이 없어 current_A 부호로 재구성한다
    (hi_correlation._add_phase와 동일 규칙).
    """
    raw = pd.read_pickle(cell_pkl)
    df = raw["cycles"] if isinstance(raw, dict) else raw
    out = []
    for cyc, grp in df.groupby("cycle"):
        if int(cyc) == 0:
            continue
        cur = grp["current_A"].to_numpy(dtype=float)
        dis_mask = cur < _PHASE_NEG
        dis = grp.loc[dis_mask].sort_values("time_s")
        if len(dis) < 30:
            continue
        t = dis["time_s"].to_numpy(dtype=float)
        i_mag = np.abs(dis["current_A"].to_numpy(dtype=float))
        dt = np.clip(np.diff(t, prepend=t[0]), 0, None)
        q_cum = np.cumsum(i_mag * dt) / 3600.0
        if len(q_cum) == 0 or q_cum[-1] < 0.05:
            continue
        out.append((int(cyc), dt, q_cum))
    return out


def _parse_offset_amps(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def _parse_conditions(raw: str) -> list[tuple[str, int, float]]:
    """"label:ref_lag:offset_amp,..." -> [(label, ref_lag, offset_amp), ...]."""
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, ref_lag_s, offset_s = chunk.split(":")
        out.append((label, int(ref_lag_s), float(offset_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", type=str, default=None,
                        help='조건별 "label:ref_lag:offset_amp" 콤마 구분 목록 '
                             '(예: "기존:0:0,신규:1:0.005"). 주면 --ref-lag/--offset-amps 무시')
    parser.add_argument("--ref-lag", type=int, default=1,
                        help="[--conditions 없을 때만] 모든 조건 공통 ref_lag (기본 1)")
    parser.add_argument("--noise-amp", type=float, default=0.03,
                        help="레퍼런스 노이즈 최대 진폭, 분수 (기본 0.03=±3%%, 고정)")
    parser.add_argument("--noise-mode", type=str, default="ou", choices=["ou", "sine"])
    parser.add_argument("--noise-period", type=float, default=200.0)
    parser.add_argument("--offset-amps", type=str, default="0,0.005",
                        help="[--conditions 없을 때만] 비교할 offset_amp 목록(A 단위), 콤마 구분")
    parser.add_argument("--ref-seed", type=int, default=20260805)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.conditions:
        conditions = _parse_conditions(args.conditions)
    else:
        conditions = [
            (("offset=0(기존)" if a == 0 else f"offset={a*1000:.0f}mA"), args.ref_lag, a)
            for a in _parse_offset_amps(args.offset_amps)
        ]

    fig, axes = plt.subplots(len(EXAMPLE_CELLS), 2, figsize=(14, 4 * len(EXAMPLE_CELLS)), squeeze=False)
    stats_rows: list[dict] = []

    for row, (ds, cell_id) in enumerate(EXAMPLE_CELLS):
        cell_pkl = CLEAN_DIR / ds / f"{cell_id}.pkl"
        if not cell_pkl.exists():
            print(f"[스킵] 없음: {cell_pkl}")
            continue
        per_cycle = _load_discharge_per_cycle(cell_pkl)
        if not per_cycle:
            print(f"[스킵] 유효 방전 사이클 없음: {cell_id}")
            continue
        cycles = np.array([c for c, _, _ in per_cycle])

        ax_l, ax_r = axes[row]
        q_dis_raw = np.array([q[-1] for _, _, q in per_cycle])

        for idx, (label, ref_lag, offset_amp) in enumerate(conditions):
            seg = QFracRefSegmenter(
                n1=0.35, n2=0.20, n_samples=2,
                ref_lag=ref_lag, noise_amp=args.noise_amp,
                noise_mode=args.noise_mode, noise_period_cycles=args.noise_period,
                ref_seed=args.ref_seed, offset_amp=offset_amp,
            )
            key = (str(cell_id), -1)
            q_ref = np.empty(len(per_cycle))
            q_ref_raw = np.empty(len(per_cycle))
            for k, (cyc, dt, q_cum) in enumerate(per_cycle):
                q_ref_val = seg._normalizer(-1, cell_id, cyc, q_cum, dt)
                noise_frac = seg._last_noise[key]
                q_ref[k] = q_ref_val
                q_ref_raw[k] = q_ref_val / (1.0 + noise_frac)

            noise_pct = (q_ref / q_ref_raw - 1.0) * 100.0
            color = _COLORS.get(offset_amp, _FALLBACK[idx % len(_FALLBACK)])
            label_full = f"{label}(lag={ref_lag}, offset={offset_amp*1000:.0f}mA)"
            ax_r.plot(cycles, noise_pct, color=color, lw=1.1,
                      label=f"{label_full} (mean={noise_pct.mean():+.2f}%p, std={noise_pct.std():.2f}%p, "
                            f"max|.|={np.abs(noise_pct).max():.2f}%p)")
            ax_l.plot(cycles, q_ref, color=color, lw=0.9, ls="--", alpha=0.85, label=f"노이즈 후 ({label_full})")
            ax_l.plot(cycles, q_ref_raw, color=color, lw=1.0, ls=":", alpha=0.6,
                      label=f"노이즈 전 (q_ref_raw, {label})")

            stats_rows.append({
                "dataset": ds, "cell_id": cell_id, "condition": label, "ref_lag": ref_lag,
                "noise_mode": args.noise_mode, "noise_amp": args.noise_amp,
                "offset_amp": offset_amp, "n_cycles": len(per_cycle),
                "noise_pct_mean": float(noise_pct.mean()), "noise_pct_std": float(noise_pct.std()),
                "noise_pct_min": float(noise_pct.min()), "noise_pct_max": float(noise_pct.max()),
                "noise_pct_absmax": float(np.abs(noise_pct).max()),
            })
            print(f"[통계] [{ds}] {cell_id}  {label_full}  mean={noise_pct.mean():+.3f}%p  "
                  f"std={noise_pct.std():.3f}%p  범위=[{noise_pct.min():+.2f}, {noise_pct.max():+.2f}]%p")

        ax_l.plot(cycles, q_dis_raw, color="gray", lw=1.0, alpha=0.5, label="실측 q_dis (그 사이클 자신)")
        ax_l.set_title(f"[{ds}] {cell_id} — 레퍼런스 용량 (Ah)")
        ax_l.set_xlabel("사이클"); ax_l.set_ylabel("Ah")
        ax_l.legend(fontsize=8)

        ax_r.axhline(0, color="gray", ls=":", lw=1)
        ax_r.set_title(f"[{ds}] {cell_id} — 조건별 비교 (%)")
        ax_r.set_xlabel("사이클"); ax_r.set_ylabel("노이즈 (%)")
        ax_r.legend(fontsize=7, loc="upper right")

    cond_tag = "-".join(c[0] for c in conditions)
    cond_desc = ", ".join(f"{c[0]}(lag={c[1]},off={c[2]*1000:g}mA)" for c in conditions)
    fig.suptitle(f"q_frac_ref 조건별 비교 (mode={args.noise_mode}, noise_amp=±{args.noise_amp*100:.0f}%, "
                 f"period={args.noise_period:.0f}cyc)\n{cond_desc}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"qfref_noise_{args.noise_mode}_noise{int(args.noise_amp*100)}pct_{cond_tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[완료] {out_path}")

    stats_df = pd.DataFrame(stats_rows)
    stats_path = OUT_DIR / f"qfref_noise_{args.noise_mode}_noise{int(args.noise_amp*100)}pct_{cond_tag}_stats.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[완료] {stats_path}")
    print("\n[요약] 조건별 전 셀 평균(사이클 단위 평균의 평균):")
    print(stats_df.groupby("condition")[["noise_pct_mean", "noise_pct_std", "noise_pct_absmax"]].mean().to_string())


if __name__ == "__main__":
    main()
