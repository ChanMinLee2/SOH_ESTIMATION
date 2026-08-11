"""
plot_qfref_noise.py — q_frac_ref 노이즈 주입 전/후 비교 플랏 (다중 noise_amp 비교).

QFracRefSegmenter._normalizer()가 실제로 반환하는 값(=세그먼트 경계를 나누는 분모)을
노이즈 주입 전(q_ref_raw = lag만큼 과거 q_tot)과 후(q_ref = q_ref_raw * (1+noise_frac))로
나란히 그린다. 실제 셀의 실측 방전 적산값(q_dis, hi_correlation.py의 q_cum[-1]과 동일
정의 — _4_data_hi/.../cycle/*.pkl에 이미 계산돼 있음)을 그대로 써서, 진짜 열화 곡선
위에 노이즈가 어떻게 얹히는지 현실적으로 보여준다.

2026-08-10 확장:
  - `--noise-amps`로 여러 진폭(기본 1/3/5/10%)을 한 그림에서 비교 — 오른쪽 패널에
    진폭별 노이즈(%) 곡선을 겹쳐 그린다(레퍼런스 용량 자체는 진폭과 무관하므로 왼쪽
    패널은 1개만).
  - 예시 셀에 "고정 바이어스가 음수 쪽 극단인 셀"을 자동으로 하나 찾아 추가한다 —
    `_noise_params_for()`의 `bias = uniform(-0.5,0.5)*noise_amp`가 셀·방향별로 결정론적
    이라(시드가 noise_amp에 안 걸림), 이 스크립트가 직접 그 원시 계수를 재계산해 가장
    음수에 가까운 셀을 고른다.
  - 사이클 단위 평균 노이즈(mean/std/min/max, %)를 각 (셀, 진폭)마다 콘솔에 출력하고
    `outputs/qfref_noise_stats.csv`에 저장한다.

사용:
  python 3_integrity/plot_qfref_noise.py
  python 3_integrity/plot_qfref_noise.py --noise-amps 0.01,0.03,0.05,0.10
  python 3_integrity/plot_qfref_noise.py --noise-mode sine --noise-period 200
  python 3_integrity/plot_qfref_noise.py --ref-lag 5 --noise-amps 0.03

주의: QFracRefSegmenter._normalizer()는 ou 모드에서 호출마다 내부 상태를 한 스텝
전진시키는 부작용이 있다 — 노이즈 값을 다시 보고 싶다고 _noise_frac()을 별도로
재호출하면 상태가 이중으로 전진해 실제 파이프라인이 쓰는 값과 달라진다. 이 스크립트는
_normalizer() 호출 후 seg._last_noise에 기록된 값만 읽어 이 문제를 피한다. 진폭별로
**별도의 QFracRefSegmenter 인스턴스**를 새로 만들어(=OU 상태를 처음부터 다시 시작)
이 부작용이 진폭 간에 서로 오염되지 않게 한다.
"""

from __future__ import annotations

import argparse
import sys
import zlib
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

CYCLE_DIR = DATA_4_HI_ROOT / "q_frac_wide" / "n1-35%_n2-20%_N-4" / "cycle"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

EXAMPLE_CELLS = [
    ("MIT", "b1c0"), ("MIT", "b2c0"), ("MIT", "b3c0"),
    ("HUST", "1-1"), ("HUST", "1-2"), ("HUST", "1-3"),
]  # 데이터셋별 최소 3개 고정 예시(음수 바이어스/진동 예시는 아래에서 자동 탐색해 추가)

_NOISE_COLORS = {0.01: "#4C72B0", 0.03: "#55A868", 0.05: "#DD8452", 0.10: "#C44E52"}
_FALLBACK_COLORS = ["#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]


def _raw_bias_factor(cell_id: str, direction: int, ref_seed: int) -> float:
    """QFracRefSegmenter._noise_params_for()의 bias 원시 계수(uniform(-0.5,0.5))만 재계산.

    noise_amp에 안 걸리는 부분이라(시드는 ref_seed:cell_id:direction만으로 결정), 진폭과
    무관하게 셀별 부호를 미리 스캔할 수 있다.
    """
    seed = zlib.crc32(f"{ref_seed}:{cell_id}:{direction}".encode())
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-0.5, 0.5))


def _find_most_negative_bias_cell(ref_seed: int, direction: int = -1) -> tuple[str, str] | None:
    """MIT/HUST 전체 셀 중 고정 바이어스 원시계수가 가장 음수인 셀 하나를 찾는다."""
    best: tuple[str, str] | None = None
    best_factor = 1.0
    for ds in ("MIT", "HUST"):
        d = CYCLE_DIR / ds
        if not d.exists():
            continue
        for pkl_path in d.glob("*.pkl"):
            cell_id = pkl_path.stem
            factor = _raw_bias_factor(cell_id, direction, ref_seed)
            if factor < best_factor:
                best_factor = factor
                best = (ds, cell_id)
    if best is not None:
        print(f"[음수 바이어스 셀 탐색] {best} 원시계수={best_factor:+.3f} "
              f"(-0.5에 가까울수록 강한 음수 바이어스)")
    return best


def _find_oscillating_bias_cell(ref_seed: int, direction: int = -1) -> tuple[str, str] | None:
    """고정 바이어스 원시계수가 0에 가장 가까운 셀을 찾는다.

    total = clip(bias + drift, ...)에서 bias가 0에 가까우면 OU 드리프트(평균회귀, 정상분포
    표준편차=noise_amp/4)가 부호를 오가는 게 그대로 total의 부호 전환으로 드러나기 쉽다 —
    "일부 구간 양수, 일부 구간 음수" 예시로 가장 적합한 후보.
    """
    best: tuple[str, str] | None = None
    best_abs = 1.0
    for ds in ("MIT", "HUST"):
        d = CYCLE_DIR / ds
        if not d.exists():
            continue
        for pkl_path in d.glob("*.pkl"):
            cell_id = pkl_path.stem
            factor = _raw_bias_factor(cell_id, direction, ref_seed)
            if abs(factor) < best_abs:
                best_abs = abs(factor)
                best = (ds, cell_id)
    if best is not None:
        print(f"[진동(0 부근 바이어스) 셀 탐색] {best} 원시계수 절댓값={best_abs:.4f} "
              f"(0에 가까울수록 드리프트가 양/음을 오갈 가능성이 큼)")
    return best


def _parse_noise_amps(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-lag", type=int, default=0,
                        help="q_frac_ref ref_lag (기본: 0 — 현재 채택된 초기 구현값)")
    parser.add_argument("--noise-amps", type=str, default="0.01,0.03,0.05,0.10",
                        help="비교할 노이즈 진폭 목록, 콤마 구분 (기본: 0.01,0.03,0.05,0.10 = ±1/3/5/10%%)")
    parser.add_argument("--noise-mode", type=str, default="ou", choices=["ou", "sine"],
                        help="노이즈 드리프트 방식 (기본: ou)")
    parser.add_argument("--noise-period", type=float, default=200.0,
                        help="노이즈 평균회귀 특성시간/파장(사이클 수, 기본 200)")
    parser.add_argument("--ref-seed", type=int, default=20260805,
                        help="노이즈 재현성 시드 (QFracRefSegmenter 기본값과 동일)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    noise_amps = _parse_noise_amps(args.noise_amps)

    # 특수 예시 셀 자동 탐색 — 중복이면 추가하지 않고 기존 셀에 태그만 덧붙임
    special_tags: dict[tuple[str, str], list[str]] = {}
    neg_cell = _find_most_negative_bias_cell(args.ref_seed)
    osc_cell = _find_oscillating_bias_cell(args.ref_seed)
    cells = list(EXAMPLE_CELLS)
    for cell, label in ((neg_cell, "음수 바이어스 예시"), (osc_cell, "진동 예시")):
        if cell is None:
            continue
        if cell not in cells:
            cells.append(cell)
        special_tags.setdefault(cell, []).append(label)

    fig, axes = plt.subplots(len(cells), 2, figsize=(14, 4 * len(cells)), squeeze=False)

    stats_rows: list[dict] = []

    for row, (ds, cell_id) in enumerate(cells):
        pkl_path = CYCLE_DIR / ds / f"{cell_id}.pkl"
        if not pkl_path.exists():
            print(f"[스킵] 없음: {pkl_path}")
            continue
        df = pd.read_pickle(pkl_path).dropna(subset=["q_dis"]).sort_values("cycle")
        cycles = df["cycle"].to_numpy()
        q_this = df["q_dis"].to_numpy(dtype=float)

        ax_l, ax_r = axes[row]
        _labels = special_tags.get((ds, cell_id))
        tag = f" [{', '.join(_labels)}]" if _labels else ""
        is_neg_example = (ds, cell_id) == neg_cell

        q_ref_raw_ref = None  # 진폭 무관 — 첫 진폭에서 한 번만 계산해 왼쪽 패널에 사용

        for amp_idx, noise_amp in enumerate(noise_amps):
            seg = QFracRefSegmenter(
                n1=0.35, n2=0.2, n_samples=4,
                ref_lag=args.ref_lag, noise_amp=noise_amp,
                noise_mode=args.noise_mode, noise_period_cycles=args.noise_period,
                ref_seed=args.ref_seed,
            )
            q_ref_raw = np.empty(len(cycles))
            q_ref     = np.empty(len(cycles))
            key = (str(cell_id), -1)
            for k, (cyc, qd) in enumerate(zip(cycles, q_this)):
                q_arr = np.array([qd])
                q_ref_val = seg._normalizer(-1, cell_id, int(cyc), q_arr)
                noise_frac = seg._last_noise[key]
                q_ref[k] = q_ref_val
                q_ref_raw[k] = q_ref_val / (1.0 + noise_frac)

            if q_ref_raw_ref is None:
                q_ref_raw_ref = q_ref_raw

            noise_pct = (q_ref / q_ref_raw - 1.0) * 100.0
            color = _NOISE_COLORS.get(noise_amp, _FALLBACK_COLORS[amp_idx % len(_FALLBACK_COLORS)])
            ax_r.plot(cycles, noise_pct, color=color, lw=1.1,
                      label=f"±{noise_amp*100:.0f}% (실효 mean={noise_pct.mean():+.2f}%p, "
                            f"std={noise_pct.std():.2f}%p)")
            # 노이즈 적용 후 실제 용량(Ah) 추이 — 왼쪽 패널에 진폭별로 겹쳐 그림(오른쪽 패널과
            # 같은 색상 사용, 얇은 점선으로 구분해 "노이즈 전" 굵은 실선과 헷갈리지 않게 함)
            ax_l.plot(cycles, q_ref, color=color, lw=0.9, ls="--", alpha=0.85,
                      label=f"노이즈 후 ±{noise_amp*100:.0f}%")

            # 부호 전환 횟수 — "일부 구간 양수, 일부 구간 음수"를 정량적으로 확인하기 위함
            _signs = np.sign(noise_pct)
            _signs = _signs[_signs != 0]
            n_sign_changes = int(np.sum(np.diff(_signs) != 0)) if len(_signs) > 1 else 0

            stats_rows.append({
                "dataset": ds, "cell_id": cell_id,
                "neg_bias_example": is_neg_example,
                "osc_bias_example": (ds, cell_id) == osc_cell,
                "ref_lag": args.ref_lag, "noise_mode": args.noise_mode,
                "noise_period_cycles": args.noise_period, "noise_amp": noise_amp,
                "n_cycles": len(cycles),
                "noise_pct_mean": float(noise_pct.mean()),
                "noise_pct_std": float(noise_pct.std()),
                "noise_pct_min": float(noise_pct.min()),
                "noise_pct_max": float(noise_pct.max()),
                "n_sign_changes": n_sign_changes,
            })
            print(f"[통계] [{ds}] {cell_id}{tag}  noise_amp=±{noise_amp*100:.0f}%  "
                  f"사이클평균={noise_pct.mean():+.3f}%p  std={noise_pct.std():.3f}%p  "
                  f"범위=[{noise_pct.min():+.2f}, {noise_pct.max():+.2f}]%p  "
                  f"부호전환={n_sign_changes}회")

        ax_l.plot(cycles, q_this, color="gray", lw=1.0, alpha=0.5, label="실측 q_dis (그 사이클 자신)")
        ax_l.plot(cycles, q_ref_raw_ref, color="#4C72B0", lw=1.3,
                  label=f"노이즈 전 (q_ref_raw, lag={args.ref_lag}, 진폭 무관)")
        ax_l.set_title(f"[{ds}] {cell_id}{tag} — 레퍼런스 용량 (Ah)")
        ax_l.set_xlabel("사이클")
        ax_l.set_ylabel("Ah")
        ax_l.legend(fontsize=8)

        ax_r.axhline(0, color="gray", ls=":", lw=1)
        ax_r.set_title(f"[{ds}] {cell_id}{tag} — 진폭별 주입 노이즈 비교 (%)")
        ax_r.set_xlabel("사이클")
        ax_r.set_ylabel("노이즈 (%)")
        ax_r.legend(fontsize=7, loc="upper right")

    amps_tag = "-".join(f"{int(a*100)}" for a in noise_amps)
    fig.suptitle(f"q_frac_ref 노이즈 주입 전/후 비교 (mode={args.noise_mode}, ref_lag={args.ref_lag}, "
                 f"noise_amps=±{{{','.join(f'{a*100:.0f}%' for a in noise_amps)}}}, "
                 f"period={args.noise_period:.0f}cyc)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / f"qfref_noise_{args.noise_mode}_lag{args.ref_lag}_amp{amps_tag}pct.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[완료] {out_path}")

    stats_df = pd.DataFrame(stats_rows)
    stats_path = OUT_DIR / f"qfref_noise_{args.noise_mode}_lag{args.ref_lag}_amp{amps_tag}pct_stats.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[완료] {stats_path}")
    print("\n[요약] 진폭별 전 셀 평균(사이클 단위 평균의 평균):")
    print(stats_df.groupby("noise_amp")[["noise_pct_mean", "noise_pct_std"]].mean().to_string())


if __name__ == "__main__":
    main()
