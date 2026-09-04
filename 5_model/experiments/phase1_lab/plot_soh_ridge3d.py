"""
5_model/experiments/phase1_lab/plot_soh_ridge3d.py

SOH(z) x Cycle(x) x <제3의 축>(y) 3D 곡면 플랏.
(사용자가 준 레퍼런스: SOH/Cycle을 고정하고 "Number of segments"를 제3의 축으로 쓴
3D 산맥형 플랏 — magma/plasma 계열 그라데이션, 각 슬라이스가 선으로 보이고 슬라이스
사이는 보간되어 빈틈 없이 하나의 곡면처럼 이어짐.)

각 리지를 독립된 커튼(폭이 채워진 사각형)으로 그리지 않고, ax.plot_surface로 인접
리지끼리 보간 연결한다 — 리지가 몇 개 안 되더라도(예: 시나리오 6개) 그 사이 공간이
매끈하게 채워져 "면"처럼 보인다. 각 리지 자체의 선(원본 곡선)은 검은 선으로 얹어서
레퍼런스처럼 슬라이스 경계가 또렷이 보이게 한다.

제3의 축은 두 가지 입력 방식을 지원해 언제든 새 축으로 확장 가능하게 설계했다:
  1. --run-dir + --ridge-by scenario : 한 run의 test_predictions.csv를 시나리오별로
     쪼개서 리지를 만든다(지금 바로 그릴 수 있는 유일한 축 — seg_name 컬럼이 이미 있음).
  2. --csv "label=path" ["label=path" ...] : 범용 모드. 리지 1개 = CSV 1개(label로 표시).
     나중에 세그먼트 길이(n2=5%/9%/20%) 등 아직 존재하지 않는 축은, 그 실험이 끝나고
     test_predictions.csv(cycle/soh_pred/soh_true 컬럼만 있으면 됨)만 있으면 이 스크립트
     코드 수정 없이 --csv로 바로 리지를 추가할 수 있다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_soh_ridge3d.py \
      --run-dir 5_model/experiments/phase1_lab/results/p1v2_runs/0827_1705_p1v2_p1v4_full_seed42 \
      --ridge-by scenario --value soh_pred \
      --out 5_model/experiments/phase1_lab/results/soh_ridge3d_scenario_pred_v4.png

  # 미래(세그먼트 길이 등 새 축) — CSV만 있으면 코드 수정 없이:
  python 5_model/experiments/phase1_lab/plot_soh_ridge3d.py \
      --csv "5%=.../n2_5pct/predictions/test_predictions.csv" \
            "9%=.../n2_9pct/predictions/test_predictions.csv" \
            "20%=.../n2_20pct/predictions/test_predictions.csv" \
      --value soh_pred --out .../soh_ridge3d_seglen_pred.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SCENARIO_ORDER = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]


def _extract_value(df: pd.DataFrame, value: str) -> np.ndarray:
    """value가 'err_pct'면 soh_true/soh_pred로부터 상대오차(%)를 직접 계산한다(cap_true_Ah로
    계산해도 동일 — cap_init이 분자/분모에서 상쇄되는 비율값이라 soh 컬럼만으로 충분).
    그 외(soh_pred/soh_true)는 CSV 컬럼을 그대로 쓴다."""
    if value == "err_pct":
        soh_true = df["soh_true"].values.astype(float)
        soh_pred = df["soh_pred"].values.astype(float)
        return np.abs(soh_pred - soh_true) / soh_true * 100.0
    return df[value].values.astype(float)


def _load_scenario_ridges(csv_path: Path, value: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """리지별 원본 (cycle, value) 배열을 그대로 반환한다(아직 binning 안 함 — 여러 리지가
    공유할 cycle bin은 plot 단계에서 한 번에 정한다)."""
    df = pd.read_csv(csv_path)
    missing = set(SCENARIO_ORDER) - set(df["seg_name"].unique())
    if missing:
        raise ValueError(f"{csv_path}: seg_name에 없는 시나리오 {missing} (필요: {SCENARIO_ORDER})")
    ridges = []
    for name in SCENARIO_ORDER:
        sub = df[df["seg_name"] == name]
        ridges.append((name, sub["cycle"].values.astype(float), _extract_value(sub, value)))
    return ridges


def _load_csv_ridges(specs: list[str], value: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ridges = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--csv 항목은 'label=path' 형식이어야 함: {spec!r}")
        label, path_str = spec.split("=", 1)
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path} 없음")
        df = pd.read_csv(path)
        ridges.append((label, df["cycle"].values.astype(float), _extract_value(df, value)))
    return ridges


def plot_ridge_surface3d(ridges: list[tuple[str, np.ndarray, np.ndarray]], value_label: str,
                          ridge_axis_label: str, title: str, out_path: Path,
                          n_bins: int = 60, cmap_name: str = "magma_r",
                          elev: float = 20, azim: float = -45) -> None:
    """ridges: [(label, cycle_arr, value_arr), ...] — 리스트 순서대로 y=0,1,2...에 배치.
    모든 리지가 공유하는 cycle bin으로 각자 재구간화한 뒤 (n_ridge, n_bins) Z grid를
    만들고 plot_surface로 그린다 — 인접한 두 리지(y, y+1) 사이는 matplotlib이 자동으로
    선형 보간해 채우므로, 리지가 6개뿐이어도 그 사이 공간이 빈틈없는 곡면으로 보인다."""
    n = len(ridges)
    x_min = min(x.min() for _, x, _ in ridges)
    x_max = max(x.max() for _, x, _ in ridges)
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    z_grid = np.full((n, n_bins), np.nan)
    for i, (_, cyc, val) in enumerate(ridges):
        idx = np.clip(np.digitize(cyc, x_edges) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            m = idx == b
            if m.any():
                z_grid[i, b] = val[m].mean()
        nan_mask = np.isnan(z_grid[i])
        if nan_mask.any() and not nan_mask.all():
            z_grid[i, nan_mask] = np.interp(
                x_centers[nan_mask], x_centers[~nan_mask], z_grid[i, ~nan_mask]
            )

    X, Y = np.meshgrid(x_centers, np.arange(n))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, z_grid, cmap=cmap_name, edgecolor="none",
                     rstride=1, cstride=1, antialiased=True, shade=True, alpha=0.95)
    # 리지 원본 라인을 아주 옅게 덧그려 슬라이스 경계를 은은하게만 표시(진한 검은 선은
    # 곡면 자체의 색 그라데이션을 가려서 요청에 따라 알파를 낮춤).
    for i in range(n):
        ax.plot(x_centers, np.full(n_bins, i), z_grid[i], color="black", alpha=0.18,
                 linewidth=0.7, zorder=10)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for label, _, _ in ridges])
    ax.set_xlabel("Cycle")
    ax.set_ylabel(ridge_axis_label)
    ax.set_zlabel(value_label)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_soh_ridge3d] saved {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOH x Cycle x <3rd axis> 3D 곡면 플랏")
    p.add_argument("--run-dir", default=None,
                   help="--ridge-by scenario 모드에서 쓸 run 디렉터리(predictions/test_predictions.csv 필요)")
    p.add_argument("--ridge-by", default=None, choices=["scenario"],
                   help="run-dir 안에서 무엇으로 리지를 나눌지. 지금은 scenario(6개)만 지원 — "
                        "다른 축(세그먼트 길이 등)은 --csv 범용 모드로 확장할 것")
    p.add_argument("--csv", nargs="+", default=None,
                   help="범용 모드: 'label=path' 여러 개. 리지 1개=CSV 1개. "
                        "새 축(세그먼트 길이/시드/데이터셋 등) 추가는 이 인자만 바꾸면 됨")
    p.add_argument("--value", default="soh_pred", choices=["soh_pred", "soh_true", "err_pct"],
                   help="z축에 쓸 값(기본 soh_pred — soh_true는 시나리오 축에서는 사실상 평평해서 "
                        "정보가 적음, cycle당 값이 시나리오 무관하게 동일하므로). "
                        "err_pct = |soh_pred-soh_true|/soh_true*100(상대오차, CSV 컬럼 없이 계산)")
    p.add_argument("--n-bins", type=int, default=60, dest="n_bins",
                   help="사이클 축 구간 개수(리지 곡선 매끄러움)")
    p.add_argument("--cmap", default=None,
                   help="기본은 --value에 따라 자동 선택: soh_pred/soh_true -> magma_r"
                        "(SOH 높은 쪽이 어둡고 낮은/degraded 쪽이 밝은 주황), "
                        "err_pct -> magma(오차 큰 쪽이 밝은 주황 — '밝음=주의'로 두 값 모두 "
                        "일관되게 유지). 직접 지정하면 이 자동선택을 덮어씀")
    p.add_argument("--elev", type=float, default=20, help="카메라 고도각")
    p.add_argument("--azim", type=float, default=-45, help="카메라 방위각(오른쪽으로 틀려면 음수를 키움, 예: -60 -> -45)")
    p.add_argument("--out", required=True, help="출력 png 경로")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.csv:
        ridges = _load_csv_ridges(args.csv, args.value)
        ridge_axis_label = "variant"
    elif args.run_dir and args.ridge_by == "scenario":
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        csv_path = run_dir / "predictions" / "test_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} 없음 — test_phase1_checkpoint.py --export-for-visualize로 먼저 만들어야 함"
            )
        ridges = _load_scenario_ridges(csv_path, args.value)
        ridge_axis_label = "scenario"
    else:
        raise SystemExit("--csv 또는 (--run-dir --ridge-by scenario) 중 하나는 지정해야 함")

    value_label = {
        "soh_pred": "Predicted SOH", "soh_true": "Observed SOH", "err_pct": "|error| (%)",
    }[args.value]
    cmap_name = args.cmap or ("magma" if args.value == "err_pct" else "magma_r")
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tag = Path(args.run_dir).name if args.run_dir else "csv"
    plot_ridge_surface3d(
        ridges, value_label=value_label, ridge_axis_label=ridge_axis_label,
        title=f"{value_label} surface — {ridge_axis_label} x Cycle ({tag})",
        out_path=out_path, n_bins=args.n_bins, cmap_name=cmap_name,
        elev=args.elev, azim=args.azim,
    )


if __name__ == "__main__":
    main()
