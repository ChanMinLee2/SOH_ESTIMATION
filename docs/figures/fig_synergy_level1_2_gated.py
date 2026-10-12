"""시너지 Level1(선형)/Level2(비선형) 통합 플랏 — 게이트 선택확률까지 한 산점도에.

그룹(=잠재적 커널 HI 후보) 하나당 점 하나:
  x = Level1 점수 = 그룹에 "추가된" 멤버들(시드 제외)의 |편상관계수| 최솟값
      ("가장 약한 멤버도 선형적으로 새 정보를 주는가")
  y = Level2 gap  = 커널(Nystroem+Ridge) train R^2 - 같은 멤버로 피팅한 선형(OLS) R^2
      ("단순 선형결합보다 진짜 비선형으로 더 잘 뽑아내는가")
  마커 크기/색   = 실제 학습된 모델에서 이 커널 HI의 게이트 선택확률
      (gates/regression_kernel_HIs.json에 이름이 없으면 "§4.3 결합 다중공선성 단계에서
      학습 전에 이미 제거됨" — 별도 회색 X로 표시, 확률 0과는 다른 범주)

기준선 x=0.1(Level1 문턱), y=0.02(Level2 문턱) — 둘 다 넘는 우상단 사분면이
docs/261001_REPORT.md에서 정의한 "진짜 시너지".

실행: python docs/figures/fig_synergy_level1_2_gated.py
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model_lib"))
sys.path.insert(0, str(PROJECT_ROOT / "7_kernel"))

from utils.io_utils import load_config  # noqa: E402
from kernel import _load_train_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import INK, SUBINK, GRID, setup_rcparams  # noqa: E402

RUN_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42"
TAG = "refact"

L1_THRESH = 0.10   # Level1 문턱(|편상관|)
L2_THRESH = 0.02   # Level2 문턱(커널R^2 - 선형R^2)


def _ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    """np.linalg.lstsq로 절편 포함 선형회귀 R^2 (kernel.py의 _fit_group_kernel과 같은 train R^2 정의)."""
    A = np.column_stack([x, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0


def _load_gate_probs(run_dir: Path) -> dict[str, float]:
    """gates/regression_kernel_HIs.json -> {kernel HI 이름: 게이트 선택확률}.
    §4.3 결합 다중공선성 단계에서 학습 전에 이미 제거된 후보는 여기 아예 없음(= 확률 0과는
    다른 범주, 호출부가 .get(name)이 None이면 "제거됨"으로 구분해서 처리)."""
    path = run_dir / "gates" / "regression_kernel_HIs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    probs: dict[str, float] = {}
    for key in data:
        if key.endswith("_names"):
            seg = key[: -len("_names")]
            names = data[key]
            vals = data[f"{seg}_probs"]
            probs.update(zip(names, vals))
    return probs


def build_points() -> list[dict]:
    warnings.filterwarnings("ignore")  # sklearn pickle 버전 경고(결과엔 영향 없음)

    cfg = load_config(str(RUN_DIR / "config.yaml"))
    from types import SimpleNamespace
    loader_args = SimpleNamespace(
        data_dir=cfg["data"]["data_dir"], seg_data_dir=cfg["data"]["seg_data_dir"],
        datasets=cfg["data"]["datasets"], split_seed=cfg["data"]["split_seed"],
        axis_config=json.dumps(cfg["scenario"]["axis_config"]), seg_axis=cfg["scenario"]["axis"],
    )
    print("[1/4] train split 로딩 중...")
    x_all, y_all, scen_idx_all, spec, names_by_seg = _load_train_split(loader_args)

    print("[2/4] synergy/kernel 산출물 로딩 중...")
    synergy = json.loads((RUN_DIR / f"synergy_groups_{TAG}_groups.json").read_text(encoding="utf-8"))
    with open(RUN_DIR / f"kernel_group_features_{TAG}_kernel.pkl", "rb") as f:
        kernel_pkl = pickle.load(f)
    gate_probs = _load_gate_probs(RUN_DIR)

    print("[3/4] 후보별 선형(OLS) R^2 계산 + Level1/Level2/게이트확률 매칭...")
    points = []
    for cand in kernel_pkl["features"]:
        name = cand["name"]
        s = cand["scenario_idx"] if "scenario_idx" in cand else spec.scenario_names.index(cand["scenario"])
        gi = int(name.rsplit("_g", 1)[1])
        members = cand["members"]
        sel = scen_idx_all == s
        x_group = x_all[sel][:, members]
        y_scen = y_all[sel]
        linear_r2 = _ols_r2(x_group, y_scen)

        scores = synergy[f"seg_{s}_group_scores"][gi]
        added_scores = scores[1:] if len(scores) > 1 else scores  # 시드(scores[0]) 제외
        level1 = float(min(abs(v) for v in added_scores))

        points.append({
            "name": name, "scenario": cand["scenario"],
            "level1": level1, "level2_gap": cand["train_r2"] - linear_r2,
            "kernel_r2": cand["train_r2"], "linear_r2": linear_r2,
            "gate_prob": gate_probs.get(name),  # None = §4.3에서 학습 전 제거됨
            "n_members": len(members),
        })
    print(f"  {len(points)}개 후보 처리 완료")
    return points


def plot_on_ax(fig: plt.Figure, ax: plt.Axes, points: list[dict], title_fontsize: float = 9.2) -> None:
    """다른 스크립트(fig4_ablation_synergy.py)가 서브플랏 하나로 재사용할 수 있도록
    실제 그리기 로직만 분리 — fig는 colorbar를 ax 옆에 붙이기 위해 필요."""
    removed = [p for p in points if p["gate_prob"] is None]
    present = [p for p in points if p["gate_prob"] is not None]

    # 사분면 배경(연하게) — 우상단 = "진짜 시너지"
    xmax = max(p["level1"] for p in points) * 1.15
    ymin = min(p["level2_gap"] for p in points) * 1.15
    ymax = max(p["level2_gap"] for p in points) * 1.15
    ax.axvspan(L1_THRESH, xmax, ymin=0, ymax=1, color="#EDE7D9", alpha=0.5, zorder=0)
    ax.axhspan(L2_THRESH, ymax, xmin=0, xmax=1, color="#EDE7D9", alpha=0.5, zorder=0)
    ax.axvline(L1_THRESH, color=SUBINK, lw=0.9, ls="--", zorder=1)
    ax.axhline(L2_THRESH, color=SUBINK, lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)

    if present:
        probs = np.array([p["gate_prob"] for p in present])
        sizes = 18 + 260 * probs  # probability 0-1 -> marker area
        sc = ax.scatter(
            [p["level1"] for p in present], [p["level2_gap"] for p in present],
            s=sizes, c=probs, cmap="YlOrBr", vmin=0, vmax=1,
            edgecolors=INK, linewidths=0.5, alpha=0.88, zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Gate selection probability (trained model)", fontsize=7.6)
        cbar.ax.tick_params(labelsize=6.8)

    if removed:
        ax.scatter(
            [p["level1"] for p in removed], [p["level2_gap"] for p in removed],
            s=30, marker="x", color=SUBINK, alpha=0.6, linewidths=1.1, zorder=2,
            label="Pruned before training (§4.3)",
        )

    ax.set_xlabel(r"Level 1 $-$ min $|$partial corr.$|$ among added members (linear synergy)")
    ax.set_ylabel(r"Level 2 $-$ kernel $R^2$ $-$ linear $R^2$ (nonlinear synergy)")
    ax.set_title("Linear vs. nonlinear synergy contribution and actual gate usage",
                 loc="left", fontsize=title_fontsize)
    ax.text(xmax * 0.985, ymax * 0.95, '"genuine synergy"\n(passes Level 1+2)',
            ha="right", va="top", fontsize=7.4, color=SUBINK, style="italic")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.8)

    handles = [Line2D([0], [0], marker="x", color=SUBINK, linestyle="", markersize=6,
                       label="Pruned before training (§4.3)")]
    ax.legend(handles=handles, loc="lower left", fontsize=6.8, frameon=False)


def plot(points: list[dict]) -> plt.Figure:
    """단독 실행용(이 스크립트를 바로 돌릴 때) — fig4_ablation_synergy.py는 plot_on_ax를 직접 씀."""
    setup_rcparams()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    plot_on_ax(fig, ax, points)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    pts = build_points()
    print("[4/4] 플랏 생성 중...")
    fig = plot(pts)
    out_png = Path(__file__).resolve().parent / "fig_synergy_level1_2_gated.png"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_png.with_suffix(".pdf"))
    print(f"saved: {out_png}\nsaved: {out_png.with_suffix('.pdf')}")

    n_real = sum(1 for p in pts if p["level1"] >= L1_THRESH and p["level2_gap"] >= L2_THRESH)
    n_removed = sum(1 for p in pts if p["gate_prob"] is None)
    print(f"\n총 후보 {len(pts)}개 중 '진짜 시너지'(Level1+2 통과): {n_real}개, "
          f"§4.3에서 학습 전 제거됨: {n_removed}개")
