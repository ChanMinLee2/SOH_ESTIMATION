"""
5_model/experiments/phase1_lab/generate_flow_plots_a.py

docs/FLOW.md에서 "A(코드 수정 없이 뽑을 수 있는 것)"로 분류한 플랏들을 한 번에 생성해
5_model/plots/ 에 저장하는 통합 스크립트. 기존 파이프라인 코드(세그멘터/모델/트레이너)는
전혀 건드리지 않고, 이미 저장된 run 산출물(predictions/gates json) + 기존 데이터 로더
(build_datasets)만 재사용한다.

포함 항목(파일명 접두어 a1~a8):
  a1  데이터셋 x 시나리오 교차표 (test split, 교락 진단용)
  a2  HI x 시나리오 유의성 판정 근거(효과크기 vs BH-adjusted p-value) — FDR은 이미
      test_hi_scenario_interaction.py의 _bh_adjust()로 계산돼 있었음(재확인 결과 별도
      보정이 필요했던 게 아니라 이미 반영돼 있었다 — docs/FLOW.md 정정 참고). 이 스크립트는
      "왜 39/64인가"(p-value가 아니라 효과크기 기준)를 시각적으로 보여주기만 한다.
  a3  셀단위 RMSE 분포 + SOH구간별 오차 + mAh 단위 MAPE/MAE (v4, test oracle)
  a4  시드 안정성(Kendall tau 시드쌍) + 시나리오간 Jaccard 퇴화진단 (v0/v2/v4, 3-seed)
  a5  v0/v2/v4 vs ctrl 비교 막대그래프(R² delta + tau delta) — 260901_REPORT.md의 기존
      3-seed 숫자를 하드코딩(5-seed 확장 전 예비 버전, 캡션에 명시)
  a6  v4 아키텍처 개략도(정적 스키매틱, 데이터 무관)
  a7  HI 처방 -> 선형회귀 전이(대조군 4종) + 처방 최소크기 스윕 + permutation importance
  a8  HI 절단오차(완전사이클 HI값 - 세그먼트 HI값, full_cycle 축 데이터 재사용)

각 함수는 독립적으로 실패해도 나머지에 영향을 안 주도록 main()에서 try/except로 감싼다.

사용 예:
  SOH_EXCLUDE_STAT_LEAK=1 python 5_model/experiments/phase1_lab/generate_flow_plots_a.py \
      --out-dir 5_model/plots
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# 한글 라벨/제목이 많아 기본 폰트로는 글리프가 깨짐(missing-glyph 사각형) — Windows에
# 기본 탑재된 맑은 고딕으로 강제 지정. 다른 OS라면 없을 수 있으니 조용히 무시.
plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from utils.io_utils import load_config  # noqa: E402
from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402
from datasets.segment_dataset import build_datasets  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

BASE = PROJECT_ROOT / "5_model" / "experiments" / "phase1_lab"
RESULTS = BASE / "results"

SCENARIO_ORDER = ["chg_lo", "chg_mid", "chg_hi", "dis_hi", "dis_mid", "dis_lo"]

V4_RUN = "p1v2_runs/0827_1705_p1v2_p1v4_full_seed42"
V4CTRL_RUN = "p1v2_runs/0901_0327_p1v2_p1v4ctrl_full_seed42_seed42"

RUN_SEEDS = {
    "v0": [
        "p1v2_runs/0821_0917_p1v2_k25_full_N2_stage12_grouped_seed42/"
        "0821_0917_p1v2_k25_full_N2_stage12_grouped_seed42",
        "p1v2_runs/0828_0239_p1v2_p1v0_full_seed0_seed0",
        "p1v2_runs/0828_0506_p1v2_p1v0_full_seed123_seed123",
    ],
    "v2": [
        "p1v2_runs/0821_1120_p1v2_k25_full_N2_stage12_kernel_v2_seed42",
        "p1v2_runs/0828_0738_p1v2_p1v2_full_seed0_seed0",
        "p1v2_runs/0828_1047_p1v2_p1v2_full_seed123_seed123",
    ],
    "v4": [
        "p1v2_runs/0827_1705_p1v2_p1v4_full_seed42",
        "p1v2_runs/0828_1957_p1v2_p1v4_full_seed0_seed0",
        "p1v2_runs/0828_2330_p1v2_p1v4_full_seed123_seed123",
    ],
}

# 260901_REPORT.md "대조군(ctrl) 비교" 절 — 3-seed 평균(v4는 seed=123 완료 후 최종치).
CTRL_DATA = {
    "v0": dict(r2=0.9182, r2_ctrl=0.9188, tau=0.1184, tau_ctrl=0.1029),
    "v2": dict(r2=0.9484, r2_ctrl=0.9474, tau=0.1862, tau_ctrl=0.1423),
    "v4": dict(r2=0.9443, r2_ctrl=0.9389, tau=0.2053, tau_ctrl=0.0881),
}

CANON_DATA_DIR = ("D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/"
                   "n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle")
CANON_SEG_DIR = ("D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/"
                  "n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg")
FULLCYCLE_DATA_DIR = "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/full_cycle/cycle"
FULLCYCLE_SEG_DIR = "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/full_cycle/seg"


def _safe_run(label: str, fn, *args, **kwargs) -> None:
    print(f"\n=== {label} ===")
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — 개별 항목 실패가 전체를 막지 않게 함
        print(f"[SKIP] {label} 실패: {type(e).__name__}: {e}")


def _dataset_of(cell_id: str) -> str:
    """cell_id 표기 규칙으로 MIT/HUST 판별(예: MIT='b1c21', HUST='1-7')."""
    if re.match(r"^b\d+c\d+$", cell_id):
        return "MIT"
    if re.match(r"^\d+-\d+$", cell_id):
        return "HUST"
    return "unknown"


def _load_regression_gate(run_rel: str) -> dict:
    p = RESULTS / run_rel / "gates" / "regression_HIs.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _rank_and_active(gate: dict, s: int, n_hi: int = 64, thresh: float = 0.9):
    ranked = gate[f"seg_{s}_ranked"]
    probs = gate[f"seg_{s}_probs"]
    rankpos = np.zeros(n_hi)
    for pos, idx in enumerate(ranked):
        rankpos[idx] = pos
    active = set(idx for idx, p in zip(ranked, probs) if p > thresh)
    return rankpos, active


def _selected_hi_per_scenario(run_rel: str, thresh: float = 0.9) -> dict[int, list[int]]:
    gate = _load_regression_gate(run_rel)
    out = {}
    for s in range(6):
        ranked = gate[f"seg_{s}_ranked"]
        probs = gate[f"seg_{s}_probs"]
        out[s] = sorted(idx for idx, p in zip(ranked, probs) if p > thresh)
    return out


# ---------------------------------------------------------------------------
# a1 — 데이터셋 x 시나리오 교차표
# ---------------------------------------------------------------------------

def plot_dataset_scenario_crosstab(out_dir: Path) -> None:
    df = pd.read_csv(RESULTS / V4_RUN / "predictions" / "test_predictions.csv")
    df["dataset"] = df["cell_id"].map(_dataset_of)
    unknown = (df["dataset"] == "unknown").sum()
    if unknown:
        print(f"  [warn] dataset 판별 실패 {unknown}건(cell_id 표기 확인 필요)")

    ct = pd.crosstab(df["seg_name"], df["dataset"]).reindex(SCENARIO_ORDER)
    print(ct)

    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#999999"])
    ax.set_title("시나리오 x 데이터셋 세그먼트 수 교차표 (test split, v4 기준)")
    ax.set_xlabel("scenario")
    ax.set_ylabel("segment count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="dataset")
    fig.tight_layout()
    out_path = out_dir / "a1_dataset_scenario_crosstab.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ct.to_csv(out_dir / "a1_dataset_scenario_crosstab.csv")
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# a2 — HI x 시나리오 유의성 판정 근거 (효과크기 vs BH-adjusted p-value)
# ---------------------------------------------------------------------------

def plot_hi_scenario_significance(out_dir: Path) -> None:
    path = RESULTS / "hi_scenario_interaction_k25_full_N2.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    per_hi = d["per_hi"]
    concepts = list(per_hi.keys())
    std_r = np.array([per_hi[c]["std_r_across_scenarios"] for c in concepts])
    p_adj = np.array([per_hi[c]["p_adj_bh"] for c in concepts])
    sig = np.array([per_hi[c]["significant"] for c in concepts])
    min_effect = d["min_effect_size"]
    alpha = d["alpha"]

    print(f"  n_hi={d['n_hi']}  n_significant(효과크기 기준)={d['n_significant']}  "
          f"n_p_significant_raw(BH<alpha)={sum(p_adj < alpha)}")
    print("  -> 거의 전부(64개 중 다수)가 BH 보정 후에도 p<alpha다(표본이 수십만~백만 행이라 "
          "p-value는 사실상 항상 유의) — 그래서 이 코드는 p-value가 아니라 효과크기"
          "(std_r_across_scenarios >= min_effect_size)를 최종 판정 기준으로 쓴다.")

    p_plot = np.clip(p_adj, 1e-300, None)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(std_r[~sig], -np.log10(p_plot[~sig]), s=10, alpha=0.6, c="gray",
               label=f"미달 ({(~sig).sum()}개)")
    ax.scatter(std_r[sig], -np.log10(p_plot[sig]), s=10, alpha=0.7, c="crimson",
               label=f"유의 ({sig.sum()}개)")
    ax.axvline(min_effect, ls="--", c="black", lw=1, label=f"효과크기 임계={min_effect}")
    ax.axhline(-np.log10(alpha), ls=":", c="blue", lw=1, label=f"BH p<{alpha}")
    ax.set_xlabel("std_r_across_scenarios (시나리오간 상관계수 표준편차 = 효과크기)")
    ax.set_ylabel("-log10(BH-adjusted p-value)")
    ax.set_title(f"HI x 시나리오 상호작용 — 최종 판정은 효과크기 기준 ({sig.sum()}/{len(concepts)}개 유의)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "a2_hi_scenario_significance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# a3 — 셀단위 RMSE 분포 + SOH구간별 오차 + mAh MAPE/MAE
# ---------------------------------------------------------------------------

def plot_cell_and_soh_bin_error(out_dir: Path) -> None:
    df = pd.read_csv(RESULTS / V4_RUN / "predictions" / "test_predictions.csv")
    df["err_ah"] = (df["cap_pred_Ah"] - df["cap_true_Ah"]).abs()
    df["err_pct"] = df["err_ah"] / df["cap_true_Ah"] * 100.0

    cell_rmse = df.groupby("cell_id").apply(
        lambda g: float(np.sqrt(np.mean((g["cap_pred_Ah"] - g["cap_true_Ah"]) ** 2)))
    )

    bins = [0.0, 0.80, 0.85, 0.90, 0.95, 1.01]
    labels = ["<0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", ">=0.95"]
    df["soh_bin"] = pd.cut(df["soh_true"], bins=bins, labels=labels, right=False)
    bin_rmse = df.groupby("soh_bin", observed=True).apply(
        lambda g: float(np.sqrt(np.mean((g["cap_pred_Ah"] - g["cap_true_Ah"]) ** 2)))
    )
    bin_counts = df.groupby("soh_bin", observed=True).size()
    bin_rmse = bin_rmse.reindex(labels)
    bin_counts = bin_counts.reindex(labels).fillna(0).astype(int)

    mape = float(df["err_pct"].mean())
    mae_mah = float(df["err_ah"].mean()) * 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].boxplot(cell_rmse.values, vert=True)
    axes[0].set_title(f"셀단위 RMSE 분포 (n={len(cell_rmse)}셀)")
    axes[0].set_ylabel("RMSE (Ah)")
    axes[0].set_xticks([])

    bars = axes[1].bar(labels, bin_rmse.values, color="#4C72B0")
    for bar, c in zip(bars, bin_counts.values):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h, f"n={c}",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_title("SOH 구간별 RMSE (풀링 R²가 안 보여주는 저SOH 구간 확인용)")
    axes[1].set_ylabel("RMSE (Ah)")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].axis("off")
    axes[2].text(
        0.05, 0.6,
        f"전체 MAPE: {mape:.2f}%\n전체 MAE: {mae_mah:.2f} mAh\n"
        f"n={len(df):,} segments / {df['cell_id'].nunique()} cells",
        fontsize=13, va="center",
    )

    fig.suptitle("v4 test(oracle) — 셀/SOH구간별 오차 + mAh 단위 병기")
    fig.tight_layout()
    out_path = out_dir / "a3_cell_and_soh_bin_error.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}  (MAPE={mape:.2f}%, MAE={mae_mah:.2f}mAh)")


# ---------------------------------------------------------------------------
# a4 — 시드 안정성(Kendall tau) + 시나리오간 Jaccard 퇴화진단
# ---------------------------------------------------------------------------

def plot_seed_stability_diagnostics(out_dir: Path) -> None:
    tau_by_version: dict[str, list[float]] = {}
    jaccard_by_version: dict[str, list[float]] = {}

    for version, runs in RUN_SEEDS.items():
        gates = [_load_regression_gate(r) for r in runs]

        taus = []
        for s in range(6):
            rankpos_list = [_rank_and_active(g, s)[0] for g in gates]
            for i in range(len(rankpos_list)):
                for j in range(i + 1, len(rankpos_list)):
                    tau, _ = kendalltau(rankpos_list[i], rankpos_list[j])
                    if not np.isnan(tau):
                        taus.append(tau)
        tau_by_version[version] = taus

        jaccards = []
        for g in gates:
            active_per_scen = [_rank_and_active(g, s)[1] for s in range(6)]
            for i in range(6):
                for j in range(i + 1, 6):
                    a, b = active_per_scen[i], active_per_scen[j]
                    union = a | b
                    jaccards.append(len(a & b) / len(union) if union else 0.0)
        jaccard_by_version[version] = jaccards

        n_unique = len(set().union(*[
            _rank_and_active(g, s)[1] for g in gates for s in range(6)
        ]))
        print(f"  {version}: tau median={np.median(taus):.3f}  "
              f"jaccard(시나리오간) median={np.median(jaccards):.3f}  "
              f"고유 활성 HI 개수(전 시드x시나리오 합집합)={n_unique}/64")

    versions = list(RUN_SEEDS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([tau_by_version[v] for v in versions], labels=versions)
    axes[0].axhline(0, color="gray", lw=0.7, ls=":")
    axes[0].set_title("시드쌍 Kendall τ 분포 (6개 시나리오 x 3개 시드쌍)")
    axes[0].set_ylabel("Kendall τ")

    axes[1].boxplot([jaccard_by_version[v] for v in versions], labels=versions)
    axes[1].set_title("시나리오간 Jaccard 분포 (퇴화 진단 — 낮을수록 시나리오별로 다르게 고름)")
    axes[1].set_ylabel("Jaccard(활성 HI, p>0.9)")

    fig.suptitle("시드 안정성 + 퇴화 진단 (v0/v2/v4, 3-seed, 독립 재계산 — 260901_REPORT.md의\n"
                 "요약 수치와 방법론이 정확히 같지 않을 수 있어 참고용)", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "a4_seed_stability_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# a5 — v0/v2/v4 vs ctrl 비교 막대그래프
# ---------------------------------------------------------------------------

def plot_ctrl_comparison_bar(out_dir: Path) -> None:
    versions = list(CTRL_DATA.keys())
    r2_delta = [CTRL_DATA[v]["r2"] - CTRL_DATA[v]["r2_ctrl"] for v in versions]
    tau_delta = [CTRL_DATA[v]["tau"] - CTRL_DATA[v]["tau_ctrl"] for v in versions]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = ["#4C72B0" if d > 0 else "#C44E52" for d in r2_delta]
    axes[0].bar(versions, r2_delta, color=colors)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("validation R² delta (실제 - ctrl)")
    for i, v in enumerate(r2_delta):
        axes[0].text(i, v, f"{v:+.4f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)

    colors2 = ["#4C72B0" if d > 0 else "#C44E52" for d in tau_delta]
    axes[1].bar(versions, tau_delta, color=colors2)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("시드간 Kendall τ delta (실제 - ctrl)")
    for i, v in enumerate(tau_delta):
        axes[1].text(i, v, f"{v:+.4f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)

    fig.suptitle("구조화된 선택 vs 무작위 대조군 — 3-seed 기준(⚠️5-seed 확장 전 예비 버전,\n"
                 "docs/FLOW.md 6-a 참고: R² 우위만으론 시드노이즈와 구분 안 됨, τ가 핵심 주장)",
                 fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "a5_ctrl_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# a6 — v4 아키텍처 개략도 (정적 스키매틱)
# ---------------------------------------------------------------------------

def plot_architecture_diagram(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF2FB", fontsize=9):
        rect = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.04,rounding_size=0.08",
                               linewidth=1.3, edgecolor="#333333", facecolor=fc)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))

    box(0.3, 5.9, 2.1, 1.0, "Raw HI (64)\nx_hi")
    box(2.9, 5.9, 2.1, 1.0, "Probe Gate\n(direction별)")
    arrow(2.4, 6.4, 2.9, 6.4)

    box(5.5, 6.6, 2.4, 1.0, "Shared Gate\n(25개, 시나리오 공통)", fc="#FDEBD3")
    box(5.5, 5.2, 2.4, 1.0, "Specific Gate\n(39개, GroupedHardConcreteGate)", fc="#FDEBD3")
    arrow(5.0, 6.6, 5.5, 7.1)
    arrow(5.0, 6.3, 5.5, 5.7)

    box(8.4, 5.9, 2.9, 1.0, "Kernel Fusion\n(RBF, 시너지그룹 -> 59개)", fc="#DFF0D8")
    arrow(7.9, 7.1, 8.4, 6.5)
    arrow(7.9, 5.7, 8.4, 6.1)

    box(5.5, 3.6, 5.8, 0.9, "concat -> cap_head (MLP)", fc="#E8DAEF")
    arrow(6.7, 5.2, 6.7, 4.5)
    arrow(9.85, 5.9, 8.6, 4.5)

    box(5.5, 2.1, 5.8, 0.9, "SOH 예측 (cap_pred)")
    arrow(8.4, 3.6, 8.4, 3.0)

    box(0.3, 1.6, 4.6, 1.5,
        "배포 라우팅:\noracle(공식 라벨, 상한선) /\nhard(분류기 argmax) /\nsoft(확률가중)",
        fc="#FADBD8", fontsize=8.5)
    arrow(4.9, 2.35, 5.5, 2.55)

    ax.set_title("v4 아키텍처 개략도 (probe gate -> shared/specific 게이트 -> "
                 "커널 융합 -> cap_head -> 배포 라우팅)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "a6_architecture_diagram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# 데이터 로딩 공통 (a7, a8에서 재사용)
# ---------------------------------------------------------------------------

def _build_raw_hi(run_rel: str, data_dir: str, seg_data_dir: str,
                   axis: str | None = None, axis_config: dict | None = None):
    """(train_ds, test_ds, concept_names, raw_train, raw_test, spec) 반환.
    raw_*: (N, 64) 원본 스케일 HI, 결측은 NaN.

    axis/axis_config를 안 주면 run_rel의 config.yaml에 저장된 축(예: q_frac_ref)을
    그대로 쓴다 — 단 data_dir/seg_data_dir가 **다른 축의 데이터**(예: full_cycle)를
    가리키는 경우 반드시 axis/axis_config도 그 축에 맞게 같이 넘겨야 한다. 데이터
    파일의 scen_idx는 그 데이터를 만들 때 쓴 축의 spec(scenario_names 개수/순서)
    기준으로만 올바르게 해석되므로, 축을 안 맞추면 seg_name이 엉뚱하게 나온다
    (예: full_cycle의 idx 0/1을 q_frac_ref spec으로 읽으면 chg_full/dis_full이
    아니라 chg_lo/chg_mid로 잘못 나옴 — 실제로 이 스크립트 개발 중 발견한 버그)."""
    cfg = load_config(str(RESULTS / run_rel / "config.yaml"))
    cfg["data"]["data_dir"] = data_dir
    cfg["data"]["seg_data_dir"] = seg_data_dir
    use_axis = axis or cfg["scenario"]["axis"]
    use_axis_config = axis_config if axis_config is not None else cfg["scenario"]["axis_config"]
    spec = get_segmenter(use_axis, {use_axis: use_axis_config}).get_spec()
    train_ds, val_ds, test_ds, norm = build_datasets(cfg, spec=spec)

    ref_seg = spec.scenario_names[0]
    suffix = f"_{ref_seg}"
    concept_names = [c[: -len(suffix)] if c.endswith(suffix) else c
                     for c in get_hi_cols_for_seg(ref_seg)]

    def _raw(ds):
        x = ds.x_hi.numpy().astype(np.float64)
        mask = ds.nan_mask.numpy().astype(bool)
        raw = x * norm.std_ + norm.mean_
        raw[~mask] = np.nan
        return raw

    return train_ds, test_ds, concept_names, _raw(train_ds), _raw(test_ds), spec


# ---------------------------------------------------------------------------
# a7 — HI 처방 -> 선형회귀 전이 + 최소크기 스윕 + permutation importance
# ---------------------------------------------------------------------------

def plot_hi_subset_transfer(out_dir: Path) -> None:
    train_ds, test_ds, concept_names, x_train, x_test, spec = _build_raw_hi(
        V4_RUN, CANON_DATA_DIR, CANON_SEG_DIR
    )
    y_train = train_ds.target.numpy().astype(np.float64)
    y_test = test_ds.target.numpy().astype(np.float64)
    seg_train = np.asarray(train_ds.seg_names)
    seg_test = np.asarray(test_ds.seg_names)

    v4_sel = _selected_hi_per_scenario(V4_RUN)
    v4ctrl_sel = _selected_hi_per_scenario(V4CTRL_RUN)
    gate_v4 = _load_regression_gate(V4_RUN)

    rng = np.random.default_rng(0)

    def fit_eval(Xtr_full, ytr, Xte_full, yte, idxs):
        if len(idxs) == 0:
            return float("nan"), None
        Xtr = np.nan_to_num(Xtr_full[:, idxs], nan=0.0)
        Xte = np.nan_to_num(Xte_full[:, idxs], nan=0.0)
        model = LinearRegression().fit(Xtr, ytr)
        pred = model.predict(Xte)
        ss_res = float(np.sum((yte - pred) ** 2))
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return r2, model

    per_scen = {}
    for s, scen in enumerate(SCENARIO_ORDER):
        m_tr = seg_train == scen
        m_te = seg_test == scen
        Xtr_full, ytr = x_train[m_tr], y_train[m_tr]
        Xte_full, yte = x_test[m_te], y_test[m_te]

        v4_idx = v4_sel[s]
        n_v4 = max(len(v4_idx), 1)
        r2_v4, model_v4 = fit_eval(Xtr_full, ytr, Xte_full, yte, v4_idx)
        rand_idx = rng.choice(64, size=n_v4, replace=False).tolist()
        r2_rand, _ = fit_eval(Xtr_full, ytr, Xte_full, yte, rand_idx)
        r2_all, _ = fit_eval(Xtr_full, ytr, Xte_full, yte, list(range(64)))
        ctrl_idx = v4ctrl_sel[s]
        r2_ctrl, _ = fit_eval(Xtr_full, ytr, Xte_full, yte, ctrl_idx)

        per_scen[scen] = dict(
            v4=r2_v4, random=r2_rand, full64=r2_all, v4ctrl=r2_ctrl,
            idx=v4_idx, model=model_v4,
            Xtr_full=Xtr_full, ytr=ytr, Xte_full=Xte_full, yte=yte,
            ranked=gate_v4[f"seg_{s}_ranked"],
        )
        print(f"  {scen}: v4(n={len(v4_idx)})R2={r2_v4:.4f}  random={r2_rand:.4f}  "
              f"full64={r2_all:.4f}  v4ctrl(n={len(ctrl_idx)})={r2_ctrl:.4f}")

    # --- (1) 대조군 4종 막대그래프 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2
    xpos = np.arange(len(SCENARIO_ORDER))
    conds = [("v4", "v4 선택"), ("random", "무작위 동일개수"),
             ("full64", "전체 64개"), ("v4ctrl", "v4-ctrl 선택")]
    for k, (cond, label) in enumerate(conds):
        vals = [per_scen[s][cond] for s in SCENARIO_ORDER]
        ax.bar(xpos + (k - 1.5) * width, vals, width, label=label)
    ax.set_xticks(xpos)
    ax.set_xticklabels(SCENARIO_ORDER)
    ax.set_ylabel("R² (선형회귀, test)")
    ax.set_title("HI 부분집합 -> 선형회귀 전이: v4가 고른 HI가 이 모델 밖에서도 유효한가")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "a7_hi_subset_linear_transfer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # --- (2) 처방 최소 크기 스윕 ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for scen in SCENARIO_ORDER:
        d = per_scen[scen]
        sizes = sorted(set([3, 5, 10, len(d["idx"])]))
        r2s = []
        for k in sizes:
            idxs = d["ranked"][:k]
            r2, _ = fit_eval(d["Xtr_full"], d["ytr"], d["Xte_full"], d["yte"], idxs)
            r2s.append(r2)
        ax.plot(sizes, r2s, marker="o", label=scen)
    ax.set_xlabel("처방 크기 (HI 개수, 랭킹 상위 k개)")
    ax.set_ylabel("R² (선형회귀, test)")
    ax.set_title("처방 최소 크기 스윕 — 실무자가 쓸 수 있는 최소 HI 개수")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "a7b_min_prescription_size_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # --- (3) permutation importance (시나리오별 소패널) ---
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for s, scen in enumerate(SCENARIO_ORDER):
        d = per_scen[scen]
        idxs, model = d["idx"], d["model"]
        if model is None or len(idxs) == 0:
            axes.flat[s].axis("off")
            continue
        Xte = np.nan_to_num(d["Xte_full"][:, idxs], nan=0.0)
        yte = d["yte"]
        base_pred = model.predict(Xte)
        ss_tot = float(np.sum((yte - yte.mean()) ** 2))
        base_r2 = 1 - float(np.sum((yte - base_pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

        importances = []
        for j in range(len(idxs)):
            Xp = Xte.copy()
            rng.shuffle(Xp[:, j])
            pred = model.predict(Xp)
            r2p = 1 - float(np.sum((yte - pred) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
            importances.append(base_r2 - r2p)

        names = [concept_names[i] for i in idxs]
        order = np.argsort(importances)[::-1]
        ax = axes.flat[s]
        ax.barh([names[o] for o in order], [importances[o] for o in order], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title(f"{scen} (base R²={base_r2:.3f})", fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle("Permutation importance — HI 하나씩 셔플했을 때 R² 하락폭(클수록 중요)")
    fig.tight_layout()
    out_path = out_dir / "a7c_permutation_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# a8 — HI 절단오차 (완전사이클 HI값 - 세그먼트 HI값)
# ---------------------------------------------------------------------------

def plot_hi_truncation_error(out_dir: Path) -> None:
    # 세그먼트(q_frac_ref) 축 — train+test 합쳐서 표본을 늘림(완전사이클 쪽과 매칭할 표본 수 확보)
    seg_train_ds, seg_test_ds, concept_names, seg_raw_tr, seg_raw_te, spec = _build_raw_hi(
        V4_RUN, CANON_DATA_DIR, CANON_SEG_DIR
    )
    seg_cell = np.concatenate([np.asarray(seg_train_ds.cell_ids), np.asarray(seg_test_ds.cell_ids)])
    seg_cycle = np.concatenate([np.asarray(seg_train_ds.cycles), np.asarray(seg_test_ds.cycles)])
    seg_name = np.concatenate([np.asarray(seg_train_ds.seg_names), np.asarray(seg_test_ds.seg_names)])
    seg_raw = np.concatenate([seg_raw_tr, seg_raw_te], axis=0)

    # 완전사이클(full_cycle) 축 — chg_full/dis_full 2개 시나리오뿐. axis를 명시적으로
    # "full_cycle"로 줘야 한다(안 주면 q_frac_ref spec으로 잘못 해석되는 버그 있었음).
    fc_train_ds, fc_test_ds, fc_concepts, fc_raw_tr, fc_raw_te, fc_spec = _build_raw_hi(
        V4_RUN, FULLCYCLE_DATA_DIR, FULLCYCLE_SEG_DIR, axis="full_cycle", axis_config={}
    )
    assert fc_concepts == concept_names, "full_cycle과 q_frac_ref의 HI 컬럼 순서가 다름"
    fc_cell = np.concatenate([np.asarray(fc_train_ds.cell_ids), np.asarray(fc_test_ds.cell_ids)])
    fc_cycle = np.concatenate([np.asarray(fc_train_ds.cycles), np.asarray(fc_test_ds.cycles)])
    fc_name = np.concatenate([np.asarray(fc_train_ds.seg_names), np.asarray(fc_test_ds.seg_names)])
    fc_raw = np.concatenate([fc_raw_tr, fc_raw_te], axis=0)
    print(f"  full_cycle 시나리오: {sorted(set(fc_name.tolist()))}")

    # (cell_id, cycle, direction) -> row index 룩업 (완전사이클은 direction당 1개뿐이라 유일)
    fc_lookup: dict[tuple, int] = {}
    for i, (c, cyc, nm) in enumerate(zip(fc_cell, fc_cycle, fc_name)):
        direction = "chg" if nm.startswith("chg") else "dis"
        fc_lookup[(c, int(cyc), direction)] = i

    n_hi = len(concept_names)
    diff_sum = np.zeros((n_hi, len(SCENARIO_ORDER)))
    diff_cnt = np.zeros((n_hi, len(SCENARIO_ORDER)))
    matched = 0
    for row in range(len(seg_cell)):
        scen = seg_name[row]
        if scen not in SCENARIO_ORDER:
            continue
        direction = "chg" if scen.startswith("chg") else "dis"
        key = (seg_cell[row], int(seg_cycle[row]), direction)
        fc_row = fc_lookup.get(key)
        if fc_row is None:
            continue
        matched += 1
        s_idx = SCENARIO_ORDER.index(scen)
        seg_vals = seg_raw[row]
        fc_vals = fc_raw[fc_row]
        valid = ~np.isnan(seg_vals) & ~np.isnan(fc_vals)
        d = np.abs(seg_vals[valid] - fc_vals[valid])
        idxs = np.where(valid)[0]
        diff_sum[idxs, s_idx] += d
        diff_cnt[idxs, s_idx] += 1
    print(f"  매칭된 세그먼트 {matched}/{len(seg_cell)}건")

    trunc_err = np.divide(diff_sum, diff_cnt, out=np.full_like(diff_sum, np.nan), where=diff_cnt > 0)

    # 카테고리별 평균 절단오차(정규화: HI 자체 스케일이 다 달라서 raw 절단오차 절대값은
    # 비교 의미가 약함 — 세그먼트값 표준편차 대비 상대 절단오차로 다시 정규화).
    seg_std = np.nanstd(seg_raw, axis=0)
    rel_trunc_err = trunc_err / np.where(seg_std[:, None] > 1e-9, seg_std[:, None], 1.0)

    categories = [c.split("_", 1)[0] for c in concept_names]
    cat_order = ["stat", "diff", "lfp", "morph"]
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_means = np.full((len(cat_order), len(SCENARIO_ORDER)), np.nan)
    for ci, cat in enumerate(cat_order):
        rows = [i for i, c in enumerate(categories) if c == cat]
        cat_means[ci] = np.nanmean(rel_trunc_err[rows], axis=0)
    im = ax.imshow(cat_means, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(SCENARIO_ORDER, rotation=30)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([c.upper() for c in cat_order])
    for ci in range(len(cat_order)):
        for si in range(len(SCENARIO_ORDER)):
            v = cat_means[ci, si]
            if not np.isnan(v):
                ax.text(si, ci, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > np.nanmax(cat_means) * 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="상대 절단오차(|세그먼트값-완전사이클값| / 세그먼트값 표준편차)")
    ax.set_title("카테고리별 HI 절단오차 — LFP가 유독 크면 0% 생존의 1순위 후보 원인 확인됨")
    fig.tight_layout()
    out_path = out_dir / "a8_hi_truncation_error.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")
    print("  카테고리별 평균 상대 절단오차(전체 시나리오 평균):",
          {cat: float(np.nanmean(cat_means[i])) for i, cat in enumerate(cat_order)})


# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="docs/FLOW.md 'A' 카테고리 플랏 통합 생성")
    p.add_argument("--out-dir", default="5_model/plots", dest="out_dir")
    p.add_argument("--only", nargs="+", default=None,
                   help="특정 항목만 실행(예: --only a1 a5). 미지정시 전체 실행")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("a1", "데이터셋x시나리오 교차표", plot_dataset_scenario_crosstab),
        ("a2", "HI x 시나리오 유의성(효과크기 vs BH p-value)", plot_hi_scenario_significance),
        ("a3", "셀/SOH구간별 오차 + mAh MAPE", plot_cell_and_soh_bin_error),
        ("a4", "시드 안정성 + Jaccard 퇴화진단", plot_seed_stability_diagnostics),
        ("a5", "v0/v2/v4 vs ctrl 비교", plot_ctrl_comparison_bar),
        ("a6", "v4 아키텍처 개략도", plot_architecture_diagram),
        ("a7", "HI 처방 전이 + 최소크기 스윕 + permutation importance", plot_hi_subset_transfer),
        ("a8", "HI 절단오차(완전사이클 대비)", plot_hi_truncation_error),
    ]
    only = set(args.only) if args.only else None

    for key, label, fn in tasks:
        if only and key not in only:
            continue
        _safe_run(f"{key} {label}", fn, out_dir)

    print(f"\n완료 — 결과: {out_dir}")


if __name__ == "__main__":
    main()
