"""
compare_runs.py — 세 시나리오 축 모델 비교 리포트.

비교 항목:
  1. HI 랭킹 유사도 (Jaccard @ k, Kendall tau)
  2. qfrac 테스트 성능 (E1: 이상적 상한선)
  3. 랜덤 세그먼트 테스트 성능 (E2/E3)
  4. 논문 게재 수준 대비 달성률 추정

사용:
  python compare_runs.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Windows 터미널 cp949 → UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "_5_data_model_scr" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 비교 대상 run 정의 ────────────────────────────────────────────────────────
RUNS = {
    "protocol": "_5_data_model_scr/0717_0324_p2_mlp_prot",
    "vwindow":  "_5_data_model_scr/0717_1341_p2_mlp_vwin",
    "rcs":      "_5_data_model_scr/0717_2003_p2_mlp_rcs",
}

# 논문 게재 목표치 (IEEE TII / J.Power Sources 수준)
TARGETS = {
    "e1_rmse":   0.015,   # qfrac test RMSE (SOH ratio) < 1.5%
    "e1_mape":   1.5,     # qfrac test MAPE < 1.5%
    "e1_r2":     0.95,    # qfrac test R² > 0.95
    "e2_rmse":   0.035,   # random seg RMSE < 3.5% (E1의 2배 허용)
    "e2_r2":     0.70,    # random seg R² > 0.70
    "e2_mape":   3.5,     # random seg MAPE < 3.5%
}

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────

def _load(run_key: str):
    d = PROJECT_ROOT / RUNS[run_key]
    metrics = json.loads((d / "metrics" / "metrics.json").read_text(encoding="utf-8"))
    rs_metrics = json.loads((d / "random_seg_test" / "metrics.json").read_text(encoding="utf-8"))
    clf_hi = json.loads((d / "gates" / "classification_HIs.json").read_text(encoding="utf-8"))
    return metrics, rs_metrics, clf_hi


# ─────────────────────────────────────────────────────────────────────────────
# HI 유사도
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_at_k(a: list[int], b: list[int], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def kendall_tau_top_k(a: list[int], b: list[int], k: int) -> float:
    """상위 k개 교집합 원소에 대해 원본 순위 기반 Kendall tau."""
    shared = [v for v in a[:k] if v in set(b[:k])]
    if len(shared) < 2:
        return float("nan")
    # 원본 리스트에서의 순위 (낮을수록 상위)
    rank_a = {v: a.index(v) for v in shared}
    rank_b = {v: b.index(v) for v in shared}
    concordant = discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            da = rank_a[shared[i]] - rank_a[shared[j]]
            db = rank_b[shared[i]] - rank_b[shared[j]]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    n = len(shared)
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else float("nan")


def hi_similarity_table(all_hi: dict) -> str:
    pairs = [
        ("protocol", "vwindow"),
        ("protocol", "rcs"),
        ("vwindow",  "rcs"),
    ]
    K = [5, 10, 20]
    lines = []
    for direction in ("charge", "discharge"):
        lines.append(f"\n  [{direction.upper()} — Top-k Jaccard / Kendall-τ]")
        lines.append(f"  {'Pair':<22}" + "".join(f"  J@{k}/τ@{k}" for k in K))
        lines.append("  " + "-" * (22 + 14 * len(K)))
        for a, b in pairs:
            key = f"{direction}_ranked"
            row = f"  {a} vs {b:<12}"
            for k in K:
                ja = jaccard_at_k(all_hi[a][key], all_hi[b][key], k)
                ta = kendall_tau_top_k(all_hi[a][key], all_hi[b][key], k)
                tau_s = f"{ta:+.2f}" if not (isinstance(ta, float) and np.isnan(ta)) else "  NaN"
                row += f"  {ja:.2f}/{tau_s}"
            lines.append(row)
    return "\n".join(lines)


def top_hi_union(all_hi: dict, direction: str, k: int = 10) -> str:
    """축별 top-k 이름을 비교해 공통·고유 피처 표시."""
    key_r = f"{direction}_ranked"
    key_n = f"{direction}_names"
    tops = {}
    for run in RUNS:
        ranked = all_hi[run][key_r][:k]
        names  = all_hi[run][key_n]
        full   = all_hi[run][key_r]
        tops[run] = {full[i]: names[i] for i in range(k)}

    sets = {r: set(tops[r]) for r in tops}
    common_all = sets["protocol"] & sets["vwindow"] & sets["rcs"]
    lines = [f"\n  [{direction.upper()} top-{k} 교집합 (3축 공통 피처)]"]
    for idx in all_hi["protocol"][key_r]:
        if idx in common_all:
            lines.append(f"    HI[{idx:2d}]  {all_hi['protocol'][key_n][all_hi['protocol'][key_r].index(idx)]}")
    if len(lines) == 1:
        lines.append("    (없음)")

    lines.append(f"\n  [{direction.upper()} 축별 top-{k} 고유 피처]")
    for run in RUNS:
        other_union = set().union(*(sets[r] for r in RUNS if r != run))
        only_this = sets[run] - other_union
        if only_this:
            names_map = {idx: tops[run][idx] for idx in only_this}
            lines.append(f"    [{run}] " + ", ".join(f"HI[{i}]={n.split('_')[1]}" for i, n in list(names_map.items())[:4]))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 성능 테이블
# ─────────────────────────────────────────────────────────────────────────────

def perf_table(all_metrics: dict, all_rs: dict) -> str:
    W = 72
    lines = []
    def h(t): lines.extend(["  " + "=" * W, f"  {t}", "  " + "=" * W])
    def r(label, vals): lines.append(f"  {label:<30}" + "".join(f"{v:>12}" for v in vals))

    h("E1: qfrac→qfrac 테스트 성능 (이상적 상한선)")
    r("", [f"[{k}]" for k in RUNS])
    r("RMSE (SOH ratio)", [f"{all_metrics[k]['test']['capacity']['rmse']:.4f}" for k in RUNS])
    r("MAE  (SOH ratio)", [f"{all_metrics[k]['test']['capacity']['mae']:.4f}"  for k in RUNS])
    r("R²",              [f"{all_metrics[k]['test']['capacity']['r2']:.4f}"    for k in RUNS])
    r("MAPE (%)",        [f"{all_metrics[k]['test']['capacity']['mape']:.2f}"  for k in RUNS])
    lines.append("")
    r("  Charge RMSE",   [f"{all_metrics[k]['test']['breakdown']['charge']['rmse']:.4f}" for k in RUNS])
    r("  Discharge RMSE",[f"{all_metrics[k]['test']['breakdown']['discharge']['rmse']:.4f}" for k in RUNS])

    lines.append("")
    h("E2/E3: qfrac→random 세그먼트 테스트 성능 (배포 시나리오)")
    r("", [f"[{k}]" for k in RUNS])
    r("RMSE (SOH ratio)", [f"{all_rs[k]['rmse']:.4f}"  for k in RUNS])
    r("MAE  (SOH ratio)", [f"{all_rs[k]['mae']:.4f}"   for k in RUNS])
    r("R²",              [f"{all_rs[k]['r2']:.4f}"    for k in RUNS])
    r("MAPE (%)",        [f"{all_rs[k]['mape']:.2f}"  for k in RUNS])
    r("routing mode",    [f"{all_rs[k]['routing']}"   for k in RUNS])
    lines.append("")
    r("  E2/E1 RMSE 배율", [f"{all_rs[k]['rmse']/all_metrics[k]['test']['capacity']['rmse']:.1f}×" for k in RUNS])

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 논문 게재 수준 달성률
# ─────────────────────────────────────────────────────────────────────────────

def publication_readiness(all_metrics: dict, all_rs: dict) -> str:
    W = 72
    lines = ["  " + "=" * W, "  논문 게재 수준 달성률 추정 (IEEE TII / J.Power Sources 기준)", "  " + "=" * W]
    lines.append(f"  {'지표':<35}  목표      " + "".join(f"  {k[:6]:>8}" for k in RUNS))
    lines.append("  " + "-" * W)

    def _score(val, target, lower_better=True):
        if lower_better:
            return min(1.0, target / val) if val > 0 else 1.0
        else:
            return min(1.0, val / target) if target > 0 else 1.0

    checks = [
        ("E1 RMSE (SOH)",   [m["test"]["capacity"]["rmse"]   for m in all_metrics.values()], TARGETS["e1_rmse"],  True,  "< 1.5%"),
        ("E1 MAPE (%)",     [m["test"]["capacity"]["mape"]   for m in all_metrics.values()], TARGETS["e1_mape"],  True,  "< 1.5%"),
        ("E1 R²",           [m["test"]["capacity"]["r2"]     for m in all_metrics.values()], TARGETS["e1_r2"],    False, "> 0.95"),
        ("E2 RMSE (SOH)",   [rs["rmse"]                      for rs in all_rs.values()],     TARGETS["e2_rmse"],  True,  "< 3.5%"),
        ("E2 R²",           [rs["r2"]                        for rs in all_rs.values()],     TARGETS["e2_r2"],    False, "> 0.70"),
        ("E2 MAPE (%)",     [rs["mape"]                      for rs in all_rs.values()],     TARGETS["e2_mape"],  True,  "< 3.5%"),
    ]

    weights = [0.20, 0.15, 0.15, 0.20, 0.15, 0.15]
    run_scores = {k: [] for k in RUNS}

    for label, vals, target, lower, target_str in checks:
        scores = [_score(v, target, lower) * 100 for v in vals]
        for k, s in zip(RUNS, scores):
            run_scores[k].append(s)
        lines.append(
            f"  {label:<35}  {target_str:<8}  " +
            "".join(f"  {s:>6.0f}%  " for s in scores)
        )

    lines.append("  " + "-" * W)

    weighted = {}
    for k, scores in run_scores.items():
        weighted[k] = sum(w * s for w, s in zip(weights, scores))
    lines.append(
        f"  {'가중 평균 달성률':<35}  {'(종합)':>8}  " +
        "".join(f"  {weighted[k]:>6.0f}%  " for k in RUNS)
    )
    lines.append("  " + "=" * W)

    lines.append("")
    lines.append("  [해석]")
    for k in RUNS:
        w = weighted[k]
        if w >= 80:
            grade = "논문 투고 가능 (Minor revision 예상)"
        elif w >= 60:
            grade = "E2 성능 개선 필요 — Major revision 가능성"
        elif w >= 40:
            grade = "상당한 개선 필요 — 현 수준으로는 투고 불가"
        else:
            grade = "근본적 재설계 필요"
        lines.append(f"    [{k:10s}]  {w:.0f}%  →  {grade}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 성능 향상 제안
# ─────────────────────────────────────────────────────────────────────────────

IMPROVEMENT_SUGGESTIONS = """
  ═══════════════════════════════════════════════════════════════════════════════
  성능 향상 제안 (우선순위 순)
  ═══════════════════════════════════════════════════════════════════════════════

  ★★★ 긴급 (E2/E3 R² < 0 → 배포 불가 상태) ★★★

  [1] test_rs seg_idx↔모델 시나리오 ID 불일치 확인 (이미 수정됨, 재측정 필요)
      · 이전 실험은 routing_mode="none"에서 방전 세그먼트가 충전 게이트로 들어갔음
      · direction_routing 수정 후 Step 9 재실행으로 실제 E2 수준 확인 선행

  [2] Random Seg 길이·위치 분포 재검토
      · 현재: min_len=5%, max_len=40%, 완전 랜덤 시작
      · 문제: 매우 짧은 세그먼트(5-10%)에서 HI 신뢰도 급락 → MAPE 16% 유발
      · 제안: min_len=15~20%로 상향, 또는 세그먼트 길이별 가중치 학습

  [3] RCS 모델 랜덤 세그먼트 내성 최악 원인 분석
      · RCS 자체가 랜덤 부분관측으로 학습됨에도 E2 RMSE=0.185 (prot의 2.2배)
      · 원인 의심: RCS 세그먼트는 CC 구간만, test_rs는 전 구간 → 분포 불일치
      · 제안: test_rs 생성 시 CC 구간만 샘플링하는 옵션 추가 (RCS 전용 E2 평가)

  ★★ 중요 (E1 R² 0.64~0.85 → 목표 0.95 미달) ★★

  [4] vwindow chg_cv 시나리오 성능 저하 (MAPE 5.4%, R²=0.195)
      · CV 구간은 전류가 급감 → stat_i_mean/i_std HI가 비정상적으로 낮아짐
      · 제안: CV 전용 HI 추가 (cv_time_frac, Q_CV/Q_total 이미 있는지 확인)
        또는 CV 세그먼트에 별도 정규화기 적용

  [5] Protocol level_step2 성능 저하 (MAPE 4.1%, R²=0.438)
      · step2 = CV 구간으로 HI 특성이 CC와 이질적
      · 제안: protocol CV step을 vwindow chg_cv처럼 별도 처리

  [6] 학습 에폭 / 조기 종료 재조정
      · 현재 val 기준: prot R²=0.745, rcs R²=0.830 → 아직 학습 여지 있을 수 있음
      · 제안: epochs=1000, patience=50으로 늘리고 LR schedule cosine restart 시도

  ★ 논문 강화 (Reviewer 요구사항)

  [7] 5-seed 반복 실험 (mean ± std) — 현재 단일 실험
      · IEEE TII 기준: 최소 3-5 seed mean/std 필수
      · 제안: for seed in [0,1,2,3,4]: train_scr --phase 1 --seed $seed && ...

  [8] 베이스라인 비교 (현재 없음)
      · 필수: Severson 2019 (linear regression on ΔQ feature)
      · 권장: XGBoost + hand-crafted HI, vanilla MLP (no gating)
      · 비교 없이는 리뷰어가 우월성 주장을 수용하지 않음

  [9] dvdq_peak_q, dvdq_valley_q 절대 Ah → q_frac 정규화
      · MIT(1.1Ah) vs HUST(1.2Ah) 절대값 차이 → HI 도메인 갭
      · 특히 크로스-데이터셋 평가 시 이 두 피처가 게이트에서 강하게 선택될수록 불리

  [10] 양방향 크로스-데이터셋 평가
       · 현재: MIT+HUST 셀 분할 (동일 분포)
       · 필요: MIT→HUST, HUST→MIT (학습 데이터와 완전히 다른 도메인)
       · 제안: is_cross_dataset_evaluate: true + datasets 순서 변경

  ═══════════════════════════════════════════════════════════════════════════════
  권장 즉시 실행 순서:
    1. Step 9 재실행 (direction_routing 수정 반영) → 실제 E2 수준 확인
    2. min_len=0.20으로 올려 test_rs 재생성 → E2 재측정
    3. 최고 E2 축(prot)으로 epochs=1000 재학습 → E1 개선 확인
    4. 베이스라인 1개 (vanilla MLP, no gate) 추가
  ═══════════════════════════════════════════════════════════════════════════════
"""


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────

def _plot_comparison(all_metrics: dict, all_rs: dict):
    if not _HAS_MPL:
        return

    axes_labels = list(RUNS.keys())
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 1: E1 성능 ────────────────────────────────────────────────────────
    metrics_e1 = {
        "RMSE": [all_metrics[k]["test"]["capacity"]["rmse"] for k in RUNS],
        "MAE":  [all_metrics[k]["test"]["capacity"]["mae"]  for k in RUNS],
        "R²":   [all_metrics[k]["test"]["capacity"]["r2"]   for k in RUNS],
    }
    for col, (metric, vals) in enumerate(metrics_e1.items()):
        ax = fig.add_subplot(gs[0, col])
        bars = ax.bar(axes_labels, vals, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        target = {"RMSE": TARGETS["e1_rmse"], "MAE": None, "R²": TARGETS["e1_r2"]}[metric]
        if target is not None:
            ax.axhline(target, color="red", lw=1.5, ls="--", label=f"목표 {target}")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"E1 {metric} (qfrac 테스트)", fontsize=10)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelsize=9)

    # ── Row 2: E2 성능 ────────────────────────────────────────────────────────
    metrics_e2 = {
        "RMSE": [all_rs[k]["rmse"] for k in RUNS],
        "R²":   [all_rs[k]["r2"]   for k in RUNS],
        "MAPE (%)": [all_rs[k]["mape"] for k in RUNS],
    }
    for col, (metric, vals) in enumerate(metrics_e2.items()):
        ax = fig.add_subplot(gs[1, col])
        bar_colors = []
        for v in vals:
            if metric == "R²":
                bar_colors.append("tab:red" if v < 0 else colors[vals.index(v) % len(colors)])
            else:
                bar_colors.append(colors[vals.index(v) % len(colors)])
        bars = ax.bar(axes_labels, vals, color=bar_colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        target = {"RMSE": TARGETS["e2_rmse"], "R²": TARGETS["e2_r2"], "MAPE (%)": TARGETS["e2_mape"]}[metric]
        ax.axhline(target, color="red", lw=1.5, ls="--", label=f"목표 {target}")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ypos = max(bar.get_height(), 0) + abs(max(vals) - min(vals)) * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"E2 {metric} (random seg 테스트)", fontsize=10)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("시나리오 축별 성능 비교  (빨간 점선 = 논문 게재 목표)", fontsize=12, y=1.01)
    out_path = OUT_DIR / "performance_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [그래프] {out_path}")


def _plot_hi_overlap(all_hi: dict):
    if not _HAS_MPL:
        return

    K_range = list(range(1, 33))
    pairs   = [("protocol", "vwindow"), ("protocol", "rcs"), ("vwindow", "rcs")]
    colors  = ["tab:blue", "tab:orange", "tab:green"]
    directions = ["charge", "discharge"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, direction in zip(axes, directions):
        for (a, b), c in zip(pairs, colors):
            key = f"{direction}_ranked"
            jac = [jaccard_at_k(all_hi[a][key], all_hi[b][key], k) for k in K_range]
            ax.plot(K_range, jac, color=c, lw=2, label=f"{a[:4]} vs {b[:4]}")
        ax.axvline(10, color="gray", ls="--", lw=0.8, label="k=10 (probe_m)")
        ax.set_xlabel("Top-k")
        ax.set_ylabel("Jaccard 유사도")
        ax.set_title(f"HI 랭킹 Jaccard 유사도 ({direction})", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("축별 probe HI 선택 일치도 (Jaccard @ k)", fontsize=11)
    plt.tight_layout()
    out_path = OUT_DIR / "hi_similarity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [그래프] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n[compare_runs] 데이터 로드 중...")
    all_metrics, all_rs, all_hi = {}, {}, {}
    for key in RUNS:
        m, rs, hi = _load(key)
        all_metrics[key] = m
        all_rs[key]      = rs
        all_hi[key]      = hi
    print("  완료\n")

    W = 74
    sep = "=" * W

    print(sep)
    print("  SCR 시나리오 축 비교 리포트")
    print(f"  대상: {', '.join(RUNS.keys())}")
    print(sep)

    # 1. 성능 테이블
    print(perf_table(all_metrics, all_rs))

    # 2. HI 유사도
    print(f"\n{sep}")
    print("  HI 랭킹 유사도")
    print(sep)
    print(hi_similarity_table(all_hi))
    print(top_hi_union(all_hi, "charge", k=10))
    print(top_hi_union(all_hi, "discharge", k=10))

    # 3. 논문 게재 수준
    print(f"\n{sep}")
    print(publication_readiness(all_metrics, all_rs))

    # 4. 성능 향상 제안
    print(IMPROVEMENT_SUGGESTIONS)

    # 5. 시각화
    if _HAS_MPL:
        print(f"\n[시각화 저장 → {OUT_DIR}]")
        _plot_comparison(all_metrics, all_rs)
        _plot_hi_overlap(all_hi)
    else:
        print("  (matplotlib 없음 — 그래프 생략)")

    # 6. JSON 요약 저장
    summary = {
        run: {
            "e1": {k: all_metrics[run]["test"]["capacity"][k] for k in ["rmse","mae","r2","mape"]},
            "e2": {k: all_rs[run][k] for k in ["rmse","mae","r2","mape","routing"]},
        }
        for run in RUNS
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[요약 JSON] {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
