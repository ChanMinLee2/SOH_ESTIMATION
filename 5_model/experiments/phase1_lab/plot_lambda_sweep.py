"""
5_model/experiments/phase1_lab/plot_lambda_sweep.py

lambda_sweep.py의 산출물(summary.csv)을 시각화한다. 학습(lambda_sweep.py)과
분석(이 파일)을 분리한 이유: summary.csv는 런이 끝날 때마다 한 줄씩 append되므로
스윕이 아직 도는 중에도(부분 결과로) 여러 번 다시 그려볼 수 있고, 플랏 스타일을
바꿀 때마다 학습을 다시 돌 필요가 없다. plot_synergy_groups.py와 같은 패턴
(matplotlib Agg, 한글 폰트 처리, 미설치 시 조용히 종료)을 재사용한다.

그림 3장을 만든다:
  1) 정규화 경로 — x=lambda_l0(로그축), y=활성 HI 개수(회귀/분류 각각, 시나리오 평균).
     --scen-k를 주면 목표선을 점선으로 같이 그림 — "공식이 목표를 실제로 맞추는지" 한눈에 확인.
  2) 성능-개수 곡선(무릎 찾기용) — x=활성 회귀 HI 개수, y=val_r2 / val_rmse. 각 점에
     lambda 값을 라벨로 붙이고, "첫점-끝점을 잇는 직선에서 가장 멀리 떨어진 점"을
     단순 elbow 후보로 표시(참고용 — 단일 시드라 최종 판단은 그림을 보고 사람이 함).
  3) 시나리오별 분해 — summary.csv의 active_reg_avg/active_cls_avg는 회귀 6개
     시나리오(chg_lo/mid/hi, dis_hi/mid/lo)·분류 2개 probe(charge/discharge)를
     평균낸 값이라 시나리오 간 편차가 안 보인다. summary.csv의 run_dir 컬럼으로 각 런의
     gates/regression_HIs.json·classification_HIs.json을 직접 읽어 8개 시나리오/probe를
     각각의 선으로 그린다(lambda_sweep.py 자체는 안 건드림 — 이미 저장된 JSON을 재활용).

사용 예:
  python 5_model/experiments/phase1_lab/plot_lambda_sweep.py \
      --input 5_model/experiments/phase1_lab/results/lambda_sweep/0824_1620_lsweep/summary.csv \
      --scen-k 25
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Windows 콘솔이 cp949일 때 한글/em-dash print가 UnicodeEncodeError로 죽는 문제 방지.
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lambda_sweep.py summary.csv 시각화")
    p.add_argument("--input", required=True, help="lambda_sweep.py가 만든 summary.csv 경로")
    p.add_argument("--out-dir", default=None, help="PNG 저장 위치 (기본: summary.csv와 같은 폴더)")
    p.add_argument("--scen-k", type=int, default=None, help="목표 활성 개수(점선 참고선)")
    p.add_argument("--cost-json", default=None,
                   help="4_hi_analysis/profile_hi_timing.py가 만든 hi_timing_cost.json 경로. "
                        "주면 성능-개수 곡선(그림2) 왼쪽 패널에 오른쪽 y축으로 계산 비용(µs, "
                        "주황색)을 겹쳐 그린다.")
    return p.parse_args()


def _load_rows(csv_path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(csv_path.open(encoding="utf-8")):
        if r.get("status") != "ok":
            continue
        try:
            r["lambda_l0"] = float(r["lambda_l0"])
            r["active_reg_avg"] = float(r["active_reg_avg"])
            r["active_cls_avg"] = float(r["active_cls_avg"])
            r["val_r2"] = float(r["val_r2"])
            r["val_rmse"] = float(r["val_rmse"])
            r["gate_saturation"] = float(r["gate_saturation"])
        except (ValueError, KeyError):
            continue
        rows.append(r)
    rows.sort(key=lambda r: r["lambda_l0"])
    return rows


def _per_scenario_counts(run_dir: Path) -> tuple[dict[str, int], dict[str, int]] | None:
    """gates/regression_HIs.json·classification_HIs.json에서 시나리오/probe별
    활성 개수(gate_prob>0.5)를 뽑는다. summary.csv의 active_*_avg가 평균낸 바로 그
    원본 — lambda_sweep.py의 _avg_active()와 threshold(0.5)를 맞춤."""
    reg_path = run_dir / "gates" / "regression_HIs.json"
    cls_path = run_dir / "gates" / "classification_HIs.json"
    if not reg_path.exists() or not cls_path.exists():
        return None
    reg_d = json.loads(reg_path.read_text(encoding="utf-8"))
    cls_d = json.loads(cls_path.read_text(encoding="utf-8"))
    reg_counts = {}
    for key in sorted(k for k in reg_d if k.endswith("_probs")):
        seg_idx = key[: -len("_probs")]
        name = reg_d.get(f"{seg_idx}_seg_name", seg_idx)
        reg_counts[name] = sum(1 for x in reg_d[key] if x > 0.5)
    cls_counts = {}
    for probe in ("charge", "discharge"):
        probs_key = f"{probe}_probs"
        if probs_key in cls_d:
            cls_counts[probe] = sum(1 for x in cls_d[probs_key] if x > 0.5)
    return reg_counts, cls_counts


def _segment_active_concepts(reg_d: dict, seg_key: str) -> list[str]:
    """seg_key(예: 'seg_0')의 활성 HI(gate_prob>0.5) 이름에서 시나리오 접미사를 떼고
    concept 이름만 반환(예: 'stat_v_mean_cw_chg_lo' -> 'stat_v_mean_cw')."""
    ranked = reg_d[f"{seg_key}_ranked"]
    probs = reg_d[f"{seg_key}_probs"]
    names = reg_d[f"{seg_key}_names"]
    seg_name = reg_d.get(f"{seg_key}_seg_name", "")
    by_idx_prob = {idx: p for idx, p in zip(ranked, probs)}
    by_idx_name = {idx: n for idx, n in zip(ranked, names)}
    suf = f"_{seg_name}"
    active = []
    for idx, prob in by_idx_prob.items():
        if prob > 0.5:
            name = by_idx_name[idx]
            concept = name[: -len(suf)] if suf and name.endswith(suf) else name
            active.append(concept)
    return active


def _segment_cost_us(active_concepts: list[str], cost: dict) -> float:
    """활성 concept 리스트의 계산 비용(µs) 합 — 전처리(vq_curve/ica_seg/morph_curves)는
    프로레이션 안 하고, 해당 카테고리(Diff/LFP 또는 Morph) 피처가 세그먼트 안에
    하나라도 활성이면 고정비용으로 1회만 더한다(hi_timing_cost.json의 note 참고)."""
    concept_mean = cost["concept_mean_us"]
    concept_cat = cost["concept_category"]
    preproc = cost["preproc_mean_us"]
    total = 0.0
    cats_present = set()
    for name in active_concepts:
        total += concept_mean.get(name, 0.0)
        cats_present.add(concept_cat.get(name))
    if "Diff" in cats_present or "LFP" in cats_present:
        total += preproc["vq_curve"] + preproc["ica_seg"]
    if "Morph" in cats_present:
        total += preproc["morph_curves"]
    return total


def _run_avg_cost_us(run_dir: Path, cost: dict) -> float | None:
    """run_dir의 gates/regression_HIs.json으로 6개 시나리오 각각의 비용을 구해 평균낸다
    (summary.csv의 active_reg_avg와 동일한 평균화 방식 — 회귀만, 분류는 안 봄)."""
    reg_path = run_dir / "gates" / "regression_HIs.json"
    if not reg_path.exists():
        return None
    reg_d = json.loads(reg_path.read_text(encoding="utf-8"))
    seg_keys = sorted(set(k[: -len("_probs")] for k in reg_d if k.endswith("_probs")))
    if not seg_keys:
        return None
    costs = [_segment_cost_us(_segment_active_concepts(reg_d, sk), cost) for sk in seg_keys]
    return sum(costs) / len(costs)


def _elbow_index(xs: list[float], ys: list[float]) -> int:
    """첫점-끝점 직선에서 수직거리가 가장 먼 점의 인덱스(단순 kneedle 근사)."""
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm == 0:
        return 0
    best_i, best_d = 0, -1.0
    for i, (x, y) in enumerate(zip(xs, ys)):
        d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def main() -> None:
    args = _parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return

    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(in_path)
    if not rows:
        print(f"[plot] {in_path}에 status=ok인 행이 없습니다 - 스윕이 아직 안 끝났거나 전부 실패.")
        return
    print(f"[plot] {len(rows)}개 성공 런 로드")

    lambdas = [r["lambda_l0"] for r in rows]
    active_reg = [r["active_reg_avg"] for r in rows]
    active_cls = [r["active_cls_avg"] for r in rows]
    val_r2 = [r["val_r2"] for r in rows]
    val_rmse = [r["val_rmse"] for r in rows]

    cost_data = None
    if args.cost_json:
        cost_data = json.loads(Path(args.cost_json).read_text(encoding="utf-8"))
    costs_us = None
    if cost_data is not None:
        costs_us = []
        for r in rows:
            run_dir_str = r.get("run_dir", "")
            c = _run_avg_cost_us(Path(run_dir_str), cost_data) if run_dir_str else None
            costs_us.append(c)
        if all(c is None for c in costs_us):
            print("[plot] --cost-json은 줬지만 어떤 run_dir에서도 비용을 계산 못 했습니다 - "
                  "gates/regression_HIs.json 경로 확인 필요. cost 오버레이 생략.")
            costs_us = None

    # ---- 그림 1: 정규화 경로 (lambda -> 활성 개수) ----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambdas, active_reg, "o-", color="#2166ac", label="회귀 활성 HI(gate_prob>0.5)")
    ax1.plot(lambdas, active_cls, "s--", color="#b2182b", label="분류 활성 HI(gate_prob>0.5)")
    if args.scen_k is not None:
        ax1.axhline(args.scen_k, color="gray", linestyle=":", linewidth=1,
                     label=f"목표(--scen-k={args.scen_k})")
    ax1.set_xscale("log")
    ax1.set_xlabel("lambda_l0 (log scale)")
    ax1.set_ylabel("활성 HI 개수")
    ax1.set_title("정규화 경로: lambda_l0 -> 활성 HI 개수")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    fig1.tight_layout()
    p1 = out_dir / "lambda_sweep_path.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    # ---- 그림 2: 성능-개수 곡선 (무릎 찾기) ----
    order = sorted(range(len(rows)), key=lambda i: active_reg[i])
    xs = [active_reg[i] for i in order]
    ys_r2 = [val_r2[i] for i in order]
    ys_rmse = [val_rmse[i] for i in order]
    lam_sorted = [lambdas[i] for i in order]
    elbow_i = _elbow_index(xs, ys_r2) if len(xs) >= 3 else None

    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    axL.plot(xs, ys_r2, "o-", color="#2166ac", label="val R2")
    for x, y, lam in zip(xs, ys_r2, lam_sorted):
        axL.annotate(f"{lam:g}", (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
    if elbow_i is not None:
        axL.scatter([xs[elbow_i]], [ys_r2[elbow_i]], color="red", zorder=5, s=80,
                     label=f"elbow 후보(lambda={lam_sorted[elbow_i]:g})")
    axL.set_xlabel("활성 회귀 HI 개수")
    axL.set_ylabel("val R2", color="#2166ac")
    axL.tick_params(axis="y", labelcolor="#2166ac")
    axL.set_title("성능-개수 곡선 (val R2 + 계산 비용)" if costs_us else "성능-개수 곡선 (val R2)")
    axL.grid(True, alpha=0.3)

    if costs_us:
        ys_cost = [costs_us[i] for i in order]
        axL2 = axL.twinx()
        valid = [(x, c) for x, c in zip(xs, ys_cost) if c is not None]
        if valid:
            axL2.plot([x for x, _ in valid], [c for _, c in valid], "o--",
                      color="#e67e22", label="계산 비용(µs, 회귀 6시나리오 평균)", alpha=0.85)
        axL2.set_ylabel("계산 비용 (µs)", color="#e67e22")
        axL2.tick_params(axis="y", labelcolor="#e67e22")
        h1, l1 = axL.get_legend_handles_labels()
        h2, l2 = axL2.get_legend_handles_labels()
        axL.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    else:
        axL.legend()

    axR.plot(xs, ys_rmse, "o-", color="#b2182b")
    for x, y, lam in zip(xs, ys_rmse, lam_sorted):
        axR.annotate(f"{lam:g}", (x, y), fontsize=7, textcoords="offset points", xytext=(4, 4))
    axR.set_xlabel("활성 회귀 HI 개수")
    axR.set_ylabel("val RMSE")
    axR.set_title("성능-개수 곡선 (val RMSE)")
    axR.grid(True, alpha=0.3)

    fig2.tight_layout()
    p2 = out_dir / "lambda_sweep_knee.png"
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # ---- 그림 3: 시나리오별 분해 (평균이 가리는 시나리오 간 편차 확인) ----
    p3 = None
    reg_series: dict[str, list[tuple[float, int]]] = {}
    cls_series: dict[str, list[tuple[float, int]]] = {}
    for r in rows:
        run_dir_str = r.get("run_dir", "")
        if not run_dir_str:
            continue
        counts = _per_scenario_counts(Path(run_dir_str))
        if counts is None:
            continue
        reg_counts, cls_counts = counts
        for name, n in reg_counts.items():
            reg_series.setdefault(name, []).append((r["lambda_l0"], n))
        for name, n in cls_counts.items():
            cls_series.setdefault(name, []).append((r["lambda_l0"], n))

    if reg_series or cls_series:
        fig3, (bxL, bxR) = plt.subplots(1, 2, figsize=(13, 5))
        reg_colors = plt.cm.tab10(range(max(len(reg_series), 1)))
        for (name, pts), color in zip(sorted(reg_series.items()), reg_colors):
            pts.sort()
            bxL.plot([x for x, _ in pts], [y for _, y in pts], "o-", color=color, label=name)
        bxL.plot(lambdas, active_reg, "k--", linewidth=1.5, alpha=0.6, label="평균(회귀)")
        bxL.set_xscale("log")
        bxL.set_xlabel("lambda_l0 (log scale)")
        bxL.set_ylabel("활성 HI 개수")
        bxL.set_title("회귀 6개 시나리오별 활성 개수")
        bxL.legend(fontsize=8)
        bxL.grid(True, which="both", alpha=0.3)

        cls_colors = {"charge": "#2166ac", "discharge": "#b2182b"}
        for name, pts in sorted(cls_series.items()):
            pts.sort()
            bxR.plot([x for x, _ in pts], [y for _, y in pts], "o-",
                     color=cls_colors.get(name), label=name)
        bxR.set_xscale("log")
        bxR.set_xlabel("lambda_l0 (log scale)")
        bxR.set_ylabel("활성 HI 개수")
        bxR.set_title("분류 charge/discharge별 활성 개수")
        bxR.legend(fontsize=8)
        bxR.grid(True, which="both", alpha=0.3)

        fig3.tight_layout()
        p3 = out_dir / "lambda_sweep_per_scenario.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
    else:
        print("[plot] run_dir의 gates JSON을 못 찾아 시나리오별 분해 그림은 건너뜀")

    print(f"[plot] 저장: {p1}")
    print(f"[plot] 저장: {p2}")
    if p3 is not None:
        print(f"[plot] 저장: {p3}")
    print("\n[plot] lambda 오름차순 요약:")
    print(f"{'lambda':>10} {'active_reg':>10} {'active_cls':>10} {'val_r2':>8} {'val_rmse':>9} {'sat':>8}")
    for r in rows:
        print(f"{r['lambda_l0']:>10g} {r['active_reg_avg']:>10.1f} {r['active_cls_avg']:>10.1f} "
              f"{r['val_r2']:>8.4f} {r['val_rmse']:>9.5f} {r['gate_saturation']:>8.4f}")
    if elbow_i is not None:
        print(f"\n[plot] elbow 후보: lambda={lam_sorted[elbow_i]:g}, "
              f"활성 개수={xs[elbow_i]:.1f}, val_r2={ys_r2[elbow_i]:.4f} "
              f"(단일 시드 기준 참고용 - 최종 판단은 그림을 보고 결정 권장)")


if __name__ == "__main__":
    main()
