"""
5_model/experiments/phase1_lab/plot_multicollinearity_reduction.py

v3(build_synergy_groups.py --global-dedup)가 실제로 그룹 간 다중공선성을 줄였는지
시각화한다. 기존(v0/v1/v2, 그룹 내부만 배제)과 v3(그 시나리오 전체와 배제)의
"그룹 간 최대 잔여 상관"을 나란히 비교한다.

지표: HI 하나마다 "자기 그룹이 아닌 다른 그룹 멤버 중 가장 상관이 높은 것과의
|raw corr|"(cross-group max correlation) — v0/v1/v2는 이 값이 redundancy_threshold(0.9)를
넘는 HI가 있어도 못 잡는게 알려진 한계였고, v3는 이 값 자체를 낮추는 게 목표다.

phase1_trainer_v2.py와 분리된 독립 스크립트 — 두 synergy_groups json과 raw 데이터만
읽고 학습은 하지 않는다.

사용 예(--seg-axis/--axis-config/--data-dir/--seg-data-dir은 표준 조합이면 생략 가능 —
기본값 자동 적용):
  python 5_model/experiments/phase1_lab/plot_multicollinearity_reduction.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --split-seed 42 \
      --before 5_model/experiments/phase1_lab/results/outputs/synergy_groups_k25_full_N2_groups_noleak.json \
      --after  5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from build_synergy_groups import _load_all_scenarios  # noqa: E402

# 루트는 data_directories.py의 DATA_4_HI_ROOT_STR에서 가져온다(build_synergy_groups.py/
# lambda_sweep.py와 동일 이유 — PC마다 실제 드라이브가 다름).
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

# seg-axis/axis-config도 이 세션 전체에서 한 번도 안 바뀐 고정값 — 위 데이터 경로와 세트로
# 묶인 값이라(다른 조합이면 데이터 경로도 같이 바뀌어야 함) 다른 조합을 쓰려면 셋 다
# 함께 오버라이드해야 한다.
DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="다중공선성 제거 전/후(v0~v2 vs v3) 비교 플랏")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR)
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--before", required=True, help="기존(v0/v1/v2) synergy_groups json")
    p.add_argument("--after", required=True, help="v3(--global-dedup) synergy_groups json")
    p.add_argument("--redundancy-threshold", type=float, default=0.9)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _cross_group_max_corr(groups: list[list[int]], raw_corr: np.ndarray, n_hi: int) -> np.ndarray:
    """HI별로 '자기 그룹이 아닌 다른 그룹 멤버'와의 |raw corr| 최댓값을 반환((n_hi,))."""
    group_of = {}
    for gi, g in enumerate(groups):
        for m in g:
            group_of[m] = gi
    out = np.zeros(n_hi)
    for i in range(n_hi):
        gi = group_of.get(i)
        others = [j for j in range(n_hi) if group_of.get(j) != gi]
        if others:
            out[i] = np.max(np.abs(raw_corr[i, others]))
    return out


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

    x_all, y_all, scen_idx_all, spec, names_by_seg = _load_all_scenarios(args)
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))

    all_before, all_after = [], []
    n_scen = spec.n_scenarios
    fig, axes = plt.subplots(1, n_scen, figsize=(4 * n_scen, 4), sharey=True)
    if n_scen == 1:
        axes = [axes]

    for s in range(n_scen):
        key = f"seg_{s}_groups"
        if key not in before or key not in after:
            continue
        sel = scen_idx_all == s
        x_s = x_all[sel]
        n_hi = x_s.shape[1]
        raw_corr = np.nan_to_num(np.corrcoef(x_s, rowvar=False), nan=0.0)

        # v3.1(global_dedup) 산출물이면 사전 가지치기로 탈락해 사후 편입된 HI가
        # "seg_{s}_group_attached"에 별도로 있다 — 이걸 members와 합쳐야 그 HI가
        # 자기 그룹 소속으로 올바르게 잡힌다(안 그러면 group_of가 None이 돼 모든
        # 다른 그룹과 "교차 위반"으로 잘못 카운트됨).
        after_attached = after.get(f"seg_{s}_group_attached")
        after_groups_full = (
            [list(g) + list(a) for g, a in zip(after[key], after_attached)]
            if after_attached else after[key]
        )
        cb = _cross_group_max_corr(before[key], raw_corr, n_hi)
        ca = _cross_group_max_corr(after_groups_full, raw_corr, n_hi)
        all_before.append(cb)
        all_after.append(ca)

        ax = axes[s]
        ax.hist(cb, bins=20, range=(0, 1), alpha=0.5, color="#b2182b", label="기존(그룹 내부만)")
        ax.hist(ca, bins=20, range=(0, 1), alpha=0.5, color="#2166ac", label="v3(전역 배제)")
        ax.axvline(args.redundancy_threshold, color="black", linestyle="--", linewidth=1)
        ax.set_title(before.get(f"seg_{s}_seg_name", f"seg_{s}"), fontsize=10)
        ax.set_xlabel("그룹 간 최대 |raw corr|")
        if s == 0:
            ax.set_ylabel("HI 개수")
            ax.legend(fontsize=8)

    fig.suptitle("다중공선성 제거 효과 — 그룹 간 최대 잔여 상관계수 분포(점선=배제 문턱 0.9)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.after).parent
    out_path = out_dir / "multicollinearity_reduction.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    cb_all = np.concatenate(all_before) if all_before else np.array([])
    ca_all = np.concatenate(all_after) if all_after else np.array([])
    print(f"[plot] 저장: {out_path}")
    print(f"[plot] 전체 HI 기준 — 기존: 그룹 간 최대상관 평균 {cb_all.mean():.4f}, "
          f"{args.redundancy_threshold} 초과 {int((cb_all >= args.redundancy_threshold).sum())}개")
    print(f"[plot]              v3: 그룹 간 최대상관 평균 {ca_all.mean():.4f}, "
          f"{args.redundancy_threshold} 초과 {int((ca_all >= args.redundancy_threshold).sum())}개")


if __name__ == "__main__":
    main()
