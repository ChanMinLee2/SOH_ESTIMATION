"""
5_model/experiments/phase1_lab/test_hi_scenario_interaction.py

HI x 시나리오 상호작용 통계 검정 — v4의 선행 작업(docs/260824_RESULTS.md 기준 5,
"차기 버전 설계안 v4" 참고). 모델과 완전히 분리된 절차로, "이 HI가 시나리오마다
SOH와의 관계가 실제로 다른가"를 통계적으로 검정한다. 학습을 안 거치므로 게이트가
우연히 그렇게 갈랐는지, 데이터 자체에 진짜 구조가 있는지 독립적으로 확인 가능.

방법: Fisher z-변환 상관계수 동일성 검정. 각 raw HI 개념(concept)에 대해, 6개
시나리오 중 두 개씩 짝지어(15쌍) "그 시나리오에서 HI-SOH 상관계수가 통계적으로
같다고 볼 수 있는가"를 검정한다(귀무가설: r_a = r_b). 15쌍 중 최소 p-value를 그
HI의 "상호작용 증거"로 삼고, 64개 HI에 대해 Benjamini-Hochberg로 다중비교 보정한다.

이 스크립트의 산출물(JSON)은:
  1) v4의 shared_gate/scen_gates 라우팅 결정(유의미한 HI만 시나리오별 게이트 유지)
  2) plot_hi_scenario_interaction.py의 입력
에 쓰인다. train split만 사용(val/test 누수 없음) — build_synergy_groups.py와
동일한 데이터 로더(_load_all_scenarios)를 그대로 재사용(중복 구현 금지 원칙).

사용 예(--seg-axis/--axis-config/--data-dir/--seg-data-dir은 표준 조합이면 생략 가능 —
기본값 자동 적용):
  python 5_model/experiments/phase1_lab/test_hi_scenario_interaction.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --split-seed 42 --alpha 0.05 --tag k25_full_N2
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
RESULTS_DIR = _HERE / "results"

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

from build_synergy_groups import _load_all_scenarios  # noqa: E402 — 중복 구현 금지, 그대로 재사용

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
    p = argparse.ArgumentParser(description="HI x 시나리오 상호작용 통계 검정 (Fisher z, train만 사용)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR)
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05,
                   help="BH 보정 후 유의성 판정 기준(기본 0.05) — 참고용, 표본이 "
                        "수십만~백만 행이라 참값과 무관하게 거의 항상 유의하게 나온다 "
                        "(실측: 초기 테스트에서 66/66 전부 유의). shared_gate 배정은 "
                        "이것만으로 하지 말고 --min-effect-size와 함께 봐야 한다.")
    p.add_argument("--min-effect-size", type=float, default=0.1, dest="min_effect_size",
                   help="시나리오 간 상관계수의 표준편차(std_r_across_scenarios)가 이 값 "
                        "미만이면 '실질적으로 불변'으로 본다. 기본 0.1 = Cohen 상관계수 "
                        "'작음' 관행(이 문서 시너지 Level1의 |r|>=0.1 문턱과 동일 기준 재사용, "
                        "일관성 유지). 실측 분포가 0.043~0.207 사이에 뚜렷한 이봉분포 없이 "
                        "연속적으로 퍼져 있어(자연스러운 무릎이 없음) 0.05는 63/64가 통과해 "
                        "변별력이 거의 없었다 — 0.1은 약 39/64 통과로 그나마 실질적인 분리가 "
                        "생긴다. 15개 쌍 중 최댓값이 아니라 표준편차를 쓰는 이유: 최댓값은 "
                        "비교 15개 중 극값을 고르는 것이라 진짜 차이가 없어도 순전히 표본 "
                        "변동만으로 부풀려진다(order-statistics 효과).")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _fisher_z_test(r_a: float, n_a: int, r_b: float, n_b: int) -> float:
    """두 독립 표본 상관계수가 같다는 귀무가설의 양측 p-value."""
    r_a = np.clip(r_a, -0.999999, 0.999999)
    r_b = np.clip(r_b, -0.999999, 0.999999)
    if n_a <= 3 or n_b <= 3:
        return 1.0
    z_a, z_b = np.arctanh(r_a), np.arctanh(r_b)
    se = np.sqrt(1.0 / (n_a - 3) + 1.0 / (n_b - 3))
    if se <= 0:
        return 1.0
    z = (z_a - z_b) / se
    from scipy.stats import norm
    return float(2 * (1 - norm.cdf(abs(z))))


def _bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg 다중비교 보정. 반환값은 각 원소 순서 그대로의 adjusted p."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = n - rank + 1
        val = min(prev, pvals[idx] * n / i)
        adj[idx] = val
        prev = val
    return adj


def main() -> None:
    args = _parse_args()
    x_all, y_all, seg_idx_all, spec, names_by_seg = _load_all_scenarios(args)
    n_hi = x_all.shape[1]
    n_scen = spec.n_scenarios

    # concept 이름(시나리오 접미사 없는 raw HI 개념 이름) — seg_0 기준으로 접미사만 제거
    seg0_names = names_by_seg[0]
    seg0_suffix = f"_{spec.scenario_names[0]}"
    concepts = [n[:-len(seg0_suffix)] if n.endswith(seg0_suffix) else n for n in seg0_names]

    # 시나리오별 (r, n) 계산
    r_by_scen = np.zeros((n_hi, n_scen))
    n_by_scen = np.zeros(n_scen, dtype=int)
    for s in range(n_scen):
        sel = seg_idx_all == s
        n_by_scen[s] = int(sel.sum())
        x_s, y_s = x_all[sel], y_all[sel]
        for i in range(n_hi):
            c = np.corrcoef(x_s[:, i], y_s)[0, 1]
            r_by_scen[i, s] = 0.0 if np.isnan(c) else c

    pairs = list(itertools.combinations(range(n_scen), 2))
    min_p_raw = np.ones(n_hi)
    worst_pair = [(-1, -1)] * n_hi
    for i in range(n_hi):
        for a, b in pairs:
            p = _fisher_z_test(r_by_scen[i, a], n_by_scen[a], r_by_scen[i, b], n_by_scen[b])
            if p < min_p_raw[i]:
                min_p_raw[i] = p
                worst_pair[i] = (a, b)

    p_adj = _bh_adjust(min_p_raw.tolist())

    result = {}
    for i in range(n_hi):
        a, b = worst_pair[i]
        std_r = float(np.std(r_by_scen[i]))
        result[concepts[i]] = {
            "r_by_scenario": {spec.scenario_names[s]: float(r_by_scen[i, s]) for s in range(n_scen)},
            "std_r_across_scenarios": std_r,
            "min_p_raw": float(min_p_raw[i]),
            "p_adj_bh": float(p_adj[i]),
            "worst_pair": [spec.scenario_names[a], spec.scenario_names[b]] if a >= 0 else None,
            "worst_pair_delta_r": float(abs(r_by_scen[i, a] - r_by_scen[i, b])) if a >= 0 else 0.0,
            "p_significant": bool(p_adj[i] < args.alpha),
            "effect_size_meaningful": bool(std_r >= args.min_effect_size),
            # v4가 실제로 쓰는 최종 판정: p-value(대표본이라 신뢰 낮음)만으론 안 정하고
            # 효과크기(std_r)를 주 기준으로 삼는다 — p_significant는 참고용 부가 정보.
            "significant": bool(std_r >= args.min_effect_size),
        }

    n_sig = sum(1 for v in result.values() if v["significant"])
    n_p_sig = sum(1 for v in result.values() if v["p_significant"])
    payload = {
        "tag": args.tag, "alpha": args.alpha, "min_effect_size": args.min_effect_size,
        "n_scenarios": n_scen, "scenario_names": spec.scenario_names, "n_hi": n_hi,
        "n_significant": n_sig, "n_p_significant_raw": n_p_sig,
        "per_hi": result,
        "note": "significant=True(=std_r_across_scenarios >= min_effect_size)인 HI만 "
                "v4에서 기존 scen_gates에 남기고, 나머지는 shared_gate로 통합한다. "
                "p_significant(BH 보정 p-value 기준)는 표본이 매우 커서(수십만~백만 행) "
                "참고용일 뿐 판정에 안 쓴다 — n_p_significant_raw와 n_significant의 "
                "차이가 크면(실측: 66/66 vs 소수) 그 증거.",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"hi_scenario_interaction_{args.tag}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[interaction] HI {n_hi}개 중 p-value 기준 유의: {n_p_sig}개(참고용, 대표본이라 "
          f"과다검출 위험) / 효과크기(std_r>={args.min_effect_size}) 기준 유의: {n_sig}개(실제 판정)")
    print(f"[interaction] 저장: {out_path}")
    top5 = sorted(result.items(), key=lambda kv: -kv[1]["std_r_across_scenarios"])[:5]
    print("\n[interaction] 시나리오 간 편차(std_r) 가장 큰 HI 5개:")
    for name, v in top5:
        print(f"  {name:30s} std_r={v['std_r_across_scenarios']:.4f} "
              f"worst_pair={v['worst_pair']}(Δr={v['worst_pair_delta_r']:.3f}) "
              f"r={v['r_by_scenario']}")


if __name__ == "__main__":
    main()
