"""
ShrinkageHardConcreteGate 체크포인트에서 shared_log_alpha / delta_log_alpha[s]를
직접 읽어서(test_phase1_checkpoint.py의 모델 재구성 없이, state_dict 텐서만 로드)
delta의 퍼짐(std)이 shared 대비 얼마나 큰지, 그리고 HardConcrete keep-prob으로
환산했을 때 시나리오별 실제 게이트 확률이 shared 단독 추정과 얼마나 벌어지는지 계산.

Run: "C:/Users/ksshin/.conda/envs/LFP_SOH_ESTIMATION/python.exe" 4_hi_analysis/shrinkage_delta_inspect.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "5_model" / "experiments" / "phase1_lab" / "results" / "p1v2_runs"

RUNS = {
    0.01: "0913_1015_p1v2_p1v4_rawonly_shrink0p01_seed42",
    0.05: "0913_1257_p1v2_p1v4_rawonly_shrink0p05_seed42",
    0.2:  "0913_1516_p1v2_p1v4_rawonly_shrink0p2_seed42",
    1.0:  "0913_1724_p1v2_p1v4_rawonly_shrink1p0_seed42",
}

# hard_concrete.py의 BETA/GAMMA/ZETA (학습 종료 시점 beta_min=0.1 사용 — summary.json 기준)
BETA, GAMMA, ZETA = 0.1, -0.1, 1.1


def keep_prob(log_alpha: np.ndarray) -> np.ndarray:
    """P(gate active) = P(stretched-sigmoid > 0), HardConcrete 논문 식 (베타=temperature)."""
    shift = BETA * math.log(-GAMMA / ZETA)
    return 1.0 / (1.0 + np.exp(-(log_alpha - shift)))


def main():
    print(f"{'lambda_shrink':>13} | {'std(shared)':>11} | {'std(delta)':>10} | {'mean|delta|':>11} | "
          f"{'keep(shared)':>12} | {'mean keep(s+d)':>14} | {'gap(pp)':>7}")
    for lam, run_dir in sorted(RUNS.items()):
        ckpt_path = RUNS_DIR / run_dir / "checkpoints" / "best_by_saturation.pt"
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck["model_state"]
        shared = sd["scen_gates.shared_log_alpha"].numpy()       # (66,)
        delta = sd["scen_gates.delta_log_alpha"].numpy()         # (6, 66)

        kp_shared = keep_prob(shared)                             # shared만 썼을 때 예상 keep-prob, (66,)
        kp_per_scen = keep_prob(shared[None, :] + delta)          # 실제 시나리오별 keep-prob, (6,66)

        mean_kp_shared = kp_shared.mean()
        mean_kp_actual = kp_per_scen.mean()  # 6개 시나리오 평균
        gap_pp = (mean_kp_actual - mean_kp_shared) * 100

        print(f"{lam:>13} | {shared.std():>11.4f} | {delta.std():>10.4f} | {np.abs(delta).mean():>11.4f} | "
              f"{mean_kp_shared:>12.4f} | {mean_kp_actual:>14.4f} | {gap_pp:>6.2f}p")

    print("\n[해석] std(delta) vs std(shared) 비율이 클수록 shared_log_alpha 단독 비교가 부정확해짐.")
    print("       gap(pp) = shared만으로 추정한 평균 keep-prob과, 실제 6개 시나리오 keep-prob 평균의 차이")
    print("       (0에 가까우면 Jensen gap 무시 가능, 커지면 delta 분포까지 봐야 함).")


if __name__ == "__main__":
    main()
