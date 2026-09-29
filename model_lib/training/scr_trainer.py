"""
SCR Trainer.

2026-09-25: `SCRTrainer` 클래스(구식 `fit()`/`fit_laplace()`/`run_overfit_test()` 루프)
삭제 — 유일한 호출자였던 `model_lib/legacy/train_scr.py`(Stage0)가 이미 삭제됐고,
`8_train/train.py`는 처음부터 `SCRTrainer`를 인스턴스화하지 않고 자기 자신의 학습
루프를 직접 구현한다(이 파일에서 재사용하는 건 아래 `L0LambdaScheduler`뿐). 같이
쓰이던 `fit_laplace()`(Laplace UQ 적합)의 유일한 소비자도 사라져서
`model_lib/utils/uncertainty.py`/`model_lib/evaluation/scr_evaluator.py`의
`predict_dataset_uq`/`plot_uq`도 같은 라운드에 정리했다(REFACTORING.md 참고).
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# lambda_l0 스케줄러
# ---------------------------------------------------------------------------

class L0LambdaScheduler:
    """
    Phase 1 전용 lambda_l0 스케줄러.

    schedule 옵션:
      none           : 매 에폭 target 고정 (기존 동작)
      delayed_warmup : 처음 warmup_epochs 동안 0, 이후 ramp_epochs에 걸쳐 선형 증가
      exp_ramp       : 0 → target을 제곱 커브로 증가 (total_epochs 기준)
      cyclic         : 0 ↔ target 코사인 사이클 반복 (cycle_epochs 주기)
    """

    def __init__(self, target: float, loss_cfg: dict, total_epochs: int):
        self.target       = target
        self.schedule     = loss_cfg.get("lambda_l0_schedule", "none")
        self.warmup_ep    = loss_cfg.get("lambda_l0_warmup_epochs", 50)
        self.ramp_ep      = loss_cfg.get("lambda_l0_ramp_epochs",   50)
        self.cycle_ep     = loss_cfg.get("lambda_l0_cycle_epochs",  100)
        self.total_epochs = total_epochs

    def get(self, epoch: int) -> float:
        """현재 에폭에서의 effective lambda_l0 반환."""
        t = self.target
        if self.schedule == "none" or t == 0.0:
            return t

        if self.schedule == "delayed_warmup":
            if epoch < self.warmup_ep:
                return 0.0
            progress = (epoch - self.warmup_ep) / max(self.ramp_ep, 1)
            return t * min(progress, 1.0)

        if self.schedule == "exp_ramp":
            progress = epoch / max(self.total_epochs - 1, 1)
            return t * (progress ** 2)

        if self.schedule == "cyclic":
            phase = (epoch % self.cycle_ep) / self.cycle_ep
            return t * 0.5 * (1.0 - math.cos(math.pi * phase))

        return t  # fallback
