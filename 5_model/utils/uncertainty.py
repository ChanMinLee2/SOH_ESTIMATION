"""
uncertainty.py — Last-Layer Laplace Approximation (LLA) for SCRModel.

외부 의존성 없이 PyTorch + scipy만으로 구현.
임의의 cap_head 타입에 적용 가능 (MLP, ResNet-Tab, Transformer, iTransformer, FTTransformer).

수식:
  φ(x)  : cap_head 최종 레이어 직전 피처 (d,) — 바이어스 확장 후 (d+1,)
  H     : (d+1, d+1)  = (1/σ_n²) Σ φᵢφᵢᵀ + λI
  σ²(x*) = σ_n² + φ(x*)ᵀ H⁻¹ φ(x*)

Usage:
    uq = LaplaceUQ(model, device)
    uq.fit(train_loader)
    uq.optimize_prior_precision(val_loader)  # optional
    uq.save(run_dir / "checkpoints" / "laplace_uq.pt")

    uq = LaplaceUQ.load(model, path, device)
    mean, std = uq.predict(batch)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _get_last_linear(cap_head: nn.Module) -> nn.Linear:
    """cap_head에서 마지막 nn.Linear 레이어를 반환한다."""
    # ResNet-Tab / Transformer / iTransformer 모두 .output = nn.Linear
    if hasattr(cap_head, "output") and isinstance(cap_head.output, nn.Linear):
        return cap_head.output
    # MLPHead: self.net = nn.Sequential(..., nn.Linear(d, 1))
    if hasattr(cap_head, "net") and isinstance(cap_head.net, nn.Sequential):
        for layer in reversed(cap_head.net):
            if isinstance(layer, nn.Linear):
                return layer
    raise AttributeError(
        f"cap_head({type(cap_head).__name__})에서 마지막 nn.Linear를 찾을 수 없습니다."
    )


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# LaplaceUQ
# ---------------------------------------------------------------------------

class LaplaceUQ:
    """
    Last-Layer Laplace Approximation.

    Args:
        model           : 학습 완료된 SCRModel
        device          : 추론 디바이스
        prior_precision : 가중치 사전 분포 정밀도 λ (optimize_prior_precision으로 튜닝 가능)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        prior_precision: float = 1.0,
    ) -> None:
        self.model = model
        self.device = device
        self.prior_precision = prior_precision
        self._L: Optional[torch.Tensor] = None   # lower-triangular Cholesky of H (CPU)
        self._sigma_n: float = 1.0               # 노이즈 표준편차 추정값
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature collection via forward hook
    # ------------------------------------------------------------------

    def _collect_phi(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """로더 전체에서 φ_aug = [φ; 1] 피처와 target을 수집한다."""
        last_lin = _get_last_linear(self.model.cap_head)
        feats, targets = [], []

        def _hook(_, inp, __):
            feats.append(inp[0].detach().cpu())

        handle = last_lin.register_forward_hook(_hook)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch_d = _batch_to_device(batch, self.device)
                self.model(batch_d)
                targets.append(batch["target"].cpu())
        handle.remove()

        phi = torch.cat(feats, dim=0)                           # (N, d)
        phi_aug = torch.cat([phi, torch.ones(len(phi), 1)], 1)  # (N, d+1)
        t = torch.cat(targets, dim=0)                           # (N,)
        return phi_aug, t

    def _collect_phi_batch(self, batch: dict) -> torch.Tensor:
        """단일 배치에서 φ_aug 피처를 수집한다 (추론용)."""
        last_lin = _get_last_linear(self.model.cap_head)
        feats: list[torch.Tensor] = []

        def _hook(_, inp, __):
            feats.append(inp[0])

        handle = last_lin.register_forward_hook(_hook)
        with torch.no_grad():
            self.model(_batch_to_device(batch, self.device))
        handle.remove()

        phi = feats[0]                                                  # (B, d)
        ones = torch.ones(phi.shape[0], 1, device=phi.device)
        return torch.cat([phi, ones], dim=1)                            # (B, d+1)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        noise_std: Optional[float] = None,
    ) -> "LaplaceUQ":
        """
        LLA 사후 분포를 학습 데이터로 추정한다.

        noise_std=None 이면 MAP 예측값과 실제 target의 잔차 표준편차로 자동 추정.
        """
        phi_aug, t = self._collect_phi(train_loader)   # CPU

        # σ_n 추정
        if noise_std is None:
            last_lin  = _get_last_linear(self.model.cap_head)
            W = last_lin.weight.data.cpu()              # (1, d)
            b = last_lin.bias.data.cpu()                # (1,)
            phi = phi_aug[:, :-1]                       # (N, d)
            pred = (phi @ W.T).squeeze(1) + b.squeeze(0)
            noise_std = float((t - pred).std().clamp(min=1e-6))
        self._sigma_n = noise_std

        self._fit_H(phi_aug)
        self._fitted = True
        print(
            f"[LaplaceUQ] fit 완료: d_feat={phi_aug.shape[1]-1}, "
            f"σ_n={noise_std:.4f}, λ={self.prior_precision:.4f}, "
            f"N={phi_aug.shape[0]}"
        )
        return self

    def _fit_H(self, phi_aug: torch.Tensor) -> None:
        """H = (1/σ_n²) XᵀX + λI 의 Cholesky 인수분해를 저장한다."""
        d1 = phi_aug.shape[1]
        H = (phi_aug.T @ phi_aug) / (self._sigma_n ** 2) + \
            self.prior_precision * torch.eye(d1)
        try:
            self._L = torch.linalg.cholesky(H)
        except torch.linalg.LinAlgError:
            H += 1e-4 * torch.eye(d1)
            self._L = torch.linalg.cholesky(H)

    # ------------------------------------------------------------------
    # Optimize prior precision
    # ------------------------------------------------------------------

    def optimize_prior_precision(
        self,
        val_loader: DataLoader,
        grid: Optional[list[float]] = None,
    ) -> float:
        """검증 NLL 최소화로 prior_precision을 그리드 탐색한다."""
        if not self._fitted:
            raise RuntimeError("fit() 먼저 호출하세요.")
        if grid is None:
            grid = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

        phi_aug, t_val = self._collect_phi(val_loader)
        t_np = t_val.numpy()

        best_nll, best_lam = float("inf"), self.prior_precision
        for lam in grid:
            self.prior_precision = lam
            self._fit_H(phi_aug)
            mu, std = self._predict_from_phi(phi_aug)
            nll = _nll(t_np, mu.numpy(), std.numpy())
            if nll < best_nll:
                best_nll, best_lam = nll, lam

        self.prior_precision = best_lam
        self._fit_H(phi_aug)   # best_lam으로 복원
        print(f"[LaplaceUQ] 최적 prior_precision={best_lam:.4f}  val_NLL={best_nll:.4f}")
        return best_lam

    # ------------------------------------------------------------------
    # Internal predict
    # ------------------------------------------------------------------

    def _predict_from_phi(
        self, phi_aug: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_lin = _get_last_linear(self.model.cap_head)
        W = last_lin.weight.data.cpu()          # (1, d)
        b = last_lin.bias.data.cpu()            # (1,)
        phi = phi_aug[:, :-1]
        mu = (phi @ W.T).squeeze(1) + b.squeeze(0)   # (B,)

        # σ²(x*) = σ_n² + φᵀ H⁻¹ φ  via Cholesky solve
        v = torch.linalg.solve_triangular(
            self._L, phi_aug.T, upper=False
        )                                              # (d+1, B)
        pred_var = self._sigma_n ** 2 + (v ** 2).sum(dim=0)  # (B,)
        pred_std = pred_var.clamp(min=0.0).sqrt()
        return mu, pred_std

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        배치에 대해 (mean, std)를 반환한다 (정규화 SOH 공간).

        Returns:
            mean : (B,)  SCRModel cap_pred와 동일
            std  : (B,)  Laplace 사후 표준편차
        """
        if not self._fitted:
            raise RuntimeError("fit() 먼저 호출하세요.")

        out = self.model(_batch_to_device(batch, self.device))
        mean = out["cap_pred"].cpu()

        phi_aug = self._collect_phi_batch(batch).cpu()
        _, std = self._predict_from_phi(phi_aug)
        return mean, std

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if not self._fitted:
            raise RuntimeError("fit() 전에는 저장할 수 없습니다.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "L":               self._L,
                "sigma_n":         self._sigma_n,
                "prior_precision": self.prior_precision,
            },
            path,
        )
        print(f"[LaplaceUQ] saved → {path}")

    @classmethod
    def load(
        cls,
        model: nn.Module,
        path: Path,
        device: torch.device,
    ) -> "LaplaceUQ":
        data = torch.load(path, map_location="cpu")
        uq = cls(model, device, prior_precision=float(data["prior_precision"]))
        uq._L       = data["L"]
        uq._sigma_n = float(data["sigma_n"])
        uq._fitted  = True
        print(f"[LaplaceUQ] loaded ← {path}")
        return uq


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------

def calibration_metrics(
    means: np.ndarray,
    stds: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """
    UQ 캘리브레이션 지표를 계산한다.

    Returns dict with:
        picp_68/90/95 : 각 신뢰 수준의 커버리지
        mpiw_90       : 90% 구간 평균 폭
        nll           : 평균 가우시안 NLL
        mean_std      : 평균 σ
    """
    from scipy.stats import norm as _norm

    results: dict = {}
    for a in (0.68, 0.90, 0.95):
        z = float(_norm.ppf((1 + a) / 2))
        lo = means - z * stds
        hi = means + z * stds
        picp = float(((targets >= lo) & (targets <= hi)).mean())
        pct  = int(a * 100)
        results[f"picp_{pct}"] = round(picp, 4)
        if a == 0.90:
            results["mpiw_90"] = round(float((2 * z * stds).mean()), 6)

    results["nll"]      = round(float(_nll(targets, means, stds)), 4)
    results["mean_std"] = round(float(stds.mean()), 6)
    return results


def _nll(
    targets: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> float:
    var = np.maximum(stds ** 2, 1e-8)
    return float(0.5 * np.mean(np.log(2 * np.pi * var) + (targets - means) ** 2 / var))


def plot_calibration(
    means: np.ndarray,
    stds: np.ndarray,
    targets: np.ndarray,
    path: Path,
) -> None:
    """캘리브레이션 곡선 (expected vs actual coverage) + σ 분포 히스토그램을 저장한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[LaplaceUQ] matplotlib 없음 — 캘리브레이션 플롯 생략")
        return

    from scipy.stats import norm as _norm

    alpha_levels = np.linspace(0.05, 0.99, 40)
    actual_cov = []
    for a in alpha_levels:
        z = float(_norm.ppf((1 + a) / 2))
        lo = means - z * stds
        hi = means + z * stds
        actual_cov.append(float(((targets >= lo) & (targets <= hi)).mean()))
    actual_cov = np.array(actual_cov)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(alpha_levels, actual_cov, "b-o", ms=3, lw=1.5, label="LLA")
    ax.fill_between(alpha_levels, alpha_levels, actual_cov, alpha=0.15, color="blue")
    area = float(np.trapz(np.abs(actual_cov - alpha_levels), alpha_levels))
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Actual coverage")
    ax.set_title(f"Calibration curve  (ECE={area:.3f})")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax2 = axes[1]
    ax2.hist(stds, bins=50, color="steelblue", alpha=0.75, edgecolor="white")
    ax2.axvline(stds.mean(), color="red", lw=1.5, linestyle="--", label=f"mean={stds.mean():.4f}")
    ax2.set_xlabel("Predicted std σ")
    ax2.set_ylabel("Count")
    ax2.set_title("Uncertainty distribution")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LaplaceUQ] calibration plot saved → {path}")
