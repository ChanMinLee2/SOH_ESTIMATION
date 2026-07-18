"""
SCR Trainer.

Implements train / validate / early stopping loop.
Saves best checkpoint and returns final selected HI indices.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.scr_loss import SCRLoss
from utils.metrics import rmse, mae, r2
from utils.tqdm_utils import trange, tqdm, write as tqdm_write


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


class SCRTrainer:

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        output_dir: Path,
        device: torch.device,
        normalizer=None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = device
        self.normalizer = normalizer

        tr_cfg   = cfg["training"]
        loss_cfg = cfg["loss"]

        self.loss_fn = SCRLoss(
            lambda_scen=loss_cfg["lambda_scen"],
            lambda_l0=loss_cfg["lambda_l0"],
        ).to(device)

        self.l0_scheduler = L0LambdaScheduler(
            target=loss_cfg["lambda_l0"],
            loss_cfg=loss_cfg,
            total_epochs=tr_cfg["epochs"],
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tr_cfg["lr"],
            weight_decay=tr_cfg["weight_decay"],
        )

        self.epochs = tr_cfg["epochs"]
        self.batch_size = tr_cfg["batch_size"]
        self.warmup_epochs = tr_cfg.get("warmup_epochs", 5)
        self.patience = tr_cfg.get("early_stop_patience", 30)
        self.grad_clip = tr_cfg.get("grad_clip", 1.0)
        self.log_interval = tr_cfg.get("log_interval", 10)

        self.scheduler = self._build_scheduler(tr_cfg)

        self.best_val_rmse = float("inf")
        self.best_epoch = 0
        self._no_improve = 0

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    def _build_scheduler(self, tr_cfg: dict):
        sched_name = tr_cfg.get("scheduler", "cosine")
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - self.warmup_epochs,
                eta_min=1e-6,
            )
        if sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        return None

    def _lr_warmup(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            lr = self.cfg["training"]["lr"] * (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    # ------------------------------------------------------------------
    # Train / val epochs
    # ------------------------------------------------------------------
    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = mse_sum = ce_sum = l0_sum = 0.0
        n = 0
        all_preds, all_targets = [], []
        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch, self.model)
            losses["total"].backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bs = batch["target"].size(0)
            total_loss += losses["total"].item() * bs
            mse_sum    += losses["mse"].item() * bs
            ce_sum     += losses["ce"].item() * bs
            l0_sum     += losses["l0"].item() * bs
            n += bs
            all_preds.append(outputs["cap_pred"].detach().cpu())
            all_targets.append(batch["target"].cpu())

        p = torch.cat(all_preds).numpy()
        t = torch.cat(all_targets).numpy()
        return {
            "loss": total_loss / n,
            "mse":  mse_sum / n,
            "ce":   ce_sum / n,
            "l0":   l0_sum / n,
            "rmse": float(rmse(t, p)),
            "r2":   float(r2(t, p)),
        }

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        preds, targets = [], []
        mse_sum = ce_sum = 0.0
        n = 0
        for batch in loader:
            batch = self._to_device(batch)
            outputs = self.model(batch)
            preds.append(outputs["cap_pred"].cpu())
            targets.append(batch["target"].cpu())
            bs = batch["target"].size(0)
            mse_sum += torch.nn.functional.mse_loss(
                outputs["cap_pred"], batch["target"]
            ).item() * bs
            ce_sum += torch.nn.functional.cross_entropy(
                outputs["level_logits"], batch["level"]
            ).item() * bs
            n += bs

        p = torch.cat(preds).numpy()
        t = torch.cat(targets).numpy()
        mse_v = mse_sum / n
        ce_v  = ce_sum  / n
        return {
            "loss": mse_v + self.loss_fn.lambda_scen * ce_v,
            "mse":  mse_v,
            "rmse": float(rmse(t, p)),
            "mae":  float(mae(t, p)),
            "r2":   float(r2(t, p)),
            "ce":   ce_v,
        }

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> nn.Module:
        ckpt_path = self.output_dir / "checkpoints" / "best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        log_path = self.output_dir / "logs" / "train_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _init_log_csv(log_path)

        # Optional: 본 학습 전 의도적 과적합 테스트
        tr_cfg = self.cfg["training"]
        if tr_cfg.get("run_overfit_test", False):
            n_ov  = tr_cfg.get("overfit_test_samples", 1024)
            ep_ov = tr_cfg.get("overfit_test_epochs",  300)
            tqdm_write(f"[overfit_test] 시작: {n_ov}샘플 × {ep_ov}에폭 (규제 없음)")
            self.run_overfit_test(train_loader, n_samples=n_ov, max_epochs=ep_ov)

        for epoch in trange(self.epochs, desc="SCR training"):
            self._lr_warmup(epoch)

            # lambda_l0 스케줄 업데이트
            eff_l0 = self.l0_scheduler.get(epoch)
            self.loss_fn.lambda_l0 = eff_l0

            t0 = time.time()
            tr = self._train_epoch(train_loader)
            vl = self._val_epoch(val_loader)

            if epoch >= self.warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            improved = vl["rmse"] < self.best_val_rmse
            if improved:
                self.best_val_rmse = vl["rmse"]
                self.best_epoch = epoch
                self._no_improve = 0
                _save_model(self.model, ckpt_path, self.cfg, self.normalizer)
            else:
                self._no_improve += 1

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            _append_log_csv(log_path, epoch + 1, tr, vl, lr, eff_l0, elapsed)

            if (epoch + 1) % self.log_interval == 0 or improved:
                tqdm_write(
                    f"epoch {epoch+1:4d} | "
                    f"tr_loss={tr['loss']:.4f} mse={tr['mse']:.4f} r2={tr['r2']:.4f} "
                    f"ce={tr['ce']:.4f} l0={tr['l0']:.4f} λ_l0={eff_l0:.4f} | "
                    f"val_loss={vl['loss']:.4f} rmse={vl['rmse']:.4f} r2={vl['r2']:.4f} | "
                    f"lr={lr:.2e} t={elapsed:.1f}s"
                    + (" *" if improved else "")
                )

            if self._no_improve >= self.patience:
                tqdm_write(f"Early stopping at epoch {epoch+1} (best={self.best_epoch+1})")
                break

        # Restore best
        _load_model(self.model, ckpt_path)
        tqdm_write(f"Best val RMSE: {self.best_val_rmse:.4f} at epoch {self.best_epoch+1}")
        return self.model

    # ------------------------------------------------------------------
    # Laplace UQ fitting
    # ------------------------------------------------------------------

    def fit_laplace(
        self,
        train_loader,
        val_loader,
        uq_cfg: dict,
    ):
        """
        학습 완료된 모델에 Last-Layer Laplace Approximation을 적합한다.

        Args:
            train_loader : 학습 DataLoader (H 추정용)
            val_loader   : 검증 DataLoader (prior_precision 최적화용)
            uq_cfg       : yaml uq: 섹션 딕셔너리

        Returns:
            LaplaceUQ 인스턴스 (fitted)
        """
        from utils.uncertainty import LaplaceUQ

        prior_precision = uq_cfg.get("prior_precision", 1.0)
        noise_std       = uq_cfg.get("noise_std", None)       # None → 잔차 자동 추정
        optimize_prior  = uq_cfg.get("optimize_prior", True)

        uq = LaplaceUQ(self.model, self.device, prior_precision=prior_precision)
        uq.fit(train_loader, noise_std=noise_std)
        if optimize_prior:
            uq.optimize_prior_precision(val_loader)
        return uq

    # ------------------------------------------------------------------
    # Overfit test utilities
    # ------------------------------------------------------------------

    def _collect_small_batch(self, loader: DataLoader, n_samples: int) -> dict:
        """train_loader에서 처음 n_samples개 샘플을 하나의 배치로 반환."""
        all_tensors: dict[str, list] = {}
        n_collected = 0
        for batch in loader:
            for k, v in batch.items():
                all_tensors.setdefault(k, []).append(v)
            n_collected += next(iter(batch.values())).size(0)
            if n_collected >= n_samples:
                break
        result = {}
        for k, vs in all_tensors.items():
            cat = torch.cat(vs)
            result[k] = cat[:n_samples]
        return result

    def run_overfit_test(
        self,
        train_loader: DataLoader,
        n_samples: int = 1024,
        max_epochs: int = 300,
        lr: float = 1e-3,
    ) -> dict:
        """
        의도적 과적합 테스트: 소량 샘플에 대해 dropout=0, weight_decay=0으로 학습.
        모델 용량이 충분하면 R²→1에 수렴; 정체되면 용량 부족 또는 태스크 노이즈 바닥.
        결과: logs/overfit_test.csv
        반환: {"max_r2": float, "epoch_max_r2": int, "final_r2": float}
        """
        import copy

        tiny = self._collect_small_batch(train_loader, n_samples)
        tiny_dev = {k: v.to(self.device) for k, v in tiny.items()}

        test_model = copy.deepcopy(self.model).to(self.device)
        for m in test_model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

        opt = torch.optim.AdamW(test_model.parameters(), lr=lr, weight_decay=0.0)

        log_path = self.output_dir / "logs" / "overfit_test.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "loss", "r2"])

        max_r2 = float("-inf")
        epoch_max_r2 = 0
        r2_val = float("nan")
        loss_val = float("nan")

        test_model.train()
        for epoch in range(max_epochs):
            opt.zero_grad()
            outputs = test_model(tiny_dev)
            loss = torch.nn.functional.mse_loss(outputs["cap_pred"], tiny_dev["target"])
            loss.backward()
            opt.step()

            loss_val = loss.item()
            p_np = outputs["cap_pred"].detach().cpu().numpy()
            t_np = tiny_dev["target"].cpu().numpy()
            r2_val = float(r2(t_np, p_np))

            if r2_val > max_r2:
                max_r2 = r2_val
                epoch_max_r2 = epoch + 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch + 1, f"{loss_val:.6f}", f"{r2_val:.6f}"])

        tqdm_write(
            f"[overfit_test] {n_samples}샘플 × {max_epochs}에폭 완료 | "
            f"max R²={max_r2:.4f} @ epoch {epoch_max_r2} | final R²={r2_val:.4f}"
        )
        tqdm_write(f"[overfit_test] → {log_path}")
        return {"max_r2": max_r2, "epoch_max_r2": epoch_max_r2, "final_r2": r2_val}


def _save_model(model: nn.Module, path: Path, cfg=None, normalizer=None) -> None:
    ckpt = {"model_state": model.state_dict()}
    if cfg is not None:
        ckpt["cfg"] = cfg
    if normalizer is not None:
        ckpt["norm_mean"]           = normalizer.mean_
        ckpt["norm_std"]            = normalizer.std_
        ckpt["norm_cap_init_mean"]  = normalizer.cap_init_mean_
        ckpt["norm_cap_init_std"]   = normalizer.cap_init_std_
    torch.save(ckpt, path)


def _load_model(model: nn.Module, path: Path) -> None:
    if path.exists():
        data = torch.load(path, map_location="cpu")
        state = data["model_state"] if isinstance(data, dict) and "model_state" in data else data
        model.load_state_dict(state)


_LOG_COLS = ["epoch", "tr_loss", "tr_mse", "tr_ce", "tr_l0", "lambda_l0", "tr_rmse", "tr_r2",
             "val_loss", "val_mse", "val_rmse", "val_mae", "val_r2", "lr", "elapsed_s"]


def _init_log_csv(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_LOG_COLS)


def _append_log_csv(path: Path, epoch: int, tr: dict, vl: dict,
                    lr: float, eff_l0: float, elapsed: float) -> None:
    row = [
        epoch,
        f"{tr['loss']:.6f}", f"{tr['mse']:.6f}", f"{tr['ce']:.6f}", f"{tr['l0']:.6f}",
        f"{eff_l0:.6f}",
        f"{tr.get('rmse', float('nan')):.6f}", f"{tr.get('r2', float('nan')):.6f}",
        f"{vl['loss']:.6f}", f"{vl['mse']:.6f}", f"{vl['rmse']:.6f}",
        f"{vl['mae']:.6f}",  f"{vl['r2']:.6f}",
        f"{lr:.6e}", f"{elapsed:.2f}",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
