"""
SCR Trainer.

Implements train / validate / early stopping loop.
Saves best checkpoint and returns final selected HI indices.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.scr_loss import SCRLoss
from utils.metrics import rmse, mae, r2
from utils.tqdm_utils import trange, tqdm, write as tqdm_write


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

        tr_cfg = cfg["training"]
        self.loss_fn = SCRLoss(
            lambda_scen=cfg["loss"]["lambda_scen"],
            lambda_l0=cfg["loss"]["lambda_l0"],
        ).to(device)

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

        return {
            "loss": total_loss / n,
            "mse":  mse_sum / n,
            "ce":   ce_sum / n,
            "l0":   l0_sum / n,
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

        for epoch in trange(self.epochs, desc="SCR training"):
            self._lr_warmup(epoch)

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
            _append_log_csv(log_path, epoch + 1, tr, vl, lr, elapsed)

            if (epoch + 1) % self.log_interval == 0 or improved:
                tqdm_write(
                    f"epoch {epoch+1:4d} | "
                    f"tr_loss={tr['loss']:.4f} mse={tr['mse']:.4f} ce={tr['ce']:.4f} l0={tr['l0']:.4f} | "
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


def _save_model(model: nn.Module, path: Path, cfg=None, normalizer=None) -> None:
    ckpt = {"model_state": model.state_dict()}
    if cfg is not None:
        ckpt["cfg"] = cfg
    if normalizer is not None:
        ckpt["norm_mean"]        = normalizer.mean_
        ckpt["norm_std"]         = normalizer.std_
        ckpt["norm_target_mean"] = normalizer.target_mean_
        ckpt["norm_target_std"]  = normalizer.target_std_
    torch.save(ckpt, path)


def _load_model(model: nn.Module, path: Path) -> None:
    if path.exists():
        data = torch.load(path, map_location="cpu")
        state = data["model_state"] if isinstance(data, dict) and "model_state" in data else data
        model.load_state_dict(state)


_LOG_COLS = ["epoch", "tr_loss", "tr_mse", "tr_ce", "tr_l0",
             "val_loss", "val_mse", "val_rmse", "val_mae", "val_r2", "lr", "elapsed_s"]


def _init_log_csv(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_LOG_COLS)


def _append_log_csv(path: Path, epoch: int, tr: dict, vl: dict,
                    lr: float, elapsed: float) -> None:
    row = [
        epoch,
        f"{tr['loss']:.6f}", f"{tr['mse']:.6f}", f"{tr['ce']:.6f}", f"{tr['l0']:.6f}",
        f"{vl['loss']:.6f}", f"{vl['mse']:.6f}", f"{vl['rmse']:.6f}",
        f"{vl['mae']:.6f}",  f"{vl['r2']:.6f}",
        f"{lr:.6e}", f"{elapsed:.2f}",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
