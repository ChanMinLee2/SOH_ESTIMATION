"""Trainer: end-to-end training loop for DFRModel.

Features:
  - Cosine LR schedule with linear warm-up
  - Router temperature annealing (Gumbel temperature decay per epoch)
  - Early stopping on validation loss
  - tqdm progress bars: outer epoch bar + inner batch bar
  - Best-checkpoint saving to _5_data_model/checkpoints/
  - Per-epoch logging to _5_data_model/logs/
"""

from __future__ import annotations
import csv
import logging
import math
import pathlib
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.tqdm_utils import tqdm

from models.dfr_model import DFRModel
from training.loss import DFRLoss
from utils.io_utils import save_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _anneal_temperature(start: float, end: float, decay: float, epoch: int) -> float:
    return max(end, start * (decay ** epoch))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training controller for DFRModel.

    Args:
        model:      DFRModel instance.
        criterion:  DFRLoss instance.
        optimizer:  PyTorch optimizer.
        cfg:        Full config dict.
        output_dir: Base output directory (_5_data_model/).
        device:     torch device string.
    """

    def __init__(
        self,
        model: DFRModel,
        criterion: DFRLoss,
        optimizer: torch.optim.Optimizer,
        cfg: Dict,
        output_dir: pathlib.Path,
        device: str = "cpu",
    ) -> None:
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.cfg       = cfg
        self.device    = device

        tcfg = cfg["training"]
        rcfg = cfg["router"]

        self.epochs        = tcfg["epochs"]
        self.patience      = tcfg["early_stop_patience"]
        self.base_lr       = tcfg["lr"]
        self.warmup_epochs = tcfg.get("warmup_epochs", 5)
        self.grad_clip     = tcfg.get("grad_clip", 1.0)
        self.log_interval  = tcfg.get("log_interval", 10)

        self.temp_start = rcfg["gumbel_temp_start"]
        self.temp_end   = rcfg["gumbel_temp_end"]
        self.temp_decay = rcfg["temp_decay"]

        self.ckpt_dir = output_dir / "checkpoints"
        self.log_dir  = output_dir / "logs"
        self.fig_dir  = output_dir / "figures"
        for d in [self.ckpt_dir, self.log_dir, self.fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch    = 0
        self.no_improve    = 0

        self._train_losses:  List[float] = []
        self._val_losses:    List[float] = []
        self._train_rmse:    List[float] = []
        self._val_rmse:      List[float] = []
        self._temperatures:  List[float] = []

        self._log_file = self.log_dir / "train_log.csv"
        self._init_log_file()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _init_log_file(self) -> None:
        with open(self._log_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "lr", "temperature",
                "train_loss", "train_mse", "train_sparse", "train_rmse",
                "val_loss", "val_rmse",
                "mean_active_groups", "elapsed_s",
            ])

    def _log_epoch(self, row: List) -> None:
        with open(self._log_file, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ------------------------------------------------------------------
    # Per-epoch passes
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        temperature: float,
        epoch_idx: int,
    ) -> Tuple[float, float, float, float]:
        self.model.train()
        total_loss = total_mse = total_sparse = 0.0
        all_pred, all_true = [], []

        pbar = tqdm(
            loader,
            desc=f"  train ep{epoch_idx + 1:>4d}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in pbar:
            x_global = batch["x_global"].to(self.device)
            gf = [t.to(self.device) for t in batch["group_features"]]
            nm = [t.to(self.device) for t in batch["nan_masks"]]
            target = batch["target"].to(self.device)

            batch_dev = {"x_global": x_global, "group_features": gf, "nan_masks": nm}

            self.optimizer.zero_grad()
            pred, gates = self.model(batch_dev, temperature=temperature)
            loss, breakdown = self.criterion(pred, target, gates)
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            n = len(target)
            total_loss   += breakdown["total"]  * n
            total_mse    += breakdown["mse"]    * n
            total_sparse += breakdown["sparse"] * n
            all_pred.append(pred.detach().cpu().numpy())
            all_true.append(target.cpu().numpy())

            pbar.set_postfix({"loss": f"{breakdown['total']:.4f}"})

        N = len(loader.dataset)
        rmse_val = float(np.sqrt(np.mean(
            (np.concatenate(all_pred) - np.concatenate(all_true)) ** 2
        )))
        return total_loss / N, total_mse / N, total_sparse / N, rmse_val

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        epoch_idx: int,
        split: str = "val",
    ) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        all_pred, all_true, all_gates = [], [], []

        pbar = tqdm(
            loader,
            desc=f"  {split:>5s} ep{epoch_idx + 1:>4d}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in pbar:
            x_global = batch["x_global"].to(self.device)
            gf = [t.to(self.device) for t in batch["group_features"]]
            nm = [t.to(self.device) for t in batch["nan_masks"]]
            target = batch["target"].to(self.device)

            batch_dev = {"x_global": x_global, "group_features": gf, "nan_masks": nm}
            pred, gates = self.model(batch_dev, hard=False)
            loss, breakdown = self.criterion(pred, target, gates)

            n = len(target)
            total_loss += breakdown["total"] * n
            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())
            all_gates.append((gates > 0.5).float().cpu().numpy())

            pbar.set_postfix({"loss": f"{breakdown['total']:.4f}"})

        N = len(loader.dataset)
        y_pred    = np.concatenate(all_pred)
        y_true    = np.concatenate(all_true)
        gates_arr = np.concatenate(all_gates, axis=0)

        rmse_val    = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mean_active = float(gates_arr.sum(axis=-1).mean())
        return total_loss / N, rmse_val, mean_active

    # ------------------------------------------------------------------
    # Main train loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """Run the full training loop with tqdm epoch progress bar.

        Returns summary dict with best epoch, best val_loss, history.
        """
        logger.info("Starting training — %d epochs, device=%s", self.epochs, self.device)
        self.model.to(self.device)

        epoch_bar = tqdm(
            range(self.epochs),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            t0 = time.time()

            temperature = _anneal_temperature(
                self.temp_start, self.temp_end, self.temp_decay, epoch
            )
            lr = _cosine_lr(
                self.optimizer, epoch, self.epochs, self.warmup_epochs, self.base_lr,
            )

            tr_loss, tr_mse, tr_sparse, tr_rmse = self._train_epoch(
                train_loader, temperature, epoch
            )
            val_loss, val_rmse, mean_active = self._eval_epoch(
                val_loader, epoch
            )

            elapsed = time.time() - t0
            self._train_losses.append(tr_loss)
            self._val_losses.append(val_loss)
            self._train_rmse.append(tr_rmse)
            self._val_rmse.append(val_rmse)
            self._temperatures.append(temperature)

            # Update epoch bar postfix (always visible)
            is_best = val_loss < self.best_val_loss
            epoch_bar.set_postfix({
                "tr_rmse":  f"{tr_rmse:.4f}",
                "val_rmse": f"{val_rmse:.4f}",
                "active":   f"{mean_active:.1f}/{self.model.hi_info.n_groups}",
                "T":        f"{temperature:.2f}",
                "best":     "✓" if is_best else "",
            })

            self._log_epoch([
                epoch + 1, f"{lr:.2e}", f"{temperature:.3f}",
                f"{tr_loss:.6f}", f"{tr_mse:.6f}", f"{tr_sparse:.6f}", f"{tr_rmse:.6f}",
                f"{val_loss:.6f}", f"{val_rmse:.6f}",
                f"{mean_active:.1f}", f"{elapsed:.1f}",
            ])

            if (epoch + 1) % self.log_interval == 0:
                logger.info(
                    "Ep%3d | lr=%.2e T=%.2f | "
                    "train loss=%.5f rmse=%.5f | val loss=%.5f rmse=%.5f | "
                    "active=%.1f | %.1fs",
                    epoch + 1, lr, temperature,
                    tr_loss, tr_rmse, val_loss, val_rmse,
                    mean_active, elapsed,
                )

            # Best checkpoint
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch + 1
                self.no_improve    = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, val_loss,
                    extra={"temperature": temperature},
                    save_path=self.ckpt_dir / "best.pth",
                )
            else:
                self.no_improve += 1

            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, val_loss,
                    extra={"temperature": temperature},
                    save_path=self.ckpt_dir / f"epoch_{epoch+1:04d}.pth",
                )

            if self.no_improve >= self.patience:
                epoch_bar.set_description("Training (early stop)")
                epoch_bar.close()
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1, self.patience,
                )
                break

        logger.info(
            "Training complete. Best val_loss=%.6f at epoch %d.",
            self.best_val_loss, self.best_epoch,
        )
        self._save_figures()

        return {
            "best_epoch":    self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses":  self._train_losses,
            "val_losses":    self._val_losses,
            "train_rmse":    self._train_rmse,
            "val_rmse":      self._val_rmse,
            "temperatures":  self._temperatures,
        }

    def _save_figures(self) -> None:
        try:
            from utils.visualization import plot_training_curves, plot_temperature_schedule
            plot_training_curves(
                self._train_losses, self._val_losses,
                self._train_rmse,   self._val_rmse,
                self.fig_dir / "training_curves.png",
            )
            plot_temperature_schedule(
                self._temperatures,
                self.fig_dir / "temperature_schedule.png",
            )
        except Exception as e:
            logger.warning("Could not save training figures: %s", e)
