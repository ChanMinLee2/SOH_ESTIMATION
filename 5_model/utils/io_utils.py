"""I/O utilities: saving/loading checkpoints, configs, and results."""

from __future__ import annotations
import json
import pathlib
import pickle
from typing import Any, Dict
import yaml
import torch


def load_config(config_path: str | pathlib.Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    extra: Dict[str, Any],
    save_path: pathlib.Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **extra,
        },
        save_path,
    )


def load_checkpoint(
    checkpoint_path: pathlib.Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def save_json(data: Dict[str, Any], save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_pickle(obj: Any, save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: pathlib.Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
