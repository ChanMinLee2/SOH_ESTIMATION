"""I/O utilities: saving/loading checkpoints, configs, and results."""

from __future__ import annotations
import json
import pathlib
import pickle
from typing import Any, Dict
import yaml
import torch


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """섹션 단위가 아니라 키 단위 재귀 병합. override가 항상 이긴다 —
    같은 키가 양쪽에 dict로 있으면 재귀 병합, 아니면 override 값으로 덮어씀."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    config_path: str | pathlib.Path,
    fixed_path: str | pathlib.Path | None = None,
) -> Dict[str, Any]:
    """실험 yaml을 로드한다.

    `fixed_path`를 명시하지 않으면 `config_path`와 같은 폴더의 `fixed.yaml`을 자동으로
    찾아 베이스로 병합한다(있을 때만 — 없으면 기존처럼 `config_path` 단독 결과를 그대로
    반환). `config_path`가 이미 모든 키를 갖춘 완결형 yaml(기존 `scr.yaml`/`exp_*.yaml`
    등)이면 병합해도 결과가 동일하다 — 겹치는 키는 항상 `config_path` 쪽이 이긴다.
    `main.yaml`처럼 실험별로 자주 바꾸는 필드만 담은 slim yaml은 이 병합으로 나머지
    필드를 `fixed.yaml`에서 채운다(docs/params.md "config.yaml 구조 개편안" 참고).
    """
    config_path = pathlib.Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if fixed_path is None:
        fixed_path = config_path.parent / "fixed.yaml"
    fixed_path = pathlib.Path(fixed_path)

    if fixed_path.exists() and fixed_path.resolve() != config_path.resolve():
        with open(fixed_path, "r", encoding="utf-8") as f:
            fixed_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(fixed_cfg, cfg)

    return cfg


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
