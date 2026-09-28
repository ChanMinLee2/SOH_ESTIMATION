"""I/O utilities: loading/saving a run's resolved config as yaml."""

from __future__ import annotations
import pathlib
from typing import Any, Dict
import yaml


def load_config(config_path: str | pathlib.Path) -> Dict[str, Any]:
    """train.py가 save_config()로 저장해 둔 그 run의 config.yaml을 그대로 읽는다
    (test.py 전용 — 프리셋 yaml을 골라 쓰는 용도는 2026-09-27 폐기, parameters.py:
    P1_MODEL_CONFIG가 유일한 학습 설정 소스)."""
    config_path = pathlib.Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


