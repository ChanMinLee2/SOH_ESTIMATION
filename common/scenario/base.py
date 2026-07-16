"""
common/scenario/base.py — Step 4/5 공유 계약 타입.

ScenarioSpec: Step 4가 생성하고 Step 5가 소비하는 JSON 아티팩트.
              이 파일 하나로 모델 구조(gate 수, 분류 클래스 수, 라우팅 테이블)가
              전부 결정된다.

SegmentRecord: Segmenter → HI 추출기 사이의 중간 표현.
               슬라이스된 V/I/t/Q 배열을 담아 HI 계산 함수에 바로 넘길 수 있다.

Segmenter: 모든 축 구현체가 상속하는 ABC.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# SegmentRecord — 사이클 내 한 세그먼트 슬라이스
# ---------------------------------------------------------------------------

@dataclass
class SegmentRecord:
    cell_id: str
    cycle: int
    seg_local_id: int   # 사이클 내 세그먼트 순번 (0-indexed)
    scenario_id: int    # 0..n_scenarios-1  → Stage B gate 라우팅 타깃
    latent_class: int   # 0..n_classes-1    → Stage A 분류 타깃
    direction: int      # +1=충전  /  -1=방전
    v: np.ndarray       # voltage slice
    i: np.ndarray       # |current| slice
    dt: np.ndarray      # time delta slice
    q: np.ndarray       # cumulative capacity slice (Ah)
    meta: dict = field(default_factory=dict)  # {q_frac_lo, q_frac_hi, seg_name, ...}


# ---------------------------------------------------------------------------
# ScenarioSpec — Step 4 출력 / Step 5 입력 계약
# ---------------------------------------------------------------------------

@dataclass
class ScenarioSpec:
    axis: str                        # "qfrac" | "protocol" | "vwindow" | "rcs" | "cluster"
    n_scenarios: int                 # Stage B gate 개수
    scenario_names: list[str]        # 길이 = n_scenarios
    n_classes: int                   # Stage A 분류 클래스 수
    class_names: list[str]           # 길이 = n_classes
    routing: list[list[int]]         # routing[dir_idx][latent_class] = scenario_id
                                     #   dir_idx: 0=충전, 1=방전
    classifier_default: str          # "mlp_probe" | "rule" | "centroid" | "none"
    params: dict = field(default_factory=dict)  # 축별 재현 파라미터

    # ------------------------------------------------------------------
    # 편의 프로퍼티
    # ------------------------------------------------------------------

    @property
    def charge_scenario_ids(self) -> list[int]:
        """충전 방향(dir_idx=0)에 배정된 scenario_id 목록."""
        return list(self.routing[0])

    @property
    def discharge_scenario_ids(self) -> list[int]:
        """방전 방향(dir_idx=1)에 배정된 scenario_id 목록."""
        return list(self.routing[1])

    def scenario_to_dir_class(self, scenario_id: int) -> tuple[int, int]:
        """scenario_id → (dir_idx, latent_class). 없으면 ValueError."""
        for dir_idx, row in enumerate(self.routing):
            for cls, sid in enumerate(row):
                if sid == scenario_id:
                    return dir_idx, cls
        raise ValueError(f"scenario_id={scenario_id} not found in routing {self.routing}")

    def direction_class_to_scenario(self, dir_idx: int, latent_class: int) -> int:
        """(dir_idx, latent_class) → scenario_id."""
        return self.routing[dir_idx][latent_class]

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "axis":               self.axis,
            "n_scenarios":        self.n_scenarios,
            "scenario_names":     self.scenario_names,
            "n_classes":          self.n_classes,
            "class_names":        self.class_names,
            "routing":            self.routing,
            "classifier_default": self.classifier_default,
            "params":             self.params,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "ScenarioSpec":
        data = json.loads(Path(path).read_text())
        return cls(
            axis=data["axis"],
            n_scenarios=data["n_scenarios"],
            scenario_names=data["scenario_names"],
            n_classes=data["n_classes"],
            class_names=data["class_names"],
            routing=data["routing"],
            classifier_default=data["classifier_default"],
            params=data.get("params", {}),
        )

    def hash(self) -> str:
        """구조적 동일성 비교용 16자 SHA-256 prefix."""
        canon = json.dumps({
            "axis":           self.axis,
            "n_scenarios":    self.n_scenarios,
            "scenario_names": self.scenario_names,
            "n_classes":      self.n_classes,
            "routing":        self.routing,
        }, sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Segmenter ABC
# ---------------------------------------------------------------------------

class Segmenter(ABC):
    name: str = "base"

    def fit(self, train_cells) -> None:
        """오프라인 보정이 필요한 축(cluster, vwindow auto_range)만 구현.
        무상태 축(qfrac, protocol)은 no-op."""

    @abstractmethod
    def iter_segments(
        self,
        cell_id: str,
        cycle: int,
        dis_v: np.ndarray,
        dis_i: np.ndarray,
        dis_dt: np.ndarray,
        dis_q: np.ndarray,
        chg_v: np.ndarray | None = None,
        chg_i: np.ndarray | None = None,
        chg_dt: np.ndarray | None = None,
        chg_q: np.ndarray | None = None,
    ) -> Iterator[SegmentRecord]:
        """사이클 1개의 dis/chg 배열을 받아 SegmentRecord를 yield."""

    @abstractmethod
    def get_spec(self) -> ScenarioSpec:
        """이 Segmenter가 생성하는 ScenarioSpec 반환."""

    def save_artifacts(self, out_dir: Path) -> None:
        """spec JSON + 축별 아티팩트(centroids 등)를 out_dir에 저장."""
        self.get_spec().save(Path(out_dir) / "scenario_spec.json")
