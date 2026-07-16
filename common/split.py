"""
common/split.py — 셀 단위 train/val/test split manifest 관리.

Step 4 이전에 한 번 생성하고, Step 4 (cluster.fit) + Step 5 (build_datasets) 가
동일 manifest를 참조함으로써 클러스터 보정의 test 누출을 방지한다.

사용:
  from common.split import create_manifest, load_manifest

  # Step 4 이전에 한 번 실행
  manifest = create_manifest(
      data_dir=Path("_4_data_hi/clean"),
      datasets=["HUST", "MIT"],
      train=0.6, val=0.2, seed=42,
  )
  manifest.save(Path("_4_data_hi/split_manifest.json"))

  # Step 4/5에서 로드
  manifest = load_manifest(Path("_4_data_hi/split_manifest.json"))
  train_cells_hust = manifest.train_cells("HUST")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SplitManifest:
    seed: int
    train_ratio: float
    val_ratio: float
    # {"MIT": {"train": [...], "val": [...], "test": [...]}, "HUST": {...}}
    splits: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def train_cells(self, dataset: str) -> list[str]:
        return self.splits.get(dataset, {}).get("train", [])

    def val_cells(self, dataset: str) -> list[str]:
        return self.splits.get(dataset, {}).get("val", [])

    def test_cells(self, dataset: str) -> list[str]:
        return self.splits.get(dataset, {}).get("test", [])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "splits": self.splits,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"[split] manifest saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        data = json.loads(Path(path).read_text())
        return cls(
            seed=data["seed"],
            train_ratio=data["train_ratio"],
            val_ratio=data["val_ratio"],
            splits=data["splits"],
        )


def create_manifest(
    data_dir: Path,
    datasets: list[str],
    train: float = 0.6,
    val: float = 0.2,
    seed: int = 42,
) -> SplitManifest:
    """
    data_dir 하위의 각 dataset 폴더에서 pkl 파일 목록을 읽어 셀 단위 split을 생성.
    data_dir: _4_data_hi/clean 또는 _4_data_hi (wide pkl 위치)
    """
    rng = np.random.default_rng(seed)
    splits: dict[str, dict[str, list[str]]] = {}

    for ds in datasets:
        ds_dir = Path(data_dir) / ds
        if not ds_dir.exists():
            print(f"[split] WARNING: {ds_dir} not found, skipping {ds}")
            continue
        cells = sorted(p.stem for p in ds_dir.glob("*.pkl"))
        if not cells:
            print(f"[split] WARNING: no pkl files in {ds_dir}")
            continue
        cells_arr = np.array(cells)
        rng.shuffle(cells_arr)
        n = len(cells_arr)
        n_train = int(n * train)
        n_val = int(n * val)
        splits[ds] = {
            "train": cells_arr[:n_train].tolist(),
            "val":   cells_arr[n_train: n_train + n_val].tolist(),
            "test":  cells_arr[n_train + n_val:].tolist(),
        }
        print(f"[split] {ds}: {n} cells → train={n_train}, val={n_val}, test={n-n_train-n_val}")

    return SplitManifest(seed=seed, train_ratio=train, val_ratio=val, splits=splits)


def load_manifest(path: Path) -> SplitManifest | None:
    """path가 없으면 None 반환 (하위호환: manifest 없으면 현행 seed split 사용)."""
    p = Path(path)
    if not p.exists():
        return None
    return SplitManifest.load(p)
