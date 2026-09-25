"""
common/scenario/ — 시나리오 축 레지스트리.

2026-09-24: q_frac_ref(v4 정식/프로덕션 축) 하나만 남기고 나머지 비-정식 축(qfrac 직접
선택/protocol/vwindow/rcs/random/random_grid/cluster/test_rs/q_frac_wide 직접 선택/
q_abs/vqslope/full_cycle)을 REGISTRY에서 뺐다 — 실사용이 q_frac_ref뿐이라 git
히스토리로 코드를 남기고 작업 트리에서는 지웠다. `qfrac.py`/`q_frac_wide.py` 파일
자체는 삭제하지 않았는데, `q_frac_wide`가 `q_frac_ref`의 부모 클래스이고(상속 관계라
파일이 없으면 q_frac_ref 자체가 깨짐), `qfrac.py`의 `QFracSegmenter`는
`model_lib/datasets/segment_dataset.py`/`model_lib/models/scr_model.py`가 기본
ScenarioSpec fallback으로 내부적으로 계속 쓰기 때문이다(더는 --seg-axis로 직접
선택은 안 되지만, 코드 의존성은 남아있음).

사용:
  from common.scenario import get_segmenter
  seg = get_segmenter("q_frac_ref", cfg={"q_frac_ref": {"n1": 0.35, "n2": 0.20}})
"""

from .base import ScenarioSpec, SegmentRecord, Segmenter
from .q_frac_ref import QFracRefSegmenter

REGISTRY: dict[str, type] = {
    "q_frac_ref": QFracRefSegmenter,
}


def get_segmenter(name: str, cfg: dict | None = None) -> Segmenter:
    """
    이름으로 Segmenter 인스턴스를 생성해 반환.

    cfg: scr.yaml의 scenario 섹션 전체를 넘기면 해당 축의 kwargs만 추출.
    예: cfg={"q_frac_ref": {"n1": 0.35, "n2": 0.20}}
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown scenario axis '{name}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    cls = REGISTRY[name]
    axis_kwargs = (cfg or {}).get(name, {})
    return cls(**axis_kwargs)


__all__ = [
    "ScenarioSpec",
    "SegmentRecord",
    "Segmenter",
    "QFracRefSegmenter",
    "REGISTRY",
    "get_segmenter",
]
