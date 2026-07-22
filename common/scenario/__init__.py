"""
common/scenario/ — 시나리오 축 레지스트리.

사용:
  from common.scenario import get_segmenter
  seg = get_segmenter("qfrac", cfg={"qfrac": {"dis_bounds": [0,0.4,0.7,1.0]}})
  seg = get_segmenter("protocol", cfg={"protocol": {"max_steps": 3}})
"""

from .base import ScenarioSpec, SegmentRecord, Segmenter
from .qfrac import QFracSegmenter

REGISTRY: dict[str, type] = {
    "qfrac": QFracSegmenter,
}

try:
    from .protocol import ProtocolSegmenter
    REGISTRY["protocol"] = ProtocolSegmenter
except Exception:  # pragma: no cover
    pass

try:
    from .vwindow import VWindowSegmenter
    REGISTRY["vwindow"] = VWindowSegmenter
except Exception:
    pass

try:
    from .rcs import RCSSegmenter
    REGISTRY["rcs"] = RCSSegmenter
except Exception:
    pass

try:
    from .cluster import ClusterSegmenter
    REGISTRY["cluster"] = ClusterSegmenter
except Exception:
    pass

try:
    from .test_rs import TestRSSegmenter
    REGISTRY["test_rs"] = TestRSSegmenter
except Exception:
    pass

try:
    from .q_frac_wide import QFracWideSegmenter
    REGISTRY["q_frac_wide"] = QFracWideSegmenter
except Exception:
    pass


def get_segmenter(name: str, cfg: dict | None = None) -> Segmenter:
    """
    이름으로 Segmenter 인스턴스를 생성해 반환.

    cfg: scr.yaml의 scenario 섹션 전체를 넘기면 해당 축의 kwargs만 추출.
    예: cfg={"qfrac": {"dis_bounds": [0,0.4,0.7,1.0]}, "protocol": {"max_steps":2}}
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
    "QFracSegmenter",
    "QFracWideSegmenter",
    "REGISTRY",
    "get_segmenter",
]
