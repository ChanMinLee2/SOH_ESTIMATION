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
    # "random" 별칭: 같은 RCSSegmenter를 다른 파라미터 프리셋(예: assign="none")으로 쓸 때
    # 기존 "rcs" 캐시(_4_data_hi/rcs/)와 경로 충돌 없이 분리 저장하기 위함.
    # --axis-config에 "axis_name": "random"을 반드시 포함해야 실제로 분리된다
    # (안 넣으면 RCSSegmenter 기본값 axis_name="rcs"가 적용돼 그대로 rcs/ 경로를 씀) —
    # docs/260811_RESULTS.md 실험 2 참고.
    REGISTRY["random"] = RCSSegmenter
    # "random_grid" 별칭: placement="grid"(결정론적 등간격 배치, 커버리지 100% 보장)로 쓸 때
    # "random"(랜덤 배치, 평균 커버리지 <100%) 캐시와 분리 저장. --axis-config에
    # "axis_name": "random_grid"와 "placement": "grid"를 반드시 함께 포함해야 한다.
    REGISTRY["random_grid"] = RCSSegmenter
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

try:
    from .q_frac_ref import QFracRefSegmenter
    REGISTRY["q_frac_ref"] = QFracRefSegmenter
except Exception:
    pass

try:
    from .vqslope import VQSlopeSegmenter
    REGISTRY["vqslope"] = VQSlopeSegmenter
except Exception:
    pass

try:
    from .q_abs import QAbsSegmenter
    REGISTRY["q_abs"] = QAbsSegmenter
except Exception:
    pass

try:
    from .full_cycle import FullCycleSegmenter
    REGISTRY["full_cycle"] = FullCycleSegmenter
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
    "QFracRefSegmenter",
    "QAbsSegmenter",
    "REGISTRY",
    "get_segmenter",
]
