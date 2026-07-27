"""
common/scenario/_random_seg.py — 존 내부 랜덤 세그먼트 추출 헬퍼 (옵션).

q_frac_wide / vqslope 두 축이 `random_segment=True` 옵션에서 공유하는 로직.

[핵심 원칙 — 라벨(SOH) 독립성 유지]
  창 길이를 "고정 관측 포인트 수(seg_len_pts)"로 잡는다. 이 값은 스니펫 내용만으로
  계산 가능하고 그 사이클의 실측 총 용량(q_tot ≈ SOH)을 참조하지 않으므로, 세그먼트
  정의가 라벨의 함수가 되지 않는다.

  **클리핑 절대 금지**: 존 포인트 수가 seg_len_pts보다 작으면, 창 길이를 존에 맞춰
  줄이지 않는다(줄이면 길이가 SOH의 함수가 되어 가짜 HI-SOH 상관 유입). 대신 그 존
  전체를 1개 세그먼트로 쓰는 fallback을 취한다.

  존이 min_pts 미만이면 창을 만들지 않는다(선택 편향이지 누수 아님).

[누락 비율]
  랜덤 추출은 존의 일부 포인트를 어느 창에도 포함시키지 않을 수 있다. 이 "샘플링되지
  못한 포인트 비율"을 시나리오별로 집계해 텍스트로 공개한다(재현성·투명성).
"""

from __future__ import annotations

import numpy as np


def sample_random_windows(
    zone_idx: np.ndarray,
    seg_len_pts: int,
    n_samples: int,
    min_pts: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], int, int]:
    """존에 속한 원시 포인트 인덱스 배열에서 고정 길이 랜덤 창 추출.

    Args:
        zone_idx   : 존에 속한 원시 포인트의 원본 인덱스 (오름차순, 연속 슬라이스 아님).
        seg_len_pts: 창의 고정 관측 포인트 수 (>= min_pts 권장).
        n_samples  : 뽑을 창 개수 (존이 창보다 짧으면 fallback 1개).
        min_pts    : 세그먼트 최소 포인트 수.
        rng        : 재현성 난수 생성기.

    Returns:
        (windows, n_covered, n_total)
          windows   : 각 창의 원본 인덱스 배열 리스트.
          n_covered : 어느 창에든 포함된 존 포인트 수 (창 겹침은 1회 집계).
          n_total   : 존 전체 포인트 수.
    """
    zone_idx = np.asarray(zone_idx)
    n = int(len(zone_idx))
    if n < min_pts:
        return [], 0, n

    windows: list[np.ndarray] = []
    if n <= seg_len_pts:
        # fallback: 존 전체 1개 (클리핑 아님 — 존 경계 그대로 사용)
        windows.append(zone_idx)
    else:
        max_start = n - seg_len_pts
        for _ in range(max(1, n_samples)):
            s = int(rng.integers(0, max_start + 1))   # 존-local 시작 위치
            windows.append(zone_idx[s:s + seg_len_pts])

    covered: set[int] = set()
    for w in windows:
        covered.update(int(x) for x in w)
    return windows, len(covered), n
