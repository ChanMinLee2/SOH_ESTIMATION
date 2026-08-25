"""중간 산출물(pkl) 저장 경로의 단일 진입점.

2026-08-08: 디스크 공간 확보를 위해 아래 세 데이터 루트를 C→D 드라이브로 이전하며,
이전에는 26개+ 파일이 각자 `PROJECT_ROOT / "_4_data_hi"`를 독립적으로 하드코딩하고 있었다
(공유 상수 모듈 없음 — 나중에 드라이브를 또 옮기려면 그 개수만큼 다시 고쳐야 하는 문제).
이 모듈이 그 유일한 진입점이다 — **드라이브를 다시 옮길 때는 여기 `_D_ROOT`만 고치면 된다.**

각 사용처는:
    from data_directories import DATA_4_HI_ROOT
또는(5_model/*.py처럼 PROJECT_ROOT와 join되는 상대경로 문자열이 필요한 경우):
    from data_directories import DATA_4_HI_ROOT_STR
"""

from __future__ import annotations

from pathlib import Path

_D_ROOT = Path(r"D:\chanminLee\LFP_SOH_prediction_v2")

# 외부 드라이브 루트 자체(원본 대용량 데이터 폴백용, 예: _1_data_unified가 로컬에 없을 때
# 2_preprocess/preprocess.py가 여기서 찾는다). _D_ROOT의 공개 별칭 — 하위 상수
# (DATA_4_HI_ROOT 등)로 커버되지 않는, 드라이브 루트 자체가 필요한 극소수 사용처 전용.
EXTERNAL_DATA_ROOT = _D_ROOT

# _2_data_clean: 2026-08-08 실사조사 결과 코드 어디서도 참조하지 않는 레거시 데이터로
# 확인됨(2_preprocess/preprocess.py 실제 출력은 이미 _4_data_hi/clean 밑에 통합돼 있음).
# 여기 상수는 문서화 목적으로만 남겨둔다 — 실제로 참조하는 코드는 없다.
DATA_2_CLEAN_ROOT = _D_ROOT / "_2_data_clean"

# _4_data_hi: 전처리 완료본(clean/clean_noshape) + 축별 세그먼트/사이클 추출 결과.
DATA_4_HI_ROOT = _D_ROOT / "_4_data_hi"
# segment_dataset.py의 `PROJECT_ROOT / data_cfg["seg_data_dir"]` 같은 join에 쓰기 위한
# 슬래시 문자열 버전 — pathlib는 join 우변이 절대경로면 좌변(PROJECT_ROOT)을 무시하고
# 우변을 그대로 쓰므로, 이 문자열을 상대경로 자리에 넣으면 PROJECT_ROOT 무관하게 D로 간다.
DATA_4_HI_ROOT_STR = str(DATA_4_HI_ROOT).replace("\\", "/")

# 4_hi_analysis 내부 pkl 캐시(hi_features*.pkl, outputs/**/*.pkl) — .py 코드 파일은
# 그대로 C:(임포트·실행 위치)에 있고, 이 상수는 그 코드가 저장/조회하는 pkl 캐시 위치만
# 가리킨다(4_hi_analysis/hi_correlation.py, seg_diagnose.py 참고).
PKL_CACHE_ROOT = _D_ROOT / "4_hi_analysis"
