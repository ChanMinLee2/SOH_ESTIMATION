"""
5_model/experiments/phase1_lab/build_synergy_groups.py

Phase1 학습 이전에 실행하는 "시너지 그룹" 사전 구성 스크립트 (기존 5_model 코드 무변경).

설계 배경 (세션 논의 요약):
  - 다중공선성 제거를 먼저 하고 시너지를 나중에 찾으면, "클러스터 대표"를 뭘로 뽑을지가
    시너지 정보 없이 정해져야 하는 순환 논리 문제가 생긴다.
  - 그래서 순서를 뒤집지 않고, 편상관계수(partial correlation) 필터 하나로
    "다중공선성 배제"와 "시너지 후보 발굴"을 동시에 수행한다:
      - 이미 그룹에 있는 멤버와 거의 같은 정보(|raw corr| >= redundancy-threshold)인 후보는
        애초에 후보에서 제외 (= 다중공선성 배제, Stage4의 클러스터 임계값과 동일 기준 재사용)
      - 살아남은 후보 중, 그룹으로 conditioning했을 때도 target과의 관계가 여전히/더 강하게
        남는(편상관계수가 큰) 후보를 추가 (= 시너지 있는 조합 우선)
  - 시간복잡도: 그룹 성장 단계마다 전체 후보를 다 정밀 검사(회귀 기반 편상관계수)하지 않고,
    먼저 "그 seed와의 단순 상관계수"로 상위 --prefilter-top-m개만 추린 뒤에만 정밀 계산 —
    O(N^2 * S) -> O(N^2) 필터 + O(N * M * S) 본검사로 완화(N=HI 후보 수, S=표본 수, M=prefilter 폭).

알고리즘 (시나리오별로 독립 수행):
  1. 전체 HI를 target과의 단순 상관계수 |r| 내림차순으로 정렬 -> seed 순서.
  2. 아직 어느 그룹에도 안 속한 seed를 하나씩 꺼내 새 그룹 시작.
  3. 그 그룹을 최대 --max-group-size(기본 4)까지 그리디로 채움:
     a. 미배정 후보 중, 지금 그룹의 어느 멤버와도 |raw corr| < --redundancy-threshold(기본 0.9)인
        것만 남김 (다중공선성 배제).
     b. 그중 |raw corr(candidate, target)|가 큰 상위 --prefilter-top-m개만 추림 (저비용 필터).
     c. 그 M개에 대해서만 편상관계수(그룹 멤버 전체로 target/후보를 회귀한 잔차의 상관)를 계산 —
        가장 큰 후보를 채택. 채택 기준(|편상관계수|) < --min-partial-corr면 이 그룹은 그만 채움.
  4. 모든 HI가 정확히 하나의 그룹에 배정될 때까지 반복 (약한 HI는 크기 1짜리 그룹으로 남음).

출력은 기존 gates JSON과 같은 "seg_{s}_..." 키 컨벤션을 따른다 — 나중에 Phase1이 이 파일을
그대로 읽어 그룹 단위로 후보를 제한하도록 확장할 때 다른 스크립트와 동일한 패턴을 쓸 수 있게.

사용 예(--seg-axis/--axis-config/--data-dir/--seg-data-dir 전부 표준 조합(q_frac_ref,
n1=0.35/n2=0.20/n_samples=2)이면 생략 가능 — 기본값 자동 적용, 다른 조합이면 넷 다 같이 오버라이드):
  python 5_model/experiments/phase1_lab/build_synergy_groups.py \
      --model-config 5_model/config/main_qfref_S_p60.yaml \
      --split-seed 42 --tag k25_full_N2_groups
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

import sys
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from log_utils import append_log_entry, current_command_str  # noqa: E402

# 루트는 data_directories.py의 DATA_4_HI_ROOT_STR에서 가져온다 — PC마다 실제 드라이브가
# 다르므로(lambda_sweep.py와 동일 이유) 절대경로를 여기 하드코딩하지 않는다. 표준 조합
# (q_frac_ref, n1=0.35/n2=0.20/n_samples=2) 기준 기본값이고, 다른 조합이면 CLI로 오버라이드.
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

# seg-axis/axis-config도 이 세션 전체에서 한 번도 안 바뀐 고정값 — lambda_sweep.py의
# FIXED_AXIS_PARAMS와 동일 이유로 기본값을 준다. DEFAULT_DATA_DIR/DEFAULT_SEG_DATA_DIR과
# 세트로 묶인 값이라(다른 조합이면 데이터 경로도 같이 바뀌어야 함) 다른 조합을 쓰려면
# 셋 다 함께 오버라이드해야 한다.
DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})

# analyze_hi_synergy.py와 동일한 이유로 torch 의존 import(load_config/build_datasets/
# get_segmenter)는 모듈 최상단에 두지 않는다 — 이 스크립트는 멀티프로세싱을 안 쓰지만,
# 규칙을 통일해두면 나중에 병렬화가 필요해져도 안전하다.
try:
    from tqdm import tqdm as _tqdm

    def tqdm(iterable=None, **kwargs):
        return _tqdm(iterable, **kwargs)

    def tqdm_write(msg: str) -> None:
        _tqdm.write(msg)
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter([])

    def tqdm_write(msg: str) -> None:
        print(msg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 이전 HI 시너지 그룹 사전 구성 (다중공선성 배제 필터 통합)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="cycle pkl 경로")
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR, help="seg pkl 경로")
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-group-size", type=int, default=4,
                   help="그룹당 최대 HI 개수 (기본 4)")
    p.add_argument("--redundancy-threshold", type=float, default=0.9,
                   help="|raw corr| >= 이 값이면 같은 그룹에 같이 못 들어감 (Stage4 클러스터 threshold와 동일 기준)")
    p.add_argument("--min-partial-corr", type=float, default=0.02,
                   help="그룹 성장을 멈추는 기준 — 편상관계수 절댓값이 이보다 작으면 더 안 채움")
    p.add_argument("--prefilter-top-m", type=int, default=15,
                   help="그룹 성장 단계마다 정밀 검사(편상관계수)할 후보 수 상한 — "
                        "먼저 단순 상관계수로 이 개수만 추린 뒤에만 정밀 계산 (시간복잡도 완화)")
    p.add_argument("--global-dedup", action="store_true", dest="global_dedup",
                   help="v3 전용: 그룹 성장을 시작하기 전에 raw HI끼리 1:1로 |corr|>=threshold인 "
                        "쌍을 먼저 정리한다(타깃과의 단순상관이 더 낮은 쪽을 후보군에서 제외, "
                        "가장 상관 높았던 survivor에 사후 귀속). 이러면 그룹 성장 단계에 진입하는 "
                        "survivor들끼리는 서로 |corr|<threshold가 항상 보장돼 그룹 간 다중공선성이 "
                        "원리적으로 발생할 수 없다(구 버전의 순서 의존적 '브릿지 HI' 문제 해소). "
                        "기본값 꺼짐 = 기존(v0/v1/v2) 동작과 100% 동일.")
    p.add_argument("--shuffle-from", default=None, dest="shuffle_from",
                   help="v-ctrl 전용: 이 경로의 synergy_groups_*.json이 가진 시나리오별 "
                        "그룹 크기 분포를 그대로 두고, 멤버만 무작위로 재배정한다 — "
                        "진짜 편상관 기반 그리디 알고리즘(build_groups)은 아예 안 돌리고 "
                        "건너뛴다. 최종 그룹 개수·크기가 참조 파일과 동일해서 커널 HI "
                        "개수(=피처 개수)가 v2/v3와 정확히 같아진다(대조군 성립 조건).")
    p.add_argument("--shuffle-seed", type=int, default=42, dest="shuffle_seed",
                   help="--shuffle-from 전용 무작위 배정 시드")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _load_all_scenarios(args) -> tuple:
    from utils.io_utils import load_config
    from datasets.segment_dataset import build_datasets
    from common.scenario import get_segmenter
    from utils.hi_schema import get_hi_cols_for_seg

    cfg = load_config(args.model_config)
    cfg.setdefault("data", {})
    cfg["data"]["data_dir"] = args.data_dir
    cfg["data"]["seg_data_dir"] = args.seg_data_dir
    cfg["data"]["datasets"] = args.datasets
    cfg["data"]["split_seed"] = args.split_seed

    axis_cfg = json.loads(args.axis_config)
    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()
    train_ds, _val_ds, _test_ds, _norm = build_datasets(cfg, spec=spec)

    x_all = train_ds.x_hi.numpy()
    y_all = train_ds.target.numpy()
    scen_idx_all = train_ds.scen_idx.numpy()
    # 시나리오별 실제 HI 공식 이름 (get_hi_cols_for_seg는 seg 접미사만 다르고 순서는
    # 항상 동일 — hi_schema.py 참고) — hi_00 같은 자리표시자 대신 diff_dqdv_area_chg_lo처럼
    # 바로 읽을 수 있는 이름을 쓰기 위해 시나리오별로 하나씩 만들어둔다.
    names_by_seg = {s: get_hi_cols_for_seg(name) for s, name in enumerate(spec.scenario_names)}
    return x_all, y_all, scen_idx_all, spec, names_by_seg


# ---------------------------------------------------------------------------
# 편상관계수 (그룹 전체로 conditioning) — 작은 회귀 잔차의 상관
# ---------------------------------------------------------------------------

def _residualize(y: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
    """conditioning 열들(+절편)로 y를 회귀한 뒤 잔차 반환. conditioning이 비어있으면 y 그대로."""
    if conditioning.shape[1] == 0:
        return y
    A = np.column_stack([conditioning, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _prune_redundant_raw(
    marg_signed: np.ndarray, raw_corr: np.ndarray, redundancy_threshold: float,
) -> tuple[list[int], dict[int, int]]:
    """그룹 성장을 시작하기 전에 raw HI끼리 |raw corr|>=threshold인 쌍을 미리 정리한다.

    **연결요소(Union-Find) 방식** — 순차 그리디(각 HI를 "그 시점까지의 survivor 목록"하고만
    비교)를 먼저 시도했으나 실측(전체 raw 데이터)에서 실패했다: survivor끼리는 0건으로
    완벽했지만, 탈락한 HI를 사후에 최적 survivor의 그룹에 "귀속"시키는 단계에서 133건의
    새 위반이 나왔다 — 탈락한 HI는 귀속될 때 그 survivor하고만 비교됐지, 최종적으로 다른
    그룹에 남는 HI들과는 한 번도 비교된 적이 없었기 때문(같은 종류의 순서 의존적 누락이
    한 단계 아래로 옮겨간 것). **완전한 해법은 전체 64개 HI로 그래프를 만들어(|raw
    corr|>=threshold인 쌍끼리 변) 연결요소를 구하는 것** — 정의상 서로 다른 연결요소에
    속한 두 HI 사이에는 (경유하는 다른 HI가 있든 없든) 직접 변이 존재하지 않으므로
    |raw corr|<threshold가 무조건 보장된다. 각 연결요소 안에서 타깃과의 단순상관
    |marg_signed|가 가장 큰 HI를 대표(survivor)로 뽑아 그룹 성장에 참여시키고, 나머지는
    그 대표가 속한 최종 그룹에 귀속시킨다 — survivor든 귀속된 HI든 관계없이, 서로 다른
    그룹에 속한 임의의 두 HI는 항상 서로 다른 연결요소 출신이라 |corr|<threshold가
    보장된다(생존자만이 아니라 전체 64개 HI에 대해 완전하다)."""
    n_hi = len(marg_signed)
    parent = list(range(n_hi))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n_hi):
        for j in range(i + 1, n_hi):
            if abs(raw_corr[i, j]) >= redundancy_threshold:
                _union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n_hi):
        components.setdefault(_find(i), []).append(i)

    survivors: list[int] = []
    attach_to: dict[int, int] = {}
    for members in components.values():
        rep = max(members, key=lambda i: abs(marg_signed[i]))
        survivors.append(int(rep))
        for m in members:
            if m != rep:
                attach_to[int(m)] = int(rep)
    return survivors, attach_to


def _partial_corr(y: np.ndarray, candidate: np.ndarray, group_x: np.ndarray) -> float:
    if group_x.shape[1] == 0:
        c = np.corrcoef(candidate, y)[0, 1]
        return 0.0 if np.isnan(c) else float(c)
    ry = _residualize(y, group_x)
    rc = _residualize(candidate, group_x)
    if np.std(ry) < 1e-8 or np.std(rc) < 1e-8:
        return 0.0
    c = np.corrcoef(ry, rc)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


# ---------------------------------------------------------------------------
# 그리디 그룹 구성
# ---------------------------------------------------------------------------

def build_groups(
    x: np.ndarray,
    y: np.ndarray,
    max_group_size: int,
    redundancy_threshold: float,
    min_partial_corr: float,
    prefilter_top_m: int,
    global_dedup: bool = False,
) -> list[dict]:
    """global_dedup=False(기존, v0/v1/v2 그대로): 다중공선성 배제를 "현재 그룹 멤버"까지만
    검사한다 — 이미 완성된 다른 그룹의 멤버와 겹쳐도 못 잡는다(그룹 간 중복 미검사, 알려진
    한계). global_dedup=True(v3.1): 그룹 성장을 시작하기 전에 `_prune_redundant_raw`로
    raw HI끼리 1:1 다중공선성을 먼저 완전히 정리한다(사전 가지치기) — 그룹 성장에 참여하는
    survivor들끼리는 이미 서로 |corr|<threshold가 보장되므로, 성장 단계 가드는 "현재 그룹
    멤버만" 봐도 충분하다(v0/v1/v2와 동일한 저비용 체크로 되돌아감).

    **왜 사전 가지치기인가(v3.1, 구버전 시드-병합 방식을 대체)**: 이전 버전(v3)은 새 시드를
    뽑을 때 "이미 배정된 HI와 겹치면 그 그룹에 편입"하는 식이었는데, 이건 그룹이 형성되는
    *순서*에 결과가 좌우됐다 — 실측 결과 HI 하나가 서로 다른 두 그룹 모두와 |corr|>=0.9인
    "브릿지" 케이스가 남았고, 그 100%가 "이미 배정된 쪽으로 먼저 편입되고 나면 그 뒤에
    처리되는 다른 그룹과의 관계는 검사할 기회 자체가 없는" 순서 의존적 누락이었다. 사전
    가지치기는 그룹 형성 자체가 시작되기 *전에* raw HI 후보군을 "서로 |corr|<threshold인
    survivor들"로 확정해버리므로, 이 순서 의존성이 원리적으로 없다(완전성 증명은
    `_prune_redundant_raw` 참고). 탈락한 HI는 버리지 않고, 가장 상관 높았던 survivor가
    최종적으로 속한 그룹에 그룹 성장 종료 후 "attached"로 사후 편입한다(모델 입력 커버리지
    유지, 단 시너지 성장 점수(Level1)에는 포함 안 시켜 지표를 오염시키지 않는다)."""
    n_hi = x.shape[1]

    # 시드 순서: 단순 상관계수(부호 있음, 정렬은 절댓값 기준) — 필터에도 재사용
    marg_signed = np.zeros(n_hi)
    for i in range(n_hi):
        c = np.corrcoef(x[:, i], y)[0, 1]
        marg_signed[i] = 0.0 if np.isnan(c) else c

    # 원시 상관행렬 — 다중공선성 배제 가드용 (한 번만 계산, O(N^2 * S))
    raw_corr = np.corrcoef(x, rowvar=False)
    raw_corr = np.nan_to_num(raw_corr, nan=0.0)

    assigned = [False] * n_hi
    group_of: dict[int, int] = {}
    groups: list[dict] = []

    attach_to: dict[int, int] = {}
    if global_dedup:
        survivors, attach_to = _prune_redundant_raw(marg_signed, raw_corr, redundancy_threshold)
        for d in attach_to:
            assigned[d] = True  # 성장 후보에서 제외 — 사후 편입 대상으로만 남김
        survivor_set = set(survivors)
        seed_order = [int(i) for i in np.argsort(-np.abs(marg_signed)) if int(i) in survivor_set]
    else:
        seed_order = list(np.argsort(-np.abs(marg_signed)))

    for seed in seed_order:
        if assigned[seed]:
            continue

        members = [int(seed)]
        scores = [float(marg_signed[seed])]
        assigned[seed] = True
        group_of[seed] = len(groups)  # 이번에 append될 그룹의 인덱스(아래에서 실제 append)

        while len(members) < max_group_size:
            group_x = x[:, members]

            # 다중공선성 배제: survivor끼리는 사전 가지치기로 이미 |corr|<threshold가
            # 보장되므로(global_dedup=True) "현재 그룹 멤버만" 봐도 충분하다 —
            # global_dedup=False(기존 v0/v1/v2)일 때도 원래부터 이 체크였으므로 동일 코드 경로.
            eligible = [
                c for c in range(n_hi)
                if not assigned[c]
                and all(abs(raw_corr[c, m]) < redundancy_threshold for m in members)
            ]
            if not eligible:
                break

            # b) 저비용 사전 필터: 단순 |상관계수(candidate, target)| 상위 M개만 정밀 검사 대상으로
            eligible.sort(key=lambda c: -abs(marg_signed[c]))
            shortlist = eligible[:prefilter_top_m]

            # c) 정밀 검사: 그룹 전체로 conditioning한 편상관계수
            best_cand, best_score = None, min_partial_corr
            for cand in shortlist:
                pc = _partial_corr(y, x[:, cand], group_x)
                if abs(pc) > abs(best_score):
                    best_score, best_cand = pc, cand

            if best_cand is None:
                break
            members.append(int(best_cand))
            scores.append(float(best_score))
            assigned[best_cand] = True
            group_of[best_cand] = len(groups)  # 이 그룹이 append될 인덱스(seed와 동일 규칙)

        groups.append({"members": members, "scores": scores, "attached": []})

    # 사후 편입: 사전 가지치기로 탈락한 HI를 가장 상관 높았던 survivor의 최종 그룹에 붙인다.
    # attach_to의 파트너는 항상 survivor이므로(구현상 탈락한 HI는 survivors에 못 들어감)
    # group_of[partner]는 항상 존재한다.
    for d, partner in attach_to.items():
        target_gi = group_of.get(partner)
        if target_gi is not None:
            groups[target_gi]["attached"].append(int(d))

    return groups


def build_groups_shuffled(
    x: np.ndarray, y: np.ndarray, group_sizes: list[int], seed: int,
) -> list[dict]:
    """v-ctrl 전용 — 진짜 편상관 그리디를 안 돌리고, 참조 그룹의 크기 분포만 그대로 두고
    멤버를 무작위로 재배정한다. scores는 실제 계산 값(첫 멤버=단순상관, 이후=편상관)을
    그대로 채워서, "이 무작위 그룹도 어차피 시너지 점수는 낮다"는 걸 사후에 확인할 수
    있게 해둔다(학습에는 안 쓰임, 진단용)."""
    n_hi = x.shape[1]
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_hi).tolist()

    groups: list[dict] = []
    pos = 0
    for size in group_sizes:
        members = order[pos:pos + size]
        pos += size
        if not members:
            continue
        scores = [float(_partial_corr(y, x[:, members[0]], x[:, :0]))]
        for i in range(1, len(members)):
            scores.append(_partial_corr(y, x[:, members[i]], x[:, members[:i]]))
        groups.append({"members": members, "scores": scores})
    return groups


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    x_all, y_all, scen_idx_all, spec, names_by_seg = _load_all_scenarios(args)

    ref_report = None
    if args.shuffle_from:
        ref_report = json.loads(Path(args.shuffle_from).read_text(encoding="utf-8"))
        print(f"[groups] v-ctrl 모드: {args.shuffle_from}의 그룹 크기 분포를 그대로 쓰고 "
              f"멤버만 무작위 재배정(shuffle-seed={args.shuffle_seed})")

    report: dict = {"tag": args.tag, "max_group_size": args.max_group_size,
                     "redundancy_threshold": args.redundancy_threshold,
                     "min_partial_corr": args.min_partial_corr,
                     "prefilter_top_m": args.prefilter_top_m,
                     "global_dedup": args.global_dedup,
                     "shuffle_from": args.shuffle_from, "shuffle_seed": args.shuffle_seed}

    all_group_sizes: list[int] = []
    for s, seg_name in enumerate(tqdm(spec.scenario_names, desc="시나리오별 그룹 구성", unit="scenario")):
        sel = scen_idx_all == s
        x_scen, y_scen = x_all[sel], y_all[sel]
        if x_scen.shape[0] < 20:
            tqdm_write(f"[groups] {seg_name}: 표본 부족({x_scen.shape[0]}) — 스킵")
            continue

        if ref_report is not None:
            if f"seg_{s}_groups" not in ref_report:
                tqdm_write(f"[groups] {seg_name}: 참조 파일에 없음 — 스킵")
                continue
            ref_sizes = [len(g) for g in ref_report[f"seg_{s}_groups"]]
            groups = build_groups_shuffled(
                x_scen, y_scen, ref_sizes, seed=args.shuffle_seed + s,
            )
        else:
            groups = build_groups(
                x_scen, y_scen,
                max_group_size=args.max_group_size,
                redundancy_threshold=args.redundancy_threshold,
                min_partial_corr=args.min_partial_corr,
                prefilter_top_m=args.prefilter_top_m,
                global_dedup=args.global_dedup,
            )
        groups.sort(key=lambda g: -abs(g["scores"][0]))  # seed 개별 중요도 순으로 그룹 정렬

        report[f"seg_{s}_seg_name"] = seg_name
        report[f"seg_{s}_groups"] = [g["members"] for g in groups]
        report[f"seg_{s}_group_names"] = [[names_by_seg[s][i] for i in g["members"]] for g in groups]
        report[f"seg_{s}_group_scores"] = [g["scores"] for g in groups]
        # v3.1 전용(global_dedup): 사전 가지치기로 탈락해 이 그룹에 사후 편입된 HI —
        # 시너지 성장(Level1)과 커널 피처 구성(build_kernel_group_features.py) 둘 다에
        # 안 쓰인다. 다중공선성 장부(이 HI가 어느 그룹 소속인지)와 x_hi 자체의 독립 게이트
        # 커버리지 용도로만 남겨둔다 — members와 합치면 그룹 크기가 2~29개로 들쭉날쭉해져
        # Level2 gap 비교의 교란변수가 된다는 게 실측으로 확인돼(docs/260827_RESULTS.md
        # "v3 커널 피처 재생성" 절) 합치지 않는 쪽으로 확정됐다. global_dedup=False면
        # 항상 빈 리스트.
        report[f"seg_{s}_group_attached"] = [g.get("attached", []) for g in groups]
        report[f"seg_{s}_group_attached_names"] = [
            [names_by_seg[s][i] for i in g.get("attached", [])] for g in groups
        ]

        sizes = [len(g["members"]) for g in groups]
        all_group_sizes += sizes
        n_multi = sum(1 for sz in sizes if sz > 1)
        tqdm_write(
            f"[groups] {seg_name}: HI {x_scen.shape[1]}개 -> 그룹 {len(groups)}개 "
            f"(2개 이상 묶인 그룹 {n_multi}개, 최대크기 {max(sizes)}, 평균크기 {np.mean(sizes):.2f})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"synergy_groups_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[groups] 저장: {out_path}")

    mean_size = float(np.mean(all_group_sizes)) if all_group_sizes else 0.0
    n_hi_total = len(all_group_sizes)
    n_groups_total = sum(1 for s in range(spec.n_scenarios) if f"seg_{s}_groups" in report
                          for _ in report[f"seg_{s}_groups"])
    append_log_entry(
        tag=f"synergy_groups_{args.tag}",
        purpose="Phase1 이전 HI 시너지 그룹 사전 구성 (편상관계수 필터 = 다중공선성 배제 + 시너지 발굴 통합)",
        command=current_command_str(),
        result_files=[str(out_path)],
        key_metrics=f"전체 HI {n_hi_total}개 -> 그룹 {n_groups_total}개, 평균 그룹 크기 {mean_size:.2f}",
        interpretation=(
            "평균 그룹 크기가 1에 가까우면 대부분 HI가 독립적(다중공선성/시너지 둘 다 약함), "
            "4에 가까우면 대부분 HI가 큰 시너지 그룹으로 묶임 — Stage4 클러스터 개수(39~55/64)와 "
            "함께 보면 이 그룹 구조가 타당한지 교차검증 가능."
        ),
    )


if __name__ == "__main__":
    main()
