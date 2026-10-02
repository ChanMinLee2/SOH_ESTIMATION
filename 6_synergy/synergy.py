"""
6_synergy/synergy.py

Phase1 학습 이전에 실행하는 "시너지 그룹" 사전 구성 스크립트 (기존 model_lib 코드 무변경).

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

2026-10-01: --out-dir를 제외한 모든 CLI 인자를 제거했다
실행은 그냥:
    python 6_synergy/synergy.py

main()이 호출하는 핵심 흐름(아래 번호는 main() 본문의 동일 번호 주석과 대응):
  1) _load_all_scenarios        — (interaction.py 공용) train split 데이터 로드
  2) build_groups/build_groups_shuffled — 시나리오별로 그룹 구성(v-ctrl이면 후자)
  3) 저장                       — synergy_groups_{tag}.json
  4) append_log_entry           — docs/phase1_lab/RESULTS_LOG.md에 실험 기록 자동 추가
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "model_lib" / "results"

# data_directories/parameters/log_utils/common/models/datasets 등은 pip install -e .로
# 어디서든 바로 import된다(pyproject.toml 참고) — sys.path 조작 불필요.
from log_utils import append_log_entry, current_command_str
import parameters as P  # 축/실행 파라미터 단일 소스

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
    # 2026-10-01: --out-dir 하나만 남기고 전부 제거 — 나머지는 이미 parameters.py가
    # 단일 소스인 값의 CLI 통로였을 뿐이다(hi_correlation.py/interaction.py 정리와
    # 동일 원칙). --out-dir만 예외인 이유는 모듈 docstring 참고(run_pipeline.py가
    # 여러 스텝이 공유하는 실험 폴더를 넘겨주는 용도라 parameters.py로 복원 불가).
    p = argparse.ArgumentParser(description="Phase1 이전 HI 시너지 그룹 사전 구성 (다중공선성 배제 필터 통합)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="산출물 저장 위치(기본: results/) — run_pipeline.py가 Step 9 학습 "
                        "run 폴더로 넘길 때 씀(2026-09-19).")
    return p.parse_args()


# _load_all_scenarios는 5_interaction/interaction.py(Step 5, synergy.py보다 먼저
# 실행됨) 소유 — 파이프라인 실행 순서상 더 앞 단계가 "기반" 코드를 갖고 뒤 단계가
# 가져다 쓰는 게 자연스럽다(2026-09-30, 원래는 반대 방향이었음 — synergy.py가
# 정의하고 interaction.py가 가져다 썼는데, Step 번호와 의존 방향이 거꾸로였다는
# 지적을 받아 교정). 중복 구현 금지 원칙은 그대로 유지.
sys.path.insert(0, str(PROJECT_ROOT / "5_interaction"))
from interaction import _load_all_scenarios  # noqa: E402


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
    
    완전한 해법은 전체 64개 HI로 그래프를 만들어(|raw corr|>=threshold인 쌍끼리 변) 
    연결요소를 구하는 것 — 정의상 서로 다른 연결요소에 속한 두 HI 사이에는 
    (경유하는 다른 HI가 있든 없든) 직접 변이 존재하지 않으므로
    |raw corr|<threshold가 무조건 보장된다. 
    
    각 연결요소 안에서 타깃과의 단순상관 |marg_signed|가 가장 큰 HI를 대표(survivor)로 뽑아 
    그룹 성장에 참여시키고, 나머지는 그 대표가 속한 최종 그룹에 귀속시킨다 — survivor든 귀속된 HI든 관계없이, 
    서로 다른 그룹에 속한 임의의 두 HI는 항상 서로 다른 연결요소 출신이라 |corr|<threshold가
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

    # parameters.py에서 그대로 읽는 실행 파라미터 — --out-dir 외엔 전부 여기서 해석
    # (main() 위 docstring 참고).
    seg_axis = P.FIXED_SEG_AXIS
    axis_config = json.dumps(P.ACTIVE_AXIS_CONFIG)
    data_dir = P.FIXED_CANONICAL_DATA_DIR
    seg_data_dir = P.FIXED_CANONICAL_SEG_DATA_DIR
    datasets = P.FIXED_CANONICAL_DATASETS
    split_seed = P.ACTIVE_SPLIT_SEED if P.ACTIVE_SPLIT_SEED is not None else P.FIXED_DEFAULT_SEED
    max_group_size = P.ACTIVE_MAX_GROUP_SIZE
    redundancy_threshold = P.ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD
    min_partial_corr = P.FIXED_MIN_PARTIAL_CORR
    prefilter_top_m = P.FIXED_PREFILTER_TOP_M
    global_dedup = P.FIXED_GLOBAL_DEDUP
    shuffle_from = P.FIXED_SYNERGY_SHUFFLE_FROM
    shuffle_seed = P.FIXED_SHUFFLE_SEED
    tag = P.FIXED_SYNERGY_TAG or f"{P.ACTIVE_P1_TAG}_groups"

    # 1) _load_all_scenarios — train split 데이터 로드(interaction.py 소유, 중복 구현 금지) —
    # 여기 CLI는 없앴으므로 필요한 필드만 담은 SimpleNamespace를 대신 넘긴다.
    _loader_args = SimpleNamespace(
        data_dir=data_dir, seg_data_dir=seg_data_dir, datasets=datasets,
        split_seed=split_seed, axis_config=axis_config, seg_axis=seg_axis,
    )
    x_all, y_all, scen_idx_all, spec, names_by_seg, _cell_ids = _load_all_scenarios(_loader_args)

    ref_report = None
    if shuffle_from:
        ref_report = json.loads(Path(shuffle_from).read_text(encoding="utf-8"))
        print(f"[groups] v-ctrl 모드: {shuffle_from}의 그룹 크기 분포를 그대로 쓰고 "
            f"멤버만 무작위 재배정(shuffle-seed={shuffle_seed})")

    report: dict = {"tag": tag, "max_group_size": max_group_size,
                    "redundancy_threshold": redundancy_threshold,
                    "min_partial_corr": min_partial_corr,
                    "prefilter_top_m": prefilter_top_m,
                    "global_dedup": global_dedup,
                    "shuffle_from": shuffle_from, "shuffle_seed": shuffle_seed}

    # 2) build_groups/build_groups_shuffled — 시나리오별로 그룹 구성(v-ctrl이면 후자)
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
                x_scen, y_scen, ref_sizes, seed=shuffle_seed + s,
            )
        else:
            groups = build_groups(
                x_scen, y_scen,
                max_group_size=max_group_size,
                redundancy_threshold=redundancy_threshold,
                min_partial_corr=min_partial_corr,
                prefilter_top_m=prefilter_top_m,
                global_dedup=global_dedup,
            )
        groups.sort(key=lambda g: -abs(g["scores"][0]))  # seed 개별 중요도 순으로 그룹 정렬

        report[f"seg_{s}_seg_name"] = seg_name
        report[f"seg_{s}_groups"] = [g["members"] for g in groups]
        report[f"seg_{s}_group_names"] = [[names_by_seg[s][i] for i in g["members"]] for g in groups]
        report[f"seg_{s}_group_scores"] = [g["scores"] for g in groups]
        # v3.1 전용(global_dedup): 사전 가지치기로 탈락해 이 그룹에 사후 편입된 HI —
        # 시너지 성장(Level1)과 커널 피처 구성(kernel.py) 둘 다에
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

    # 3) 저장 — synergy_groups_{tag}.json
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synergy_groups_{tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[groups] 저장: {out_path}")

    # 4) append_log_entry — docs/phase1_lab/RESULTS_LOG.md에 실험 기록 자동 추가
    mean_size = float(np.mean(all_group_sizes)) if all_group_sizes else 0.0
    n_hi_total = len(all_group_sizes)
    n_groups_total = sum(1 for s in range(spec.n_scenarios) if f"seg_{s}_groups" in report
                          for _ in report[f"seg_{s}_groups"])
    append_log_entry(
        tag=f"synergy_groups_{tag}",
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
