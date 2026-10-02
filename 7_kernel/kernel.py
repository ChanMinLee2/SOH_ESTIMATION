"""
7_kernel/kernel.py

synergy.py가 만든 그룹(다중공선성 배제 + 편상관계수 시너지 필터를 통과한
시나리오별 HI 묶음, 크기 2 이상만 대상)을 "그룹당 새 HI 하나"로 물리적으로 융합하는 스크립트.
그룹 멤버 HI들 -> SOH를 RBF 커널(Nystroem 근사 + Ridge)로 fit해서, 그 예측값을 그룹의
"커널 HI"로 쓴다 — 편상관계수(선형)로는 못 잡는 비선형 시너지를 명시적으로 캡처하기 위함.
raw HI는 대체하지 않고 그대로 둔 채 별도 블록으로 "추가"한다(train.py가
scr_model.py의 독립 게이트 scen_kernel_gates에 연결 — 설계 배경/이력은
docs/260820_RESULTS.md 참고).

이 스크립트가 하는 다중공선성 관리는 이제 3단계다(2026-09-18에 3차 추가):
  1. (synergy.py가 이미 함) 그룹 *내부* raw HI 중복 배제.
  2. 이 스크립트: 커널 HI *끼리*(시나리오 다른 그룹끼리도 원본 HI가 겹치면 커널값이
     비슷할 수 있어 전체 train pooled로 재검사) 다중공선성 배제(--redundancy-threshold).
  3. (신규) raw HI(64) + 이 시나리오가 만든 커널 HI를 합쳐서, **시나리오별로**(2차와 달리
     pooled 아님) |r|>=0.95인 각 쌍의 "패자"를 정해 제거한다 — degree(다른 HI와도
     0.95를 넘는 관계 개수)가 더 많은 쪽 우선 제거, 동일하면 타깃(SOH) 상관계수가 더
     낮은 쪽 제거 — `kernel_group_features_{tag}_combined_redundancy.json`으로 저장,
     train.py --combined-redundancy-json이 실제 배제를 적용한다.
  raw HI와 그걸로 만든 커널 HI 사이의 다중공선성은 이제 3차에서 검사한다(과거엔 검사하지
  않던 알려진 한계였음, docs/260820_RESULTS.md 참고).

  또한 2026-09-18부터 x_kernel 소비 방식이 바뀌었다(train.py의
  _apply_kernel_features, 요구사항1) — 커널 HI 컬럼은 이제 "그 컬럼을 만든 시나리오"의
  행에만 값이 채워지고 다른 시나리오 행에서는 항상 0이다(전에는 모든 행에 모든 커널
  모델을 적용해 다른 시나리오용 모델을 분포 밖(OOD) 입력에 적용한 의미 없는 값이 섞여
  있었다). 이 pkl의 mean/std(정규화 통계)도 그에 맞춰 own-scenario 행만으로 계산한다
  (2차 다중공선성 배제 판단 자체는 여전히 pooled kernel_vals를 씀 — 그 부분은 변경 없음).

출력 1(pickle, JSON이 아닌 이유: sklearn 파이프라인 객체를 그대로 저장해 재적용해야 함):
  {
    "tag": str, "n_features": int,
    "alpha": float, "gamma": float|None, "n_components": int,
    "redundancy_threshold": float, "max_features": int|None,
    "features": [
      {"name": str, "scenario": str, "members": [raw HI idx...],
       "member_names": [...], "train_r2": float, "model": Pipeline,
       "mean": float, "std": float}, ...
    ],
  }

출력 2(신규, JSON) kernel_group_features_{tag}_combined_redundancy.json:
  {
    "tag": str, "threshold": 0.95,
    "by_scenario": {
      seg_name: {"removed_raw_idx": [...], "removed_raw_names": [...],
                 "removed_kernel_names": [...], "n_total_checked": int,
                 "n_edges": int}, ...
    },
  }

2026-10-02: --out-dir를 제외한 모든 CLI 인자를 제거했다
실행은 그냥:
    python 7_kernel/kernel.py

2026-10-02: main()을 단일 책임 원칙에 따라 8개 서브함수로 분리했다(동작 변화
없는 순수 구조 정리, train.py의 t1~t7과 동일 원칙) —
k1_resolve_params_and_paths(out_dir/tag/synergy-groups-json 등 경로 해석) ->
k2_load_data(train split + HI 카테고리 비용 + synergy.py 산출물 로드) ->
k3_fit_group_kernels(시나리오별 그룹 -> Nystroem+Ridge 커널 HI 피팅) ->
k4_dedupe_kernels(2차 배제 — 커널끼리 pooled 상관) ->
k5_apply_feature_cap(max-features 캡, 시나리오별 라운드로빈) ->
k6_compute_normalization(최종 채택된 커널 HI의 own-scenario mean/std) ->
k7_build_combined_redundancy(3차 배제 — raw+kernel 결합, 시나리오별) ->
k8_save_results(pkl/json 저장 + 콘솔 요약 + RESULTS_LOG 기록). 각 단계는
SimpleNamespace로 다음 단계에 필요한 값만 넘기고, `candidates`/`rejected`/
`final` 같은 리스트는 여러 단계가 같은 리스트 객체를 참조로 공유하며 그 자리에서
append/mutate한다(원래 main() 하나였을 때와 동일한 공유 방식, 복사 없음).
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

from utils.hi_schema import get_hi_cost_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "model_lib" / "results"

# data_directories/parameters/log_utils/common/models/datasets 등은 pip install -e .로
# 어디서든 바로 import된다(pyproject.toml 참고) — sys.path 조작 불필요.
from log_utils import append_log_entry, current_command_str
import parameters as P  # 축/실행 파라미터 단일 소스

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
    # 2026-10-02: --out-dir 하나만 남기고 전부 제거 — 나머지는 이미 parameters.py가
    # 단일 소스인 값의 CLI 통로였을 뿐이다(hi_correlation.py/interaction.py/synergy.py
    # 정리와 동일 원칙). --out-dir만 예외인 이유는 모듈 docstring 참고.
    p = argparse.ArgumentParser(
        description="시너지 그룹(크기 2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, "
                     "추가로 넣음) + 2차 다중공선성 배제 + 정규화 통계 저장"
    )
    p.add_argument("--out-dir", default=None, dest="out_dir",
                    help="산출물 저장 위치(기본: results/) — run_pipeline.py가 Step 8 학습 "
                         "run 폴더로 넘길 때 씀. synergy-groups-json 자동 탐색 기준 폴더도 "
                         "이 값과 같다(synergy.py가 같은 폴더에 저장하므로).")
    return p.parse_args()


def _load_train_split(args) -> tuple:
    # build_datasets/get_segmenter/get_hi_cols_for_seg는 torch 의존 import라 모듈
    # 최상단에 안 둔다(analyze_hi_synergy.py와 동일 이유 — 모듈 docstring 위쪽 참고).
    # 2026-10-02: 같이 있던 `import copy`(torch와 무관)만 top-level로 올리고 이
    # 셋은 그대로 남김 — 기존 설계 의도를 그대로 유지.
    from datasets.segment_dataset import build_datasets
    from common.scenario import get_segmenter
    from utils.hi_schema import get_hi_cols_for_seg

    cfg = copy.deepcopy(P.P1_MODEL_CONFIG)
    cfg["data"]["data_dir"] = args.data_dir
    cfg["data"]["seg_data_dir"] = args.seg_data_dir
    cfg["data"]["datasets"] = args.datasets
    cfg["data"]["split_seed"] = args.split_seed

    axis_cfg = json.loads(args.axis_config)
    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()
    train_ds, _val_ds, _test_ds, _norm = build_datasets(cfg, spec=spec)

    x_all = (train_ds.x_hi * train_ds.nan_mask).numpy()  # NaN 위치는 0으로 (forward()와 동일 처리)
    y_all = train_ds.target.numpy()
    scen_idx_all = train_ds.scen_idx.numpy()
    names_by_seg = {s: get_hi_cols_for_seg(name) for s, name in enumerate(spec.scenario_names)}
    return x_all, y_all, scen_idx_all, spec, names_by_seg


def _fit_group_kernel(
    x_group: np.ndarray, y: np.ndarray, alpha: float, gamma: float | None,
    n_components: int, random_state: int,
):
    """Nystroem(RBF 근사) + Ridge. 시나리오당 표본이 수만~수십만 행이라 KernelRidge의
    O(n^2) 그람 행렬은 메모리가 안 감당된다(예: n=278186 -> 288GiB) — Nystroem이
    랜드마크 n_components개만 서브샘플해 커널을 저차원으로 근사한 뒤 그 위에서 선형
    회귀(Ridge)를 푸는, 대규모 데이터의 표준적인 RBF 커널 근사 방식. 개념(비선형 RBF
    조합)은 KernelRidge와 동일하고 .predict() 인터페이스도 동일하다."""

    n_components_eff = min(n_components, x_group.shape[0])
    model = make_pipeline(
        Nystroem(kernel="rbf", gamma=gamma, n_components=n_components_eff, random_state=random_state),
        Ridge(alpha=alpha),
    )
    model.fit(x_group, y)
    pred = model.predict(x_group)
    return model, float(r2_score(y, pred))


def _residualize(y: np.ndarray, conditioning: np.ndarray) -> np.ndarray:
    """synergy.py의 동명 함수와 동일 로직(중복 재구현이지만 두 스크립트가
    서로 import하는 관계가 아니라 독립 유지 — 로직이 5줄짜리라 모듈 결합보다 낫다고 판단)."""
    if conditioning.shape[1] == 0:
        return y
    A = np.column_stack([conditioning, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def _raw_conditioned_partial_corr(y: np.ndarray, kernel_pred: np.ndarray, x_group: np.ndarray) -> float:
    """v3 전용: 커널 예측값이 '자기 그룹의 raw 멤버로 이미 설명되는 부분'을 빼고도
    SOH와 관계가 남는지 검사. 낮으면(raw로 이미 설명됨) 커널이 raw 대비 새 정보를
    거의 안 준다는 뜻 — raw-커널 간 중복으로 간주해 후보에서 제외한다."""
    ry = _residualize(y, x_group)
    rk = _residualize(kernel_pred, x_group)
    if np.std(ry) < 1e-8 or np.std(rk) < 1e-8:
        return 0.0
    c = np.corrcoef(ry, rk)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _round_robin_select(
    kept: list[int], candidates: list[dict], cap: int,
) -> list[int]:
    """시나리오별 쿼터 라운드로빈으로 kept(다중공선성 배제를 통과한 후보 인덱스)에서
    최대 cap개를 고른다. 전역 train_r2 랭킹으로 한 번에 자르면 특정 시나리오의 그룹이
    전부 R^2가 낮아 최종본에 하나도 안 남을 수 있다(synergy.py로 어렵게 찾은
    그 시나리오 그룹 정보가 통째로 버려짐) — 시나리오마다 "남은 후보 중 최선" 하나씩
    돌아가며 채워 최소 floor(cap/n_scenarios)개는 보장한다."""
    by_scenario: dict[int, list[int]] = {}
    for i in kept:
        by_scenario.setdefault(candidates[i]["scenario_idx"], []).append(i)
    for s in by_scenario:
        by_scenario[s].sort(key=lambda i: -candidates[i]["train_r2"])

    selected: list[int] = []
    scenario_order = sorted(by_scenario.keys())
    while len(selected) < cap and any(by_scenario[s] for s in scenario_order):
        for s in scenario_order:
            if not by_scenario[s]:
                continue
            selected.append(by_scenario[s].pop(0))
            if len(selected) >= cap:
                break
    return selected


def k1_resolve_params_and_paths() -> SimpleNamespace:
    """1) 파라미터/경로 결정 — out_dir과 synergy_groups_json 경로 등을 parameters.py에서 해석."""
    args = _parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR

    # parameters.py에서 그대로 읽는 실행 파라미터 — --out-dir 외엔 전부 여기서 해석
    # (모듈 docstring 참고).
    split_seed = P.ACTIVE_SPLIT_SEED if P.ACTIVE_SPLIT_SEED is not None else P.FIXED_DEFAULT_SEED
    alpha = P.FIXED_KERNEL_ALPHA
    gamma = P.FIXED_KERNEL_GAMMA
    n_components = P.FIXED_KERNEL_N_COMPONENTS
    redundancy_threshold = P.ACTIVE_SYNERGY_REDUNDANCY_THRESHOLD
    max_features = P.FIXED_KERNEL_MAX_FEATURES
    min_raw_partial_corr = P.FIXED_MIN_RAW_PARTIAL_CORR
    combined_redundancy_threshold = P.FIXED_COMBINED_REDUNDANCY_THRESHOLD
    tag = P.FIXED_KERNEL_TAG or f"{P.ACTIVE_P1_TAG}_kernel"
    # synergy.py(Step 6)와 동일한 태그 파생 규칙 — run_pipeline.py가 Step 6~7에 같은
    # --out-dir(run_dir)을 넘겨주므로, 그 폴더의 synergy_groups_{synergy_tag}.json이
    # 곧 이번 실행의 입력이다(오버라이드하고 싶으면 FIXED_KERNEL_SYNERGY_GROUPS_JSON).
    synergy_tag = P.FIXED_SYNERGY_TAG or f"{P.ACTIVE_P1_TAG}_groups"
    synergy_groups_json = P.FIXED_KERNEL_SYNERGY_GROUPS_JSON or str(out_dir / f"synergy_groups_{synergy_tag}.json")

    return SimpleNamespace(
        out_dir=out_dir, split_seed=split_seed, alpha=alpha, gamma=gamma,
        n_components=n_components, redundancy_threshold=redundancy_threshold,
        max_features=max_features, min_raw_partial_corr=min_raw_partial_corr,
        combined_redundancy_threshold=combined_redundancy_threshold,
        tag=tag, synergy_groups_json=synergy_groups_json,
    )


def k2_load_data(params: SimpleNamespace) -> SimpleNamespace:
    """2) 데이터 로드 — train split(x_all/y_all/scen_idx_all/spec/names_by_seg) + raw HI 비용
    + synergy.py(Step 6) 산출물(groups_data)을 읽는다."""
    # _load_train_split은 SimpleNamespace를 인자로 받는다(interaction.py/synergy.py의
    # _load_all_scenarios 호출과 동일 패턴) — CLI가 사라졌으므로 필요한 필드만 구성.
    _loader_args = SimpleNamespace(
        data_dir=P.FIXED_CANONICAL_DATA_DIR, seg_data_dir=P.FIXED_CANONICAL_SEG_DATA_DIR,
        datasets=P.FIXED_CANONICAL_DATASETS, split_seed=params.split_seed,
        axis_config=json.dumps(P.ACTIVE_AXIS_CONFIG), seg_axis=P.FIXED_SEG_AXIS,
    )
    x_all, y_all, scen_idx_all, spec, names_by_seg = _load_train_split(_loader_args)
    # raw HI 카테고리 비용(stat/diff/lfp/morph) — seg 이름과 무관하게 순서/값이 동일하므로
    # 하나만 구해서 재사용(get_hi_cost_vector는 접두 seg별로 컬럼 "이름"만 다르고 카테고리
    # 순서/비용값 자체는 고정).
    raw_hi_costs = get_hi_cost_vector("dis_hi")
    groups_data = json.loads(Path(params.synergy_groups_json).read_text(encoding="utf-8"))

    return SimpleNamespace(
        x_all=x_all, y_all=y_all, scen_idx_all=scen_idx_all, spec=spec,
        names_by_seg=names_by_seg, raw_hi_costs=raw_hi_costs, groups_data=groups_data,
    )


def k3_fit_group_kernels(params: SimpleNamespace, data: SimpleNamespace) -> SimpleNamespace:
    """3) 그룹 -> 커널 HI 피팅 — 시나리오별 synergy 그룹(크기 2+)마다 Nystroem+Ridge를 학습시켜
    candidates를 만든다(크기1/raw-중복 그룹은 rejected에 기록만 하고 스킵)."""
    candidates: list[dict] = []
    rejected: list[dict] = []  # 어떤 HI 조합이 어느 단계에서 왜 탈락했는지(plot_kernel_rejected.py용)
    n_skipped_size1 = 0
    for s, seg_name in enumerate(tqdm(data.spec.scenario_names, desc="그룹 -> 커널 HI 피팅", unit="scenario")):
        key = f"seg_{s}_groups"
        if key not in data.groups_data:
            continue
        sel = data.scen_idx_all == s
        if sel.sum() < 20:
            tqdm_write(f"[kernel] {seg_name}: 표본 부족({int(sel.sum())}) — 스킵")
            continue
        x_scen, y_scen = data.x_all[sel], data.y_all[sel]

        groups = data.groups_data[key]
        n_fit = 0
        n_skipped_raw_dup = 0
        for gi, members in enumerate(groups):
            # v3.1(global_dedup) 산출물의 "attached"(사전 가지치기로 탈락해 이 그룹에
            # 사후 편입된 HI)는 여기서 일부러 안 쓴다 — attached는 대표와 항상 |raw
            # corr|>=0.9인 거의 동일 신호라 커널 입력에 추가해도 새 정보가 거의 없는 반면,
            # 그룹마다 커널 입력 폭이 2~29개로 들쭉날쭉해져 (a) RBF 커널이 서로 거의
            # 동일한 컬럼 수십 개로 학습되는 통계적으로 불안정한 상황을 만들고 (b) Level2
            # gap 비교 시 그룹 크기 자체가 교란변수가 된다(실측: 재비교에서 확인됨). attached는
            # groups json에 "이 그룹 소속"이라는 라벨로만 남아 다중공선성 배제 보장(0건)엔
            # 그대로 기여한다 — raw HI 커버리지도 x_hi가 이미 64개 전부 독립적으로 갖고
            # 있어 손실이 없다(커널 융합은 대체가 아니라 추가이므로).
            if len(members) < 2:
                # 크기 1 그룹 = raw HI가 이미 x_hi에 그대로 있으므로 커널 융합 대상에서
                # 제외(안 그러면 자기 자신의 단조 변환에 가까운 슬롯만 하나 더 늘어남).
                n_skipped_size1 += 1
                continue
            x_group = x_scen[:, members]
            model, r2 = _fit_group_kernel(
                x_group, y_scen, params.alpha, params.gamma, params.n_components, params.split_seed,
            )
            if params.min_raw_partial_corr is not None:
                kernel_pred = model.predict(x_group)
                pc = _raw_conditioned_partial_corr(y_scen, kernel_pred, x_group)
                if abs(pc) < params.min_raw_partial_corr:
                    n_skipped_raw_dup += 1
                    rejected.append({
                        "name": f"kernel_{seg_name}_g{gi}", "scenario": seg_name,
                        "member_names": [data.names_by_seg[s][i] for i in members],
                        "train_r2": r2, "reason": "raw_conditioned_filter",
                        "detail": f"raw-conditioned partial corr {pc:.4f} < 문턱 {params.min_raw_partial_corr}"
                                  "(자기 그룹 raw 멤버로 이미 설명됨, 커널이 새 정보를 거의 안 줌)",
                    })
                    continue
            member_cost = float(np.mean([data.raw_hi_costs[m] for m in members]))
            candidates.append({
                "name": f"kernel_{seg_name}_g{gi}",
                "scenario": seg_name,
                "scenario_idx": s,
                "members": [int(m) for m in members],
                "member_names": [data.names_by_seg[s][i] for i in members],
                "model": model,
                "train_r2": r2,
                "cost": member_cost,  # L0 페널티용: 멤버 raw HI 카테고리 비용의 평균(scr_loss.py)
            })
            n_fit += 1
        extra = f", raw-중복 {n_skipped_raw_dup}개 탈락" if params.min_raw_partial_corr is not None else ""
        tqdm_write(f"[kernel] {seg_name}: 그룹 {len(groups)}개(크기1 {len(groups) - n_fit - n_skipped_raw_dup}개 스킵{extra}) "
                    f"-> 커널 HI {n_fit}개 피팅 완료")

    if not candidates:
        raise RuntimeError("생성된 커널 후보가 없습니다 — synergy-groups-json 내용을 확인하세요.")

    return SimpleNamespace(candidates=candidates, rejected=rejected, n_skipped_size1=n_skipped_size1)


def k4_dedupe_kernels(params: SimpleNamespace, data: SimpleNamespace, fit: SimpleNamespace) -> SimpleNamespace:
    """4) 2차 다중공선성 배제 — 전체 train(모든 시나리오 pooled)에서 커널 값끼리 상관을 계산해
    임계값 이상이면 train_r2가 낮은 쪽을 fit.rejected로 보낸다(그룹 내부 중복은 synergy.py가
    이미 걸렀지만, 시나리오가 다른 그룹끼리는 원본 HI가 겹치면 커널 변환 후에도 비슷한 값이
    나올 수 있어 여기서 다시 검사)."""
    kernel_vals = np.zeros((data.x_all.shape[0], len(fit.candidates)), dtype=np.float64)
    for j, c in enumerate(fit.candidates):
        kernel_vals[:, j] = c["model"].predict(data.x_all[:, c["members"]])

    corr = np.corrcoef(kernel_vals, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    order = sorted(range(len(fit.candidates)), key=lambda i: -fit.candidates[i]["train_r2"])
    kept: list[int] = []
    for i in order:
        conflict = [k for k in kept if abs(corr[i, k]) >= params.redundancy_threshold]
        if not conflict:
            kept.append(i)
        else:
            worst = max(conflict, key=lambda k: abs(corr[i, k]))
            c = fit.candidates[i]
            fit.rejected.append({
                "name": c["name"], "scenario": c["scenario"], "member_names": c["member_names"],
                "train_r2": c["train_r2"], "reason": "kernel_kernel_dedup",
                "detail": f"커널끼리 |corr|={abs(corr[i, worst]):.4f}>={params.redundancy_threshold} "
                          f"vs 이미 채택된 {fit.candidates[worst]['name']}({fit.candidates[worst]['member_names']})",
            })
    n_dropped_corr = len(fit.candidates) - len(kept)

    return SimpleNamespace(kernel_vals=kernel_vals, kept=kept, n_dropped_corr=n_dropped_corr)


def k5_apply_feature_cap(params: SimpleNamespace, fit: SimpleNamespace, dedup: SimpleNamespace) -> SimpleNamespace:
    """5) max-features 캡 적용 — 지정됐으면 시나리오별 라운드로빈으로 상한까지만 남기고
    나머지는 fit.rejected에 쿼터초과로 기록한다(안 주면 다중공선성 배제를 통과한 건 전부 유지)."""
    cap = params.max_features if params.max_features is not None else len(dedup.kept)
    final_idx = _round_robin_select(dedup.kept, fit.candidates, cap)
    n_dropped_cap = max(0, len(dedup.kept) - len(final_idx))
    final = [fit.candidates[i] for i in final_idx]
    for i in dedup.kept:
        if i not in final_idx:
            c = fit.candidates[i]
            fit.rejected.append({
                "name": c["name"], "scenario": c["scenario"], "member_names": c["member_names"],
                "train_r2": c["train_r2"], "reason": "max_features_quota",
                "detail": f"--max-features {params.max_features} 쿼터 초과(다중공선성 배제는 통과)",
            })

    return SimpleNamespace(final=final, final_idx=final_idx, n_dropped_cap=n_dropped_cap)


def k6_compute_normalization(data: SimpleNamespace, dedup: SimpleNamespace, cap: SimpleNamespace) -> dict:
    """6) 정규화 통계 계산 — 커널 예측값(SOH 스케일)을 원래 HI의 z-score 스케일에 맞추기 위해,
    최종 채택된 커널 HI마다 own-scenario 분포 기준 mean/std를 구해 cap.final에 제자리로 채운다."""
    # 커널 예측값은 SOH를 직접 예측하도록 fit돼 스케일이 SOH 자체(대략 0.7~1.05)를 따른다.
    # 원래 x_hi는 z-score(평균0/표준편차1)라 스케일이 전혀 다름 — 여기서 train 기준
    # mean/std를 구해 저장해두고, 적용 시점(train.py)에서 (v - mean) / std로 표준화한다
    # (val/test엔 이 train 통계를 그대로 적용, fit은 안 함 — 누수 없음).
    # 2026-09-18(v4 로직 수정, 요구사항1 "각 시나리오에서 만든 커널만 그 시나리오에서
    # 사용"과 일관성): mean/std는 이 커널이 *실제로 쓰이는* own-scenario 행만으로 계산한다.
    # kernel_vals는 전체 x_all(모든 시나리오 pooled)에 predict()한 값이라 -- k4의 2차
    # 다중공선성 배제(커널끼리, pooled 비교) 판단에는 그대로 쓰지만, 정규화는 own-scenario
    # 분포를 반영해야 한다. 안 그러면 "전혀 다른 시나리오들이 뒤섞인 분포" 기준으로
    # z-score해서 실제 사용될 own-scenario 값이 이상하게 치우친 스케일로 들어간다.
    final_idx_arr = np.array(cap.final_idx)
    final_vals = dedup.kernel_vals[:, final_idx_arr]
    means = np.zeros(len(cap.final))
    stds = np.ones(len(cap.final))
    for k, f in enumerate(cap.final):
        own_vals = final_vals[data.scen_idx_all == f["scenario_idx"], k]
        means[k] = float(own_vals.mean())
        sd = float(own_vals.std())
        stds[k] = sd if sd > 1e-8 else 1.0  # 상수에 가까운 피처 0-division 방지
    for f, m, sd in zip(cap.final, means, stds):
        f["mean"] = float(m)
        f["std"] = float(sd)

    n_final_by_scenario = {
        data.spec.scenario_names[s]: sum(1 for f in cap.final if f["scenario_idx"] == s)
        for s in range(data.spec.n_scenarios)
    }
    return n_final_by_scenario


def k7_build_combined_redundancy(params: SimpleNamespace, data: SimpleNamespace, cap: SimpleNamespace) -> dict:
    """7) 3차 결합(raw+kernel) 다중공선성 배제 — 시나리오별로(2차와 달리 pooled 아님, 그
    시나리오 데이터에서만) raw HI(64)+이 시나리오가 만든 커널 HI를 합쳐 |r|>=threshold 쌍의
    '패자'를 기록한다(실제 마스킹 적용은 train.py --combined-redundancy-json이 담당,
    scr_model.py는 무변경 — 여긴 기록만 한다)."""
    # 2026-09-18, 요구사항2. |r|>=0.95인 각 쌍에 대해 "패자"를 정해 제거한다(사용자 피드백,
    # 2026-09-18 정정 — 처음엔 "쌍이 있으면 둘 다 제거"였는데, 그러면 서로 얽힌 쌍이 많을수록
    # 무차별로 다 날아가 버려서 아래 규칙으로 변경):
    #   1) 다른 HI와도 |r|>=0.95인 관계 개수(= 이 컴포넌트 안에서의 degree)가 더 많은 쪽을
    #      제거(더 많이 겹치는 쪽이 더 중복도가 높다고 보고 우선 정리).
    #   2) degree가 같으면, 타깃(SOH)과의 단순상관 |target_corr|가 더 낮은 쪽을 제거
    #      ("자체 상관계수가 더 높은 HI를 살려").
    #   3) 그래도 같으면(초저확률) 인덱스가 더 큰 쪽을 제거(결정성 확보용 임의 규칙).
    COMBINED_REDUNDANCY_THRESHOLD = params.combined_redundancy_threshold  # 기본 0.95,
        # fig3(hi_design_rationale)/redundancy_gate_resolution.py와 동일 관례 —
        # 2026-09-19부터 --combined-redundancy-threshold로 CLI 노출(예전엔 하드코딩)
    n_raw = data.x_all.shape[1]
    combined_redundancy: dict[str, dict] = {}
    for s, seg_name in enumerate(data.spec.scenario_names):
        sel = data.scen_idx_all == s
        if sel.sum() < 20:
            continue
        own_feats = [f for f in cap.final if f["scenario_idx"] == s]
        x_raw_scen = data.x_all[sel]
        y_scen = data.y_all[sel]
        if own_feats:
            x_kernel_scen = np.column_stack([
                f["model"].predict(x_raw_scen[:, f["members"]]) for f in own_feats
            ])
            combined = np.column_stack([x_raw_scen, x_kernel_scen])
        else:
            combined = x_raw_scen
        combined_names = list(data.names_by_seg[s]) + [f["name"] for f in own_feats]

        corr = np.corrcoef(combined, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        target_corr = np.array([
            np.corrcoef(combined[:, k], y_scen)[0, 1] for k in range(combined.shape[1])
        ])
        target_corr = np.nan_to_num(target_corr, nan=0.0)

        n = combined.shape[1]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if abs(corr[i, j]) >= COMBINED_REDUNDANCY_THRESHOLD]
        degree = np.zeros(n, dtype=int)
        for i, j in edges:
            degree[i] += 1
            degree[j] += 1

        removed_set: set[int] = set()
        for i, j in edges:
            if degree[i] != degree[j]:
                loser = i if degree[i] > degree[j] else j
            elif abs(target_corr[i]) != abs(target_corr[j]):
                loser = i if abs(target_corr[i]) < abs(target_corr[j]) else j
            else:
                loser = max(i, j)
            removed_set.add(loser)
        removed = sorted(removed_set)

        removed_raw_idx = [i for i in removed if i < n_raw]
        removed_kernel_names = [combined_names[i] for i in removed if i >= n_raw]
        combined_redundancy[seg_name] = {
            "removed_raw_idx": removed_raw_idx,
            "removed_raw_names": [combined_names[i] for i in removed_raw_idx],
            "removed_kernel_names": removed_kernel_names,
            "n_total_checked": combined.shape[1],
            "n_edges": len(edges),
        }

    return combined_redundancy


def k8_save_results(
    params: SimpleNamespace, fit: SimpleNamespace, dedup: SimpleNamespace,
    cap: SimpleNamespace, n_final_by_scenario: dict, combined_redundancy: dict,
) -> None:
    """8) 저장 — combined_redundancy json + kernel pkl 저장, 콘솔 요약 출력, RESULTS_LOG 기록."""
    params.out_dir.mkdir(parents=True, exist_ok=True)
    combined_redundancy_out_path = (
        params.out_dir / f"kernel_group_features_{params.tag}_combined_redundancy.json"
    )
    combined_redundancy_out_path.write_text(
        json.dumps({"tag": params.tag, "threshold": params.combined_redundancy_threshold,
                    "by_scenario": combined_redundancy}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    n_removed_raw_total = sum(len(v["removed_raw_idx"]) for v in combined_redundancy.values())
    n_removed_kernel_total = sum(len(v["removed_kernel_names"]) for v in combined_redundancy.values())
    print(f"[kernel] 결합(raw+kernel) 다중공선성 배제(시나리오별, |r|>={params.combined_redundancy_threshold}): "
          f"raw {n_removed_raw_total}개/kernel {n_removed_kernel_total}개 제거 대상 "
          f"-> {combined_redundancy_out_path}")

    out_path = params.out_dir / f"kernel_group_features_{params.tag}.pkl"
    artifact = {
        "tag": params.tag,
        "n_features": len(cap.final),
        "alpha": params.alpha,
        "gamma": params.gamma,
        "n_components": params.n_components,
        "redundancy_threshold": params.redundancy_threshold,
        "max_features": params.max_features,
        "min_raw_partial_corr": params.min_raw_partial_corr,
        "features": [
            {k: v for k, v in f.items() if k != "scenario_idx"} for f in cap.final
        ],
    }
    with open(out_path, "wb") as fh:
        pickle.dump(artifact, fh)

    # 2026-09-21: 탈락 목록(raw_conditioned_filter/kernel_kernel_dedup/max_features_quota
    # 사유별) 파일 저장은 제거 — 원래 읽던 plot_kernel_rejected.py가 이미 삭제돼 아무도
    # 이 파일을 다시 읽지 않았다(파이프라인 실행 전 정리 커밋에서 함께 삭제됨). 개수
    # 요약만 콘솔에 남기고 `rejected` 리스트 자체(사유 상세)는 저장하지 않는다.

    avg_r2 = float(np.mean([f["train_r2"] for f in cap.final]))
    print(f"\n[kernel] 후보 {len(fit.candidates)}개(크기1 그룹 {fit.n_skipped_size1}개 스킵) "
          f"-> 2차 다중공선성 배제로 {dedup.n_dropped_corr}개 제거 "
          f"-> {'상한(' + str(params.max_features) + ')으로 ' + str(cap.n_dropped_cap) + '개 추가 제거 -> ' if params.max_features else ''}"
          f"최종 {len(cap.final)}개, 평균 train R^2={avg_r2:.4f}")
    print("[kernel] 시나리오별 최종 커널 HI 개수: " +
          ", ".join(f"{k}={v}" for k, v in n_final_by_scenario.items()))
    print(f"[kernel] 저장: {out_path}")
    print(f"[kernel] 탈락 {len(fit.rejected)}개 (파일 저장 안 함 — 콘솔 요약만)")

    append_log_entry(
        tag=f"kernel_group_features_{params.tag}",
        purpose="시너지 그룹(크기2+)을 RBF 커널로 그룹당 1개 HI로 융합(raw HI는 유지, 추가) "
                "+ 2차 다중공선성 배제 + 정규화 통계 저장 + 3차 결합(raw+kernel) 다중공선성 "
                "배제(시나리오별, 2026-09-18 신규)",
        command=current_command_str(),
        result_files=[str(out_path), str(combined_redundancy_out_path)],
        key_metrics=(f"후보 {len(fit.candidates)}개 -> 최종 {len(cap.final)}개, "
                     f"평균 train R^2={avg_r2:.4f}, 시나리오별 개수={n_final_by_scenario}, "
                     f"결합 다중공선성 배제: raw {n_removed_raw_total}개/kernel {n_removed_kernel_total}개"),
        interpretation=(
            "이 pkl은 train.py --kernel-features-pkl로 넘기면 x_hi(raw HI)는 그대로 "
            "두고 x_kernel(정규화된 커널 융합값)을 별도 게이트(scen_kernel_gates)로 추가한다 "
            "— raw HI와 커널 HI를 동시에 쓰는 게 목적. 평균 train R^2가 각 그룹 멤버 HI 개별 "
            "상관보다 뚜렷이 높다면 비선형 시너지가 실제로 존재한다는 신호. "
            "_combined_redundancy.json은 train.py --combined-redundancy-json으로 "
            "넘기면 시나리오별로 raw+kernel 통틀어 |r|>=0.95인 HI를 전부 그 시나리오에서 배제한다."
        ),
    )


def main() -> None:
    # 1) 파라미터/경로 결정
    params = k1_resolve_params_and_paths()  
    
    # 2) train split + raw HI 비용 + synergy 그룹 로드
    data = k2_load_data(params)  
    
    # 3) 그룹 -> 커널 HI 피팅
    fit = k3_fit_group_kernels(params, data)  
    
    # 4) 2차 배제 — 커널끼리 pooled 상관
    dedup = k4_dedupe_kernels(params, data, fit)
    
    # 5) max-features 캡(시나리오별 라운드로빈) 
    cap = k5_apply_feature_cap(params, fit, dedup) 
    
    # 6) own-scenario 정규화 통계
    n_final_by_scenario = k6_compute_normalization(data, dedup, cap)  
    
    # 7) 3차 배제 — raw+kernel 결합, 시나리오별
    combined_redundancy = k7_build_combined_redundancy(params, data, cap) 
    
    # 8) 저장 + 로그
    k8_save_results(params, fit, dedup, cap, n_final_by_scenario, combined_redundancy)  


if __name__ == "__main__":
    main()
