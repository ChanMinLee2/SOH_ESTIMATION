"""
5_model/experiments/phase1_lab/plot_hi_ridge3d.py

HI 원본값(z) x Cycle(x) x 시나리오(y) 3D 곡면 플랏 — plot_soh_ridge3d.py와 같은 렌더링
(fine grid + gaussian smoothing + plot_surface)을 재사용하되, z축을 SOH/오차가 아니라
HI 원본값(정규화 해제된 실제 스케일)으로 바꾼 버전.

64개 HI 중 카테고리(STAT/DIFF/LFP/MORPH)별 2개씩 총 8개를 기본값으로 그린다 — 어떤 HI를
쓸지는 --hi "category:concept" 인자로 자유롭게 바꿀 수 있다.

데이터 출처: 모델 체크포인트가 필요 없다 — run_dir의 config.yaml로 데이터만 재구성해서
SegmentDataset의 정규화된 x_hi를 SegmentNormalizer.mean_/std_로 역변환해 원본 스케일
HI 값을 얻는다(원래 NaN이었던 값은 다시 NaN으로 복원해 스무딩 시 실제 관측치만 쓰도록 함).

사용 예:
  python 5_model/experiments/phase1_lab/plot_hi_ridge3d.py \
      --run-dir 5_model/experiments/phase1_lab/results/p1v2_runs/0827_1705_p1v2_p1v4_full_seed42 \
      --data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/cycle" \
      --seg-data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200/seg" \
      --out-dir 5_model/experiments/phase1_lab/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

import numpy as np  # noqa: E402

from utils.io_utils import load_config  # noqa: E402
from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402
from datasets.segment_dataset import build_datasets  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

from plot_soh_ridge3d import plot_ridge_surface3d, SCENARIO_ORDER  # noqa: E402

# 카테고리별 2개씩, 총 8개 — 물리적으로 알아보기 쉬운 것 위주로 선택(--select default).
DEFAULT_HI_PICKS: dict[str, list[str]] = {
    "stat":  ["v_mean_cw", "v_std"],
    "diff":  ["dqdv_peak_h", "dqdv_area"],
    "lfp":   ["plateau_frac", "inflect_v"],
    "morph": ["vt_dtw", "vq_dtw"],
}


def _rank_by_scenario_deviation(
    x_raw: "np.ndarray", seg_names: "np.ndarray", concept_names: list[str],
) -> dict[str, float]:
    """concept -> 시나리오간 편차 점수 = std(시나리오별 평균) / std(전체 유효값).
    스케일이 완전히 다른 HI끼리(예: 0~1 비율 vs 수백 단위 통계량) 비교 가능하게
    "그 HI 자체의 전체 변동폭 대비 시나리오간 평균이 얼마나 벌어지는가"로 정규화한
    scale-free 지표(ANOVA의 between/total 분산비와 같은 발상)."""
    scores: dict[str, float] = {}
    for idx, concept in enumerate(concept_names):
        col = x_raw[:, idx]
        valid = ~np.isnan(col)
        if not valid.any():
            scores[concept] = float("nan")
            continue
        total_std = float(np.std(col[valid]))
        if total_std < 1e-12:
            scores[concept] = 0.0
            continue
        scen_means = []
        for scen in SCENARIO_ORDER:
            m = valid & (seg_names == scen)
            if m.any():
                scen_means.append(float(np.mean(col[m])))
        if len(scen_means) < 2:
            scores[concept] = float("nan")
            continue
        scores[concept] = float(np.std(scen_means)) / total_std
    return scores


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HI 원본값 x Cycle x 시나리오 3D 곡면 플랏(카테고리당 2개, 기본 8개)"
    )
    p.add_argument("--run-dir", required=True, dest="run_dir",
                   help="results/p1v2_runs/<run> — config.yaml만 씀(체크포인트/모델 불필요)")
    p.add_argument("--data-dir", default=None, dest="data_dir")
    p.add_argument("--seg-data-dir", default=None, dest="seg_data_dir")
    p.add_argument("--hi", nargs="+", default=None,
                   help="'category:concept' 형식으로 직접 지정(예: stat:v_std). "
                        "미지정 시 --select 기준으로 자동 선정")
    p.add_argument("--select", default="default", choices=["default", "variance"],
                   help="--hi 미지정 시 선정 기준. default=미리 정한 8개(하드코딩). "
                        "variance=카테고리별로 시나리오간 편차(scale-free) 최대/최소 "
                        "HI를 각각 1개씩 자동 선정(카테고리당 2개, 총 8개)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-bins", type=int, default=60, dest="n_bins")
    p.add_argument("--elev", type=float, default=20)
    p.add_argument("--azim", type=float, default=-45)
    p.add_argument("--out-dir", required=True, dest="out_dir")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    cfg = load_config(str(run_dir / "config.yaml"))
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
        print(f"[plot_hi_ridge3d] data_dir 오버라이드: {args.data_dir}")
    if args.seg_data_dir is not None:
        cfg["data"]["seg_data_dir"] = args.seg_data_dir
        print(f"[plot_hi_ridge3d] seg_data_dir 오버라이드: {args.seg_data_dir}")

    spec = get_segmenter(
        cfg["scenario"]["axis"], {cfg["scenario"]["axis"]: cfg["scenario"]["axis_config"]}
    ).get_spec()
    train_ds, val_ds, test_ds, norm = build_datasets(cfg, spec=spec)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    ref_seg = spec.scenario_names[0]
    suffix = f"_{ref_seg}"
    concept_names = [c[: -len(suffix)] if c.endswith(suffix) else c
                     for c in get_hi_cols_for_seg(ref_seg)]

    x_norm = ds.x_hi.numpy().astype(np.float64)          # (N, N_HI) z-scored
    nan_mask = ds.nan_mask.numpy().astype(bool)           # (N, N_HI) True=valid
    x_raw = x_norm * norm.std_ + norm.mean_                # 역변환(원본 스케일 복원)
    x_raw[~nan_mask] = np.nan                               # 원래 결측이던 값은 다시 NaN으로

    cycles = np.asarray(ds.cycles, dtype=float)
    seg_names = np.asarray(ds.seg_names)
    missing_scen = set(SCENARIO_ORDER) - set(seg_names.tolist())
    if missing_scen:
        raise ValueError(f"seg_name에 없는 시나리오 {missing_scen} (필요: {SCENARIO_ORDER})")

    tags: dict[tuple[str, str], str] = {}
    if args.hi:
        picks: list[tuple[str, str]] = []
        for item in args.hi:
            if ":" not in item:
                raise ValueError(f"--hi 항목은 'category:concept' 형식이어야 함: {item!r}")
            cat, concept = item.split(":", 1)
            picks.append((cat, concept))
    elif args.select == "variance":
        scores = _rank_by_scenario_deviation(x_raw, seg_names, concept_names)
        by_cat: dict[str, list[str]] = {}
        for concept in concept_names:
            cat = concept.split("_", 1)[0]
            by_cat.setdefault(cat, []).append(concept)
        picks = []
        print("[plot_hi_ridge3d] 카테고리별 시나리오간 편차(scale-free) 최대/최소 선정:")
        for cat, concepts_in_cat in by_cat.items():
            valid_concepts = [c for c in concepts_in_cat if not np.isnan(scores[c])]
            ranked = sorted(valid_concepts, key=lambda c: scores[c])
            worst, best = ranked[0], ranked[-1]  # 최소(가장 평평) / 최대(가장 요동)
            # concept_names는 이미 "{cat}_"로 시작하므로 접두어를 다시 안 붙이게 그대로 씀
            worst_short = worst[len(f"{cat}_"):]
            best_short  = best[len(f"{cat}_"):]
            print(f"  {cat}: max={best_short}({scores[best]:.3f})  min={worst_short}({scores[worst]:.3f})")
            picks.append((cat, best_short))
            tags[(cat, best_short)] = "maxvar"
            picks.append((cat, worst_short))
            tags[(cat, worst_short)] = "minvar"
    else:
        picks = [(cat, name) for cat, names in DEFAULT_HI_PICKS.items() for name in names]

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat, concept in picks:
        # concept_names는 category 접두어가 붙은 채로 저장돼 있음(예: "stat_v_mean_cw") —
        # native seg df 컬럼이 {cat}_{key} 형태라 세그먼트 접미어만 떼면 이 형태가 남는다.
        full_concept = concept if concept.startswith(f"{cat}_") else f"{cat}_{concept}"
        if full_concept not in concept_names:
            raise ValueError(f"{full_concept!r}가 HI 목록에 없음(leak 제외 설정 확인) — 후보: {concept_names}")
        idx = concept_names.index(full_concept)

        ridges = []
        for scen in SCENARIO_ORDER:
            m = seg_names == scen
            c = cycles[m]
            v = x_raw[m, idx]
            valid = ~np.isnan(v)
            ridges.append((scen, c[valid], v[valid]))

        tag = tags.get((cat, concept))
        fname = f"hi_ridge3d_scenario_{cat}_{tag}_{concept}.png" if tag \
            else f"hi_ridge3d_scenario_{cat}_{concept}.png"
        out_path = out_dir / fname
        plot_ridge_surface3d(
            ridges, value_label=f"{cat.upper()}: {concept}", ridge_axis_label="scenario",
            title=f"{concept} ({cat.upper()}, {tag or ''}) surface — scenario x Cycle ({run_dir.name})",
            out_path=out_path, n_bins=args.n_bins, cmap_name="viridis",
            elev=args.elev, azim=args.azim,
        )


if __name__ == "__main__":
    main()
