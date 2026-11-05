"""
docs/figures/fig7_synergy_cache.py — Figure 7(synergy gain) 데이터 캐시.

2026-11-04: 메인 단일 데이터셋 run(1031_0117 MIT / 1031_0431 HUST / 1031_0938 TJU)의
synergy 그룹과 저장된 커널 HI 모델(kernel_group_features_*_kernel.pkl의 Nystroem+Ridge)을
그대로 다시 불러, 그룹(크기 2 이상)마다
  - 멤버 raw HI 각각의 |r(HI, SOH)|           (train, test)
  - 커널 HI(=저장된 모델의 predict)의 |r(kernel, SOH)| (train, test)
  - 같은 멤버의 선형 결합(train 최소제곱)의 |r(linear, SOH)| (train, test)
  - 그룹 성장 시 기록된 점수(첫 멤버=단순상관, 이후=편상관; synergy_groups json)
  - 커널 게이트 확률(gates/regression_kernel_HIs.json, 최종 선택 여부)
를 계산해 _cache/fig7_synergy_cache.json에 저장한다. 그림은 fig7_synergy.py가 그린다.

kernel pkl에는 1차 필터(raw-conditioned partial corr >= 0.10)를 통과한 커널만 있다 — 그
필터에서 탈락한 그룹은 여기서 같은 Nystroem+Ridge 설정으로 다시 피팅해 "dropped"로 표시한다.

torch/sklearn 의존이라 학습 환경 파이썬으로 실행:
    C:\\Users\\ksshin\\.conda\\envs\\LFP_SOH_ESTIMATION\\python.exe docs/figures/fig7_synergy_cache.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline

from datasets.segment_dataset import build_datasets
from common.scenario import get_segmenter
from utils.hi_schema import get_hi_cols_for_seg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "model_lib" / "results" / "p1v2_runs"
OUT = Path(__file__).resolve().parent / "_cache" / "fig7_synergy_cache.json"

RUNS = {
    "MIT": "1031_0117_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_mit_seed42",
    "HUST": "1031_0431_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_hust_seed42",
    "TJU": "1031_0938_p1v2_hi63_scen_trm_union_nowarm_l0-2e-4_n2-20_tju_seed42",
}


def _r(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


# x_hi는 66열(누설 HI 3개 포함)이지만 synergy/kernel의 멤버 인덱스는 이 3개를 뺀 63열 기준이다
# (seed 단순상관이 synergy json의 첫 점수와 일치하는 것으로 확인).
LEAK_HIS = ("stat_q_abs", "stat_energy_seg", "diff_dqdv_area")


def _xy(ds, keep: np.ndarray):
    x = (ds.x_hi * ds.nan_mask).numpy()[:, keep]   # kernel.py::_load_train_split와 동일 처리
    return x, ds.target.numpy(), ds.scen_idx.numpy()


def one_run(name: str, run_dir: Path) -> list[dict]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    axis = cfg["scenario"]["axis"]
    spec = get_segmenter(axis, {axis: cfg["scenario"]["axis_config"]}).get_spec()
    train_ds, _val, test_ds, _norm = build_datasets(cfg, spec=spec)
    cols = get_hi_cols_for_seg(spec.scenario_names[0])
    keep = np.array([i for i, c in enumerate(cols)
                     if c[: -len(spec.scenario_names[0]) - 1] not in LEAK_HIS])
    assert len(keep) == 63, len(keep)
    xtr, ytr, str_ = _xy(train_ds, keep)
    xte, yte, ste = _xy(test_ds, keep)

    groups = json.loads(next(run_dir.glob("synergy_groups_*.json")).read_text(encoding="utf-8"))
    kpkl = pickle.load(open(next(run_dir.glob("kernel_group_features_*_kernel.pkl")), "rb"))
    kernels = {f["name"]: f for f in kpkl["features"]}
    alpha, gamma, n_comp = kpkl["alpha"], kpkl["gamma"], kpkl["n_components"]
    comb = json.loads(next(run_dir.glob("*_combined_redundancy.json")).read_text(encoding="utf-8"))
    gk = json.loads((run_dir / "gates" / "regression_kernel_HIs.json").read_text(encoding="utf-8"))
    gate_prob = {}
    k = 0
    while f"seg_{k}_names" in gk:
        for n, p in zip(gk[f"seg_{k}_names"], gk[f"seg_{k}_probs"]):
            gate_prob[n] = float(p)
        k += 1

    rows = []
    for s, seg in enumerate(spec.scenario_names):
        key = f"seg_{s}_groups"
        if key not in groups:
            continue
        mtr, mte = str_ == s, ste == s
        removed_k = set(comb["by_scenario"].get(seg, {}).get("removed_kernel_names", []))
        for gi, (members, names, scores) in enumerate(zip(groups[key], groups[f"seg_{s}_group_names"],
                                                          groups[f"seg_{s}_group_scores"])):
            if len(members) < 2:
                continue
            kname = f"kernel_{seg}_g{gi}"
            if kname in kernels:
                model, status = kernels[kname]["model"], "kept"
                if kname in removed_k:
                    status = "removed_combined"
            else:
                model = make_pipeline(Nystroem(kernel="rbf", gamma=gamma, n_components=n_comp,
                                               random_state=cfg["data"].get("split_seed", 42)),
                                      Ridge(alpha=alpha))
                model.fit(xtr[mtr][:, members], ytr[mtr])
                status = "dropped_partial"
            ktr = model.predict(xtr[mtr][:, members])
            kte = model.predict(xte[mte][:, members])
            # 같은 멤버의 선형 결합(최소제곱) — 커널(비선형 융합)이 선형 결합보다 더 주는지 비교용
            lin = LinearRegression().fit(xtr[mtr][:, members], ytr[mtr])
            ltr = lin.predict(xtr[mtr][:, members])
            lte = lin.predict(xte[mte][:, members])
            rows.append({
                "dataset": name, "scenario": seg, "kernel": kname, "status": status,
                "gate_prob": gate_prob.get(kname),
                "members": [nm[: -len(seg) - 1] for nm in names],
                "growth_scores": scores,
                "member_r_train": [_r(xtr[mtr][:, m], ytr[mtr]) for m in members],
                "member_r_test": [_r(xte[mte][:, m], yte[mte]) for m in members],
                "kernel_r_train": _r(ktr, ytr[mtr]),
                "kernel_r_test": _r(kte, yte[mte]),
                "linear_r_train": _r(ltr, ytr[mtr]),
                "linear_r_test": _r(lte, yte[mte]),
                "n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                "train_r2_saved": kernels[kname]["train_r2"] if kname in kernels else None,
            })
            # 정합성 확인: seed 단순상관 == synergy json 첫 점수
            assert abs(rows[-1]["member_r_train"][0] - scores[0]) < 1e-3, (kname, rows[-1]["member_r_train"][0], scores[0])
    print(f"[{name}] {len(rows)} multi-member groups "
          f"({sum(r['status'] == 'kept' for r in rows)} kept)")
    return rows


def main():
    rows = []
    for name, run in RUNS.items():
        rows += one_run(name, RUNS_DIR / run)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
