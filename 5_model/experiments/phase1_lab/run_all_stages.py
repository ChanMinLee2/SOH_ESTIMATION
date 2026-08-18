"""
5_model/experiments/phase1_lab/run_all_stages.py

Phase1 독립검증 로드맵 Stage 0~6을 한 번에 실행하는 마스터 오케스트레이터
(docs/PHASE1_REDESIGN.md §3 참고).

    Stage0  현재 구조(train_scr.py) 기준선 다중시드 수렴성 측정
    Stage1+2  phase1_trainer_v2.py(체크포인트 기준 변경 + temperature annealing)
              로 같은 seed들 재실행 -> 수렴성 재측정 -> Stage0 대비 개선 여부 비교
    Stage3  Stage1+2 결과를 다중시드 앙상블(Borda count)해 "가짜 Phase1 run
            디렉터리"로 합성 (Phase2 --gates-from 그대로 사용 가능)
    Stage4  HI 상관 클러스터링으로 "남은 불안정성이 진짜인지" 재해석
    Stage5  시나리오별 HI 조합 시너지 스크리닝 (MI + greedy + ablation)
    Stage6  = Stage3 산출물(합성 gates 디렉터리)이 최종 "확정 산출물" — 경로만 재출력

각 단계는 이미 만들어진 독립 스크립트를 subprocess로 그대로 호출한다(이 파일
자체는 오케스트레이션만 담당, 로직 중복 없음). 각 하위 스크립트가 실행 끝에
docs/phase1_lab/RESULTS_LOG.md에 자기 결과를 자동으로 append하므로, 이 파일은
전체 실행에 대한 요약 엔트리 하나만 마지막에 추가한다.

사용 예:
  python 5_model/experiments/phase1_lab/run_all_stages.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-4_lag-0_noise-3%_ou-200/cycle" \
      --seg-data-dir "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-4_lag-0_noise-3%_ou-200/seg" \
      --scen-k 25 --seeds 42 0 123 7 2024 --beta-min 0.1 \
      --k-values 5 15 25 --synergy-k 15 --tag k25_full
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LAB_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LAB_DIR / "results"

sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tqdm_utils import tqdm, write as tqdm_write  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402
from log_utils import append_log_entry, current_command_str  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 Stage0~6 통합 검증 오케스트레이터")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True)
    p.add_argument("--data-dir", required=True, help="Stage4/5용 cycle pkl 경로")
    p.add_argument("--seg-data-dir", required=True, help="Stage4/5용 seg pkl 경로")
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--charge-m", type=int, default=None)
    p.add_argument("--discharge-m", type=int, default=None)
    p.add_argument("--scen-k", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=300,
                   help="Stage1+2(v2) epochs 상한 (기본 300 — cfg 기본 500 대신). "
                        "Stage0(baseline)은 --model-config 자체의 early_stop_patience를 "
                        "따르므로, Stage0도 줄이려면 5_model/config/main_qfref_S_p60.yaml "
                        "처럼 patience를 낮춘 config를 --model-config로 지정할 것")
    p.add_argument("--v2-patience", type=int, default=60,
                   help="Stage1+2(v2) 조기종료 patience (기본 60, phase1_trainer_v2.py --patience)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Stage1+2(v2) batch_size 오버라이드 (성능 영향 가능성 있음 — §시간단축 옵션 참고)")
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 15, 25],
                   help="Stage0/1+2/4 수렴성 분석에 쓸 k 목록")
    p.add_argument("--synergy-k", type=int, default=None,
                   help="Stage5 조합탐색 top-k (기본: --scen-k)")
    p.add_argument("--corr-threshold", type=float, default=0.9)
    p.add_argument("--synergy-scenarios", nargs="+", default=None,
                   help="Stage5 대상 시나리오 목록 (기본: 축의 전체 시나리오)")
    p.add_argument("--synergy-workers", type=int, default=None)
    p.add_argument("--parallel", type=int, default=1, help="Stage0/1+2 시드별 동시 실행 수")
    p.add_argument("--split-seed", type=int, default=None, help="Stage4/5 데이터 로딩용 (기본: seeds[0])")
    p.add_argument("--skip-stages", nargs="+", default=[], choices=["0", "12", "3", "4", "5"],
                   help="건너뛸 단계 번호 (예: 이미 Stage0을 돌려놨으면 --skip-stages 0)")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _run(cmd: list[str], desc: str) -> int:
    tqdm_write(f"\n{'='*70}\n[{desc}]\n$ {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        tqdm_write(f"[{desc}] 실패 (exit={result.returncode})")
    return result.returncode


def main() -> None:
    args = _parse_args()
    split_seed = args.split_seed if args.split_seed is not None else args.seeds[0]
    py = sys.executable

    common_axis = ["--seg-axis", args.seg_axis, "--axis-config", args.axis_config]
    common_mk = []
    if args.charge_m is not None: common_mk += ["--charge-m", str(args.charge_m)]
    if args.discharge_m is not None: common_mk += ["--discharge-m", str(args.discharge_m)]
    if args.scen_k is not None: common_mk += ["--scen-k", str(args.scen_k)]

    stages = ["0", "12", "3", "4", "5"]
    active_stages = [s for s in stages if s not in args.skip_stages]

    baseline_manifest = RESULTS_DIR / f"convergence_manifest_{args.tag}_baseline.json"
    stage12_manifest = RESULTS_DIR / f"convergence_manifest_{args.tag}_stage12.json"

    with tqdm(total=len(active_stages), desc="Phase1 Stage0-6", unit="stage") as pbar:

        # ── Stage 0: 현재 구조 기준선 ────────────────────────────────────────
        if "0" in active_stages:
            pbar.set_description("Stage0 기준선 다중시드")
            _run([py, str(LAB_DIR / "run_convergence_seeds.py"), "--trainer", "baseline",
                  "--model-config", args.model_config, *common_axis, *common_mk,
                  "--seeds", *[str(s) for s in args.seeds], "--tag", f"{args.tag}_baseline",
                  "--parallel", str(args.parallel)],
                 "Stage0: baseline 다중시드 학습")
            _run([py, str(LAB_DIR / "analyze_convergence.py"),
                  "--manifest", str(baseline_manifest), "--k-values", *[str(k) for k in args.k_values]],
                 "Stage0: baseline 수렴성 분석")
            pbar.update(1)

        # ── Stage 1+2: 체크포인트 기준 변경 + temperature annealing ──────────
        if "12" in active_stages:
            pbar.set_description("Stage1+2 개선판 다중시드")
            v2_extra = ["--max-epochs", str(args.max_epochs), "--patience", str(args.v2_patience)]
            if args.batch_size is not None:
                v2_extra += ["--batch-size", str(args.batch_size)]
            _run([py, str(LAB_DIR / "run_convergence_seeds.py"), "--trainer", "v2",
                  "--beta-min", str(args.beta_min), *v2_extra,
                  "--model-config", args.model_config, *common_axis, *common_mk,
                  "--seeds", *[str(s) for s in args.seeds], "--tag", f"{args.tag}_stage12",
                  "--parallel", str(args.parallel)],
                 "Stage1+2: 개선판 다중시드 학습")
            _run([py, str(LAB_DIR / "analyze_convergence.py"),
                  "--manifest", str(stage12_manifest), "--k-values", *[str(k) for k in args.k_values]],
                 "Stage1+2: 개선판 수렴성 분석")

            # baseline 대비 개선폭 즉시 비교 출력
            if baseline_manifest.exists():
                base_report = RESULTS_DIR / f"convergence_report_{args.tag}_baseline.json"
                s12_report = RESULTS_DIR / f"convergence_report_{args.tag}_stage12.json"
                if base_report.exists() and s12_report.exists():
                    b = json.loads(base_report.read_text(encoding="utf-8"))
                    n = json.loads(s12_report.read_text(encoding="utf-8"))
                    tqdm_write("\n[Stage0 -> Stage1+2] Kendall τ 개선폭 (시나리오별):")
                    for sid, bd in b["per_scenario"].items():
                        nd = n["per_scenario"].get(sid, {})
                        if nd:
                            delta = nd["mean_kendall_tau"] - bd["mean_kendall_tau"]
                            tqdm_write(f"  {bd['seg_name']:10s}  {bd['mean_kendall_tau']:.4f} -> "
                                       f"{nd['mean_kendall_tau']:.4f}  (Δ={delta:+.4f})")
            pbar.update(1)

        # ── Stage 3: 다중시드 앙상블 -> 합성 gates 디렉터리 ───────────────────
        if "3" in active_stages:
            pbar.set_description("Stage3 앙상블 합성")
            _run([py, str(LAB_DIR / "materialize_ensemble_gates.py"),
                  "--manifest", str(stage12_manifest), "--out-tag", f"{args.tag}_ensemble"],
                 "Stage3: 앙상블 gates 합성")
            pbar.update(1)

        # ── Stage 4: 상관 클러스터링 재해석 ──────────────────────────────────
        if "4" in active_stages:
            pbar.set_description("Stage4 클러스터링 재해석")
            _run([py, str(LAB_DIR / "analyze_hi_clusters.py"),
                  "--manifest", str(stage12_manifest),
                  "--model-config", args.model_config, *common_axis,
                  "--data-dir", args.data_dir, "--seg-data-dir", args.seg_data_dir,
                  "--datasets", *args.datasets, "--corr-threshold", str(args.corr_threshold),
                  "--k-values", *[str(k) for k in args.k_values],
                  "--split-seed", str(split_seed), "--tag", f"{args.tag}_clusters"],
                 "Stage4: 상관 클러스터링 재해석")
            pbar.update(1)

        # ── Stage 5: HI 조합 시너지 (시나리오별) ─────────────────────────────
        if "5" in active_stages:
            pbar.set_description("Stage5 조합 시너지")
            scenarios = args.synergy_scenarios
            if scenarios is None:
                spec = get_segmenter(args.seg_axis, {args.seg_axis: json.loads(args.axis_config)}).get_spec()
                scenarios = spec.scenario_names
            synergy_k = args.synergy_k or args.scen_k or 15
            for sname in tqdm(scenarios, desc="  Stage5 시나리오별 시너지", unit="scenario", leave=False):
                cmd = [py, str(LAB_DIR / "analyze_hi_synergy.py"),
                       "--model-config", args.model_config, *common_axis,
                       "--data-dir", args.data_dir, "--seg-data-dir", args.seg_data_dir,
                       "--datasets", *args.datasets, "--scenario", sname, "--k", str(synergy_k),
                       "--split-seed", str(split_seed), "--tag", f"{args.tag}_{sname}"]
                if args.synergy_workers is not None:
                    cmd += ["--workers", str(args.synergy_workers)]
                _run(cmd, f"Stage5: {sname} 시너지 스크리닝")
            pbar.update(1)

    ensemble_dir = RESULTS_DIR / "ensembles" / f"{args.tag}_ensemble"
    print(f"\n{'='*70}")
    print(f"[Stage0-6 완료] tag={args.tag}")
    print(f"  Stage6 최종 산출물(Phase2 --gates-from 대상): {ensemble_dir}")
    print(f"  전체 기록: docs/phase1_lab/RESULTS_LOG.md")
    print(f"{'='*70}")

    append_log_entry(
        tag=f"full_pipeline_{args.tag}",
        purpose=f"Stage0-6 통합 실행 ({len(args.seeds)}개 seed: {args.seeds})",
        command=current_command_str(),
        result_files=[str(RESULTS_DIR / f"convergence_report_{args.tag}_baseline.md"),
                      str(RESULTS_DIR / f"convergence_report_{args.tag}_stage12.md"),
                      str(RESULTS_DIR / f"cluster_report_{args.tag}_clusters.json"),
                      str(ensemble_dir)],
        key_metrics="개별 하위 스크립트 로그 항목 참고(위쪽에 각각 자동 기록됨)",
        interpretation=f"Stage6 산출물({ensemble_dir})을 --gates-from으로 Phase2와 비교(§4 A/B/C 실험)로 다음 단계 진행.",
    )


if __name__ == "__main__":
    main()
