"""
5_model/experiments/phase1_lab/test_phase1_checkpoint.py

phase1_trainer_v2.py가 저장한 체크포인트(checkpoints/best_by_saturation.pt)를 test split에서
평가하고, 대표 셀의 용량곡선 비교 플랏을 그린다.

왜 5_model/test_scr.py를 그대로 못 쓰는가: test_scr.py는 "Phase 2" 체크포인트(norm_mean/std와
cfg를 통째로 저장, classifier 라우팅 지원)를 가정한다. phase1_trainer_v2.py의 체크포인트는
{"model_state","epoch","gate_saturation","val_rmse"}만 저장하고, SCRModel도 scen_group_ids
(v3)/shared_hi_mask(v4)/n_kernel_hi(커널 피처 버전)로 구성될 수 있어 test_scr.py의 모델
재구성 로직(이 세 kwarg를 모르는 채로 SCRModel을 만듦)과 맞지 않는다 — 그대로 로드하면
norm_mean 키 누락으로 즉시 죽거나, 아키텍처 불일치로 load_state_dict가 깨진다.

대신 phase1_trainer_v2.py와 완전히 동일한 방식으로 cfg/spec/데이터/모델을 재구성한 뒤(중복
구현 금지 — train_scr.py/test_scr.py/phase1_trainer_v2.py의 기존 함수만 재사용), 평가 자체는
test_scr.py와 같은 SCREvaluator(5_model/evaluation/scr_evaluator.py)를 그대로 쓴다.

run_dir/config.yaml(트레이너가 저장한, 완전히 해석된 cfg)과 run_dir/p1v2_summary.json
(synergy_groups_json/kernel_features_pkl 경로)을 자동으로 읽으므로 v0/v2/v3/v-ctrl/v0-ctrl
run은 --run-dir와 --rep-cells만 주면 된다. v4(shared_gate)는 interaction_json이
p1v2_summary.json에 기록되지 않으므로 --interaction-json으로 학습 때 쓴 파일을 다시 지정해야
한다.

사용 예:
  python 5_model/experiments/phase1_lab/test_phase1_checkpoint.py \
      --run-dir 5_model/experiments/phase1_lab/results/p1v2_runs/0828_1549_p1v2_p1v0ctrl_full_seed42 \
      --rep-cells b1c0 b1c1

  # v4(shared_gate) checkpoint는 --interaction-json도 같이 지정:
  python 5_model/experiments/phase1_lab/test_phase1_checkpoint.py \
      --run-dir 5_model/experiments/phase1_lab/results/p1v2_runs/<v4_run> \
      --interaction-json 5_model/experiments/phase1_lab/results/hi_scenario_interaction_k25_full_N2.json \
      --rep-cells b1c0

  # 여러 run(v0/v2/v3/v4)을 plot_phase1_capacity_comparison.py로 한 그림에 비교하려면
  # --export-for-visualize를 추가해 각 run_dir에 metrics/predictions/routing을 채워둔다:
  python 5_model/experiments/phase1_lab/test_phase1_checkpoint.py \
      --run-dir <run> --rep-cells b1c0 --export-for-visualize
"""

from __future__ import annotations

import argparse
import json
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

import torch  # noqa: E402

from utils.io_utils import load_config  # noqa: E402
from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402
from datasets.segment_dataset import build_datasets  # noqa: E402
from models.scr_model import SCRModel  # noqa: E402
from evaluation.scr_evaluator import SCREvaluator  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

import train_scr as _base  # noqa: E402 (synergy group 로더 재사용)
import test_scr as _tbase  # noqa: E402 (_resolve_device/_pick_rep_cells 재사용)
from phase1_trainer_v2 import _apply_kernel_features  # noqa: E402 (중복 구현 금지)


class _KernelAugmentedDataset(torch.utils.data.Dataset):
    """SCREvaluator.predict_dataset은 표준 DataLoader(+scr_evaluator._collate)를 쓰는데,
    SegmentDataset.__getitem__(datasets/segment_dataset.py)은 x_kernel을 모르는 고정
    dict만 반환한다 — x_kernel은 FastTensorLoader(트레이너 전용)만 hasattr(ds,"x_kernel")로
    감지해서 배치에 넣어준다. 그래서 커널 있는 checkpoint(v2/v3/v4)를 SCREvaluator로 평가하면
    forward에서 KeyError: 'x_kernel'이 난다. segment_dataset.py/scr_evaluator.py는 건드리지
    않고(중복 구현/기존 스크립트 수정 금지), 여기서만 __getitem__에 x_kernel을 끼워 넣는
    얇은 래퍼로 우회한다. __getattr__로 나머지 속성(cell_ids/cycles/seg_names 등)은 원본
    데이터셋에 그대로 위임 — _pick_rep_cells 등 기존 코드가 요구하는 속성 접근에 영향 없음."""

    def __init__(self, base_ds):
        self._base = base_ds

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        item["x_kernel"] = self._base.x_kernel[idx]
        return item

    def __getattr__(self, name):
        return getattr(self._base, name)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="phase1_trainer_v2.py 체크포인트 test 평가 + 대표 셀 용량곡선 비교 플랏"
    )
    p.add_argument("--run-dir", required=True, dest="run_dir",
                   help="results/p1v2_runs/<run> 디렉터리(config.yaml/p1v2_summary.json이 있는 곳)")
    p.add_argument("--checkpoint", default=None,
                   help="기본값: <run-dir>/checkpoints/best_by_saturation.pt")
    p.add_argument("--interaction-json", default=None, dest="interaction_json",
                   help="v4(shared_gate) checkpoint 전용 — p1v2_summary.json에 기록되지 않으므로 "
                        "학습 때 준 test_hi_scenario_interaction.py 산출물을 다시 지정해야 함")
    p.add_argument("--regression-model", default="mlp", dest="regression_model",
                   choices=["mlp", "transformer", "i_transformer", "resnet_tab", "ft_transformer"],
                   help="학습 때 --regression-model을 오버라이드했다면 동일하게 지정 "
                        "(config.yaml에는 반영 안 돼 있음 — 기본값 mlp면 신경 안 써도 됨)")
    p.add_argument("--rep-cells", nargs="+", default=None, dest="rep_cells",
                   help="비교 플랏을 그릴 셀 ID(들). 미지정 시 데이터셋별 1개 자동 선정")
    p.add_argument("--data-dir", default=None, dest="data_dir",
                   help="config.yaml의 data.data_dir 오버라이드 — run마다 학습 당시 머신의 "
                        "경로(상대경로 또는 다른 드라이브)가 그대로 박혀있어, 이 스크립트를 "
                        "돌리는 머신에 그 경로가 없으면 필요")
    p.add_argument("--seg-data-dir", default=None, dest="seg_data_dir",
                   help="config.yaml의 data.seg_data_dir 오버라이드 (위와 동일 이유)")
    p.add_argument("--device", default="auto")
    p.add_argument("--export-for-visualize", action="store_true", dest="export_for_visualize",
                   help="visualize_results.py의 RunBundle이 읽을 수 있도록 "
                        "metrics/metrics.json, predictions/test_predictions.csv, "
                        "routing/routing_table.csv를 run_dir에 추가로 저장한다 "
                        "(plot_phase1_capacity_comparison.py로 여러 run을 비교할 때 필요)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = _tbase._resolve_device(args.device)
    print(f"[test_p1] device={device}")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} 없음 — phase1_trainer_v2.py가 만든 run 디렉터리가 맞는지 확인하세요"
        )
    cfg = load_config(str(cfg_path))
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
        print(f"[test_p1] data_dir 오버라이드: {args.data_dir}")
    if args.seg_data_dir is not None:
        cfg["data"]["seg_data_dir"] = args.seg_data_dir
        print(f"[test_p1] seg_data_dir 오버라이드: {args.seg_data_dir}")

    summary_path = run_dir / "p1v2_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    def _resolve_summary_path(v):
        # p1v2_summary.json에는 트레이너 실행 당시 cwd 기준 상대경로가 그대로 남아있을 수
        # 있어(예: "5_model/experiments/.../kernel_v3.pkl"), 이 스크립트를 다른 cwd에서
        # 실행해도 항상 찾도록 PROJECT_ROOT 기준으로 고정한다.
        if not v:
            return None
        p = Path(v)
        resolved = p if p.is_absolute() else PROJECT_ROOT / p
        if resolved.exists():
            return resolved
        # results/ 정리(v3/v4 입력만 남기고 나머지는 results/outputs/로 이동, 260827
        # 세션) 이전에 학습된 run은 summary.json에 옛 경로가 그대로 박혀있다 — 파일명만
        # 살아있는 outputs/ 하위에서 한 번 더 찾는다.
        fallback = resolved.parent / "outputs" / resolved.name
        if fallback.exists():
            print(f"[test_p1] {resolved} 없음 — {fallback}에서 발견(results/ 정리 이전 경로)")
            return fallback
        return resolved

    synergy_groups_json = _resolve_summary_path(summary.get("synergy_groups_json"))
    kernel_features_pkl = _resolve_summary_path(summary.get("kernel_features_pkl"))

    ckpt_path = (Path(args.checkpoint) if args.checkpoint
                 else run_dir / "checkpoints" / "best_by_saturation.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[test_p1] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"[test_p1] epoch={ckpt.get('epoch')} "
          f"gate_saturation={ckpt.get('gate_saturation', float('nan')):.4f} "
          f"val_rmse={ckpt.get('val_rmse', float('nan')):.6f}")

    spec = get_segmenter(
        cfg["scenario"]["axis"], {cfg["scenario"]["axis"]: cfg["scenario"]["axis_config"]}
    ).get_spec()
    train_ds, val_ds, test_ds, norm = build_datasets(cfg, spec=spec)

    kernel_hi_names = None
    if kernel_features_pkl:
        print(f"[test_p1] kernel-features-pkl 자동 적용(p1v2_summary.json): {kernel_features_pkl}")
        kernel_hi_names = _apply_kernel_features(
            [train_ds, val_ds, test_ds], Path(kernel_features_pkl)
        )

    scen_group_ids = None
    if synergy_groups_json:
        print(f"[test_p1] synergy-groups-json 자동 적용(p1v2_summary.json): {synergy_groups_json}")
        scen_group_ids = _base._load_synergy_group_ids(
            Path(synergy_groups_json), spec.n_scenarios, spec.scenario_names,
        )

    shared_hi_mask = None
    if args.interaction_json:
        interaction_data = json.loads(Path(args.interaction_json).read_text(encoding="utf-8"))
        ref_seg_name = spec.scenario_names[0]
        ref_cols = get_hi_cols_for_seg(ref_seg_name)
        suffix = f"_{ref_seg_name}"
        concepts_in_order = [c[: -len(suffix)] if c.endswith(suffix) else c for c in ref_cols]
        per_hi = interaction_data["per_hi"]
        shared_hi_mask = torch.tensor(
            [not per_hi.get(c, {"significant": False})["significant"] for c in concepts_in_order],
            dtype=torch.bool,
        )
        n_shared = int(shared_hi_mask.sum().item())
        print(f"[test_p1] interaction-json 적용: {args.interaction_json} "
              f"({n_shared}/{len(shared_hi_mask)}개 HI -> shared_gate)")

    lambda_scen = cfg.get("loss", {}).get("lambda_scen", 0.0)
    with_probe_mlp = lambda_scen > 0
    p1_model_cfg = {**cfg["model"], "regression_model": args.regression_model,
                     "with_raw_cnn": False, "with_raw_flat": False}

    model = SCRModel(
        d_probe=cfg["model"]["d_probe"], d_head=cfg["model"]["d_head"], dropout=cfg["model"]["dropout"],
        spec=spec, with_probe_mlp=with_probe_mlp, model_cfg=p1_model_cfg,
        scen_group_ids=scen_group_ids,
        shared_hi_mask=shared_hi_mask,
        n_kernel_hi=len(kernel_hi_names) if kernel_hi_names else 0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    if hasattr(test_ds, "x_kernel"):
        test_ds = _KernelAugmentedDataset(test_ds)

    rep_cells = args.rep_cells or _tbase._pick_rep_cells(test_ds, cfg, 1)
    print(f"[test_p1] rep_cells: {rep_cells}")

    figures_dir = run_dir / "figures"
    evaluator = SCREvaluator(
        model=model, normalizer=norm, device=device,
        figures_dir=figures_dir, rep_cells=rep_cells,
    )

    test_modes = evaluator.evaluate_modes(test_ds, modes=("oracle",))
    print(f"[test_p1] test oracle capacity metrics: {test_modes['oracle']['capacity']}")
    evaluator._plot_scatter(test_modes["oracle"]["_pred"], tag="test_oracle")
    evaluator._plot_capacity_curves(test_modes["oracle"]["_pred"])
    print(f"[test_p1] 저장: {figures_dir}")

    if args.export_for_visualize:
        _export_for_visualize(run_dir, evaluator, test_modes, spec)


def _export_for_visualize(run_dir: Path, evaluator: SCREvaluator, test_modes: dict, spec) -> None:
    """visualize_results.py의 RunBundle(__init__에서 metrics/predictions/routing을
    무조건 다 읽음)이 phase1_lab run_dir을 로드할 수 있도록, test_scr.py Phase2 run이
    남기는 것과 같은 스키마로 세 파일을 추가 저장한다. checkpoints/*.pt, config.yaml,
    scenario_spec.json, gates/*.json은 phase1_trainer_v2.py가 이미 저장해두므로 손댈 필요
    없음 — 여기서 부족한 세 파일만 채운다(RunBundle/_plot_capacity_curve_comparison 코드는
    무수정)."""
    import csv

    pred = test_modes["oracle"]["_pred"]

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = {"test": evaluator.strip_modes_for_json(test_modes)}
    (metrics_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with open(predictions_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "cycle", "seg_name", "soh_true", "soh_pred",
                    "cap_true_Ah", "cap_pred_Ah"])
        for i in range(len(pred["cell_ids"])):
            cap_init = float(pred["cap_init_raw"][i])
            soh_true = float(pred["cap_true_raw"][i])
            soh_pred = float(pred["cap_pred_raw"][i])
            w.writerow([pred["cell_ids"][i], int(pred["cycles"][i]), pred["seg_names"][i],
                        soh_true, soh_pred, soh_true * cap_init, soh_pred * cap_init])

    routing_dir = run_dir / "routing"
    routing_dir.mkdir(parents=True, exist_ok=True)
    _ref_seg = spec.scenario_names[0]
    _suffix = f"_{_ref_seg}"
    hi_names = [n[: -len(_suffix)] if n.endswith(_suffix) else n
                for n in get_hi_cols_for_seg(_ref_seg)]
    probe_json = json.loads((run_dir / "gates" / "classification_HIs.json").read_text(encoding="utf-8"))
    scen_json = json.loads((run_dir / "gates" / "regression_HIs.json").read_text(encoding="utf-8"))
    with open(routing_dir / "routing_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gate"] + hi_names)

        def _row(name: str, ranked: list[int], probs: list[float]) -> None:
            mask = ["0"] * len(hi_names)
            for idx, p in zip(ranked, probs):
                if p > 0.9:
                    mask[idx] = "1"
            w.writerow([name] + mask)

        _row("probe", probe_json.get("charge_ranked", []), probe_json.get("charge_probs", []))
        for s in range(spec.n_scenarios):
            _row(scen_json.get(f"seg_{s}_seg_name", f"seg_{s}"),
                 scen_json.get(f"seg_{s}_ranked", []), scen_json.get(f"seg_{s}_probs", []))

    print(f"[test_p1] visualize_results.py용 파일 저장: {metrics_dir}, {predictions_dir}, {routing_dir}")


if __name__ == "__main__":
    main()
