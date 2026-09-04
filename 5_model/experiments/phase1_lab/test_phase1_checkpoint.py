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

hard/soft 라우팅(2026-09-03 복원): phase1_trainer_v2.py는 lambda_scen>0이면 SCRModel에
probe_mlp라는 dual-objective 분류 헤드(probe_x+direction -> level_logits)를 CE로 함께
학습한다(scr_model.py forward 참고) — 원래 있던 별도 시나리오 분류기(train_classifier.py,
run_pipeline.py 구 Step 7)와는 목적이 다른 보조 헤드지만, 입력 형태가 완전히 같아
SCREvaluator.set_classifier(model.probe_mlp)로 그대로 꽂힌다. 이 스크립트는 checkpoint에
probe_mlp가 있으면 자동으로 oracle/hard/soft 전부 평가하고, 없으면(lambda_scen=0) oracle만
평가한다 — 분류기를 별도로 학습하는 스텝 없이도 라우팅 현실성(hard)을 볼 수 있다.

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

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

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
                   help="v4/v5(shared_gate) checkpoint 전용 — p1v2_summary.json에 기록돼 있으면 "
                        "자동 적용되고(v5 이후), 안 돼 있으면(v4 등 구버전) 학습 때 준 "
                        "test_hi_scenario_interaction.py 산출물을 다시 지정해야 함")
    p.add_argument("--specific-group-ids-json", default=None, dest="specific_group_ids_json",
                   help="v5(그룹 게이팅) checkpoint 전용 — p1v2_summary.json에 기록돼 있으면 "
                        "자동 적용됨. build_specific_component_groups.py 산출물")
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

    interaction_json = args.interaction_json or (
        str(_resolve_summary_path(summary.get("interaction_json"))) if summary.get("interaction_json") else None
    )
    shared_hi_mask = None
    if interaction_json:
        interaction_data = json.loads(Path(interaction_json).read_text(encoding="utf-8"))
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
        print(f"[test_p1] interaction-json 적용: {interaction_json} "
              f"({n_shared}/{len(shared_hi_mask)}개 HI -> shared_gate)")

    specific_group_ids_json = args.specific_group_ids_json or (
        str(_resolve_summary_path(summary.get("specific_group_ids_json")))
        if summary.get("specific_group_ids_json") else None
    )
    if specific_group_ids_json:
        # v5: build_specific_component_groups.py 산출물 — phase1_trainer_v2.py와 동일한 로직
        # (재정렬 없이 그대로 적용, seg_{s}_specific_group_ids는 이미 _specific_idx 순서와 일치).
        spec_data = json.loads(Path(specific_group_ids_json).read_text(encoding="utf-8"))
        n_specific_expected = int(shared_hi_mask.numel() - shared_hi_mask.sum().item())
        if spec_data.get("n_specific") != n_specific_expected:
            raise ValueError(
                f"--specific-group-ids-json의 n_specific({spec_data.get('n_specific')})이 "
                f"interaction_json에서 나온 specific 개수({n_specific_expected})와 다릅니다."
            )
        scen_group_ids = {
            s: spec_data[f"seg_{s}_specific_group_ids"]
            for s in range(spec.n_scenarios)
            if f"seg_{s}_specific_group_ids" in spec_data
        }
        n_groups_total = sum(spec_data.get(f"seg_{s}_n_groups", 0) for s in range(spec.n_scenarios))
        print(f"[test_p1] specific-group-ids-json 적용: {specific_group_ids_json} "
              f"({len(scen_group_ids)}/{spec.n_scenarios}개 시나리오, 총 그룹 {n_groups_total}개)")

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

    # model.probe_mlp는 Phase 1이 CE(lambda_scen>0)로 회귀와 함께 학습한 dual-objective
    # 분류 헤드다(scr_model.py forward: probe_x+direction -> level_logits). 입력 형태가
    # SCREvaluator.set_classifier()가 기대하는 [probe_x || direction] (B, N_HI+1) ->
    # (B, n_classes)와 완전히 동일해 별도 어댑터 없이 그대로 꽂힌다 — 이게 되면
    # hard(분류기 argmax 라우팅)/soft(확률가중 라우팅) 평가를 별도 분류기 학습
    # 없이(구 Step 7, train_classifier.py 제거됨) 복원할 수 있다.
    if model.probe_mlp is not None:
        evaluator.set_classifier(model.probe_mlp)
        modes = ("oracle", "hard", "soft")
        print("[test_p1] model.probe_mlp를 라우팅 분류기로 사용 — oracle/hard/soft 전부 평가")
    else:
        modes = ("oracle",)
        print("[test_p1] lambda_scen=0(probe_mlp 없음) — 이 체크포인트는 hard/soft 라우팅이 "
              "불가능해 oracle만 평가합니다.")

    test_modes = evaluator.evaluate_modes(test_ds, modes=modes)
    for m in modes:
        print(f"[test_p1] test {m} capacity metrics: {test_modes[m]['capacity']}")
        if m != "oracle":
            print(f"[test_p1] test {m} classification: {test_modes[m]['classification']}")
        evaluator._plot_scatter(test_modes[m]["_pred"], tag=f"test_{m}")
        if m != "oracle":
            evaluator._plot_confusion_matrix(test_modes[m]["_pred"], tag=f"test_{m}")
        _plot_error_heatmaps(test_modes[m]["_pred"], spec, figures_dir, tag=f"test_{m}")
    # 용량곡선(capacity_curve_*.png)은 test_scr.py와 동일한 관례로 파일명에 모드 태그가
    # 없어 한 모드만 그릴 수 있다 — 실배포 기준(hard)이 있으면 그쪽, 없으면 oracle.
    _curve_mode = "hard" if "hard" in modes else "oracle"
    evaluator._plot_capacity_curves(test_modes[_curve_mode]["_pred"])
    print(f"[test_p1] 저장: {figures_dir}")

    if args.export_for_visualize:
        _export_for_visualize(run_dir, evaluator, test_modes, spec)


def _smoothed_error_grid(x: np.ndarray, y: np.ndarray, err: np.ndarray,
                          x_edges: np.ndarray, y_edges: np.ndarray,
                          sigma) -> np.ndarray:
    """(x,y) 산점 데이터를 fine grid에 bin하고(합/개수 따로 누적), 각각 gaussian_filter로
    스무딩한 뒤 나눠서 "국소 가중평균 |오차|" 연속 그리드를 만든다. 그냥 bin 평균만 내면
    fine grid일수록 빈 칸(개수=0 → NaN)이 많아 듬성듬성해지는데, sum/count를 각각 스무딩
    후 나누면(Nadaraya-Watson류 커널 평균과 동치) 빈 칸도 이웃 값으로 자연스럽게 채워져
    imshow가 레퍼런스 이미지처럼 매끈한 그라데이션으로 보인다. sigma의 어느 축이든 0이면
    그 축으론 블렌딩하지 않는다(카테고리 x축을 서로 안 섞이게 할 때 씀)."""
    from scipy.ndimage import gaussian_filter

    n_y = len(y_edges) - 1
    n_x = len(x_edges) - 1
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_x - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, n_y - 1)

    sum_grid = np.zeros((n_y, n_x))
    cnt_grid = np.zeros((n_y, n_x))
    np.add.at(sum_grid, (y_bin, x_bin), err)
    np.add.at(cnt_grid, (y_bin, x_bin), 1)

    sum_s = gaussian_filter(sum_grid, sigma=sigma, mode="nearest")
    cnt_s = gaussian_filter(cnt_grid, sigma=sigma, mode="nearest")
    return np.divide(sum_s, cnt_s, out=np.full_like(sum_s, np.nan), where=cnt_s > 1e-6)


def _render_error_heatmap(x: np.ndarray, y_edges: np.ndarray, y: np.ndarray, err: np.ndarray,
                           x_edges: np.ndarray, sigma, figsize, xlabel: str, xticks,
                           title: str, cbar_label: str, out_path: Path, vmax: float) -> None:
    grid = _smoothed_error_grid(x, y, err, x_edges, y_edges, sigma=sigma)
    cmap = matplotlib.colormaps["jet"].copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap,
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    vmin=0, vmax=vmax, interpolation="bilinear")
    if xticks is not None:
        idx, labels = xticks
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observed capacity (Ah)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[test_p1] saved {out_path}")


def _plot_error_heatmaps(pred_dict: dict, spec, figures_dir: Path, tag: str = "test_oracle") -> None:
    """오차를 fine grid(150~250 bin)로 나눈 뒤 gaussian_filter로 보간한 연속 히트맵을
    저장한다 — 레퍼런스 이미지 (c)/(d)와 같은 매끈한 그라데이션 텍스처를 내는 게 목적.
    절대오차(Ah)와 상대오차(%, |오차|/관측용량*100 — MAPE와 같은 정의) 두 단위로, 각각
    (용량 x 시나리오)/(용량 x 사이클) 축 조합을 그려 총 4장을 만든다. 이 파이프라인엔
    잔존용량(SOC성) 예측 타깃이 없어서(예측 타깃은 SOH/최대용량 하나뿐 —
    evaluator.predict_dataset 참고) 레퍼런스의 "관측 잔존용량 x 관측 최대용량" 축 그대로는
    못 쓴다. 대신 관측 용량(Ah, 열화 축)을 공통 y축으로 두고 x축을 시나리오/사이클로
    바꿔서 본다. (용량 x 사이클)은 셀 단위로는 사실상 1차원 궤적(사이클↔용량이 거의
    결정론적)이라 순수 산점도로는 면을 못 채우는데, sum/count를 각각 스무딩해서
    나누면(커널 가중평균) 이웃 궤적 정보로 빈틈이 자연스럽게 메워진다."""
    if not _HAS_MPL:
        return

    cap_init = np.asarray(pred_dict["cap_init_raw"], dtype=float)
    cap_true = np.asarray(pred_dict["cap_true_raw"], dtype=float) * cap_init
    cap_pred = np.asarray(pred_dict["cap_pred_raw"], dtype=float) * cap_init
    err_ah = np.abs(cap_pred - cap_true)
    err_pct = err_ah / cap_true * 100.0
    cycles = np.asarray(pred_dict["cycles"], dtype=float)
    scen_idx = np.asarray(pred_dict["scen_idx"], dtype=int)

    y_edges = np.linspace(cap_true.min(), cap_true.max(), 181)

    # scen_idx는 정수라 그대로 binning하면 칼럼당 딱 1개 fine bin에만 몰려서(폭 40개 중 1개)
    # sigma_x로 스무딩해도 옆 칼럼과 안 섞이게 하면 바늘처럼 가늘어져 안 보인다 — 칼럼
    # 내부에 지터를 줘서 폭 전체(±0.45)에 데이터를 먼저 채운 뒤에 binning+스무딩해야
    # 칼럼이 꽉 찬 그라데이션으로 보인다(칼럼 간 간격 0.1은 sigma_x를 작게 잡아 유지).
    scen_names = list(spec.scenario_names)
    n_scen = len(scen_names)
    rng = np.random.default_rng(0)
    x_jittered = scen_idx.astype(float) + rng.uniform(-0.45, 0.45, size=scen_idx.shape)
    x_edges1 = np.linspace(-0.5, n_scen - 0.5, n_scen * 40 + 1)
    x_edges2 = np.linspace(cycles.min(), cycles.max(), 251)
    xticks1 = (list(range(n_scen)), scen_names)

    # 파일명 접미사 없음 = 기존 Ah 버전(이미 검증받은 파일명 그대로 유지), _pct = 신규 상대오차 버전.
    # vmax는 (전과 동일하게) 스무딩 전 raw 오차의 99th percentile로 고정 — 그리드 자체의
    # percentile을 쓰면 스무딩으로 극단치가 희석돼 색 스케일이 기존 Ah 버전과 달라진다.
    for suffix, unit, err, cbar_label in (("", "Ah", err_ah, "mean |error| (Ah)"),
                                           ("_pct", "%", err_pct, "mean |error| (%)")):
        vmax = float(np.percentile(err, 99))
        _render_error_heatmap(
            x_jittered, y_edges, cap_true, err, x_edges1, sigma=(3, 1.5),
            figsize=(max(6, n_scen * 1.4), 6), xlabel="", xticks=xticks1,
            title=f"SCR {tag} — mean |error| ({unit}), interpolated: scenario x capacity",
            cbar_label=cbar_label,
            out_path=figures_dir / f"error_heatmap_capacity_scenario{suffix}_{tag}.png",
            vmax=vmax,
        )
        _render_error_heatmap(
            cycles, y_edges, cap_true, err, x_edges2, sigma=3,
            figsize=(7, 6), xlabel="Cycle", xticks=None,
            title=f"SCR {tag} — mean |error| ({unit}), interpolated: cycle x capacity",
            cbar_label=cbar_label,
            out_path=figures_dir / f"error_heatmap_capacity_cycle{suffix}_{tag}.png",
            vmax=vmax,
        )


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
