"""
9_eval/test.py

train.py(v4, shared_gate)가 저장한 체크포인트(checkpoints/best_by_saturation.pt)를
test split에서 평가하고, 대표 셀의 용량곡선 비교 플랏을 그린다. run_dir/config.yaml(트레이너가
저장한 완전히 해석된 cfg)과 run_dir/p1v2_summary.json(kernel_features_pkl 등 경로)을 자동으로
읽는다 — interaction_json은 p1v2_summary.json에 기록되지만, 값이 없거나 산출물을 옮긴 경우엔
--interaction-json으로 다시 지정한다.

hard/soft 라우팅: train.py는 lambda_scen>0이면 SCRModel에 probe_mlp라는
dual-objective 분류 헤드(probe_x+direction -> level_logits)를 CE로 함께 학습한다(scr_model.py
forward 참고) — 입력 형태가 SCREvaluator.set_classifier()와 완전히 같아 그대로 꽂힌다. 이
스크립트는 checkpoint에 probe_mlp가 있으면 자동으로 oracle/hard/soft 전부 평가하고, 없으면
(lambda_scen=0) oracle만 평가한다 — 분류기를 별도로 학습하는 스텝 없이도 라우팅 현실성(hard)을
볼 수 있다.

사용 예(--run-dir은 2026-09-23 재배치 이전의 기존 run 경로도 그대로 쓸 수 있다 — results/
자체는 안 옮겼음):
  python 9_eval/test.py \
      --run-dir legacy_results/experiments/phase1_lab/results/p1v2_runs/<v4_run> \
      --rep-cells b1c0 b1c1

  # interaction_json이 p1v2_summary.json에 없거나 산출물을 옮겼다면 다시 지정:
  python 9_eval/test.py \
      --run-dir legacy_results/experiments/phase1_lab/results/p1v2_runs/<v4_run> \
      --interaction-json legacy_results/experiments/phase1_lab/results/hi_scenario_interaction_k25_full_N2.json \
      --rep-cells b1c0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model_lib"))
sys.path.insert(0, str(PROJECT_ROOT / "8_train"))
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

import parameters as P  # noqa: E402 — 재현성/평가 파라미터 단일 소스
from utils.io_utils import load_config  # noqa: E402
from utils.hi_schema import get_hi_cols_for_seg  # noqa: E402
from datasets.segment_dataset import build_datasets  # noqa: E402
from models.scr_model import SCRModel  # noqa: E402
from evaluation.scr_evaluator import SCREvaluator  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

from train import (  # noqa: E402 (중복 구현 금지)
    _apply_kernel_features, _build_redundancy_mask,
)


def _resolve_device(device_str: str) -> torch.device:
    """2026-09-24: model_lib/legacy/test_scr.py(Stage0, 삭제됨)에서 이전 — 이 스크립트만
    쓰는 단일 소비자라 공용 모듈로 안 빼고 로컬로 유지."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _pick_rep_cells(test_ds, cfg: dict, n_per_dataset: int = 1) -> list[str]:
    """2026-09-24: model_lib/legacy/test_scr.py(Stage0, 삭제됨)에서 이전."""
    data_cfg   = cfg["data"]
    seg_dir    = PROJECT_ROOT / data_cfg["seg_data_dir"]
    test_cells = sorted(set(test_ds.cell_ids))
    picked: list[str] = []
    for ds_name in data_cfg.get("datasets", []):
        ds_dir = seg_dir / ds_name
        if not ds_dir.exists():
            continue
        ds_cell_set = {p.stem for p in ds_dir.glob("*.pkl")}
        candidates  = [c for c in test_cells if c in ds_cell_set]
        picked.extend(candidates[:n_per_dataset])
    if not picked:
        picked = test_cells[:n_per_dataset * 2]
    return picked


class _KernelAugmentedDataset(torch.utils.data.Dataset):
    """SCREvaluator.predict_dataset은 표준 DataLoader(+scr_evaluator._collate)를 쓰는데,
    SegmentDataset.__getitem__(datasets/segment_dataset.py)은 x_kernel을 모르는 고정
    dict만 반환한다 — x_kernel은 FastTensorLoader(트레이너 전용)만 hasattr(ds,"x_kernel")로
    감지해서 배치에 넣어준다. 그래서 커널 있는 checkpoint(v4)를 SCREvaluator로 평가하면
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
        description="train.py 체크포인트 test 평가 + 대표 셀 용량곡선 비교 플랏"
    )
    p.add_argument("--run-dir", required=True, dest="run_dir",
                   help="results/p1v2_runs/<run> 디렉터리(config.yaml/p1v2_summary.json이 있는 곳)")
    p.add_argument("--checkpoint", default=None,
                   help="기본값: <run-dir>/checkpoints/best_by_saturation.pt")
    p.add_argument("--interaction-json", default=P.ACTIVE_INTERACTION_JSON, dest="interaction_json",
                   help="shared_gate(v4) 구성 — 보통 p1v2_summary.json에 기록된 경로가 자동 "
                        "적용되므로, 그 기록이 없거나 산출물을 옮겼을 때만 지정하면 됨 "
                        "(parameters.py 기본 None=자동 탐지)")
    p.add_argument("--kernel-features-pkl", default=P.ACTIVE_KERNEL_FEATURES_PKL,
                   dest="kernel_features_pkl",
                   help="p1v2_summary.json에 기록된 경로를 무시하고 이 값을 쓴다. 학습 후 "
                        "산출물을 옮긴 경우(예: run_dir 재구성) summary.json의 기록이 낡아져 "
                        "FileNotFoundError가 나는데, 그럴 때 직접 지정하는 용도 — 보통은 자동"
                        "탐지로 충분하니 안 줘도 됨(parameters.py 기본 None).")
    p.add_argument("--combined-redundancy-json", default=P.ACTIVE_COMBINED_REDUNDANCY_JSON,
                   dest="combined_redundancy_json",
                   help="위와 동일 이유의 오버라이드(kernel-features-pkl과 짝을 이루는 파일이라 "
                        "보통 같이 옮겨졌을 것). parameters.py 기본 None.")
    p.add_argument("--rep-cells", nargs="+", default=P.ACTIVE_REP_CELLS, dest="rep_cells",
                   help="비교 플랏을 그릴 셀 ID(들). 미지정 시 데이터셋별 5개 자동 선정"
                        "(2026-09-18, 기존 1개 -> 5개)")
    p.add_argument("--data-dir", default=P.ACTIVE_DATA_DIR, dest="data_dir",
                   help="config.yaml의 data.data_dir 오버라이드 — run마다 학습 당시 머신의 "
                        "경로(상대경로 또는 다른 드라이브)가 그대로 박혀있어, 이 스크립트를 "
                        "돌리는 머신에 그 경로가 없으면 필요")
    p.add_argument("--seg-data-dir", default=P.ACTIVE_SEG_DATA_DIR, dest="seg_data_dir",
                   help="config.yaml의 data.seg_data_dir 오버라이드 (위와 동일 이유)")
    p.add_argument("--device", default=P.FIXED_DEVICE or "auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"[test_p1] device={device}")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} 없음 — train.py가 만든 run 디렉터리가 맞는지 확인하세요"
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
        # 있어(예: "legacy_results/experiments/.../kernel_v3.pkl"), 이 스크립트를 다른 cwd에서
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

    kernel_features_pkl = (Path(args.kernel_features_pkl) if args.kernel_features_pkl
                            else _resolve_summary_path(summary.get("kernel_features_pkl")))
    combined_redundancy_json = (Path(args.combined_redundancy_json) if args.combined_redundancy_json
                                 else _resolve_summary_path(summary.get("combined_redundancy_json")))

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

    kernel_hi_counts = None
    kernel_costs_by_scen = None
    combined_redundancy = None
    redundancy_mask = None
    if combined_redundancy_json:
        combined_redundancy = json.loads(Path(combined_redundancy_json).read_text(encoding="utf-8"))
    if kernel_features_pkl:
        print(f"[test_p1] kernel-features-pkl 자동 적용(p1v2_summary.json): {kernel_features_pkl}")
        _, kernel_hi_counts, kernel_costs_by_scen = _apply_kernel_features(
            [train_ds, val_ds, test_ds], Path(kernel_features_pkl), spec,
            combined_redundancy=combined_redundancy,
        )
    if combined_redundancy is not None:
        # 2026-09-21: 입력 레벨 nan_mask 이중 마스킹은 제거(train.py와 동일 이유
        # — probe_x가 그 nan_mask를 공유해서 분류기 정확도를 붕괴시키는 버그였음). 학습 때와
        # 똑같이 게이트 레벨 redundancy_mask만 적용해야 체크포인트와 아키텍처가 일치한다.
        redundancy_mask = _build_redundancy_mask(combined_redundancy, spec)
        print(f"[test_p1] combined-redundancy-json 자동 적용(p1v2_summary.json): {combined_redundancy_json} "
              f"(게이트 출력 0-강제만 적용, {int((~redundancy_mask).sum().item())}개 (시나리오,HI) 조합 배제)")

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

    lambda_scen = cfg.get("loss", {}).get("lambda_scen", 0.0)
    with_probe_mlp = lambda_scen > 0
    # regression_model은 항상 cfg["model"]의 저장값(v4는 항상 "mlp") 그대로 쓴다 — 다른
    # 아키텍처(transformer 등)는 sanity-check용으로만 쓰이던 옵션이라 여기서 오버라이드할
    # 이유가 없다. with_raw_cnn/with_raw_flat도 v4에서 항상 비활성.
    p1_model_cfg = {**cfg["model"], "with_raw_cnn": False, "with_raw_flat": False}

    model = SCRModel(
        d_probe=cfg["model"]["d_probe"], d_head=cfg["model"]["d_head"], dropout=cfg["model"]["dropout"],
        spec=spec, with_probe_mlp=with_probe_mlp, model_cfg=p1_model_cfg,
        shared_hi_mask=shared_hi_mask,
        kernel_hi_counts=kernel_hi_counts,
        kernel_hi_costs=kernel_costs_by_scen,
        redundancy_mask=redundancy_mask,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    if hasattr(test_ds, "x_kernel"):
        test_ds = _KernelAugmentedDataset(test_ds)

    rep_cells = args.rep_cells or _pick_rep_cells(test_ds, cfg, 5)
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
    _plot_hi_importance_ranking(run_dir, figures_dir, spec)
    print(f"[test_p1] 저장: {figures_dir}")

    # 2026-09-06: metrics/metrics.json 등은 기본으로 항상 저장(예전엔 --export-for-visualize
    # 없이 돌리면 콘솔 출력·figures/ PNG만 남고 metrics.json 자체가 아예 안 생겨서 혼동을
    # 일으켰다). 플래그는 하위 호환을 위해 그대로 받되 더 이상 이 저장 여부를 좌우하지 않는다.
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


_CATEGORY_COLORS = {
    "stat":  "#1f77b4",
    "diff":  "#ff7f0e",
    "lfp":   "#2ca02c",
    "morph": "#9467bd",
}


def _rank_scores(names: list[str], probs: list[float]) -> dict[str, float]:
    """이름 목록을 gate_prob 내림차순으로 정렬해 rank 1(최상위)~N(최하위)을 매기고,
    score = (N - rank + 1) / N 로 정규화한다(1위=1.0, 꼴찌=1/N). N이 시나리오/시드마다
    달라도(커널 HI는 K_s가 제각각) 0~1 스케일로 비교 가능하게 하는 게 목적
    (plot_kernel_group_recipe.py의 _category_of/_strip_seg_suffix와 같은 명명 규칙 재사용)."""
    n = len(names)
    order = sorted(range(n), key=lambda i: -probs[i])
    scores = {}
    for rank, i in enumerate(order, start=1):
        scores[names[i]] = (n - rank + 1) / n
    return scores


def _plot_hi_importance_ranking(run_dir: Path, figures_dir: Path, spec) -> None:
    """gates/regression_HIs.json(raw)·regression_kernel_HIs.json(kernel)의 시나리오별
    gate_prob 랭킹에 선형가중치(1위=1.0~꼴찌=1/N, 시나리오/커널폭 무관 0~1 정규화)를 매겨
    "평균(raw) / 자체값(kernel) 중요도 점수"를 계산하고, HI 전체를 이 점수로 정렬한
    가로 막대 2개(raw/kernel)를 그린다.

    raw HI는 전 시나리오에 공통 카탈로그(N_HI개)로 존재하므로, 시나리오 접미사를 뗀
    "개념 이름"(예: stat_v_mean_cw)별로 각 시나리오에서 받은 점수를 평균한다 — 결과는
    "이 HI가 시나리오를 막론하고 평균적으로 얼마나 상위권에 뽑히는가"가 된다.
    kernel HI는 이름 자체가 시나리오 전용(kernel_{seg}_g{gi})이라 여러 시나리오에
    걸쳐 존재하지 않으므로 평균이 아니라 자기 시나리오 내 점수를 그대로 쓴다 —
    막대는 색으로 소속 시나리오를 구분해 한 그림에 모아 랭킹만 매긴다.

    2026-09-21 신규(사용자 요청) — 커널 게이트가 시나리오별 own-scenario 폭(K_s)으로
    좁혀져 있어(scr_model.py _apply_scen_kernel_gate) 원본 커널 후보 수가 시나리오/시드마다
    다른데, rank/N 정규화라 그 폭 차이와 무관하게 직접 비교 가능하다."""
    if not _HAS_MPL:
        return
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False
    raw_path = run_dir / "gates" / "regression_HIs.json"
    kernel_path = run_dir / "gates" / "regression_kernel_HIs.json"
    if not raw_path.exists():
        print(f"[test_p1] {raw_path} 없음 — HI 중요도 랭킹 플랏 스킵")
        return
    raw_json = json.loads(raw_path.read_text(encoding="utf-8"))

    # raw: 시나리오 접미사를 뗀 개념 이름별로 시나리오 간 점수를 모아 평균
    raw_scores_by_concept: dict[str, list[float]] = {}
    raw_category: dict[str, str] = {}
    for s in range(spec.n_scenarios):
        seg_name = raw_json.get(f"seg_{s}_seg_name")
        names = raw_json.get(f"seg_{s}_names")
        probs = raw_json.get(f"seg_{s}_probs")
        if not names:
            continue
        suffix = f"_{seg_name}"
        concepts = [n[: -len(suffix)] if n.endswith(suffix) else n for n in names]
        scores = _rank_scores(concepts, probs)
        for c, sc in scores.items():
            raw_scores_by_concept.setdefault(c, []).append(sc)
            raw_category.setdefault(c, c.split("_", 1)[0] if "_" in c else c)
    raw_avg = {c: sum(v) / len(v) for c, v in raw_scores_by_concept.items()}
    raw_sorted = sorted(raw_avg.items(), key=lambda kv: -kv[1])

    # kernel: 이름이 이미 시나리오 전용이라 시나리오별 점수를 그대로 모아 합침
    kernel_sorted: list[tuple[str, float, str]] = []  # (name, score, seg_name)
    if kernel_path.exists():
        kernel_json = json.loads(kernel_path.read_text(encoding="utf-8"))
        for s in range(spec.n_scenarios):
            seg_name = kernel_json.get(f"seg_{s}_seg_name")
            names = kernel_json.get(f"seg_{s}_names")
            probs = kernel_json.get(f"seg_{s}_probs")
            if not names:
                continue
            scores = _rank_scores(names, probs)
            kernel_sorted.extend((n, scores[n], seg_name) for n in names)
        kernel_sorted.sort(key=lambda t: -t[1])

    seg_colors = {name: plt.get_cmap("tab10")(i % 10)
                  for i, name in enumerate(spec.scenario_names)}

    n_raw = len(raw_sorted)
    n_ker = len(kernel_sorted)
    fig, (ax_raw, ax_ker) = plt.subplots(
        1, 2, figsize=(13, max(4.0, 0.16 * max(n_raw, n_ker, 1))),
    )

    if raw_sorted:
        y = np.arange(n_raw)
        vals = [v for _, v in raw_sorted]
        colors = [_CATEGORY_COLORS.get(raw_category[c], "#888888") for c, _ in raw_sorted]
        ax_raw.barh(y, vals, color=colors, height=0.8)
        ax_raw.set_yticks(y)
        ax_raw.set_yticklabels([c for c, _ in raw_sorted], fontsize=6.5)
        ax_raw.invert_yaxis()
        ax_raw.set_xlabel("평균 중요도 점수 (시나리오 평균, rank/N 정규화)")
        ax_raw.set_title(f"raw HI (n={n_raw}, 시나리오 {spec.n_scenarios}개 평균)", fontsize=10.5, fontweight="bold")
        cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=col, markersize=8, label=cat)
                       for cat, col in _CATEGORY_COLORS.items()]
        ax_raw.legend(handles=cat_handles, loc="lower right", fontsize=7.5, title="카테고리", title_fontsize=7.5)
    else:
        ax_raw.axis("off")

    if kernel_sorted:
        y = np.arange(n_ker)
        vals = [v for _, v, _ in kernel_sorted]
        colors = [seg_colors[seg] for _, _, seg in kernel_sorted]
        ax_ker.barh(y, vals, color=colors, height=0.8)
        ax_ker.set_yticks(y)
        ax_ker.set_yticklabels([n for n, _, _ in kernel_sorted], fontsize=6)
        ax_ker.invert_yaxis()
        ax_ker.set_xlabel("중요도 점수 (자기 시나리오 내 rank/N 정규화)")
        ax_ker.set_title(f"kernel HI (n={n_ker}, 시나리오 {spec.n_scenarios}개 전부 모음)", fontsize=10.5, fontweight="bold")
        seg_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=col, markersize=8, label=name)
                       for name, col in seg_colors.items()]
        ax_ker.legend(handles=seg_handles, loc="lower right", fontsize=7.5, title="시나리오", title_fontsize=7.5)
    else:
        ax_ker.axis("off")
        ax_ker.set_title("kernel HI (없음)", fontsize=10.5)

    fig.suptitle(f"HI 중요도 랭킹 — {run_dir.name}", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = figures_dir / "hi_importance_ranking.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[test_p1] 저장: {out_path}")


def _export_for_visualize(run_dir: Path, evaluator: SCREvaluator, test_modes: dict, spec) -> None:
    """visualize_results.py의 RunBundle(__init__에서 metrics/predictions/routing을
    무조건 다 읽음)이 phase1_lab run_dir을 로드할 수 있도록, test_scr.py Phase2 run이
    남기는 것과 같은 스키마로 세 파일을 추가 저장한다. checkpoints/*.pt, config.yaml,
    scenario_spec.json, gates/*.json은 train.py가 이미 저장해두므로 손댈 필요
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

    def _write_predictions_csv(path: Path, p: dict) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "cycle", "seg_name", "soh_true", "soh_pred",
                        "cap_true_Ah", "cap_pred_Ah"])
            for i in range(len(p["cell_ids"])):
                cap_init = float(p["cap_init_raw"][i])
                soh_true = float(p["cap_true_raw"][i])
                soh_pred = float(p["cap_pred_raw"][i])
                w.writerow([p["cell_ids"][i], int(p["cycles"][i]), p["seg_names"][i],
                            soh_true, soh_pred, soh_true * cap_init, soh_pred * cap_init])

    _write_predictions_csv(predictions_dir / "test_predictions.csv", pred)
    # 2026-09-13: oracle 외 모드(hard/soft)도 별도 CSV로 저장 -- 지금까지 hard/soft
    # per-segment 예측은 그 자리에서 PNG(error_heatmap/capacity_curve)만 그리고
    # 값 자체는 저장 안 해서, 나중에(Fig5/6 재구성 때) 스타일을 다시 맞추려 해도
    # 원본 수치에 접근할 방법이 없었다 -- 이후 실험은 재평가 없이 바로 재사용 가능.
    for m in test_modes:
        if m == "oracle":
            continue
        _write_predictions_csv(predictions_dir / f"test_predictions_{m}.csv", test_modes[m]["_pred"])

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
