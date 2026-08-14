"""
test_scr.py  —  SCR evaluation entry point.

Loads a Phase 2 checkpoint and evaluates on test split.
Produces:
  - per-split metrics table (RMSE / MAE / R2 / MAPE)
  - scatter plot: pred vs true
  - capacity curve plots for representative cells

Usage:
  python 5_model/test_scr.py
  python 5_model/test_scr.py --checkpoint _5_data_model_scr/0708_1720/checkpoints/best.pt
  python 5_model/test_scr.py --rep-cells b1c0 b1c1 1-1
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

try:
    from utils.compat import install_numpy2_shim
    install_numpy2_shim()
except ImportError:
    pass

import json
import torch
import numpy as np

from utils.io_utils import load_config
from datasets.segment_dataset import build_datasets, build_random_seg_dataset, SegmentNormalizer
from models.scr_model import SCRModel
from models.scenario_classifier import MLPProbeClassifier
from evaluation.scr_evaluator import SCREvaluator
from utils.hi_schema import N_HI, spec_from_qfrac

_proj_root_c = Path(__file__).resolve().parent.parent
if str(_proj_root_c) not in sys.path:
    sys.path.insert(0, str(_proj_root_c))
from common.scenario.base import ScenarioSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SCR model")
    p.add_argument("--config",          default="5_model/config/scr.yaml")
    p.add_argument("--checkpoint",      default=None)
    p.add_argument("--classifier-ckpt", default=None,
                   help="시나리오 분류기 체크포인트 (B안 E3 routing). "
                        "미지정 시 run_dir/classifier/clf_best.pt 자동 탐색")
    p.add_argument("--device",          default="auto")
    p.add_argument("--charge-m",    type=int, default=None,
                   help="충전 probe 상위 m개 (yaml classifier.charge_probe_m 오버라이드)")
    p.add_argument("--discharge-m", type=int, default=None,
                   help="방전 probe 상위 m개 (yaml classifier.discharge_probe_m 오버라이드)")
    p.add_argument("--scen-k",      type=int, default=None,
                   help="시나리오별 scen HI 수 (yaml regression.scen_k_count 오버라이드). "
                        "다른 k로 학습된 checkpoint를 평가할 때 필요")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _build_normalizer_from_ckpt(ckpt: dict) -> SegmentNormalizer:
    norm = SegmentNormalizer()
    norm.mean_ = ckpt["norm_mean"]
    norm.std_  = ckpt["norm_std"]
    # 구버전 체크포인트(norm_target_*) 하위호환
    norm.cap_init_mean_ = float(ckpt.get("norm_cap_init_mean",
                                          ckpt.get("norm_target_mean", 0.0)))
    norm.cap_init_std_  = float(ckpt.get("norm_cap_init_std",
                                          ckpt.get("norm_target_std",  1.0)))
    return norm


def main() -> None:
    args   = _parse_args()
    device = _resolve_device(args.device)
    print(f"[test] device={device}")

    cfg = load_config(str(PROJECT_ROOT / args.config))

    # m/k 오버라이드 (다른 m/k로 학습된 checkpoint를 같은 --config로 평가할 때)
    if args.charge_m is not None:
        cfg.setdefault("classifier", {})["charge_probe_m"] = args.charge_m
    if args.discharge_m is not None:
        cfg.setdefault("classifier", {})["discharge_probe_m"] = args.discharge_m
    if args.scen_k is not None:
        cfg.setdefault("regression", {})["scen_k_count"] = args.scen_k

    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"]
    ckpt_path  = (Path(args.checkpoint) if args.checkpoint
                  else _find_latest_checkpoint(output_dir))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[test] checkpoint: {ckpt_path}")

    ckpt      = torch.load(ckpt_path, map_location="cpu")
    cfg_saved = ckpt.get("cfg", cfg)

    # ------------------------------------------------------------------
    # 데이터 재구성
    # ------------------------------------------------------------------
    # cap_init 관련 옵션은 현재 yaml 기준으로 적용 (cfg_saved가 구버전이면 키 없음)
    for k in ("use_initial_capacity", "nominal_capacities", "datasets"):
        if k in cfg.get("data", {}):
            cfg_saved.setdefault("data", {})[k] = cfg["data"][k]

    # 데이터 경로: cfg_saved(학습 시 해결된 경로)를 현재 cfg에 전파 (null 대체)
    for k in ("data_dir", "seg_data_dir"):
        if not cfg.get("data", {}).get(k) and cfg_saved.get("data", {}).get(k):
            cfg.setdefault("data", {})[k] = cfg_saved["data"][k]

    # ------------------------------------------------------------------
    # run_dir 결정 + spec 로드 (build_datasets 전에 필요)
    # ------------------------------------------------------------------
    run_dir = ckpt_path.parent.parent

    probe_json = _resolve_gate_json(run_dir, "classification_HIs.json",
                                    "scenario_classification_HIs.json")
    scen_json  = _resolve_gate_json(run_dir, "regression_HIs.json",
                                    "scenario_regression_HIs.json")

    # spec 로드 (run_dir/scenario_spec.json → 없으면 qfrac 기본값)
    _spec_path = run_dir / "scenario_spec.json"
    if _spec_path.exists():
        spec = ScenarioSpec.load(_spec_path)
        print(f"[test] spec loaded: {_spec_path}  axis={spec.axis}")
    else:
        spec = spec_from_qfrac()
        print(f"[test] scenario_spec.json 없음 — qfrac 기본값 사용")

    train_ds, val_ds, test_ds, _ = build_datasets(cfg_saved, spec=spec)
    norm = _build_normalizer_from_ckpt(ckpt)
    for ds in (train_ds, val_ds, test_ds):
        _reapply_norm(ds, norm)

    # m / k
    cls_cfg     = cfg_saved.get("classifier", cfg.get("classifier", {}))
    reg_cfg     = cfg_saved.get("regression", cfg.get("regression", {}))
    charge_m    = cfg.get("classifier", {}).get("charge_probe_m",
                  cls_cfg.get("charge_probe_m", cls_cfg.get("probe_m_count", 1)))
    discharge_m = cfg.get("classifier", {}).get("discharge_probe_m",
                  cls_cfg.get("discharge_probe_m", cls_cfg.get("probe_m_count", 1)))
    scen_k      = cfg.get("regression", {}).get("scen_k_count",
                  reg_cfg.get("scen_k_count", 5))
    auto_mk     = cfg.get("classifier", {}).get("is_auto_mk_selection",
                  cls_cfg.get("is_auto_mk_selection", False))

    m_cfg = cfg_saved.get("model", cfg.get("model", {}))
    _is_raw_mode = m_cfg.get("regression_model") == "raw_mlp"

    if _is_raw_mode:
        # raw_mlp 모드: HI 게이트가 아예 없으므로 마스크 로드를 건너뛴다.
        print("[test] raw_mlp 모드 checkpoint 감지 — HI 게이트/분류기 없이 RawMLPModel로 평가")
        ch_mask = dis_mask = scen_masks = None

        from models.raw_mlp_model import RawMLPModel
        model = RawMLPModel(
            d_head=m_cfg.get("d_head", 128),
            dropout=m_cfg.get("dropout", 0.1),
            model_cfg=m_cfg,
            spec=spec,
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
    else:
        # gate 마스크 로드
        ch_mask, dis_mask = _load_probe_masks(probe_json, charge_m, discharge_m, auto=auto_mk)
        scen_masks        = _load_scen_masks(scen_json, spec.n_scenarios, N_HI, scen_k, auto=auto_mk)

        if probe_json: print(f"[test] probe gate JSON: {probe_json}")
        if scen_json:  print(f"[test] scen  gate JSON: {scen_json}")
        if not probe_json and not scen_json:
            print("[test] gates JSON 없음 — L0 게이트 그대로 사용")

        # ------------------------------------------------------------------
        # 모델 재구성
        # ------------------------------------------------------------------
        state_keys   = set(ckpt["model_state"].keys())
        ckpt_has_l0  = ("charge_probe_gate.log_alpha" in state_keys or
                        "probe_gate.log_alpha" in state_keys)   # legacy compat

        # cap_head 구조를 yaml이 아니라 checkpoint 가중치 자체에서 추론 — 예전 Phase 1
        # checkpoint는 model_cfg가 누락된 채 학습돼(버그, train_scr.py에서 수정됨)
        # mlp_hidden_dims와 무관하게 항상 [d_head, d_head//2] 기본 shape로 저장돼 있다.
        # yaml의 mlp_hidden_dims를 그대로 믿고 재구성하면 size mismatch로 로드에 실패한다.
        _cap_dims = sorted(
            (int(_m.group(1)), ckpt["model_state"][k].shape[0])
            for k in state_keys
            if (_m := re.match(r"cap_head\.net\.(\d+)\.weight$", k))
        )
        _m_cfg_for_build = m_cfg
        if _cap_dims:
            _inferred_hidden = [d for _, d in _cap_dims[:-1]]  # 마지막 레이어(출력 dim=1) 제외
            if _inferred_hidden and _inferred_hidden != list(m_cfg.get("mlp_hidden_dims") or []):
                print(f"[test] cap_head 구조를 checkpoint 가중치에서 추론: "
                      f"mlp_hidden_dims={_inferred_hidden} (yaml 설정과 다름 — 이 checkpoint는 "
                      f"과거 구조로 학습됨)")
                _m_cfg_for_build = {**m_cfg, "mlp_hidden_dims": _inferred_hidden}

        model = SCRModel(
            d_probe=m_cfg.get("d_probe", 64),
            d_head=m_cfg.get("d_head", 128),
            dropout=m_cfg.get("dropout", 0.1),
            charge_probe_mask=ch_mask,
            discharge_probe_mask=dis_mask,
            scen_masks=scen_masks,
            model_cfg=_m_cfg_for_build,
            spec=spec,
        )
        strict_load = not (ckpt_has_l0 and ch_mask is not None)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=strict_load)
        if unexpected:
            print(f"[test] gate 파라미터 무시 (JSON 마스크 사용): {len(unexpected)}개")
        model.eval()

    # ------------------------------------------------------------------
    # 대표 셀 선정
    # ------------------------------------------------------------------
    eval_cfg  = cfg.get("evaluation", {})
    rep_cells = (eval_cfg.get("rep_cells")
                 or _pick_rep_cells(test_ds, cfg_saved,
                                    eval_cfg.get("rep_cells_per_dataset", 1)))
    print(f"[test] rep_cells: {rep_cells}")

    # ------------------------------------------------------------------
    # 평가
    # ------------------------------------------------------------------
    figures_dir     = run_dir / "figures"
    metrics_dir     = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    routing_dir     = run_dir / "routing"
    for d in (figures_dir, metrics_dir, predictions_dir, routing_dir):
        d.mkdir(parents=True, exist_ok=True)

    evaluator = SCREvaluator(
        model=model,
        normalizer=norm,
        device=device,
        figures_dir=figures_dir,
        rep_cells=rep_cells,
    )

    # ------------------------------------------------------------------
    # 시나리오 분류기 로드 (B안 — E3 routing=hard/soft 시 사용)
    # ------------------------------------------------------------------
    rs_cfg = cfg.get("test", {})
    _clf_ckpt_path = (Path(args.classifier_ckpt) if args.classifier_ckpt
                      else _resolve_classifier_ckpt(run_dir))
    if _is_raw_mode:
        # RawMLPModel은 seg_idx를 직접 입력받아 회귀 — 분류기 기반 라우팅이 무의미하므로
        # 설령 classifier 체크포인트가 남아 있어도(예: 분류기 스텝 수동 실행) 무시하고
        # oracle만 평가.
        print("[test] raw_mlp 모드 — 분류기 로드 생략, oracle 모드만 평가")
    elif _clf_ckpt_path is not None and _clf_ckpt_path.exists():
        try:
            _clf_ckpt = torch.load(_clf_ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            _clf_ckpt = torch.load(_clf_ckpt_path, map_location="cpu")
        _clf_type = _clf_ckpt.get("clf_type", "mlp")
        if _clf_type == "cnn":
            from models.scenario_classifier import CNNProbeClassifier
            _clf = CNNProbeClassifier(
                n_hi=_clf_ckpt["n_hi"],
                n_classes=_clf_ckpt["n_classes"],
                d_hidden=_clf_ckpt["d_hidden"],
            )
        else:
            _clf = MLPProbeClassifier(
                n_hi=_clf_ckpt["n_hi"],
                n_classes=_clf_ckpt["n_classes"],
                d_hidden=_clf_ckpt["d_hidden"],
            )
        _clf.load_state_dict(_clf_ckpt["clf_state"])
        evaluator.set_classifier(_clf)
        print(f"[test] 분류기 로드: {_clf_ckpt_path}  type={_clf_type}  "
              f"val_acc={_clf_ckpt.get('val_acc', float('nan')):.4f}")
    else:
        _loc = _clf_ckpt_path if _clf_ckpt_path is not None else (
            f"{run_dir/'classifier'/'clf_best.pt'} 또는 gates_from 폴더의 classifier/clf_best.pt")
        print(f"[test] 분류기 체크포인트 없음 ({_loc}) — oracle 모드만 실행.")

    # ------------------------------------------------------------------
    # 통합 평가: 학습된 분류기로 분류 → 라우팅 → 회귀 (oracle / hard / soft)
    #   oracle : 정답 seg_idx 라우팅 (회귀 상한선)
    #   hard   : 분류기 argmax 라우팅 (실배포)   ← 분류기 있을 때만
    #   soft   : 분류기 확률 가중 평균           ← 분류기 있을 때만
    # ------------------------------------------------------------------
    import json as _json
    has_clf = evaluator._classifier is not None
    route_modes = ("oracle", "hard", "soft") if has_clf else ("oracle",)
    print(f"\n[test] 통합 평가 모드: {route_modes}  (분류기 {'있음' if has_clf else '없음'})")

    test_modes  = evaluator.evaluate_modes(test_ds,  modes=route_modes)
    train_modes = evaluator.evaluate_modes(train_ds, modes=("oracle",))
    val_modes   = evaluator.evaluate_modes(val_ds,   modes=("oracle",))

    _eff = evaluator._compute_efficiency(test_modes["oracle"]["_pred"])
    metrics_out = {
        "test":  evaluator.strip_modes_for_json(test_modes,  _eff),
        "train": evaluator.strip_modes_for_json(train_modes),
        "val":   evaluator.strip_modes_for_json(val_modes),
    }
    (metrics_dir / "metrics.json").write_text(
        _json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[test] saved {metrics_dir / 'metrics.json'}")

    # 모드별 플롯: scatter(분류 색상) + confusion (hard/soft)
    for mode in route_modes:
        pred = test_modes[mode]["_pred"]
        evaluator._plot_scatter(pred, tag=f"test_{mode}")
        if mode != "oracle":
            evaluator._plot_confusion_matrix(pred, tag=f"test_{mode}")
    # 용량 곡선은 실배포(hard) 기준, 분류기 없으면 oracle
    _curve_mode = "hard" if has_clf else "oracle"
    evaluator._plot_capacity_curves(test_modes[_curve_mode]["_pred"])

    # Routing heatmap — raw_mlp는 HI 게이트/선택 자체가 없어 스킵 (heatmap이 무의미)
    if not _is_raw_mode:
        probe_sel = _selected_from_masks(ch_mask, dis_mask)   # union for heatmap
        scen_sel  = ({s: scen_masks[s].nonzero(as_tuple=False).squeeze(1).tolist()
                      for s in range(spec.n_scenarios)} if scen_masks is not None else None)
        evaluator.plot_routing_heatmap(routing_dir,
                                       probe_sel=probe_sel,
                                       scen_sel=scen_sel)

        # HI 카테고리 랭킹 히트맵 — regression_HIs.json의 랭크순 이름을 그대로 사용
        if scen_json and scen_json.exists():
            _rj = json.loads(scen_json.read_text())
            scen_ranked_names = {
                s: _rj.get(f"seg_{s}_names", []) for s in range(spec.n_scenarios)
            }
            evaluator.plot_hi_category_heatmap(routing_dir, scen_ranked_names,
                                               top_k=None)

    # Laplace UQ (train_scr.py Phase 2에서 uq.enabled=true 시 생성됨) — oracle 회귀 기준
    uq_ckpt = run_dir / "checkpoints" / "laplace_uq.pt"
    if uq_ckpt.exists():
        from utils.uncertainty import LaplaceUQ
        uq = LaplaceUQ.load(model, uq_ckpt, device)
        uq_pred = evaluator.predict_dataset_uq(test_ds, uq)
        evaluator.save_uq_metrics(uq_pred, metrics_dir)
        evaluator.plot_uq(uq_pred, figures_dir)
    # predictions CSV: 실배포(hard) 예측 저장 → level_pred가 실제 분류기 출력
    _save_mode = "hard" if has_clf else "oracle"
    evaluator.save_predictions(test_modes[_save_mode]["_pred"], predictions_dir)

    # ------------------------------------------------------------------
    # Summary (모드별 회귀 + 분류)
    # ------------------------------------------------------------------
    print("\n=== SCR Evaluation Summary (test) ===")
    header = f"{'mode':8s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}  {'MAPE':>8s}  {'clf_acc':>8s}"
    print(header); print("-" * len(header))
    for mode in route_modes:
        cap = test_modes[mode]["capacity"]
        acc = test_modes[mode]["classification"].get("accuracy")
        acc_s = f"{acc:8.4f}" if isinstance(acc, (int, float)) else f"{'N/A':>8s}"
        print(f"{mode:8s}  {cap['rmse']:8.4f}  {cap['mae']:8.4f}"
              f"  {cap['r2']:8.4f}  {cap.get('mape', float('nan')):8.2f}  {acc_s}")

    probe_his = model.get_selected_probe_his()
    scen_his  = model.get_selected_scen_his()
    avg_scen  = np.mean([len(v) for v in scen_his.values()])
    _seg_names_test = model.spec.scenario_names
    print(f"\nActive charge probe HIs    : {len(probe_his['charge'])}/{N_HI}")
    print(f"Active discharge probe HIs : {len(probe_his['discharge'])}/{N_HI}")
    print(f"Avg scen HIs/seg           : {avg_scen:.1f}/{N_HI}")
    for s, idxs in scen_his.items():
        seg_lbl = _seg_names_test[s] if s < len(_seg_names_test) else f"scen_{s}"
        print(f"  {seg_lbl:12s}: {len(idxs)} HIs selected")

    # ------------------------------------------------------------------
    # random_segment_test (선택)
    # ------------------------------------------------------------------
    rs_cfg = cfg.get("test", {})
    if rs_cfg.get("random_segment_test", False):
        _run_random_segment_test(
            cfg=cfg,
            model=model,
            norm=norm,
            evaluator=evaluator,
            run_dir=run_dir,
            metrics_dir=metrics_dir,
        )


def _run_random_segment_test(
    cfg: dict,
    model,
    norm: SegmentNormalizer,
    evaluator,
    run_dir: Path,
    metrics_dir: Path,
) -> None:
    """랜덤 세그먼트 데이터셋으로 추가 평가 (qfrac train → random test)."""
    rs_cfg      = cfg.get("test", {})
    rs_data_dir = rs_cfg.get("random_seg_data_dir", f"{DATA_4_HI_ROOT_STR}/test_rs/seg")
    rs_datasets = rs_cfg.get("random_seg_datasets",
                             cfg.get("data", {}).get("datasets", ["MIT", "HUST"]))

    print(f"\n[random_seg_test] 시작: data={rs_data_dir}  "
          f"(oracle/hard/soft 통합 평가 — 분류기 유무에 따라 자동 결정)")

    # test_rs ScenarioSpec 로드 (구 형식: n_scenarios=2 chg/dis | 재생성: n_scenarios=6 위치기반)
    rs_spec_path = PROJECT_ROOT / rs_data_dir.replace("/seg", "") / "scenario_spec.json"
    if rs_spec_path.exists():
        from common.scenario.base import ScenarioSpec as _Spec
        rs_spec = _Spec.load(rs_spec_path)
        print(f"[random_seg_test] spec: {rs_spec_path}")
    else:
        rs_spec = None
        print(f"[random_seg_test] spec 없음 — default 사용")

    data_cfg = cfg.get("data", {})
    try:
        rs_ds = build_random_seg_dataset(
            seg_data_dir=Path(rs_data_dir),
            datasets=rs_datasets,
            normalizer=norm,       # 체크포인트 norm, refit 없음
            data_cfg=data_cfg,
            spec=rs_spec,
        )
    except RuntimeError as e:
        print(f"[random_seg_test] 데이터 로드 실패: {e}")
        return

    print(f"[random_seg_test] {len(rs_ds):,} 세그먼트")

    # ── 통합 평가: oracle / hard / soft (메인 테스트와 동일 플로우) ──────────
    # 구 test_rs(seg_idx 0/1, 2-시나리오)는 oracle 라우팅 시 방향별 보정 필요.
    # 재생성 test_rs(위치기반 6-시나리오)는 seg_idx가 이미 모델공간이라 보정 불필요.
    _is_model_space = (rs_spec is not None
                       and getattr(rs_spec, "n_scenarios", 0) == model.n_scenarios)
    has_clf  = evaluator._classifier is not None
    rs_modes = ("oracle", "hard", "soft") if has_clf else ("oracle",)
    rs_res   = evaluator.evaluate_modes(
        rs_ds, modes=rs_modes,
        direction_routing_for_oracle=not _is_model_space,
    )
    print(f"[random_seg_test] spec n_scenarios={getattr(rs_spec,'n_scenarios','?')} "
          f"→ {'모델공간(위치기반 레짐)' if _is_model_space else '방향보정(구 형식)'}")
    _rs_eff = evaluator._compute_efficiency(rs_res["oracle"]["_pred"])

    print("\n=== Random Segment Test (통합) ===")
    print(f"  segs : {len(rs_ds):,}")
    header = f"  {'mode':8s}  {'RMSE':>8s}  {'MAE':>8s}  {'R2':>8s}  {'clf_acc':>8s}"
    print(header); print("  " + "-" * (len(header) - 2))
    for mode in rs_modes:
        cap   = rs_res[mode]["capacity"]
        acc   = rs_res[mode]["classification"].get("accuracy")
        acc_s = f"{acc:8.4f}" if isinstance(acc, (int, float)) else f"{'N/A':>8s}"
        print(f"  {mode:8s}  {cap['rmse']:8.4f}  {cap['mae']:8.4f}"
              f"  {cap['r2']:8.4f}  {acc_s}")

    # 저장 (모드별 회귀 + 분류)
    rs_dir = run_dir / "random_seg_test"
    rs_dir.mkdir(exist_ok=True)
    import json as _json
    (rs_dir / "metrics.json").write_text(
        _json.dumps(evaluator.strip_modes_for_json(rs_res, _rs_eff),
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 예측 CSV: 실배포(hard) 기준, 없으면 oracle
    _rs_save = "hard" if has_clf else "oracle"
    evaluator.save_predictions(rs_res[_rs_save]["_pred"], rs_dir, tag="random_seg")

    # 모드별 scatter + confusion + 용량 곡선
    rs_fig_dir   = rs_dir / "figures"
    rs_rep_cells = _pick_rs_rep_cells(
        rs_ds,
        rs_data_dir_path=PROJECT_ROOT / rs_data_dir,
        rs_datasets=rs_datasets,
        n_per_ds=cfg.get("evaluation", {}).get("rep_cells_per_dataset", 3),
    )
    print(f"[random_seg_test] rep_cells: {rs_rep_cells}")
    for mode in rs_modes:
        evaluator.plot_for_dataset(
            rs_res[mode]["_pred"], rs_fig_dir, rs_rep_cells, tag=f"random_seg_{mode}",
        )

    print(f"[random_seg_test] 저장 → {rs_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: Path) -> Path:
    if not output_dir.exists():
        return output_dir / "checkpoints" / "final.pt"
    run_dirs   = [d for d in output_dir.iterdir()
                  if d.is_dir() and len(d.name) == 9 and d.name[4] == "_"]
    candidates = [d / "checkpoints" / "final.pt" for d in run_dirs
                  if (d / "checkpoints" / "final.pt").exists()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    old = [d / "scr_final.pt" for d in run_dirs if (d / "scr_final.pt").exists()]
    if old:
        return max(old, key=lambda p: p.stat().st_mtime)
    return output_dir / "checkpoints" / "final.pt"


def _resolve_classifier_ckpt(run_dir: Path) -> Path | None:
    """분류기 체크포인트 위치를 두 관례 모두 지원해 찾는다.

    - 기존 관례(레거시, 기존 run 전부 이 위치): {run_dir}/classifier/clf_best.pt
      (train_classifier.py --run-dir에 Phase 2 폴더를 줬을 때 저장되는 위치)
    - 신규 관례(2026-07-30 파이프라인 재배치 이후): Phase 1 폴더(gates_from)의
      classifier/clf_best.pt — 분류기가 Phase 2보다 먼저(Phase 1 직후) 학습되어
      Phase 2가 with_raw_cnn으로 그 결과를 바로 재사용할 수 있게 되면서 저장 위치도
      Phase 1 폴더로 옮겨감. _resolve_gate_json과 동일하게 config.yaml의
      data.gates_from을 따라가 찾는다.
    """
    legacy = run_dir / "classifier" / "clf_best.pt"
    if legacy.exists():
        return legacy
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    import yaml as _yaml
    with open(cfg_path, encoding="utf-8") as f:
        saved_cfg = _yaml.safe_load(f)
    gates_from = saved_cfg.get("data", {}).get("gates_from")
    if not gates_from:
        return None
    from_dir = Path(gates_from) if Path(gates_from).is_absolute() else PROJECT_ROOT / gates_from
    new_loc = from_dir / "classifier" / "clf_best.pt"
    return new_loc if new_loc.exists() else None


def _resolve_gate_json(run_dir: Path, new_name: str, old_name: str) -> Path | None:
    p = run_dir / "gates" / new_name
    if p.exists():
        return p
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    import yaml as _yaml
    with open(cfg_path, encoding="utf-8") as f:
        saved_cfg = _yaml.safe_load(f)
    if saved_cfg.get("data", {}).get("gates_ignore", False):
        return None
    gates_from = saved_cfg.get("data", {}).get("gates_from")
    if not gates_from:
        return None
    from_dir   = PROJECT_ROOT / gates_from
    search_dir = (from_dir / "gates") if (from_dir / "gates").exists() else from_dir
    for name in (new_name, old_name):
        p = search_dir / name
        if p.exists():
            return p
    return None


def _load_probe_masks(
    json_path: Path | None, charge_m: int, discharge_m: int,
    auto: bool = False, threshold: float = 0.5,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not json_path or not json_path.exists():
        return None, None
    data = json.loads(json_path.read_text())

    def _mask(indices: list[int]) -> torch.Tensor:
        m = torch.zeros(N_HI, dtype=torch.bool)
        for i in indices:
            m[i] = True
        return m

    def _threshold_mask(ranked: list[int], probs: list[float]) -> torch.Tensor:
        selected = [idx for idx, p in zip(ranked, probs) if p >= threshold]
        return _mask(selected if selected else ranked[:1])

    if "charge_ranked" in data:
        if auto:
            return (_threshold_mask(data["charge_ranked"],    data.get("charge_probs", [])),
                    _threshold_mask(data["discharge_ranked"], data.get("discharge_probs", [])))
        return (_mask(data["charge_ranked"][:charge_m]),
                _mask(data["discharge_ranked"][:discharge_m]))
    # backward compat: single ranked_indices
    indices = data.get("ranked_indices", [])
    if auto:
        probs = data.get("probs", [1.0] * len(indices))
        ch = _threshold_mask(indices, probs)
        return ch, ch
    return _mask(indices[:charge_m]), _mask(indices[:discharge_m])


def _load_scen_masks(
    json_path: Path | None, n_segs: int, n_hi: int, k: int,
    auto: bool = False, threshold: float = 0.5,
) -> torch.Tensor | None:
    if not json_path or not json_path.exists():
        return None
    data  = json.loads(json_path.read_text())
    masks = torch.zeros(n_segs, n_hi, dtype=torch.bool)
    for s in range(n_segs):
        if f"seg_{s}_ranked" in data:
            ranked = data[f"seg_{s}_ranked"]
            if auto:
                probs    = data.get(f"seg_{s}_probs", [])
                selected = [idx for idx, p in zip(ranked, probs) if p >= threshold]
                indices  = selected if selected else ranked[:1]
            else:
                indices = ranked[:k]
        else:
            indices = data.get(f"seg_{s}", [])
        for i in indices:
            masks[s, i] = True
    return masks


def _selected_from_masks(
    ch_mask: torch.Tensor | None, dis_mask: torch.Tensor | None
) -> list[int] | None:
    """Union of charge + discharge probe masks → routing heatmap의 probe row."""
    if ch_mask is None and dis_mask is None:
        return None
    union = torch.zeros(N_HI, dtype=torch.bool)
    if ch_mask  is not None: union |= ch_mask
    if dis_mask is not None: union |= dis_mask
    return union.nonzero(as_tuple=False).squeeze(1).tolist()


def _pick_rep_cells(test_ds, cfg: dict, n_per_dataset: int = 1) -> list[str]:
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


def _pick_rs_rep_cells(
    rs_ds,
    rs_data_dir_path: Path,
    rs_datasets: list[str],
    n_per_ds: int = 3,
) -> list[str]:
    """랜덤 세그먼트 데이터셋에서 데이터셋별 n_per_ds개 대표 셀 선정."""
    all_cells = sorted(set(rs_ds.cell_ids))
    picked: list[str] = []
    for ds_name in rs_datasets:
        ds_dir = rs_data_dir_path / ds_name
        if ds_dir.exists():
            ds_cells = {p.stem for p in ds_dir.glob("*.pkl")}
            candidates = [c for c in all_cells if c in ds_cells]
        else:
            candidates = all_cells
        picked.extend(candidates[:n_per_ds])
    return picked or all_cells[:n_per_ds * max(len(rs_datasets), 1)]


def _reapply_norm(ds, norm: SegmentNormalizer) -> None:
    """체크포인트 normalizer를 Dataset의 target·cap_init에 재적용.

    x_hi는 build_datasets 내부 norm과 checkpoint norm이 동일 데이터에서 fit됐으므로
    동일 값이 보장됨 → 재계산 생략. random_segment_test는 별도 경로(build_random_seg_dataset)
    에서 checkpoint norm을 직접 사용하므로 이 함수를 거치지 않는다.
    """
    ds.target = torch.tensor(
        ds.capacity_raw / np.where(ds.cap_init_raw > 0, ds.cap_init_raw, 1.0),
        dtype=torch.float32,
    )
    ds.cap_init = torch.tensor(
        norm.transform_cap_init(ds.cap_init_raw), dtype=torch.float32
    )


if __name__ == "__main__":
    main()
