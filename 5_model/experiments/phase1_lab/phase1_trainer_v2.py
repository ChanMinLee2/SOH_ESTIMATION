"""
5_model/experiments/phase1_lab/phase1_trainer_v2.py

Stage1(체크포인트 선택 기준 변경) + Stage2(temperature annealing)를 적용한
독립 Phase1 트레이너.

기존 5_model/train_scr.py, 5_model/training/scr_trainer.py,
5_model/models/hard_concrete.py는 단 한 줄도 수정하지 않는다 — 전부 그대로
IMPORT해서 재사용(SCRModel/SCRLoss/build_datasets/L0LambdaScheduler/게이트 JSON
저장 함수)하고, "에폭을 몇 번 돌고 어느 시점을 최종본으로 저장할지"를 결정하는
학습 루프만 이 파일이 독자적으로 새로 짠다.

기존 SCRTrainer.fit() 대비 차이점 (docs/PHASE1_REDESIGN.md §3 Stage1/2):
  1. 체크포인트 선택 기준: val RMSE 최소가 아니라 "게이트 포화도"
     (gate_prob이 애매한 [0.1, 0.9] 구간에 남아있는 비율)가 가장 낮은 시점을
     저장한다. 단, L0 램프(warmup+ramp)가 완전히 끝난 이후 에폭만 후보로 삼는다
     — 이래야 "페널티가 실제로 적용된 상태"에서 고른 게 보장된다(기존 버그의
     직접적인 원인이었던 "페널티 걸리기 전에 저장" 문제를 구조적으로 차단).
  2. Temperature annealing: HardConcreteGate.BETA는 클래스 상수지만
     self.BETA로 접근하므로 인스턴스별 오버라이드가 가능하다(hard_concrete.py
     무변경). L0 램프 구간과 맞물려 2/3 -> --beta-min으로 선형으로 낮춰
     게이트를 더 확실하게 0/1로 밀어붙인다.

출력 레이아웃은 기존 Phase1 run과 100% 동일하게 맞춘다(gates/classification_HIs.json,
gates/regression_HIs.json, scenario_spec.json, logs/train_log_v2.csv) — 그래야
analyze_convergence.py / materialize_ensemble_gates.py를 무수정으로 재사용할 수 있다.

사용 예:
  python 5_model/experiments/phase1_lab/phase1_trainer_v2.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --axis-config '{"n1":0.35,"n2":0.20,"ref_lag":0,"noise_amp":0.03,"noise_mode":"ou","noise_period_cycles":200,"n_samples":4}' \
      --scen-k 25 --seed 42 --split-seed 42 --beta-min 0.1 --tag stage12_k25
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from utils.io_utils import load_config, save_config  # noqa: E402
from utils.hi_schema import N_HI, get_hi_cols_for_seg  # noqa: E402
from utils.metrics import rmse as _rmse, r2 as _r2  # noqa: E402
from utils.tqdm_utils import trange, write as tqdm_write  # noqa: E402
from datasets.segment_dataset import build_datasets, FastTensorLoader  # noqa: E402
from models.scr_model import SCRModel  # noqa: E402
from training.scr_loss import SCRLoss  # noqa: E402
from training.scr_trainer import L0LambdaScheduler  # noqa: E402
from common.scenario import get_segmenter  # noqa: E402

# train_scr.py는 스크립트지만 __main__ 가드가 있어 import해도 안전 — JSON 저장
# 함수만 재사용(중복 구현 금지 원칙, docs/PHASE1_REDESIGN.md 참고).
import train_scr as _base  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "p1v2_runs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 v2 — 체크포인트 기준 변경 + temperature annealing")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--axis-config", required=True)
    p.add_argument("--charge-m", type=int, default=None)
    p.add_argument("--discharge-m", type=int, default=None)
    p.add_argument("--scen-k", type=int, default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--data-dir", default=None, help="cycle pkl 경로 (cfg['data']['data_dir'] 오버라이드)")
    p.add_argument("--seg-data-dir", default=None, help="seg pkl 경로 (cfg['data']['seg_data_dir'] 오버라이드)")
    p.add_argument("--beta-min", type=float, default=0.1, help="annealing 종착 BETA(기본 0.1, 원래값 2/3)")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="cfg의 training.epochs(기본 500) 대신 쓸 상한. 시간 단축용 — "
                        "--patience 조기종료가 그 전에 걸리면 이 값까지 안 감")
    p.add_argument("--patience", type=int, default=60,
                   help="L0 완전 램프 이후, 게이트 포화도가 이 에폭 수만큼 연속 개선 없으면 "
                        "조기 종료(기본 60 — main_qfref_S_p60.yaml과 동일 근거). "
                        "베스트 체크포인트 자체는 patience 길이와 무관하게 항상 보존되므로 "
                        "결과에는 영향 없고 학습 시간만 줄어듦. 0이면 조기종료 비활성(항상 --max-epochs까지)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="cfg의 training.batch_size 오버라이드. 배치가 클수록 에폭당 배치 수가 "
                        "줄어 Stage A/B 게이트 루프(방향×시나리오별 서브포워드)의 배치당 "
                        "오버헤드 총합이 줄어든다 — 값 자체이 학습 결과를 바꿀 수 있으니 "
                        "(선형 학습이 아니라 완전 무해하지는 않음) 처음 켤 때는 baseline과 "
                        "1개 seed로 비교 검증 권장")
    p.add_argument("--regression-model", default="mlp",
                   choices=["mlp", "transformer", "i_transformer", "resnet_tab", "ft_transformer"],
                   help="Phase1 cap_head 종류. 원래 train_scr.py는 이걸 항상 'mlp'로 강제한다"
                        "(design intent: 게이트 선택이 최종 헤드 아키텍처와 무관해야 함) — "
                        "이 오버라이드는 그 전제가 실제로 맞는지 검증하기 위한 sanity check 전용. "
                        "기본값(mlp)이면 기존 train_scr.py Phase1과 동일하게 동작.")
    p.add_argument("--synergy-groups-json", default=None, dest="synergy_groups_json",
                   help="build_synergy_groups.py 산출물(synergy_groups_*.json) 경로. 주어지면 "
                        "scen_gates가 시나리오별 그룹 계층 게이트(GroupedHardConcreteGate)로 "
                        "학습됨 — train_scr.py의 --synergy-groups-json과 동일 로더 재사용")
    p.add_argument("--tag", required=True)
    return p.parse_args()


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _gate_saturation_fraction(model: SCRModel) -> float:
    """게이트 확률이 애매한 [0.1,0.9] 구간에 있는 비율 — 낮을수록 더 확실하게 이산화됨."""
    probs = []
    for gate in [model.charge_probe_gate, model.discharge_probe_gate, *model.scen_gates]:
        probs.append(gate.gate_prob().detach().cpu())
    p = torch.cat(probs)
    return float(((p > 0.1) & (p < 0.9)).float().mean().item())


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(str(PROJECT_ROOT / args.model_config))
    axis_cfg = json.loads(args.axis_config)
    cfg.setdefault("scenario", {})["axis"] = args.seg_axis
    cfg["scenario"]["axis_config"] = axis_cfg
    cfg.setdefault("data", {})["split_seed"] = args.split_seed
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.seg_data_dir is not None:
        cfg["data"]["seg_data_dir"] = args.seg_data_dir

    cls_cfg = cfg.setdefault("classifier", {})
    reg_cfg = cfg.setdefault("regression", {})
    if args.charge_m is not None: cls_cfg["charge_probe_m"] = args.charge_m
    if args.discharge_m is not None: cls_cfg["discharge_probe_m"] = args.discharge_m
    if args.scen_k is not None: reg_cfg["scen_k_count"] = args.scen_k
    charge_m = cls_cfg.get("charge_probe_m", 10)
    discharge_m = cls_cfg.get("discharge_probe_m", 10)
    scen_k = reg_cfg.get("scen_k_count", 5)

    spec = get_segmenter(args.seg_axis, {args.seg_axis: axis_cfg}).get_spec()

    # 데이터 경로: train_scr.py의 자동 경로계산(_axis_dir) 로직을 재사용하지 않으므로,
    # yaml에 없으면 --data-dir/--seg-data-dir로 직접 줘야 한다.
    if not cfg["data"].get("data_dir") or not cfg["data"].get("seg_data_dir"):
        raise RuntimeError(
            "cfg['data']['data_dir']/['seg_data_dir']가 비어 있습니다 — "
            "이 v2 트레이너는 train_scr.py의 자동 경로계산 로직을 재사용하지 않으므로, "
            "--model-config에 이미 박혀있지 않다면 --data-dir/--seg-data-dir을 직접 넘겨주세요."
        )

    train_ds, val_ds, _test_ds, norm = build_datasets(cfg, spec=spec)
    tr_cfg = cfg["training"]
    if args.batch_size is not None:
        print(f"[p1v2] batch_size 오버라이드: {tr_cfg['batch_size']} -> {args.batch_size}")
        tr_cfg["batch_size"] = args.batch_size
    train_loader = FastTensorLoader(train_ds, tr_cfg["batch_size"], shuffle=True)
    val_loader = FastTensorLoader(val_ds, tr_cfg["batch_size"], shuffle=False)

    lambda_scen = cfg.get("loss", {}).get("lambda_scen", 0.0)
    with_probe_mlp = lambda_scen > 0
    # 원래 train_scr.py는 여기를 항상 "mlp"로 강제한다(design intent) — 이 v2 트레이너는
    # --regression-model로 그 전제를 sanity-check할 수 있게 열어둔다(기본값은 "mlp"라
    # 오버라이드 안 주면 기존과 100% 동일 동작).
    p1_model_cfg = {**cfg["model"], "regression_model": args.regression_model,
                     "with_raw_cnn": False, "with_raw_flat": False}

    scen_group_ids = None
    if args.synergy_groups_json:
        scen_group_ids = _base._load_synergy_group_ids(
            Path(args.synergy_groups_json), spec.n_scenarios, spec.scenario_names,
        )
        n_groups_total = sum(len(set(g)) for g in scen_group_ids.values())
        print(f"[p1v2] synergy-groups-json 적용: {args.synergy_groups_json} "
              f"({len(scen_group_ids)}/{spec.n_scenarios}개 시나리오, 총 그룹 {n_groups_total}개)")

    model = SCRModel(
        d_probe=cfg["model"]["d_probe"], d_head=cfg["model"]["d_head"], dropout=cfg["model"]["dropout"],
        spec=spec, with_probe_mlp=with_probe_mlp, model_cfg=p1_model_cfg,
        scen_group_ids=scen_group_ids,
    ).to(device)

    loss_cfg = cfg["loss"]
    loss_fn = SCRLoss(lambda_scen=lambda_scen, lambda_l0=loss_cfg["lambda_l0"]).to(device)

    if loss_cfg.get("lambda_l0_auto", False):
        avg_m = (charge_m + discharge_m) / 2
        probe_scale = 10 / max(avg_m, 1)
        scen_scale = 10 / max(scen_k, 1)
        auto_lambda = round(max(1e-4, min(0.01 * math.sqrt(probe_scale * scen_scale), 0.5)), 5)
        loss_cfg["lambda_l0"] = auto_lambda
        print(f"[p1v2] lambda_l0_auto: charge_m={charge_m} discharge_m={discharge_m} scen_k={scen_k} -> {auto_lambda}")

    epochs = args.max_epochs if args.max_epochs is not None else tr_cfg["epochs"]
    if args.max_epochs is not None:
        print(f"[p1v2] epochs 상한 오버라이드: {tr_cfg['epochs']} -> {epochs}")
    warmup_ep = tr_cfg.get("warmup_epochs", 10)
    l0_scheduler = L0LambdaScheduler(target=loss_cfg["lambda_l0"], loss_cfg=loss_cfg, total_epochs=epochs)
    l0_warmup_ep = loss_cfg.get("lambda_l0_warmup_epochs", 50)
    l0_ramp_ep = loss_cfg.get("lambda_l0_ramp_epochs", 50)
    l0_fully_ramped_ep = l0_warmup_ep + l0_ramp_ep  # 이 에폭부터 체크포인트 후보로 인정

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_ep, 1), eta_min=1e-6)

    beta_default = 2.0 / 3.0
    beta_min = args.beta_min

    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = RESULTS_DIR / f"{timestamp}_p1v2_{args.tag}_seed{args.seed}"
    (output_dir / "gates").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoints" / "best_by_saturation.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    spec.save(output_dir / "scenario_spec.json")

    log_path = output_dir / "logs" / "train_log_v2.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,lambda_l0,beta,tr_rmse,tr_r2,val_rmse,val_r2,gate_saturation,is_selected\n")

    best_sat = float("inf")
    best_epoch = -1
    no_improve = 0  # L0 완전 램프 이후, best_sat 갱신 없이 지난 에폭 수

    for epoch in trange(epochs, desc=f"[p1v2:{args.tag}] seed={args.seed}"):
        if epoch < warmup_ep:
            lr = tr_cfg["lr"] * (epoch + 1) / warmup_ep
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        eff_l0 = l0_scheduler.get(epoch)
        loss_fn.lambda_l0 = eff_l0

        # Stage2: L0 램프 구간과 맞물린 BETA annealing (경계 밖에서는 각각 상수 유지)
        if epoch < l0_warmup_ep:
            beta_now = beta_default
        elif epoch < l0_fully_ramped_ep:
            frac = (epoch - l0_warmup_ep) / max(l0_ramp_ep, 1)
            beta_now = beta_default + (beta_min - beta_default) * frac
        else:
            beta_now = beta_min
        for gate in [model.charge_probe_gate, model.discharge_probe_gate, *model.scen_gates]:
            gate.BETA = beta_now

        # ---- train epoch ----
        model.train()
        tr_preds, tr_targets = [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            losses = loss_fn(out, batch, model)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg.get("grad_clip", 1.0))
            optimizer.step()
            tr_preds.append(out["cap_pred"].detach().cpu())
            tr_targets.append(batch["target"].cpu())
        tr_p, tr_t = torch.cat(tr_preds).numpy(), torch.cat(tr_targets).numpy()
        tr_rmse_v, tr_r2_v = float(_rmse(tr_t, tr_p)), float(_r2(tr_t, tr_p))

        if epoch >= warmup_ep:
            scheduler.step()

        # ---- val epoch ----
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch)
                val_preds.append(out["cap_pred"].cpu())
                val_targets.append(batch["target"].cpu())
        val_p, val_t = torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy()
        val_rmse_v, val_r2_v = float(_rmse(val_t, val_p)), float(_r2(val_t, val_p))

        sat = _gate_saturation_fraction(model)

        # Stage1: 체크포인트 선택 = "L0가 완전히 램프된 이후" 구간에서 포화도 최소
        is_selected = False
        if epoch >= l0_fully_ramped_ep:
            if sat < best_sat:
                best_sat = sat
                best_epoch = epoch
                is_selected = True
                no_improve = 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "gate_saturation": sat,
                            "val_rmse": val_rmse_v}, ckpt_path)
            else:
                no_improve += 1

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{eff_l0:.6f},{beta_now:.4f},{tr_rmse_v:.6f},{tr_r2_v:.6f},"
                    f"{val_rmse_v:.6f},{val_r2_v:.6f},{sat:.6f},{int(is_selected)}\n")

        if (epoch + 1) % 10 == 0 or is_selected:
            tqdm_write(f"epoch {epoch+1:4d}  lambda_l0={eff_l0:.4f}  beta={beta_now:.3f}  "
                       f"tr_r2={tr_r2_v:.4f}  val_r2={val_r2_v:.4f}  sat={sat:.3f}" + (" *selected*" if is_selected else ""))

        # 조기종료: best_sat 갱신 없이 --patience 에폭이 지나면 중단. 이후 남은 에폭을
        # 더 돌아도 이미 저장된 best_sat 체크포인트가 바뀌지 않으므로(항상 진짜 best만
        # 저장) 결과에는 영향 없이 시간만 절약된다 — SCRTrainer의 patience와 동일 원리.
        if args.patience > 0 and no_improve >= args.patience:
            best_ep_str = str(best_epoch + 1) if best_epoch >= 0 else "없음(한 번도 개선 안 됨)"
            tqdm_write(f"[p1v2] 조기종료: epoch {epoch+1} (best epoch={best_ep_str}, "
                       f"{args.patience}에폭 연속 포화도 개선 없음)")
            break

    if best_epoch < 0:
        tqdm_write("[p1v2] 경고: L0 완전 램프 이후 구간에서 포화도가 한 번도 개선되지 않음 — "
                   "마지막으로 돈 에폭을 그대로 채택합니다(epochs를 늘리거나 beta_min을 더 낮춰보세요).")
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "gate_saturation": sat,
                    "val_rmse": val_rmse_v}, ckpt_path)
        best_epoch = epoch

    # ---- 최종 선택 체크포인트 복원 후 게이트 JSON 저장 (기존 함수 그대로 재사용) ----
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    hi_cols_ref = get_hi_cols_for_seg("dis_hi")
    hi_cols_by_seg = {s: get_hi_cols_for_seg(spec.scenario_names[s]) for s in range(spec.n_scenarios)}
    _base._save_probe_masks_to_json(model, output_dir / "gates" / "classification_HIs.json", hi_cols_ref)
    _base._save_scen_masks_to_json(model, output_dir / "gates" / "regression_HIs.json", hi_cols_by_seg)
    _base._plot_gate_probs(
        model, output_dir / "gates" / "gate_probs.png", hi_cols_ref,
        charge_m, discharge_m, scen_k,
    )

    cfg.setdefault("data", {})["exclude_stat_leak"] = None  # v2 트레이너 표식용, 필요시 실값으로 교체
    save_config(cfg, output_dir / "config.yaml")

    summary = {
        "tag": args.tag, "seed": args.seed, "split_seed": args.split_seed,
        "selected_epoch": best_epoch, "gate_saturation": best_sat,
        "beta_min": beta_min, "l0_fully_ramped_epoch": l0_fully_ramped_ep,
        "output_dir": str(output_dir),
        "synergy_groups_json": args.synergy_groups_json,
        "synergy_n_groups": ({s: max(g) + 1 for s, g in scen_group_ids.items()}
                              if scen_group_ids else None),
    }
    (output_dir / "p1v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[p1v2] 선택된 epoch={best_epoch} (gate_saturation={best_sat:.4f})")
    print(f"[p1v2] run dir: {output_dir}")


if __name__ == "__main__":
    main()
