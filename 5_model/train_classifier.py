"""
train_classifier.py — Direction-aware scenario classifier (B안).

HI 세그먼트 데이터로 시나리오 레짐(level) 분류 모델을 독립 학습한다.
회귀 모델(SCRModel Phase 1/2)과 완전히 분리된 독립 모듈이며,
test 시 routing=hard/soft 실험(E3)에서만 사용된다.

입력 : probe_x (N_HI=64,) — Phase 1 probe gate 상위 m개만 active, 나머지 0
       (classification_HIs.json 에서 charge/discharge 방향별 마스크 로드)
       마스크 파일이 없으면 x_hi 전체 사용 (fallback)
출력 : n_classes logits  (qfrac: 3, protocol: axis별)
손실 : CrossEntropyLoss 전용
저장 : {run_dir}/classifier/clf_best.pt

사용:
  python 5_model/train_classifier.py --run-dir _5_data_model_scr/0716_1200_p2_mlp_prot
  python 5_model/train_classifier.py --config 5_model/config/scr.yaml \\
                                     --run-dir _5_data_model_scr/0716_1200_p2_mlp_prot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.io_utils import load_config
from datasets.segment_dataset import build_datasets, FastTensorLoader
from models.scenario_classifier import MLPProbeClassifier, CNNProbeClassifier
from utils.hi_schema import N_HI, spec_from_qfrac
from common.scenario.base import ScenarioSpec
from utils.tqdm_utils import trange, write as tqdm_write


def _collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="시나리오 분류기 학습 (B안 — 회귀와 완전 분리)")
    p.add_argument("--config",   default="5_model/config/scr.yaml")
    p.add_argument("--run-dir",  default=None,
                   help="Phase 2 run 폴더 경로. 분류기를 {run_dir}/classifier/에 저장. "
                        "미지정 시 output_dir 내 가장 최신 폴더 사용")
    p.add_argument("--device",   default="auto")
    p.add_argument("--epochs",   type=int, default=None,
                   help="최대 에폭 (yaml training.epochs 오버라이드)")
    p.add_argument("--lr",       type=float, default=None,
                   help="학습률 (yaml training.lr 오버라이드)")
    p.add_argument("--d-hidden", type=int, default=64,
                   help="MLPProbeClassifier hidden dim (기본 64)")
    p.add_argument("--exclude-cv", action="store_true", dest="exclude_cv",
                   help="train_scr.py --exclude-cv 로 학습된 run의 '_ccOnly' 데이터 경로 사용 "
                        "(spec 기반 자동 경로 재구성에 접미사 추가). yaml data.exclude_cv 로도 설정 가능")
    return p.parse_args()


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _find_latest_run_dir(output_dir: Path) -> Path | None:
    dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime)
    return dirs[-1] if dirs else None


def _axis_dir_from_spec(spec: "ScenarioSpec") -> str:
    """spec.axis + spec.params → _4_data_hi 하위 경로 문자열."""
    p = spec.params or {}
    # random_segment=True 면 _random-L{seg_len_pts} suffix (hi_correlation._rand_suffix 와 동일)
    rand_sfx = f"_random-L{int(p.get('seg_len_pts', 20))}" if p.get("random_segment", False) else ""
    if spec.axis == "q_frac_wide":
        n1 = int(round(p.get("n1", 0.4) * 100))
        n2 = int(round(p.get("n2", 0.2) * 100))
        ns = int(p.get("n_samples", 4))
        return f"q_frac_wide/n1-{n1}%_n2-{n2}%_N-{ns}{rand_sfx}"
    if spec.axis == "q_abs":
        ms = int(round(p.get("mid_start", 0.20) * 100))
        me = int(round(p.get("mid_end", 0.50) * 100))
        sl = int(round(p.get("seg_len", 0.15) * 100))
        ns = int(p.get("n_samples", 4))
        return f"q_abs/ms-{ms}%_me-{me}%_sl-{sl}%_N-{ns}{rand_sfx}"
    if spec.axis == "vqslope":
        mode = str(p.get("mode", "dva")).lower()
        ns   = int(p.get("n_samples", 1))
        return f"vqslope/{mode}_N-{ns}{rand_sfx}"
    return spec.axis



def _load_probe_masks(
    run_dir: Path, cfg: dict, device: torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Phase 1 classification_HIs.json에서 charge/discharge probe 마스크 로드.
    마스크는 float (N_HI,): 선택된 HI 위치=1.0, 나머지=0.0.
    파일 없으면 (None, None) 반환 → training loop에서 x_hi 전체 사용.
    """
    probe_json = run_dir / "gates" / "classification_HIs.json"
    if not probe_json.exists():
        return None, None

    data = json.loads(probe_json.read_text(encoding="utf-8"))
    clf_cfg = cfg.get("classifier", {})
    ch_m    = clf_cfg.get("charge_probe_m",    3)
    dis_m   = clf_cfg.get("discharge_probe_m", 2)

    ch_mask  = torch.zeros(N_HI, dtype=torch.float32)
    dis_mask = torch.zeros(N_HI, dtype=torch.float32)
    for i in data.get("charge_ranked",    [])[:ch_m]:
        ch_mask[i]  = 1.0
    for i in data.get("discharge_ranked", [])[:dis_m]:
        dis_mask[i] = 1.0

    return ch_mask.to(device), dis_mask.to(device)


def _apply_probe_mask(
    x_hi: torch.Tensor,
    direction: torch.Tensor,
    ch_mask: torch.Tensor | None,
    dis_mask: torch.Tensor | None,
) -> torch.Tensor:
    """direction-aware probe masking → probe_x (B, N_HI)."""
    if ch_mask is None:
        return x_hi
    probe_x = torch.zeros_like(x_hi)
    ch_sel  = direction > 0
    dis_sel = direction <= 0
    if ch_sel.any():
        probe_x[ch_sel]  = x_hi[ch_sel]  * ch_mask
    if dis_sel.any():
        probe_x[dis_sel] = x_hi[dis_sel] * dis_mask
    return probe_x


# ---------------------------------------------------------------------------
# 학습 기록 CSV — scr_trainer.py의 train_log.csv 와 동일한 패턴
# ---------------------------------------------------------------------------

_LOG_COLS = ["epoch", "tr_loss", "tr_acc", "val_loss", "val_acc", "lr", "elapsed_s"]


def _init_log_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(_LOG_COLS)


def _append_log_csv(
    path: Path, epoch: int, tr_loss: float, tr_acc: float,
    vl_loss: float, vl_acc: float, lr: float, elapsed: float,
) -> None:
    row = [
        epoch,
        f"{tr_loss:.6f}", f"{tr_acc:.6f}",
        f"{vl_loss:.6f}", f"{vl_acc:.6f}",
        f"{lr:.6e}", f"{elapsed:.2f}",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main() -> None:
    args   = _parse_args()
    device = _resolve_device(args.device)
    print(f"[clf] device={device}")

    cfg = load_config(str(PROJECT_ROOT / args.config))

    # run_dir 결정
    output_dir = PROJECT_ROOT / cfg["data"]["output_dir"]
    if args.run_dir:
        run_dir = PROJECT_ROOT / args.run_dir
    else:
        run_dir = _find_latest_run_dir(output_dir)
    if run_dir is None or not run_dir.exists():
        raise RuntimeError(
            "run_dir을 찾을 수 없습니다. --run-dir 옵션으로 Phase 2 폴더를 지정하세요."
        )
    print(f"[clf] run_dir: {run_dir}")

    # raw_mlp 모드 run — RawMLPModel은 seg_idx를 직접 입력받아 회귀하므로
    # HI 기반 방향/시나리오 분류기가 의미가 없다. 학습을 건너뛴다.
    _saved_cfg_path = run_dir / "config.yaml"
    if _saved_cfg_path.exists():
        _saved_cfg = load_config(str(_saved_cfg_path))
        if _saved_cfg.get("model", {}).get("regression_model") == "raw_mlp":
            print("[clf] raw_mlp 모드 run 감지 — 시나리오 분류기 학습 생략 "
                  "(RawMLPModel은 라우팅이 필요 없음). test_scr.py는 자동으로 oracle만 평가합니다.")
            return

    # spec 로드
    spec_path = run_dir / "scenario_spec.json"
    if spec_path.exists():
        spec = ScenarioSpec.load(spec_path)
        print(f"[clf] spec: axis={spec.axis}  n_classes={spec.n_classes}  "
              f"n_scenarios={spec.n_scenarios}")
    else:
        spec = spec_from_qfrac()
        print(f"[clf] scenario_spec.json 없음 — qfrac 기본값 사용")

    _data_cfg = cfg.setdefault("data", {})
    _axis_dir = _axis_dir_from_spec(spec)

    # --exclude-cv: train_scr.py --exclude-cv 로 학습된 run은 '_ccOnly' 경로를 썼으므로
    # spec 기반 재구성에도 동일 접미사를 붙여야 한다 (spec.params에는 이 정보가 없음 —
    # segmenter 생성자 kwargs가 아니라서 저장되지 않으므로 CLI/yaml로 명시적으로 전달).
    _exclude_cv = args.exclude_cv or bool(cfg.get("data", {}).get("exclude_cv", False))
    if _exclude_cv:
        _axis_dir = f"{_axis_dir}_ccOnly"

    if spec.axis in ("q_frac_wide", "q_abs", "vqslope"):
        # spec.params에서 태그 재구성 — 저장된 config 경로(구버전)보다 항상 우선
        _data_cfg["seg_data_dir"] = f"_4_data_hi/{_axis_dir}/seg"
        _data_cfg["data_dir"]     = f"_4_data_hi/{_axis_dir}/cycle"
    else:
        # 저장된 config에서 경로 복원 시도, 없으면 spec.axis 기반 fallback
        saved_cfg_path = run_dir / "config.yaml"
        if saved_cfg_path.exists():
            saved_cfg = load_config(str(saved_cfg_path))
            for k in ("data_dir", "seg_data_dir"):
                if saved_cfg.get("data", {}).get(k):
                    _data_cfg[k] = saved_cfg["data"][k]
        if not _data_cfg.get("seg_data_dir"):
            _data_cfg["seg_data_dir"] = f"_4_data_hi/{_axis_dir}/seg"
        if not _data_cfg.get("data_dir"):
            _data_cfg["data_dir"] = f"_4_data_hi/{_axis_dir}/cycle"

    print(f"[clf] data_dir={_data_cfg['data_dir']}  exclude_cv={_exclude_cv}")

    # 데이터 로드 (같은 config → Phase 2와 동일한 split/정규화)
    train_ds, val_ds, _, norm = build_datasets(cfg, spec=spec)

    # Phase 1 probe 마스크 로드 (classification_HIs.json)
    ch_mask, dis_mask = _load_probe_masks(run_dir, cfg, device)
    if ch_mask is not None:
        clf_cfg = cfg.get("classifier", {})
        print(f"[clf] probe mask loaded: charge_m={int(ch_mask.sum())}  "
              f"discharge_m={int(dis_mask.sum())}  (from {run_dir / 'gates' / 'classification_HIs.json'})")
    else:
        print("[clf] classification_HIs.json 없음 → x_hi 전체 사용 (fallback)")

    # 하이퍼파라미터
    train_cfg = cfg.get("training", {})
    epochs   = args.epochs or min(train_cfg.get("epochs", 500), 200)
    lr       = args.lr    or train_cfg.get("lr", 1e-3)
    patience = train_cfg.get("early_stop_patience", 20)
    batch_sz = train_cfg.get("batch_size", 2048)
    d_hidden = args.d_hidden

    # 분류기 타입: mlp(기본) | cnn (원시 V/I 곡선 CNN + HI probe 융합)
    clf_type = cfg.get("classifier", {}).get("type", "mlp").lower()

    print(f"[clf] type={clf_type}  n_classes={spec.n_classes}  d_hidden={d_hidden}  "
          f"lr={lr}  epochs={epochs}  patience={patience}")
    print(f"[clf] train={len(train_ds):,}  val={len(val_ds):,} segments")

    if clf_type == "cnn":
        # CNNProbeClassifier: probe_x(N_HI) + CNN(x_raw) + direction 내부 융합
        clf     = CNNProbeClassifier(N_HI, spec.n_classes, d_hidden=d_hidden).to(device)
        clf_n_hi = N_HI
    else:
        # MLPProbeClassifier: [probe_x || direction] = N_HI+1 입력 (기존 동작 유지)
        clf     = MLPProbeClassifier(N_HI + 1, spec.n_classes, d_hidden=d_hidden).to(device)
        clf_n_hi = N_HI + 1
    opt       = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # FastTensorLoader: collate 오버헤드 제거. CNN 분류기는 x_raw 필요 → include_raw=True.
    train_loader = FastTensorLoader(train_ds, batch_sz, shuffle=True,  include_raw=True)
    val_loader   = FastTensorLoader(val_ds,   batch_sz, shuffle=False, include_raw=True)

    clf_dir = run_dir / "classifier"
    clf_dir.mkdir(exist_ok=True)

    log_path = run_dir / "logs" / "classifier_logs.csv"
    _init_log_csv(log_path)
    print(f"[clf] 학습 기록 → {log_path}")

    best_val_acc = -1.0
    no_improve   = 0
    t0 = time.time()

    pbar = trange(1, epochs + 1, desc="Classifier training")
    for epoch in pbar:
        # ── train ────────────────────────────────────────────────────────
        clf.train()
        tr_loss = tr_correct = tr_total = 0
        for batch in train_loader:
            x     = batch["x_hi"].to(device)
            dir_t = batch["direction"].to(device)
            lbl   = batch["level"].to(device)
            probe = _apply_probe_mask(x, dir_t, ch_mask, dis_mask)
            if clf_type == "cnn":
                binp   = {"x_raw": batch["x_raw"].to(device), "direction": dir_t}
                logits = clf.classify(probe, binp)               # (B, n_classes)
            else:
                inp    = torch.cat([probe, dir_t.unsqueeze(1)], dim=1)  # (B, N_HI+1)
                logits = clf(inp)
            loss   = criterion(logits, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss    += loss.item() * x.size(0)
            tr_correct += (logits.argmax(1) == lbl).sum().item()
            tr_total   += x.size(0)
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # ── val ──────────────────────────────────────────────────────────
        clf.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x     = batch["x_hi"].to(device)
                dir_t = batch["direction"].to(device)
                lbl   = batch["level"].to(device)
                probe = _apply_probe_mask(x, dir_t, ch_mask, dis_mask)
                if clf_type == "cnn":
                    binp   = {"x_raw": batch["x_raw"].to(device), "direction": dir_t}
                    logits = clf.classify(probe, binp)               # (B, n_classes)
                else:
                    inp    = torch.cat([probe, dir_t.unsqueeze(1)], dim=1)  # (B, N_HI+1)
                    logits = clf(inp)
                vl_loss    += criterion(logits, lbl).item() * x.size(0)
                vl_correct += (logits.argmax(1) == lbl).sum().item()
                vl_total   += x.size(0)
        vl_loss /= vl_total
        vl_acc   = vl_correct / vl_total

        _append_log_csv(log_path, epoch, tr_loss, tr_acc, vl_loss, vl_acc,
                        lr, time.time() - t0)

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(tr_acc=f"{tr_acc:.4f}", val_acc=f"{vl_acc:.4f}",
                             best=f"{max(best_val_acc, vl_acc):.4f}")

        if epoch % 10 == 0 or epoch == 1:
            tqdm_write(f"[clf] epoch {epoch:>4d}  "
                      f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                      f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
                      f"({time.time()-t0:.0f}s)")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({
                "clf_state": clf.state_dict(),
                "clf_type":  clf_type,   # "mlp" | "cnn"
                "n_hi":      clf_n_hi,   # mlp: probe_x(N_HI)+direction(1) | cnn: probe_x(N_HI)
                "n_classes": spec.n_classes,
                "d_hidden":  d_hidden,
                "val_acc":   vl_acc,
                "epoch":     epoch,
                "axis":      spec.axis,
            }, clf_dir / "clf_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm_write(f"[clf] Early stop @ epoch {epoch}  best_val_acc={best_val_acc:.4f}")
                break

    if hasattr(pbar, "close"):
        pbar.close()
    elapsed = time.time() - t0
    tqdm_write(f"\n[clf] 완료  best_val_acc={best_val_acc:.4f}  총 {elapsed:.0f}s")
    tqdm_write(f"[clf] 저장 → {clf_dir / 'clf_best.pt'}")
    tqdm_write(f"[clf] 학습 기록 저장 → {log_path}")


if __name__ == "__main__":
    main()
