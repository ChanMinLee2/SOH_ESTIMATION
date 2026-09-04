"""
SCR Evaluator.

Produces per run folder:
  figures/
    scatter_test.png
    capacity_curve_*.png
    confusion_matrix_test.png
  metrics/
    metrics.json          capacity + breakdown + scenario + efficiency
  routing/
    routing_heatmap.png   scen(6) x HI(65) activation map
    routing_table.csv     selected HI list per scenario
  predictions/
    test_predictions.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import compute_metrics
from utils.hi_schema import get_hi_cost_vector, get_hi_cols_for_seg, N_HI
from datasets.segment_dataset import SegmentDataset, SegmentNormalizer


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_COST_VEC = np.array(get_hi_cost_vector("dis_hi"), dtype=np.float32)  # (65,)


class SCREvaluator:

    def __init__(
        self,
        model: nn.Module,
        normalizer: SegmentNormalizer,
        device: torch.device,
        figures_dir: Path,
        rep_cells: Sequence[str] = (),
    ):
        self.model = model.to(device)
        self.normalizer = normalizer
        self.device = device
        self.figures_dir = figures_dir
        self.rep_cells = list(rep_cells)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Spec-derived dimensions (avoid hardcoding N_SEGS/N_LEVELS)
        self._n_scenarios = model.n_scenarios
        self._n_classes   = model.n_classes
        self._seg_names   = model.spec.scenario_names
        self._class_names = model.spec.class_names
        self._spec        = model.spec   # routing table 접근용
        self._classifier  = None         # B안: set_classifier()로 주입

    def set_classifier(self, clf: torch.nn.Module) -> None:
        """B안: 독립 학습된 시나리오 분류기를 주입한다 (routing=hard/soft 시 사용)."""
        self._classifier = clf.to(self.device)
        self._classifier.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_dataset(
        self,
        ds: SegmentDataset,
        batch_size: int = 512,
        routing_mode: str = "none",
        direction_routing: bool = False,
    ) -> dict:
        """
        routing_mode:
          "none" — 방향(direction)만으로 헤드 선택, 분류기 우회 (기본)
          "hard" — 분류기 argmax → 단일 시나리오 헤드 (분류기 활성화 필요)
          "soft" — 분류기 확률 가중 평균 (분류기 활성화 필요)

        direction_routing:
          True  — routing_mode="none"일 때 방향별 첫 번째 시나리오로 scen_idx 보정
                  (test_rs처럼 scen_idx 0/1만 있어 모델 시나리오 ID와 불일치할 때 사용)
          False — scen_idx 그대로 사용 (qfrac 정규 테스트 기본값)
        """
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
        self.model.eval()

        preds_norm, trues_norm = [], []
        level_preds, level_trues = [], []
        scen_idxs, directions = [], []
        probe_zs, scen_zs = [], []

        # routing table: (n_dir, n_classes) — hard/soft 라우팅용
        _use_clf = (routing_mode != "none" and self._classifier is not None)
        _routing_t = None
        if _use_clf and hasattr(self._spec, "routing"):
            # spec.routing은 jagged list일 수 있으므로 padding 후 수동 채움
            _r = self._spec.routing
            _n_dir = len(_r)
            _n_cls = max(len(row) for row in _r)
            _routing_t = torch.zeros(_n_dir, _n_cls, dtype=torch.long, device=self.device)
            for _d, _row in enumerate(_r):
                for _c, _sid in enumerate(_row):
                    _routing_t[_d, _c] = _sid

        # direction-only fallback: routing_mode="none"일 때 방향별 첫 시나리오로 보정
        # (test_rs: scen_idx 0/1 → 모델 시나리오 ID 불일치 해소)
        _dir_t = None
        if direction_routing and routing_mode == "none" and hasattr(self._spec, "routing"):
            _dir_t = torch.tensor(
                [row[0] for row in self._spec.routing],
                dtype=torch.long, device=self.device,
            )  # (n_dir,): charge→routing[0][0], discharge→routing[1][0]

        for batch in loader:
            batch_d = {k: v.to(self.device) for k, v in batch.items()}
            B = batch_d["x_hi"].size(0)

            if _dir_t is not None:
                # direction_routing 보정: 방향에 맞는 첫 번째 모델 시나리오로 재매핑
                dir_idx = (batch_d["direction"] <= 0).long()   # 0=charge, 1=discharge
                batch_d["scen_idx"] = _dir_t[dir_idx]

            if _use_clf and _routing_t is not None:
                x_hi    = batch_d["x_hi"]
                dir_t   = batch_d["direction"]
                dir_idx = (dir_t <= 0).long()              # 0=charge, 1=discharge

                # Phase 1 probe mask 적용 — train_classifier.py와 동일한 입력
                probe_x_clf = self.model.get_probe_x(x_hi, dir_t, batch_d["scen_idx"])
                from models.scenario_classifier import CNNProbeClassifier
                if isinstance(self._classifier, CNNProbeClassifier):
                    # CNN: probe_x + x_raw + direction 내부 융합 (batch_d에 x_raw 포함)
                    clf_logits = self._classifier.classify(probe_x_clf, batch_d)  # (B, n_classes)
                else:
                    clf_inp    = torch.cat([probe_x_clf, dir_t.unsqueeze(1)], dim=1)  # (B, N_HI+1)
                    clf_logits = self._classifier(clf_inp)    # (B, n_classes)

                if routing_mode == "hard":
                    class_pred = clf_logits.argmax(1)                     # (B,)
                    batch_d["scen_idx"] = _routing_t[dir_idx, class_pred]  # (B,)
                    out = self.model(batch_d)
                    lv_pred = class_pred.cpu().numpy()

                else:  # soft
                    clf_probs = torch.softmax(clf_logits, dim=1)  # (B, n_classes)
                    cap_cls   = []
                    for c in range(self._n_classes):
                        b_c = dict(batch_d)
                        b_c["scen_idx"] = _routing_t[dir_idx, c]   # (B,)
                        out_c = self.model(b_c)
                        cap_cls.append(out_c["cap_pred"])
                    cap_stack = torch.stack(cap_cls, dim=1)        # (B, n_classes)
                    cap_merged = (clf_probs * cap_stack).sum(1)    # (B,)
                    # argmax for reporting
                    class_pred = clf_logits.argmax(1)
                    lv_pred = class_pred.cpu().numpy()
                    out = {
                        "cap_pred": cap_merged,
                        "level_logits": clf_logits,
                        "probe_z": torch.zeros(B, N_HI, device=self.device),
                        "scen_z":  torch.zeros(B, N_HI, device=self.device),
                    }
            else:
                out = self.model(batch_d)
                lv_pred = out["level_logits"].argmax(1).cpu().numpy()

            preds_norm.append(out["cap_pred"].cpu().numpy())
            trues_norm.append(batch["target"].numpy())
            level_preds.append(lv_pred)
            level_trues.append(batch["level"].numpy())
            scen_idxs.append(batch_d["scen_idx"].cpu().numpy())   # 라우팅 후 수정된 값 저장
            directions.append(batch["direction"].numpy())
            probe_zs.append(out["probe_z"].cpu().numpy())   # (B, N_HI)
            scen_zs.append(out["scen_z"].cpu().numpy())     # (B, N_HI)

        preds_norm   = np.concatenate(preds_norm)
        trues_norm   = np.concatenate(trues_norm)
        level_preds  = np.concatenate(level_preds)
        level_trues  = np.concatenate(level_trues)
        scen_idxs     = np.concatenate(scen_idxs)
        directions   = np.concatenate(directions)
        probe_zs     = np.concatenate(probe_zs)   # (N, 65)
        scen_zs      = np.concatenate(scen_zs)    # (N, 65)

        # target은 SOH ratio (∈ (0,1]) — inverse_target 불필요
        cap_pred_raw = preds_norm   # SOH ratio
        cap_true_raw = trues_norm   # SOH ratio

        return {
            "cap_pred_raw":  cap_pred_raw,   # SOH ratio
            "cap_true_raw":  cap_true_raw,   # SOH ratio
            "cap_pred_norm": preds_norm,
            "cap_true_norm": trues_norm,
            "cap_init_raw":  ds.cap_init_raw,  # Ah per sample (SOH→Ah 변환용)
            "level_pred":    level_preds,
            "level_true":    level_trues,
            "scen_idx":       scen_idxs,
            "direction":     directions,
            "probe_z":       probe_zs,
            "scen_z":        scen_zs,
            "cell_ids":      ds.cell_ids,
            "cycles":        ds.cycles,
            "seg_names":     ds.seg_names,
            "cap_raw":       ds.capacity_raw,
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------
    def plot_for_dataset(
        self,
        pred_dict: dict,
        out_dir: Path,
        rep_cells: list[str],
        tag: str = "random_seg",
    ) -> None:
        """
        pred_dict에 대한 scatter + 용량 곡선 + confusion matrix를 out_dir에 저장.
        self.figures_dir / self.rep_cells를 임시 교체하여 기존 메서드를 재활용한다.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        _orig_figs = self.figures_dir
        _orig_rep  = self.rep_cells
        self.figures_dir = out_dir
        self.rep_cells   = rep_cells
        self._plot_scatter(pred_dict, tag=tag)
        self._plot_capacity_curves(pred_dict)
        self._plot_confusion_matrix(pred_dict, tag=tag)
        self.figures_dir = _orig_figs
        self.rep_cells   = _orig_rep

    def evaluate(
        self,
        train_ds: SegmentDataset,
        val_ds: SegmentDataset,
        test_ds: SegmentDataset,
        batch_size: int = 512,
    ) -> dict[str, dict]:
        results = {}
        for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            pred = self.predict_dataset(ds, batch_size)
            metrics = compute_metrics(pred["cap_true_raw"], pred["cap_pred_raw"])
            results[split_name] = {**metrics, **pred}
            print(f"[eval] {split_name}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        self._plot_scatter(results["test"], tag="test")
        self._plot_capacity_curves(results["test"])
        self._plot_confusion_matrix(results["test"], tag="test")
        return results

    # ------------------------------------------------------------------
    # 통합 평가: 학습된 분류기로 분류 → 라우팅 → 회귀 (oracle/hard/soft)
    # ------------------------------------------------------------------
    _MODE_ROUTING = {"oracle": "none", "hard": "hard", "soft": "soft"}

    def evaluate_modes(
        self,
        ds: SegmentDataset,
        modes: Sequence[str] = ("oracle", "hard", "soft"),
        batch_size: int = 512,
        direction_routing_for_oracle: bool = False,
    ) -> dict:
        """각 라우팅 모드로 분류→회귀 평가.

        modes:
          oracle : 정답 scen_idx로 라우팅 (분류 100% 가정 → 회귀 상한선)
          hard   : 학습된 분류기 argmax로 라우팅 (실배포 시나리오)
          soft   : 분류기 확률 가중 평균
        Returns {mode: {"capacity","breakdown","classification","_pred"}}.
        분류기가 없으면 hard/soft는 호출자가 modes에서 제외해야 한다.
        """
        out: dict = {}
        for mode in modes:
            rmode = self._MODE_ROUTING[mode]
            dr = direction_routing_for_oracle and (rmode == "none")
            pred = self.predict_dataset(
                ds, batch_size, routing_mode=rmode, direction_routing=dr
            )
            entry = {
                "capacity":  self._capacity_metrics(pred),
                "breakdown": self._compute_breakdown(pred),
            }
            if mode == "oracle":
                entry["classification"] = {
                    "accuracy": 1.0,
                    "note": "oracle: ground-truth routing (정답 레짐 라벨 사용)",
                }
            else:
                cm = self.classification_metrics(pred)
                entry["classification"] = cm if cm is not None else {
                    "note": "정답 레짐 라벨 없음 — 분류 정확도 산출 불가",
                }
            entry["_pred"] = pred
            out[mode] = entry
        return out

    @staticmethod
    def strip_modes_for_json(modes_result: dict, efficiency: dict | None = None) -> dict:
        """evaluate_modes 결과에서 _pred(대용량 배열) 제거 + efficiency 부착."""
        clean = {
            mode: {k: v for k, v in entry.items() if k != "_pred"}
            for mode, entry in modes_result.items()
        }
        if efficiency is not None:
            clean["efficiency"] = efficiency
        return clean

    # ------------------------------------------------------------------
    # Save metrics JSON (capacity + breakdown + scenario + efficiency)
    # ------------------------------------------------------------------
    def save_metrics(self, results: dict, metrics_dir: Path) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for split in ("train", "val", "test"):
            p = results[split]
            out[split] = {
                "capacity":  self._capacity_metrics(p),
                "breakdown": self._compute_breakdown(p),
                "scenario":  self._compute_scenario_metrics(p),
                "efficiency": self._compute_efficiency(p),
            }
        path = metrics_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[eval] saved {path}")

    def _capacity_metrics(self, p: dict) -> dict:
        m = compute_metrics(p["cap_true_raw"], p["cap_pred_raw"])
        return {k: float(v) for k, v in m.items()}

    def _compute_breakdown(self, p: dict) -> dict:
        t, pred = p["cap_true_raw"], p["cap_pred_raw"]
        d = p["direction"]
        lv = p["level_true"]

        out: dict = {}
        for tag, sel in [("charge", d > 0), ("discharge", d < 0)]:
            if sel.sum() > 1:
                m = compute_metrics(t[sel], pred[sel])
                out[tag] = {k: float(v) for k, v in m.items()}

        for i, name in enumerate(self._class_names):
            sel = lv == i
            if sel.sum() > 1:
                m = compute_metrics(t[sel], pred[sel])
                out[f"level_{name.lower()}"] = {k: float(v) for k, v in m.items()}
        return out

    def _compute_scenario_metrics(self, p: dict) -> dict:
        y_true = p["level_true"]
        y_pred = p["level_pred"]

        # CE 비활성 시 level_logits=zeros → level_pred=0 (argmax) → 무의미한 값
        # all-zero logits 감지: pred가 모두 동일하면 분류기 미학습으로 처리
        if len(np.unique(y_pred)) <= 1:
            return {"disabled": True, "note": "CE loss disabled — classification not trained"}

        acc = float((y_true == y_pred).mean())
        n_cls = self._n_classes
        cls_names = self._class_names
        per_class = []
        for i in range(n_cls):
            sel = y_true == i
            per_class.append(float((y_pred[sel] == i).mean()) if sel.sum() > 0 else float("nan"))

        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, pr in zip(y_true, y_pred):
            cm[t][pr] += 1

        return {
            "accuracy": acc,
            "per_class_accuracy": {cls_names[i]: per_class[i] for i in range(n_cls)},
            "confusion_matrix": cm.tolist(),
        }

    def classification_metrics(self, pred_dict: dict) -> dict | None:
        """라우팅 분류기의 레짐(level) 분류 성능 (routing=hard/soft 테스트 전용).

        반환:
          accuracy               전체 평균 정확도
          per_class_accuracy     클래스(lo/mid/hi)별 recall
          per_direction_accuracy 충전/방전별 정확도
          per_scenario_accuracy  시나리오별(방향×레벨 → scenario_name) 정확도
          confusion_matrix       n_classes × n_classes
          n_samples              샘플 수
        분류가 비활성(level_pred 균일)이면 None 반환.
        """
        y_true = np.asarray(pred_dict.get("level_true", []))
        y_pred = np.asarray(pred_dict.get("level_pred", []))
        if len(y_true) == 0 or len(np.unique(y_pred)) <= 1:
            return None
        # 정답 라벨이 단일값(placeholder, 예: test_rs n_classes=1)이면 정확도 무의미
        if len(np.unique(y_true)) <= 1:
            uniq, cnt = np.unique(y_pred, return_counts=True)
            return {
                "note": "정답 레짐 라벨이 단일값(placeholder) — 분류 정확도 무의미. "
                        "위치기반 레짐 라벨 포함해 데이터 재생성 필요.",
                "predicted_distribution": {int(u): int(c) for u, c in zip(uniq, cnt)},
                "n_samples": int(len(y_true)),
            }

        n_cls     = self._n_classes
        cls_names = self._class_names
        direction = np.asarray(pred_dict.get("direction", np.zeros_like(y_true)))

        overall = float((y_true == y_pred).mean())

        # 클래스(레벨)별 recall
        per_class: dict = {}
        for i in range(n_cls):
            sel = y_true == i
            per_class[cls_names[i]] = (
                float((y_pred[sel] == i).mean()) if sel.sum() > 0 else None
            )

        # 방향별 정확도
        per_direction: dict = {}
        for tag, sel in [("charge", direction > 0), ("discharge", direction < 0)]:
            if sel.sum() > 0:
                per_direction[tag] = float((y_true[sel] == y_pred[sel]).mean())

        # 시나리오별 정확도 (true 시나리오 = routing[dir_idx][level_true])
        per_scenario: dict = {}
        routing = getattr(self._spec, "routing", None)
        if routing is not None:
            counts: dict = {}
            correct: dict = {}
            for dir_v, lt, lp in zip(direction, y_true, y_pred):
                dir_idx = 0 if dir_v > 0 else 1
                try:
                    sid = routing[dir_idx][int(lt)]
                except (IndexError, TypeError, ValueError):
                    continue
                name = (self._seg_names[sid]
                        if 0 <= sid < len(self._seg_names) else f"scen_{sid}")
                counts[name]  = counts.get(name, 0) + 1
                correct[name] = correct.get(name, 0) + int(lt == lp)
            per_scenario = {k: float(correct[k] / counts[k]) for k in counts}

        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, pr in zip(y_true, y_pred):
            if 0 <= int(t) < n_cls and 0 <= int(pr) < n_cls:
                cm[int(t)][int(pr)] += 1

        return {
            "accuracy":               overall,
            "per_class_accuracy":     per_class,
            "per_direction_accuracy": per_direction,
            "per_scenario_accuracy":  per_scenario,
            "confusion_matrix":       cm.tolist(),
            "n_samples":              int(len(y_true)),
        }

    def _compute_efficiency(self, p: dict) -> dict:
        probe_act = (p["probe_z"] > 0)   # (N, 65) bool
        scen_act  = (p["scen_z"] > 0)    # (N, 65) bool
        computed  = probe_act | scen_act  # union = actually computed

        avg_probe    = float(probe_act.sum(axis=1).mean())
        avg_scen     = float(scen_act.sum(axis=1).mean())
        avg_computed = float(computed.sum(axis=1).mean())
        avg_cost     = float((computed.astype(np.float32) @ _COST_VEC).mean())
        max_cost     = float(_COST_VEC.sum())

        return {
            "avg_probe_his":    avg_probe,
            "avg_scen_his":     avg_scen,
            "avg_computed_his": avg_computed,
            "avg_cost":         avg_cost,
            "max_cost":         max_cost,
            "cost_reduction_pct": float((1 - avg_cost / max_cost) * 100),
        }

    # ------------------------------------------------------------------
    # UQ predict + plot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_dataset_uq(self, ds, uq, batch_size: int = 512) -> dict:
        """
        LaplaceUQ를 사용해 (mean, std) 예측을 수행한다.

        Returns:
            predict_dataset()과 동일한 dict에 'cap_std_raw' 키 추가.
        """
        from torch.utils.data import DataLoader

        pred = self.predict_dataset(ds, batch_size)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate)
        all_stds = []
        for batch in loader:
            _, std = uq.predict(batch)   # (B,)
            all_stds.append(std.numpy())
        pred["cap_std_raw"] = np.concatenate(all_stds)
        return pred

    def save_uq_metrics(self, pred_dict: dict, metrics_dir: Path) -> None:
        """UQ 캘리브레이션 지표를 metrics/uq_metrics.json 에 저장한다."""
        from utils.uncertainty import calibration_metrics

        stds = pred_dict.get("cap_std_raw")
        if stds is None:
            return
        m = calibration_metrics(
            means=pred_dict["cap_pred_raw"],
            stds=stds,
            targets=pred_dict["cap_true_raw"],
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        path = metrics_dir / "uq_metrics.json"
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(m, f, indent=2)
        print(f"[eval] saved {path}")
        print(f"[eval] UQ - picp_90={m['picp_90']:.3f}  mpiw_90={m['mpiw_90']:.5f}"
              f"  nll={m['nll']:.4f}  mean_std={m['mean_std']:.5f}")

    def plot_uq(self, pred_dict: dict, figures_dir: Path) -> None:
        """캘리브레이션 곡선 + σ 분포 히스토그램을 figures/calibration.png 에 저장한다."""
        from utils.uncertainty import plot_calibration

        stds = pred_dict.get("cap_std_raw")
        if stds is None:
            return
        plot_calibration(
            means=pred_dict["cap_pred_raw"],
            stds=stds,
            targets=pred_dict["cap_true_raw"],
            path=figures_dir / "calibration.png",
        )

    # ------------------------------------------------------------------
    # Save predictions CSV
    # ------------------------------------------------------------------
    def save_predictions(
        self, pred_dict: dict, predictions_dir: Path, tag: str = "test"
    ) -> None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        path = predictions_dir / f"{tag}_predictions.csv"

        probe_active_n = (pred_dict["probe_z"] > 0).sum(axis=1)
        scen_active_n  = (pred_dict["scen_z"]  > 0).sum(axis=1)
        cap_init_ah    = pred_dict["cap_init_raw"]   # Ah per sample

        soh_true = pred_dict["cap_true_raw"]
        soh_pred = pred_dict["cap_pred_raw"]
        cap_true_ah = soh_true * cap_init_ah
        cap_pred_ah = soh_pred * cap_init_ah

        has_uq = "cap_std_raw" in pred_dict
        std_col = pred_dict["cap_std_raw"].tolist() if has_uq else None

        rows = zip(
            pred_dict["cell_ids"],
            pred_dict["cycles"],
            pred_dict["seg_names"],
            soh_true.tolist(),
            soh_pred.tolist(),
            cap_true_ah.tolist(),
            cap_pred_ah.tolist(),
            cap_init_ah.tolist(),
            pred_dict["level_true"].tolist(),
            pred_dict["level_pred"].tolist(),
            probe_active_n.tolist(),
            scen_active_n.tolist(),
            std_col if has_uq else [None] * len(soh_true),
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "cell_id", "cycle", "seg_name",
                "soh_true", "soh_pred", "soh_error",
                "cap_true_Ah", "cap_pred_Ah", "cap_init_Ah",
                "level_true", "level_pred",
                "probe_active_n", "scen_active_n",
            ]
            if has_uq:
                header.append("soh_std")
            w.writerow(header)
            for cell, cyc, seg, st, sp, ct, cp, ci, lt, lp, pa, sa, sg in rows:
                row = [
                    cell, cyc, seg,
                    f"{st:.6f}", f"{sp:.6f}", f"{sp - st:.6f}",
                    f"{ct:.6f}", f"{cp:.6f}", f"{ci:.6f}",
                    lt, lp, pa, sa,
                ]
                if has_uq:
                    row.append(f"{sg:.6f}")
                w.writerow(row)
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Routing heatmap + table
    # ------------------------------------------------------------------
    def plot_routing_heatmap(
        self,
        routing_dir: Path,
        probe_sel: list[int] | None = None,
        scen_sel: dict[int, list[int]] | None = None,
    ) -> None:
        routing_dir.mkdir(parents=True, exist_ok=True)

        # Use JSON-provided selections if available; fall back to model gate values
        if probe_sel is None:
            probe_sel = self.model.get_selected_probe_his()
        if scen_sel is None:
            scen_sel = self.model.get_selected_scen_his()

        n_scen = self._n_scenarios
        act_matrix = np.zeros((n_scen, N_HI), dtype=np.float32)
        for s in range(n_scen):
            for i in scen_sel.get(s, []):
                act_matrix[s, i] = 1.0

        probe_row = np.zeros(N_HI, dtype=np.float32)
        for i in probe_sel:
            probe_row[i] = 1.0

        full_matrix = np.vstack([probe_row[np.newaxis, :], act_matrix])
        row_labels  = ["probe"] + self._seg_names

        # HI short names (strip seg suffix, keep category+key)
        hi_cols = get_hi_cols_for_seg("dis_hi")  # reference seg
        hi_short = [c.rsplit("_dis_hi", 1)[0] for c in hi_cols]

        # --- PNG ---
        if _HAS_MPL:
            fig, ax = plt.subplots(figsize=(max(14, N_HI * 0.18), 4))
            im = ax.imshow(full_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_xticks(range(N_HI))
            ax.set_xticklabels(hi_short, rotation=90, fontsize=5.5)
            ax.set_title("SCR Routing: probe + scenario × HI activation")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            fig.tight_layout()
            path_png = routing_dir / "routing_heatmap.png"
            fig.savefig(path_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[eval] saved {path_png}")

        # --- CSV ---
        path_csv = routing_dir / "routing_table.csv"
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gate"] + hi_short)
            for row_name, row in zip(row_labels, full_matrix):
                w.writerow([row_name] + [int(v) for v in row])
        print(f"[eval] saved {path_csv}")

    # ------------------------------------------------------------------
    # HI 카테고리 랭킹 히트맵 — 시나리오별 상위 HI가 어느 카테고리(A=stat/B=diff/
    # C=lfp/D=morph)에 속하는지 랭크 순으로 시각화 (docs/260811_RESULTS.md 참고)
    # ------------------------------------------------------------------
    _CAT_PREFIX = {"stat": "A", "diff": "B", "lfp": "C", "morph": "D"}
    _CAT_COLOR  = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f", "#ffffff"]  # A/B/C/D/(빈칸)
    _CAT_LABELS = ["A: stat(통계)", "B: diff(미분)", "C: lfp(LFP특징)", "D: morph(형태)"]

    def plot_hi_category_heatmap(
        self,
        routing_dir: Path,
        scen_ranked_names: dict[int, list[str]],   # {scen_idx: [rank순 HI이름, ...]}
        top_k: int | None = None,
    ) -> None:
        """행=HI 랭크 순위, 열=시나리오, 셀 색=HI 카테고리(A/B/C/D)."""
        if not _HAS_MPL:
            return
        n_scen   = self._n_scenarios
        max_rank = top_k or max((len(v) for v in scen_ranked_names.values()), default=0)
        if max_rank == 0:
            return
        cat_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        mat = np.full((max_rank, n_scen), 4, dtype=int)   # 4 = 빈칸(랭크 부족/미분류)
        for s in range(n_scen):
            for rank, name in enumerate(scen_ranked_names.get(s, [])[:max_rank]):
                prefix = name.split("_", 1)[0]
                mat[rank, s] = cat_idx.get(self._CAT_PREFIX.get(prefix, ""), 4)

        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        cmap = ListedColormap(self._CAT_COLOR)
        fig, ax = plt.subplots(figsize=(max(4, n_scen * 1.3), max(6, max_rank * 0.16)))
        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=4)
        ax.set_xticks(range(n_scen))
        ax.set_xticklabels(self._seg_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("HI 랭크 (0=최상위)")
        ax.set_title("시나리오별 HI 랭킹 — 카테고리 구성")
        handles = [Patch(color=self._CAT_COLOR[i], label=l) for i, l in enumerate(self._CAT_LABELS)]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        path = routing_dir / "hi_category_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self, pred_dict: dict, tag: str = "test") -> None:
        if not _HAS_MPL:
            return
        y_true = pred_dict["level_true"]
        y_pred = pred_dict["level_pred"]
        if len(np.unique(y_pred)) <= 1:
            return  # CE 비활성 — confusion matrix 생략

        n_cls = self._n_classes
        cls_names = self._class_names
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        acc = (y_true == y_pred).mean()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(cls_names)
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(cls_names)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(f"Level confusion ({tag})  acc={acc:.3f}")
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=10, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        path = self.figures_dir / f"confusion_matrix_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    def _plot_scatter(self, pred_dict: dict, tag: str = "test") -> None:
        if not _HAS_MPL:
            return
        p = pred_dict["cap_pred_raw"]
        t = pred_dict["cap_true_raw"]

        # 분류(routing) 활성 시: 정답/오답 색상으로 분류 성능을 산점도에 표시
        y_true = np.asarray(pred_dict.get("level_true", []))
        y_pred = np.asarray(pred_dict.get("level_pred", []))
        clf_active = (len(y_pred) == len(t) and len(t) > 0
                      and len(np.unique(y_pred)) > 1)

        fig, ax = plt.subplots(figsize=(5, 5))
        if clf_active:
            correct = (y_true == y_pred)
            acc = float(correct.mean())
            ax.scatter(t[~correct], p[~correct], s=6, alpha=0.5, c="#d62728",
                       label=f"misclassified ({int((~correct).sum())})")
            ax.scatter(t[correct], p[correct], s=5, alpha=0.4, c="#2ca02c",
                       label=f"correct ({int(correct.sum())})")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
            title = f"SCR {tag} — pred vs true  (level acc={acc:.3f})"
        else:
            ax.scatter(t, p, s=5, alpha=0.4)
            title = f"SCR {tag} — pred vs true"

        lim = [min(t.min(), p.min()) * 0.98, max(t.max(), p.max()) * 1.02]
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlabel("True SOH")
        ax.set_ylabel("Predicted SOH")
        ax.set_title(title)
        ax.set_xlim(lim); ax.set_ylim(lim)
        path = self.figures_dir / f"scatter_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval] saved {path}")

    # ------------------------------------------------------------------
    # Capacity curve plots for representative cells
    # 2-row × 3-col layout: row0=Charge / row1=Discharge
    # Each row: [capacity curve | abs error | rel error]
    # Pred lines are split by segment sub-type (Low / Mid / High) rather
    # than averaged, so misrouted predictions are visible as separate lines.
    # ------------------------------------------------------------------
    def _plot_capacity_curves(self, pred_dict: dict) -> None:
        if not _HAS_MPL:
            return

        cell_ids    = np.array(pred_dict["cell_ids"])
        cycles      = np.array(pred_dict["cycles"])
        cap_init_ah = pred_dict["cap_init_raw"]           # Ah per sample
        cap_true    = pred_dict["cap_true_raw"] * cap_init_ah   # SOH→Ah
        cap_pred    = pred_dict["cap_pred_raw"] * cap_init_ah   # SOH→Ah
        directions  = pred_dict["direction"]
        scen_idxs    = pred_dict["scen_idx"]

        # (scen_idx, level_idx, display label) — derived from spec routing
        _spec = self.model.spec
        _chg_ids  = _spec.charge_scenario_ids     # dir_idx=0
        _dis_ids  = _spec.discharge_scenario_ids  # dir_idx=1
        _DIR_SEGS = {
            "Charge":    [(s, _spec.scenario_to_dir_class(s)[1],
                           _spec.class_names[_spec.scenario_to_dir_class(s)[1]])
                          for s in _chg_ids],
            "Discharge": [(s, _spec.scenario_to_dir_class(s)[1],
                           _spec.class_names[_spec.scenario_to_dir_class(s)[1]])
                          for s in _dis_ids],
        }
        _COLORS = ["tab:red", "tab:green", "tab:orange", "tab:purple",
                   "tab:brown", "tab:pink", "tab:cyan", "tab:olive"]
        _STYLES = ["--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)),
                   (0, (3, 5, 1, 5)), (0, (1, 1)), "-"]
        _LV_COLOR = {i: _COLORS[i % len(_COLORS)] for i in range(_spec.n_classes)}
        _LV_LS    = {i: _STYLES[i % len(_STYLES)] for i in range(_spec.n_classes)}

        for cell in self.rep_cells:
            sel = cell_ids == cell
            if sel.sum() == 0:
                print(f"[eval] rep cell '{cell}' not found in test split, skipping")
                continue

            c_cyc  = cycles[sel]
            c_true = cap_true[sel]
            c_pred = cap_pred[sel]
            c_dir  = directions[sel]
            c_seg  = scen_idxs[sel]

            uniq_cyc = np.unique(c_cyc)

            fig, axes = plt.subplots(2, 3, figsize=(17, 8))
            fig.suptitle(cell, fontsize=12, y=1.01)

            for row, (dir_name, seg_list) in enumerate(_DIR_SEGS.items()):
                is_charge = (dir_name == "Charge")
                dir_mask  = (c_dir > 0) if is_charge else (c_dir < 0)

                ax_cap, ax_err, ax_rel = axes[row, 0], axes[row, 1], axes[row, 2]

                # True capacity per cycle for this direction
                d_cyc  = c_cyc[dir_mask]
                d_true = c_true[dir_mask]
                true_line = np.array([
                    d_true[d_cyc == cy].mean() if (d_cyc == cy).any() else np.nan
                    for cy in uniq_cyc
                ])

                ax_cap.plot(uniq_cyc, true_line, "b-", label="True", linewidth=1.5)

                for scen_idx, lv_idx, lv_name in seg_list:
                    s_mask = dir_mask & (c_seg == scen_idx)
                    if s_mask.sum() == 0:
                        continue
                    s_cyc  = c_cyc[s_mask]
                    s_pred = c_pred[s_mask]
                    pred_line = np.array([
                        s_pred[s_cyc == cy].mean() if (s_cyc == cy).any() else np.nan
                        for cy in uniq_cyc
                    ])

                    color = _LV_COLOR[lv_idx]
                    ls    = _LV_LS[lv_idx]
                    ax_cap.plot(uniq_cyc, pred_line, color=color, linestyle=ls,
                                label=f"Pred-{lv_name}", linewidth=1.2)

                    err_abs = np.abs(pred_line - true_line)
                    err_rel = err_abs / np.where(true_line == 0, 1.0, np.abs(true_line)) * 100
                    ax_err.plot(uniq_cyc, err_abs, color=color, linestyle=ls,
                                label=lv_name, linewidth=1.0)
                    ax_rel.plot(uniq_cyc, err_rel, color=color, linestyle=ls,
                                label=lv_name, linewidth=1.0)

                ax_cap.set_xlabel("Cycle"); ax_cap.set_ylabel("Capacity (Ah)")
                ax_cap.set_title(f"{dir_name} — capacity curve")
                ax_cap.legend(fontsize=8)
                ax_err.set_xlabel("Cycle"); ax_err.set_ylabel("|Error| (Ah)")
                ax_err.set_title(f"{dir_name} — absolute error")
                ax_err.legend(fontsize=8)
                ax_rel.set_xlabel("Cycle"); ax_rel.set_ylabel("Relative error (%)")
                ax_rel.set_title(f"{dir_name} — relative error (%)")
                ax_rel.legend(fontsize=8)

            fig.tight_layout()
            safe_name = cell.replace("/", "_")
            path = self.figures_dir / f"capacity_curve_{safe_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[eval] saved {path}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
