"""
SCR Evaluator.

figures/scatter_*.png, capacity_curve_*.png, confusion_matrix_*.png는 이 모듈의
_plot_scatter/_plot_capacity_curves/_plot_confusion_matrix를 9_eval/test.py가 호출해
생성한다. metrics/metrics.json은 save_metrics()가 만든다. routing/, predictions/ 산출물은
test.py가 자체 로직으로 직접 쓴다(이 모듈은 관여하지 않음).

2026-10-04: 단일 책임 원칙에 따라 서브함수로 분리했다(동작 변화 없는 순수 구조
정리) — predict_dataset은 _build_routing_table/_predict_plain_batch/
_predict_routed_batch로, _plot_capacity_curves는 _extract_cell_series/
_plot_capacity_curve_direction으로 나눴다. 각 함수 docstring 참고.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import compute_metrics
from utils.hi_schema import get_hi_cost_vector, N_HI
from datasets.segment_dataset import SegmentDataset, SegmentNormalizer


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

_COST_VEC = np.array(get_hi_cost_vector("dis_hi"), dtype=np.float32)  # (65,)


def _infer_dataset_from_cell_id(cell_id: str) -> str:
    """cell_id 문자열만으로 MIT/HUST/TJU/CALCE를 구분한다(2026-09-08, 데이터셋별
    breakdown 추가).

    셀 명명 규칙이 4개 데이터셋 간 겹치지 않는다는 사실에 의존(1_convert/convert_*.py
    변환기들이 만든 실제 셀 이름 확인됨):
      MIT   : "b1c0", "b2c17" ...   (b + 숫자 + c + 숫자)
      HUST  : "1-1", "3-8" ...      (숫자-숫자)
      TJU   : "CY25-05_1-#1" ...    ("CY"로 시작)
      CALCE : "CS2_33", "CX2_16" ...("CS2_" 또는 "CX2_"로 시작)
    SegmentDataset이 dataset_id(float, datasets 리스트 내 순서 기반)를 이미 갖고
    있지만, predict_dataset()의 반환 dict에 아직 안 실려 있어 재배선하는 대신
    이미 있는 cell_ids 문자열에서 바로 판별한다 — 셀 이름 자체가 데이터셋마다
    고유한 접두 패턴이라 안전하다.
    """
    if cell_id.startswith("CY"):
        return "TJU"
    if cell_id.startswith("CS2_") or cell_id.startswith("CX2_"):
        return "CALCE"
    if cell_id and cell_id[0].isdigit():
        return "HUST"
    return "MIT"


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
    def _build_routing_table(self) -> torch.Tensor | None:
        """spec.routing((방향,레벨)->시나리오, jagged list일 수 있음)을 패딩된
        (n_dir, n_classes) long 텐서로 만든다 — hard/soft 라우팅 전용. spec에
        routing이 없으면 None(predict_dataset에서 분리, 동작 변화 없음)."""
        if not hasattr(self._spec, "routing"):
            return None
        _r = self._spec.routing
        n_dir = len(_r)
        n_cls = max(len(row) for row in _r)
        routing_t = torch.zeros(n_dir, n_cls, dtype=torch.long, device=self.device)
        for d, row in enumerate(_r):
            for c, sid in enumerate(row):
                routing_t[d, c] = sid
        return routing_t

    def _predict_plain_batch(self, batch_d: dict) -> tuple[dict, np.ndarray]:
        """routing_mode="none" 또는 분류기 미주입 — 방향(direction)만으로 헤드 선택,
        분류기 우회(predict_dataset에서 분리, 동작 변화 없음)."""
        out = self.model(batch_d)
        lv_pred = out["level_logits"].argmax(1).cpu().numpy()
        return out, lv_pred

    def _predict_routed_batch(
        self, batch_d: dict, routing_mode: str, routing_t: torch.Tensor,
    ) -> tuple[dict, np.ndarray]:
        """routing_mode="hard"|"soft" — 학습된 분류기로 먼저 레벨을 예측하고, routing_t로
        (방향,레벨)->시나리오를 찾아 그 시나리오 헤드로 회귀한다(predict_dataset에서
        분리, 동작 변화 없음). hard: argmax 레벨 하나로 단일 헤드만 통과. soft: 전체
        클래스에 대해 각각 통과시킨 뒤 분류기 확률로 가중평균."""
        x_hi    = batch_d["x_hi"]
        dir_t   = batch_d["direction"]
        dir_idx = (dir_t <= 0).long()              # 0=charge, 1=discharge
        B = x_hi.size(0)

        # Phase 1 probe mask 적용 — train_classifier.py와 동일한 입력
        # (2026-09-25: CNNProbeClassifier 분기 삭제 — v4는 self._classifier에
        # 항상 model.probe_mlp만 주입하므로 이 else 경로만 실제로 쓰였다.)
        probe_x_clf = self.model.get_probe_x(x_hi, dir_t, batch_d["scen_idx"])
        clf_parts  = [probe_x_clf, dir_t.unsqueeze(1)]
        # 2026-10-07: probe_kernel_gates(분류기용 커널 후보 블록)가 있는 모델이면
        # scr_model.py::forward()와 똑같이 그 블록도 입력에 이어붙여야 probe_mlp의
        # 입력 폭(N_HI+1+n_dir*max_k)과 맞는다 — get_probe_kernel_x가 probe_kernel_gates
        # 없는 모델이면 None을 돌려줘서 자동으로 기존(N_HI+1)과 동일 동작.
        if "x_kernel_probe" in batch_d:
            probe_kernel_x = self.model.get_probe_kernel_x(batch_d["x_kernel_probe"], dir_t)
            if probe_kernel_x is not None:
                clf_parts.append(probe_kernel_x)
        clf_inp    = torch.cat(clf_parts, dim=1)  # (B, N_HI+1[+n_dir*max_k])
        clf_logits = self._classifier(clf_inp)    # (B, n_classes)

        if routing_mode == "hard":
            class_pred = clf_logits.argmax(1)                     # (B,)
            batch_d["scen_idx"] = routing_t[dir_idx, class_pred]  # (B,)
            out = self.model(batch_d)
            lv_pred = class_pred.cpu().numpy()
        else:  # soft
            clf_probs = torch.softmax(clf_logits, dim=1)  # (B, n_classes)
            cap_cls   = []
            for c in range(self._n_classes):
                b_c = dict(batch_d)
                b_c["scen_idx"] = routing_t[dir_idx, c]   # (B,)
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
        return out, lv_pred

    @torch.no_grad()
    def predict_dataset(
        self,
        ds: SegmentDataset,
        batch_size: int = 512,
        routing_mode: str = "none",
    ) -> dict:
        """
        routing_mode:
          "none" — 방향(direction)만으로 헤드 선택, 분류기 우회 (기본)
          "hard" — 분류기 argmax → 단일 시나리오 헤드 (분류기 활성화 필요)
          "soft" — 분류기 확률 가중 평균 (분류기 활성화 필요)

        2026-10-04: 단일 책임 원칙에 따라 서브함수로 분리했다(동작 변화 없는 순수
        구조 정리) — _build_routing_table(라우팅 테이블) ->
        _predict_plain_batch/_predict_routed_batch(배치 1개 추론) -> 이 함수는 루프
        돌며 결과를 모으고 concat해 최종 dict로 반환하는 역할만 남는다.
        """
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
        self.model.eval()

        preds_norm, trues_norm = [], []
        level_preds, level_trues = [], []
        scen_idxs, directions = [], []
        probe_zs, scen_zs = [], []

        use_clf = (routing_mode != "none" and self._classifier is not None)
        routing_t = self._build_routing_table() if use_clf else None

        for batch in loader:
            batch_d = {k: v.to(self.device) for k, v in batch.items()}

            if use_clf and routing_t is not None:
                out, lv_pred = self._predict_routed_batch(batch_d, routing_mode, routing_t)
            else:
                out, lv_pred = self._predict_plain_batch(batch_d)

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

        # target은 SOH ratio (∈ (0,1]) — inverse_target 불필요, norm/raw가 같은 배열
        return {
            "cap_pred_raw":  preds_norm,   # SOH ratio
            "cap_true_raw":  trues_norm,   # SOH ratio
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
            "q_frac_lo":     ds.q_frac_lo,
            "cap_raw":       ds.capacity_raw,
        }

    # ------------------------------------------------------------------
    # 통합 평가: 학습된 분류기로 분류 → 라우팅 → 회귀 (oracle/hard/soft)
    # ------------------------------------------------------------------
    _MODE_ROUTING = {"oracle": "none", "hard": "hard", "soft": "soft"}

    def evaluate_modes(
        self,
        ds: SegmentDataset,
        modes: Sequence[str] = ("oracle", "hard", "soft"),
        batch_size: int = 512,
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
            pred = self.predict_dataset(ds, batch_size, routing_mode=rmode)
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

        # 데이터셋별 breakdown (2026-09-08) — 4개 데이터셋(MIT/HUST/TJU/CALCE)을
        # 함께 학습한 run에서 화학종별 성능이 얼마나 다른지 보려면 필요. cell_ids가
        # 있을 때만(=predict_dataset()이 채워준 경우) 계산한다.
        cell_ids = p.get("cell_ids")
        if cell_ids is not None and len(cell_ids) == len(t):
            ds_labels = np.array([_infer_dataset_from_cell_id(str(c)) for c in cell_ids])
            for ds_name in sorted(set(ds_labels.tolist())):
                sel = ds_labels == ds_name
                if sel.sum() > 1:
                    m = compute_metrics(t[sel], pred[sel])
                    out[f"dataset_{ds_name}"] = {k: float(v) for k, v in m.items()}
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
    def _extract_cell_series(self, pred_dict: dict, cell: str) -> dict | None:
        """pred_dict에서 cell 하나에 해당하는 행만 뽑아 capacity curve 플롯에 필요한
        배열들을 묶어 돌려준다. 그 셀이 split에 없으면 경고를 찍고 None
        (_plot_capacity_curves에서 분리, 동작 변화 없음)."""
        cell_ids = np.array(pred_dict["cell_ids"])
        sel = cell_ids == cell
        if sel.sum() == 0:
            print(f"[eval] rep cell '{cell}' not found in test split, skipping")
            return None

        cap_init_ah = pred_dict["cap_init_raw"]                  # Ah per sample
        cap_true    = pred_dict["cap_true_raw"] * cap_init_ah    # SOH→Ah
        cap_pred    = pred_dict["cap_pred_raw"] * cap_init_ah    # SOH→Ah
        return {
            "cyc":  np.array(pred_dict["cycles"])[sel],
            "true": cap_true[sel],
            "pred": cap_pred[sel],
            "dir":  pred_dict["direction"][sel],
            "qlo":  np.array(pred_dict["q_frac_lo"], dtype=np.float64)[sel],
            "seg":  np.array(pred_dict["seg_names"])[sel],
        }

    def _plot_capacity_curve_direction(
        self, ax_cap, ax_err, ax_rel, series: dict, dir_name: str, is_charge: bool,
    ) -> None:
        """한 셀의 한 방향(충전/방전)에 대해 capacity curve/절대오차/상대오차 3개
        패널을 채운다 — scen_idx(zone) 카테고리로 뭉뚱그려 평균내는 대신, q_frac_lo
        (세그먼트 시작 q-fraction, 축 설계상 사이클과 무관하게 고정)로 정렬한
        "세그먼트 순번"별로 선을 따로 그린다(2026-09-18, _plot_capacity_curves에서
        분리는 2026-10-04 — 둘 다 동작 변화 없음). True capacity는 세그먼트와 무관하게
        사이클당 하나의 값이라 한 줄만 그린다."""
        uniq_cyc = np.unique(series["cyc"])
        dir_mask = (series["dir"] > 0) if is_charge else (series["dir"] < 0)

        d_cyc  = series["cyc"][dir_mask]
        d_true = series["true"][dir_mask]
        d_pred = series["pred"][dir_mask]
        d_qlo  = series["qlo"][dir_mask]
        d_seg  = series["seg"][dir_mask]

        true_line = np.array([
            d_true[d_cyc == cy].mean() if (d_cyc == cy).any() else np.nan
            for cy in uniq_cyc
        ])
        ax_cap.plot(uniq_cyc, true_line, "b-", label="True", linewidth=1.5)

        # 세그먼트 순번: 이 방향의 고유 q_frac_lo를 오름차순 정렬 -- 축 설계상
        # 사이클 간 완전히 고정이므로 첫 사이클에서 뽑은 목록이 전체 대표값이다.
        first_cyc = uniq_cyc[0]
        order_qlo = np.sort(np.unique(d_qlo[d_cyc == first_cyc]))
        if len(order_qlo) == 0:
            order_qlo = np.sort(np.unique(d_qlo))
        n_order = max(len(order_qlo), 1)
        colors = plt.cm.viridis(np.linspace(0.05, 0.90, n_order))

        for k, qlo_val in enumerate(order_qlo):
            seg_mask = np.isclose(d_qlo, qlo_val, atol=1e-6)
            if seg_mask.sum() == 0:
                continue
            s_cyc  = d_cyc[seg_mask]
            s_pred = d_pred[seg_mask]
            pred_line = np.array([
                s_pred[s_cyc == cy].mean() if (s_cyc == cy).any() else np.nan
                for cy in uniq_cyc
            ])
            zone_name = d_seg[seg_mask][0] if seg_mask.any() else "?"
            label = f"seg{k + 1} ({zone_name})"
            color = colors[k]

            ax_cap.plot(uniq_cyc, pred_line, color=color, linewidth=1.2, label=label)

            err_abs = np.abs(pred_line - true_line)
            err_rel = err_abs / np.where(true_line == 0, 1.0, np.abs(true_line)) * 100
            ax_err.plot(uniq_cyc, err_abs, color=color, linewidth=1.0, label=label)
            ax_rel.plot(uniq_cyc, err_rel, color=color, linewidth=1.0, label=label)

        ax_cap.set_xlabel("Cycle"); ax_cap.set_ylabel("Capacity (Ah)")
        ax_cap.set_title(f"{dir_name} — capacity curve")
        ax_cap.legend(fontsize=7, ncol=2)
        ax_err.set_xlabel("Cycle"); ax_err.set_ylabel("|Error| (Ah)")
        ax_err.set_title(f"{dir_name} — absolute error")
        ax_err.legend(fontsize=7, ncol=2)
        ax_rel.set_xlabel("Cycle"); ax_rel.set_ylabel("Relative error (%)")
        ax_rel.set_title(f"{dir_name} — relative error (%)")
        ax_rel.legend(fontsize=7, ncol=2)

    def _plot_capacity_curves(self, pred_dict: dict) -> None:
        """대표 셀(self.rep_cells)마다 2행(충전/방전)×3열(capacity/절대오차/상대오차)
        figure를 저장한다.

        2026-10-04: 단일 책임 원칙에 따라 서브함수로 분리했다(동작 변화 없는 순수
        구조 정리) — _extract_cell_series(셀 하나의 데이터 추출) ->
        _plot_capacity_curve_direction(방향 하나의 3패널 채우기) -> 이 함수는 셀
        루프 + figure 생성/저장만 남는다."""
        if not _HAS_MPL:
            return

        for cell in self.rep_cells:
            series = self._extract_cell_series(pred_dict, cell)
            if series is None:
                continue

            fig, axes = plt.subplots(2, 3, figsize=(17, 8))
            fig.suptitle(cell, fontsize=12, y=1.01)

            for row, (dir_name, is_charge) in enumerate((("Charge", True), ("Discharge", False))):
                self._plot_capacity_curve_direction(
                    axes[row, 0], axes[row, 1], axes[row, 2], series, dir_name, is_charge,
                )

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
