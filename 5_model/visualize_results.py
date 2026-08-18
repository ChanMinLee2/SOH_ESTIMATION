"""
visualize_results.py — 여러 SCR 학습 run 폴더를 다축으로 비교하는 시각화 스크립트.

train_scr.py --phase 2 로 생성된 run 폴더(예: _5_data_model_scr/0724_0111_p2_res_qfw)를
여러 개 지정하면, 그 안의 metrics.json / predictions/test_predictions.csv /
routing/routing_table.csv / checkpoints/*.pt 를 읽어 하나의 비교 플롯으로 그린다.

비교 항목 (하나의 PNG, GridSpec 레이아웃):
  Row 1-3 : RMSE / MAE / MAPE  — (Overall + 시나리오별) × N_scenarios+1 서브플롯
            * hard 라우팅(실배포, 학습된 분류기 argmax) 기준.
              predictions/test_predictions.csv의 seg_name으로 직접 재집계한다
              (metrics.json의 breakdown은 charge/discharge/level_*뿐, 시나리오
              6개 개별 값이 없기 때문).
  Row 4   : 스칼라 지표 7종 — R²(hard, overall) / 분류 정확도(hard) / 인퍼런스
            시간(ms/sample) / 학습 파라미터 수 / oracle→hard RMSE 저하율(라우팅
            비용) / 평균 HI 비용(avg_cost) / random-seg(E2) RMSE(있으면)
  Row 5   : 선정 HI 자카드(Jaccard) 유사도 히트맵(이진) — run별로 1개씩, probe(방향게이트) +
            시나리오 간 (scenario × scenario) 행렬. 한 run 안에서 시나리오마다 얼마나
            다른 HI 서브셋을 선정했는지 보기 위함 (대각선=1.0, 낮을수록 시나리오 간
            HI 선택이 서로 다름). routing_table.csv의 이진(top-k 컷오프 후) 마스크만
            사용(모델 재로딩 불요).
  Row 6   : 확률 가중 Jaccard(Ruzicka 유사도) 히트맵 — Row 5와 같은 레이아웃이지만
            gates/classification_HIs.json·regression_HIs.json의 top-k로 자르기 전
            원본 gate 확률을 사용한다. 이진 버전은 top-k 경계에서 아깝게 탈락한 HI를
            무조건 "안 겹침"으로 취급하지만, 이 버전은 min(p,q)/max(p,q) 가중으로
            그 "거의 선택될 뻔한" 정도까지 반영한다(이진 마스크를 넣으면 Row 5와 동일).
  Row 7   : (--with-jacobian 지정 시) 선정 HI 자코비안(gradient) 코사인 유사도
            히트맵 — Overall + 시나리오별. 실제 테스트 샘플에서
            d(cap_pred)/d(x_hi) 를 시나리오별로 평균한 벡터끼리 비교 —
            "같은 HI를 골랐어도 실제로 비슷하게 쓰는가"까지 검증.

전제: 비교 대상 run들이 동일한 시나리오 축(scenario_spec.json의 axis / n_scenarios /
      scenario_names)을 공유하면 위 Row 1-3/5-7에 시나리오별 서브플롯·히트맵이 모두 나온다.
      축이 서로 다르면(예: q_frac_wide vs vqslope vs q_abs) 시나리오 이름·개수 자체가
      의미상 대응되지 않으므로 에러를 내지 않고 "축 간 비교 모드"로 자동 전환한다 —
      Row 1-3은 Overall만 남기고 시나리오별 열을 생략하며, Row 5-7(시나리오×시나리오
      Jaccard/자코비안 히트맵)은 축마다 시나리오 구조가 달라 비교 자체가 성립하지
      않으므로 생략한다. 대표 셀 용량곡선 비교(capacity_curve_compare_*.png)도 이 경우
      시나리오별 행 대신 시나리오 평균 1개 행으로 그린다.

결과 저장 위치:
  _5_data_model_scr/comparison/<MMDD_HHMM>_result_comparison/   (기본, --title로 접미사 교체 가능)
    result_comparison.png
    summary.json

사용:
  & python 5_model/visualize_results.py --runs "_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%" "_5_data_model_scr/0723_2356_p2_tr_qfw_35%_20%" "_5_data_model_scr/0724_0111_p2_res_qfw_35%_20%" --labels mlp transformer resnet_tab
  python 5_model/visualize_results.py --runs _5_data_model_scr/0723_1633_p2_mlp_qfw _5_data_model_scr/0724_0111_p2_res_qfw
  python 5_model/visualize_results.py --runs RUN1 RUN2 RUN3 --labels mlp resnet transformer --with-jacobian
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows 터미널 cp949 → UTF-8 강제 (콘솔 인코딩이 cp949일 때 한글/특수문자 출력 크래시 방지)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from utils.hi_schema import N_HI, RAW_CH, RAW_N
from models.scr_model import SCRModel
from common.scenario.base import ScenarioSpec

OUT_ROOT = PROJECT_ROOT / "_5_data_model_scr" / "comparison"

_COLORS = plt.get_cmap("tab10").colors


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="여러 SCR run 폴더 다축 비교 시각화")
    p.add_argument("--runs", nargs="+", required=True,
                   help="비교할 run 폴더 경로 (예: _5_data_model_scr/0724_0111_p2_res_qfw), 2개 이상")
    p.add_argument("--labels", nargs="+", default=None,
                   help="run별 표시 이름 (미지정 시 폴더명 사용, --runs와 개수 일치 필요)")
    p.add_argument("--with-jacobian", action="store_true",
                   help="실 데이터 기반 Jacobian(gradient) 코사인 유사도 패널 추가 (느림 — 데이터셋 재구축 필요)")
    p.add_argument("--checkpoint-name", default="best.pt",
                   help="run별 사용할 체크포인트 파일명 (기본 best.pt, 없으면 final.pt로 폴백)")
    p.add_argument("--infer-batch-size", type=int, default=256)
    p.add_argument("--infer-warmup", type=int, default=10)
    p.add_argument("--infer-reps", type=int, default=50)
    p.add_argument("--jacobian-max-samples", type=int, default=300,
                   help="시나리오/전체당 gradient 계산에 사용할 최대 샘플 수")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-name", default=None,
                   help="결과 폴더명 전체 오버라이드 (기본 <MMDD_HHMM>_result_comparison). "
                        "--title과 동시 지정 시 이쪽이 우선")
    p.add_argument("--title", default=None,
                   help="결과 폴더명의 'result_comparison' 부분만 대체 "
                        "(<MMDD_HHMM>_<title> 형태로 생성, 타임스탬프는 유지)")
    p.add_argument("--rep-cells", nargs="+", default=None,
                   help="SOH 예측 곡선 비교(capacity_curve_compare_*.png)에 쓸 셀 ID를 "
                        "직접 지정 (예: b1c0). 미지정 시 기존처럼 MIT/HUST 각각 최대 3개를 "
                        "자동 선정. 지정하면 데이터셋 무관하게 정확히 그 셀들만 그린다.")
    return p.parse_args()


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _normalize_legacy_metrics(metrics: dict) -> None:
    """2026-07-23 oracle/hard/soft 통합 평가 이전(protocol/vwindow/rcs 등 구형 run)
    metrics.json은 test.{capacity,breakdown,scenario,efficiency}를 최상위에 바로 두고
    분류기(classifier) 자체가 없었다(scenario.disabled=true) — oracle/hard 라우팅
    구분이 없다. 이런 run을 축 간 비교에 섞을 수 있도록, test.capacity를
    test.oracle.capacity로 감싸 새 스키마와 동일한 모양으로 맞춘다(in-place).
    test.hard가 없으므로 has_clf=False로 남아 hard 관련 필드는 그대로 None 처리된다.
    """
    test = metrics.get("test")
    if not isinstance(test, dict) or "oracle" in test or "capacity" not in test:
        return
    test["oracle"] = {"capacity": test["capacity"], "classification": {}}


# =============================================================================
# Run bundle 로딩
# =============================================================================

class RunBundle:
    """단일 run 폴더에서 읽은 모든 정보를 담는 컨테이너."""

    def __init__(self, run_dir: Path, label: str):
        self.run_dir = run_dir
        self.label = label

        self.cfg: dict = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        self.spec = ScenarioSpec.load(run_dir / "scenario_spec.json")

        self.metrics: dict = json.loads(
            (run_dir / "metrics" / "metrics.json").read_text(encoding="utf-8")
        )
        _normalize_legacy_metrics(self.metrics)
        self.has_clf = "hard" in self.metrics.get("test", {})

        self.pred_rows = self._load_predictions()
        self.routing_sets = self._load_routing_table()
        self.gate_probs = self._load_gate_probs()

        eff_path = run_dir / "random_seg_test" / "metrics.json"
        self.random_seg: dict | None = (
            json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else None
        )

        # lazily filled by _build_model_for_run
        self.model: torch.nn.Module | None = None
        self.n_params: int | None = None
        self.infer_ms_per_sample: float | None = None
        self.jacobian_profiles: dict[str, np.ndarray] | None = None

    def _load_predictions(self) -> list[dict]:
        path = self.run_dir / "predictions" / "test_predictions.csv"
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "cell_id":     r["cell_id"],
                    "cycle":       int(r["cycle"]),
                    "seg_name":    r["seg_name"],
                    "soh_true":    float(r["soh_true"]),
                    "soh_pred":    float(r["soh_pred"]),
                    "cap_true_Ah": float(r["cap_true_Ah"]),
                    "cap_pred_Ah": float(r["cap_pred_Ah"]),
                })
        return rows

    def _load_routing_table(self) -> dict[str, set[str]]:
        path = self.run_dir / "routing" / "routing_table.csv"
        out: dict[str, set[str]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            hi_names = header[1:]
            for row in reader:
                gate_name = row[0]
                active = {hi_names[i] for i, v in enumerate(row[1:]) if v == "1"}
                out[gate_name] = active
        return out

    def _load_gate_probs(self) -> dict[str, np.ndarray]:
        """gates/*.json의 0/1로 자르기 전 원본 gate 확률을 HI 인덱스(0..N_HI-1) 정렬
        벡터로 복원한다. routing_sets(이진)와 같은 라벨 체계(probe + 시나리오명)를 쓴다 —
        확률 가중 Jaccard(Ruzicka 유사도) 계산용. gates/*.json이 없으면 빈 dict."""
        out: dict[str, np.ndarray] = {}

        cls_path = self.run_dir / "gates" / "classification_HIs.json"
        if cls_path.exists():
            data = json.loads(cls_path.read_text(encoding="utf-8"))
            ch  = np.zeros(N_HI, dtype=np.float64)
            dis = np.zeros(N_HI, dtype=np.float64)
            for idx, p in zip(data.get("charge_ranked", []), data.get("charge_probs", [])):
                ch[idx] = p
            for idx, p in zip(data.get("discharge_ranked", []), data.get("discharge_probs", [])):
                dis[idx] = p
            out["probe"] = np.maximum(ch, dis)   # routing_table.csv의 union(probe) 규칙과 동일

        reg_path = self.run_dir / "gates" / "regression_HIs.json"
        if reg_path.exists():
            data = json.loads(reg_path.read_text(encoding="utf-8"))
            for s in range(self.spec.n_scenarios):
                ranked = data.get(f"seg_{s}_ranked")
                probs  = data.get(f"seg_{s}_probs")
                if ranked is None or probs is None:
                    continue
                vec = np.zeros(N_HI, dtype=np.float64)
                for idx, p in zip(ranked, probs):
                    vec[idx] = p
                sname = data.get(f"seg_{s}_seg_name", self.spec.scenario_names[s])
                out[sname] = vec
        return out


def _load_bundles(run_dirs: list[str], labels: list[str] | None) -> list[RunBundle]:
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError(f"--labels 개수({len(labels)})가 --runs 개수({len(run_dirs)})와 다릅니다.")
    bundles = []
    for i, rd in enumerate(run_dirs):
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"run 폴더를 찾을 수 없습니다: {run_dir}")
        label = labels[i] if labels else run_dir.name
        print(f"[viz] loading run: {run_dir}  (label={label})")
        bundles.append(RunBundle(run_dir, label))
    return bundles


def _check_cross_axis(bundles: list[RunBundle]) -> bool:
    """비교 대상 run들의 시나리오 축이 서로 다른지 확인한다.

    과거엔 축이 다르면 즉시 에러로 중단했지만, 축 자체가 다른 run들을 나란히 보고
    싶은 경우(예: protocol/vwindow/rcs/q_frac_wide/vqslope/q_abs 전체 비교)가 실제로
    있어 — 시나리오 이름/개수가 대응되지 않는 시나리오별 서브플롯·히트맵만 생략하고
    Overall 지표는 계속 비교할 수 있도록 완화했다. True를 반환하면 호출부가
    "축 간 비교 모드"로 렌더링한다.
    """
    ref = bundles[0].spec
    diffs = [b for b in bundles[1:] if b.spec.axis != ref.axis or b.spec.scenario_names != ref.scenario_names]
    if diffs:
        print("[viz] 비교 대상 run들의 시나리오 축이 서로 다릅니다 — 축 간 비교 모드(Overall만) 사용:")
        print(f"  [{ref.axis}] {bundles[0].label}: {ref.scenario_names}")
        for b in diffs:
            print(f"  [{b.spec.axis}] {b.label}: {b.spec.scenario_names}")
        return True
    return False


# =============================================================================
# 성능 지표 (test_predictions.csv 기반, hard 라우팅)
# =============================================================================

def _metrics_from_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100.0) if mask.any() else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _compute_capacity_panels(b: RunBundle) -> dict[str, dict[str, float]]:
    """{"Overall": {...}, scen_name: {...}, ...} — hard 라우팅 test_predictions.csv 기준."""
    rows = b.pred_rows
    all_true = np.array([r["soh_true"] for r in rows])
    all_pred = np.array([r["soh_pred"] for r in rows])
    out = {"Overall": _metrics_from_pairs(all_true, all_pred)}

    # metrics.json과의 정합성 체크 (다른 라우팅 모드 CSV가 저장된 경우 조기 경고)
    mode_key = "hard" if b.has_clf else "oracle"
    ref_rmse = b.metrics["test"][mode_key]["capacity"]["rmse"]
    if abs(out["Overall"]["rmse"] - ref_rmse) > 5e-3:
        print(f"[viz][경고] {b.label}: CSV 기반 Overall RMSE({out['Overall']['rmse']:.4f})가 "
              f"metrics.json test.{mode_key}.capacity.rmse({ref_rmse:.4f})와 어긋납니다 — "
              "predictions/test_predictions.csv가 다른 라우팅 모드로 저장됐을 수 있습니다.")

    for name in b.spec.scenario_names:
        t = np.array([r["soh_true"] for r in rows if r["seg_name"] == name])
        p = np.array([r["soh_pred"] for r in rows if r["seg_name"] == name])
        out[name] = _metrics_from_pairs(t, p) if len(t) > 1 else {
            "rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")
        }
    return out


def _scalar_metrics(b: RunBundle) -> dict[str, float | None]:
    test = b.metrics["test"]
    mode_key = "hard" if b.has_clf else "oracle"
    r2 = test[mode_key]["capacity"]["r2"]
    clf_acc = test["hard"]["classification"].get("accuracy") if b.has_clf else None

    oracle_rmse = test["oracle"]["capacity"]["rmse"]
    hard_rmse = test["hard"]["capacity"]["rmse"] if b.has_clf else None
    routing_gap_pct = (
        (hard_rmse - oracle_rmse) / oracle_rmse * 100.0
        if (hard_rmse is not None and oracle_rmse > 0) else None
    )

    eff = test.get("efficiency", {})
    avg_cost = eff.get("avg_cost")

    rs_rmse = b.random_seg["rmse"] if b.random_seg else None

    return {
        "r2": r2,
        "clf_acc": clf_acc,
        "infer_ms": b.infer_ms_per_sample,
        "n_params_k": (b.n_params / 1000.0) if b.n_params is not None else None,
        "routing_gap_pct": routing_gap_pct,
        "avg_cost": avg_cost,
        "random_seg_rmse": rs_rmse,
    }


# =============================================================================
# 모델 재구성 (인퍼런스 타이밍 / 파라미터 수 / Jacobian 용)
# =============================================================================

def _load_gate_masks(run_dir: Path, cfg: dict, n_scenarios: int
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clf_cfg = cfg.get("classifier", {})
    reg_cfg = cfg.get("regression", {})
    charge_m = clf_cfg.get("charge_probe_m", clf_cfg.get("probe_m_count", 1))
    discharge_m = clf_cfg.get("discharge_probe_m", clf_cfg.get("probe_m_count", 1))
    scen_k = reg_cfg.get("scen_k_count", 5)

    probe_data = json.loads((run_dir / "gates" / "classification_HIs.json").read_text(encoding="utf-8"))
    scen_data = json.loads((run_dir / "gates" / "regression_HIs.json").read_text(encoding="utf-8"))

    ch_mask = torch.zeros(N_HI, dtype=torch.bool)
    for i in probe_data["charge_ranked"][:charge_m]:
        ch_mask[i] = True
    dis_mask = torch.zeros(N_HI, dtype=torch.bool)
    for i in probe_data["discharge_ranked"][:discharge_m]:
        dis_mask[i] = True

    scen_masks = torch.zeros(n_scenarios, N_HI, dtype=torch.bool)
    for s in range(n_scenarios):
        for i in scen_data[f"seg_{s}_ranked"][:scen_k]:
            scen_masks[s, i] = True

    return ch_mask, dis_mask, scen_masks


def _build_model_for_run(b: RunBundle, device: torch.device, ckpt_name: str) -> None:
    """b.model / b.n_params 를 채운다."""
    ckpt_path = b.run_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        fallback = "final.pt" if ckpt_name != "final.pt" else "best.pt"
        ckpt_path = b.run_dir / "checkpoints" / fallback
        print(f"[viz] {b.label}: {ckpt_name} 없음 → {fallback} 사용")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    ch_mask, dis_mask, scen_masks = _load_gate_masks(b.run_dir, b.cfg, b.spec.n_scenarios)
    m_cfg = b.cfg.get("model", {})
    model = SCRModel(
        d_probe=m_cfg.get("d_probe", 64),
        d_head=m_cfg.get("d_head", 128),
        dropout=m_cfg.get("dropout", 0.1),
        charge_probe_mask=ch_mask,
        discharge_probe_mask=dis_mask,
        scen_masks=scen_masks,
        model_cfg=m_cfg,
        spec=b.spec,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval().to(device)

    b.model = model
    b.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[viz] {b.label}: model built ({ckpt_path.name}), "
          f"trainable params={b.n_params:,}, regression_model={m_cfg.get('regression_model', 'mlp')}")


def _benchmark_inference(b: RunBundle, device: torch.device,
                          batch_size: int, warmup: int, reps: int) -> None:
    """합성 입력(x_hi ~ N(0,1), 정규화된 HI 분포와 근사)으로 순수 forward 지연시간 측정."""
    model = b.model
    n_scen = b.spec.n_scenarios
    # generator는 CPU에 고정 — CUDA generator는 별도 초기화가 필요해 번거로우므로,
    # 항상 CPU에서 생성한 뒤 배치 전체를 목표 device로 옮긴다.
    g = torch.Generator().manual_seed(0)
    x_hi = torch.randn(batch_size, N_HI, generator=g)
    nan_mask = torch.ones(batch_size, N_HI)
    direction = torch.where(torch.rand(batch_size, generator=g) > 0.5, 1.0, -1.0)
    seg_idx = torch.randint(0, n_scen, (batch_size,), generator=g)
    cap_init = torch.randn(batch_size, generator=g)
    batch = {
        "x_hi": x_hi.to(device), "nan_mask": nan_mask.to(device),
        "direction": direction.to(device), "seg_idx": seg_idx.to(device),
        "cap_init": cap_init.to(device),
    }
    # with_raw_cnn/with_raw_flat 모델(REGRESSION_UPGRADE.md §2/§5/§8/§10)은 forward에서
    # batch["x_raw"]를 읽으므로, 합성 벤치마크 입력에도 같은 shape(B, RAW_CH, RAW_N)의
    # 더미를 채워줘야 한다.
    if getattr(model, "raw_cnn", None) is not None or getattr(model, "with_raw_flat", False):
        x_raw = torch.randn(batch_size, RAW_CH, RAW_N, generator=g)
        batch["x_raw"] = x_raw.to(device)

    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(batch)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    ms_per_batch = (t1 - t0) / reps * 1000.0
    b.infer_ms_per_sample = ms_per_batch / batch_size
    print(f"[viz] {b.label}: inference {ms_per_batch:.3f} ms/batch(bs={batch_size}) "
          f"= {b.infer_ms_per_sample:.5f} ms/sample")


# =============================================================================
# Jacobian (gradient) 프로파일 — --with-jacobian 전용
# =============================================================================

def _axis_dir_from_spec(spec: ScenarioSpec) -> str:
    if spec.axis == "q_frac_wide":
        p = spec.params or {}
        n1 = int(round(p.get("n1", 0.4) * 100))
        n2 = int(round(p.get("n2", 0.2) * 100))
        ns = int(p.get("n_samples", 4))
        min_pts = int(p.get("min_pts", 10))
        minpts_sfx = f"_minpts{min_pts}" if min_pts != 10 else ""
        # assign="none"(no_scen 대조군, docs/260816_RESULTS.md §5)이면 접미사 (hi_correlation._qfw_tag 와 동일 규칙)
        assign_sfx = "" if p.get("assign", "position_bin") == "position_bin" else "_noscen"
        return f"q_frac_wide/n1-{n1}%_n2-{n2}%_N-{ns}{minpts_sfx}{assign_sfx}"
    if spec.axis == "q_frac_ref":
        # q_frac_wide와 동일한 n1/n2/N % 표기 + lag/noise 태그
        # (hi_correlation._qfref_tag, train_classifier._axis_dir_from_spec 와 동일 규칙)
        p = spec.params or {}
        n1 = int(round(p.get("n1", 0.4) * 100))
        n2 = int(round(p.get("n2", 0.2) * 100))
        ns = int(p.get("n_samples", 4))
        lag = int(p.get("ref_lag", 0))
        noise = int(round(p.get("noise_amp", 0.03) * 100))
        nmode = str(p.get("noise_mode", "ou"))
        period = int(round(p.get("noise_period_cycles", 200.0)))
        min_pts = int(p.get("min_pts", 10))
        minpts_sfx = f"_minpts{min_pts}" if min_pts != 10 else ""
        assign_sfx = "" if p.get("assign", "position_bin") == "position_bin" else "_noscen"
        return f"q_frac_ref/n1-{n1}%_n2-{n2}%_N-{ns}{minpts_sfx}{assign_sfx}_lag-{lag}_noise-{noise}%_{nmode}-{period}"
    if spec.axis == "q_abs":
        p = spec.params or {}
        ms = int(round(p.get("mid_start", 0.20) * 100))
        me = int(round(p.get("mid_end", 0.50) * 100))
        sl = int(round(p.get("seg_len", 0.15) * 100))
        ns = int(p.get("n_samples", 4))
        return f"q_abs/ms-{ms}%_me-{me}%_sl-{sl}%_N-{ns}"
    return spec.axis


def _build_test_dataset_for_run(b: RunBundle):
    from datasets.segment_dataset import build_datasets, SegmentNormalizer

    cfg = json.loads(json.dumps(b.cfg))  # deep copy (json roundtrip은 이 cfg가 순수 dict/list/str/number라 안전)
    data_cfg = cfg.setdefault("data", {})
    axis_dir = _axis_dir_from_spec(b.spec)
    if not data_cfg.get("seg_data_dir"):
        data_cfg["seg_data_dir"] = f"{DATA_4_HI_ROOT_STR}/{axis_dir}/seg"
    if not data_cfg.get("data_dir"):
        data_cfg["data_dir"] = f"{DATA_4_HI_ROOT_STR}/{axis_dir}/cycle"

    _, _, test_ds, _ = build_datasets(cfg, spec=b.spec)

    ckpt_path = b.run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = b.run_dir / "checkpoints" / "final.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    norm = SegmentNormalizer()
    norm.mean_ = ckpt["norm_mean"]
    norm.std_ = ckpt["norm_std"]
    norm.cap_init_mean_ = float(ckpt.get("norm_cap_init_mean", 0.0))
    norm.cap_init_std_ = float(ckpt.get("norm_cap_init_std", 1.0))
    # x_hi는 build_datasets 내부에서 동일 데이터·seed로 fit된 정규화가 이미 적용되어 있어
    # 재계산이 필요 없다 (test_scr.py._reapply_norm과 동일한 전제). cap_init만 체크포인트
    # 기준으로 재적용한다.
    test_ds.cap_init = torch.tensor(
        norm.transform_cap_init(test_ds.cap_init_raw), dtype=torch.float32
    )
    return test_ds


def _compute_jacobian_profiles(b: RunBundle, device: torch.device, max_samples: int) -> None:
    """b.jacobian_profiles = {"Overall": (N_HI,), scen_name: (N_HI,), ...} 채움."""
    test_ds = _build_test_dataset_for_run(b)
    model = b.model
    seg_idx_all = test_ds.seg_idx

    def _profile(sel_idx: torch.Tensor) -> np.ndarray | None:
        if len(sel_idx) == 0:
            return None
        if len(sel_idx) > max_samples:
            perm = torch.randperm(len(sel_idx))[:max_samples]
            sel_idx = sel_idx[perm]
        x_hi = test_ds.x_hi[sel_idx].to(device).clone().requires_grad_(True)
        batch = {
            "x_hi": x_hi,
            "nan_mask": test_ds.nan_mask[sel_idx].to(device),
            "direction": test_ds.direction[sel_idx].to(device),
            "seg_idx": test_ds.seg_idx[sel_idx].to(device),
            "cap_init": test_ds.cap_init[sel_idx].to(device),
        }
        # with_raw_cnn/with_raw_flat 모델은 forward에서 batch["x_raw"]를 읽음 — SegmentDataset은
        # include_raw 여부와 무관하게 항상 x_raw를 보유하므로 그대로 슬라이싱해서 넣는다.
        if getattr(model, "raw_cnn", None) is not None or getattr(model, "with_raw_flat", False):
            batch["x_raw"] = test_ds.x_raw[sel_idx].to(device)
        out = model(batch)
        grad, = torch.autograd.grad(out["cap_pred"].sum(), x_hi)
        return grad.mean(dim=0).detach().cpu().numpy()

    profiles: dict[str, np.ndarray | None] = {}
    all_idx = torch.arange(len(test_ds))
    profiles["Overall"] = _profile(all_idx)
    for s, name in enumerate(b.spec.scenario_names):
        sel = (seg_idx_all == s).nonzero(as_tuple=True)[0]
        profiles[name] = _profile(sel)

    b.jacobian_profiles = profiles
    print(f"[viz] {b.label}: Jacobian 프로파일 계산 완료 ({len(profiles)}개 그룹)")


# =============================================================================
# 유사도 행렬
# =============================================================================

def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _weighted_jaccard(a: "np.ndarray | None", b: "np.ndarray | None") -> float:
    """확률 가중 Jaccard(Ruzicka 유사도) — a/b가 이진(0/1) 벡터면 _jaccard와 동일한 값.
    0/1 top-k 컷오프 전 원본 gate 확률을 쓰면, top-k 경계 근처에서 아깝게 탈락한 HI도
    "일부만 겹친 것"으로 반영된다(이진 Jaccard는 컷오프 반대편이면 무조건 0 취급)."""
    if a is None or b is None:
        return float("nan")
    num = float(np.minimum(a, b).sum())
    den = float(np.maximum(a, b).sum())
    return num / den if den > 1e-12 else 1.0


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _sim_matrix(bundles: list[RunBundle], key: str, kind: str) -> np.ndarray:
    n = len(bundles)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if kind == "jaccard":
                ai = bundles[i].routing_sets.get(key, set())
                aj = bundles[j].routing_sets.get(key, set())
                m[i, j] = _jaccard(ai, aj)
            else:  # jacobian
                ai = bundles[i].jacobian_profiles.get(key)
                aj = bundles[j].jacobian_profiles.get(key)
                m[i, j] = _cosine(ai, aj)
    return m


def _scenario_sim_matrix(bundle: RunBundle, labels: list[str]) -> np.ndarray:
    """단일 run 내부에서 라벨(probe/시나리오)끼리 선정 HI 자카드 유사도(이진, top-k 컷오프 후).
    낮을수록 그 두 라벨이 서로 다른 HI 서브셋을 쓴다는 뜻 — "시나리오별로 얼마나
    다른 HI를 골랐는가"를 보기 위한 행렬(대각선은 항상 1.0)."""
    n = len(labels)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            ai = bundle.routing_sets.get(labels[i], set())
            aj = bundle.routing_sets.get(labels[j], set())
            m[i, j] = _jaccard(ai, aj)
    return m


def _scenario_sim_matrix_weighted(bundle: RunBundle, labels: list[str]) -> np.ndarray:
    """_scenario_sim_matrix의 확률 가중(Ruzicka) 버전 — top-k로 자르기 전 원본 gate
    확률(bundle.gate_probs)을 사용해 top-k 경계 근처 HI의 "부분 겹침"까지 반영한다."""
    n = len(labels)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            ai = bundle.gate_probs.get(labels[i])
            aj = bundle.gate_probs.get(labels[j])
            m[i, j] = _weighted_jaccard(ai, aj)
    return m


# =============================================================================
# 시각화
# =============================================================================

def _short_label(s: str, maxlen: int = 16) -> str:
    """긴 run 라벨을 앞(타임스탬프)+뒤(모델/축 태그) 보존 축약. 예: 0724_0111_p2_res_qfw → 0724_0111…res_qfw"""
    if len(s) <= maxlen:
        return s
    head_n = maxlen // 2
    tail_n = maxlen - head_n - 1
    return f"{s[:head_n]}…{s[-tail_n:]}"


def _bar_row(fig, gs_row, col_labels: list[str], bundles: list[RunBundle],
             getter, title_prefix: str, ylabel: str, fmt: str = "{:.4f}"):
    """col_labels 개수만큼 서브플롯을 만들고, 각 서브플롯에 run별 막대를 그린다."""
    n = len(col_labels)
    for c, col_label in enumerate(col_labels):
        ax = fig.add_subplot(gs_row[c])
        vals = [getter(b, col_label) for b in bundles]
        xs = np.arange(len(bundles))
        colors = [_COLORS[i % len(_COLORS)] for i in range(len(bundles))]
        na_mask = [v is None or (isinstance(v, float) and np.isnan(v)) for v in vals]
        plot_vals = [0.0 if bad else v for bad, v in zip(na_mask, vals)]
        bars = ax.bar(xs, plot_vals, color=colors, alpha=0.85, edgecolor="k", linewidth=0.4)
        for x, bad, v, bar in zip(xs, na_mask, vals, bars):
            if bad:
                ax.text(x, 0, "N/A", ha="center", va="bottom", fontsize=7, color="red")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt.format(v), ha="center", va="bottom", fontsize=6.5)
        finite_vals = [v for bad, v in zip(na_mask, plot_vals) if not bad]
        top = max(finite_vals) if finite_vals else 1.0
        ax.set_ylim(0, top * 1.22 if top > 0 else 1.0)  # 막대 위 숫자 라벨이 제목과 안 겹치도록 여유 공간 확보
        ax.set_xticks([])
        ax.set_title(f"{title_prefix}{col_label}", fontsize=8.5)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)


def _heatmap_row(fig, gs_row, col_labels: list[str], bundles: list[RunBundle],
                  kind: str, title_prefix: str):
    n_runs = len(bundles)
    run_labels = [_short_label(b.label) for b in bundles]
    for c, col_label in enumerate(col_labels):
        ax = fig.add_subplot(gs_row[c])
        m = _sim_matrix(bundles, col_label, kind)
        im = ax.imshow(m, vmin=0 if kind == "jaccard" else -1, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(n_runs)); ax.set_xticklabels(run_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n_runs)); ax.set_yticklabels(run_labels, fontsize=6)
        for i in range(n_runs):
            for j in range(n_runs):
                v = m[i, j]
                txt = "NaN" if np.isnan(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                        color="black" if not np.isnan(v) and v > 0.4 else "dimgray")
        ax.set_title(f"{title_prefix}{col_label}", fontsize=8)
    return im if n_runs > 0 else None


def _scenario_heatmap_row(fig, gs_row_spec, bundles: list[RunBundle],
                           labels: list[str], title_prefix: str,
                           matrix_fn=_scenario_sim_matrix):
    """run별로 하나씩(열=run) labels×labels(probe+시나리오) 유사도 행렬을 그린다.

    _heatmap_row(kind="jaccard")는 "같은 시나리오를 run끼리 비교"(run×run)했지만,
    이 함수는 "한 run 안에서 시나리오끼리 비교"(scenario×scenario) — 시나리오별로
    실제로 다른 HI를 선정했는지 보기 위한 본래 의도에 맞춘 버전.

    matrix_fn: _scenario_sim_matrix(이진 Jaccard, 기본) 또는
               _scenario_sim_matrix_weighted(확률 가중 Ruzicka) 중 선택.
    """
    n_runs = len(bundles)
    inner = gridspec.GridSpecFromSubplotSpec(1, n_runs, subplot_spec=gs_row_spec, wspace=0.7)
    n_lab = len(labels)
    im = None
    for c, b in enumerate(bundles):
        ax = fig.add_subplot(inner[0, c])
        m = matrix_fn(b, labels)
        im = ax.imshow(m, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(n_lab)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n_lab)); ax.set_yticklabels(labels, fontsize=6)
        for i in range(n_lab):
            for j in range(n_lab):
                v = m[i, j]
                txt = "NaN" if np.isnan(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                        color="black" if not np.isnan(v) and v > 0.4 else "dimgray")
        ax.set_title(f"{title_prefix}{_short_label(b.label)}", fontsize=7.5)
    return im if n_runs > 0 else None


def _scalar_bar_row(fig, gs_row, defs: list[tuple[str, str, str]],
                     bundles: list[RunBundle], scalar_cache: dict) -> None:
    for c, (key, title, fmt) in enumerate(defs):
        ax = fig.add_subplot(gs_row[c])
        vals = [scalar_cache[b.label][key] for b in bundles]
        xs = np.arange(len(bundles))
        colors = [_COLORS[i % len(_COLORS)] for i in range(len(bundles))]
        na_mask = [v is None or (isinstance(v, float) and np.isnan(v)) for v in vals]
        plot_vals = [0.0 if bad else v for bad, v in zip(na_mask, vals)]
        bars = ax.bar(xs, plot_vals, color=colors, alpha=0.85, edgecolor="k", linewidth=0.4)
        for x, bad, v, bar in zip(xs, na_mask, vals, bars):
            if bad:
                ax.text(x, 0, "N/A", ha="center", va="bottom", fontsize=7, color="red")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt.format(v), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks([]); ax.set_title(title, fontsize=8.2); ax.tick_params(axis="y", labelsize=7)


def _plot_all(bundles: list[RunBundle], with_jacobian: bool, out_path: Path, cross_axis: bool = False):
    if cross_axis:
        # 축이 다르면 시나리오 이름 자체가 대응되지 않으므로 Overall 열만 남긴다.
        scen_names: list[str] = []
        col_labels = ["Overall"]
    else:
        scen_names = bundles[0].spec.scenario_names
        col_labels = ["Overall"] + list(scen_names)
    n_scen = len(scen_names)
    metric_n_cols = len(col_labels)  # RMSE/MAE/MAPE 열 수 — cross_axis면 Overall 1개뿐
    jaccard_labels = ["probe"] + list(scen_names)

    scalar_defs_all = [
        ("r2", "R² (overall)", "{:.4f}"),
        ("clf_acc", "분류 정확도 (hard)", "{:.4f}"),
        ("infer_ms", "인퍼런스 (ms/sample)", "{:.4f}"),
        ("n_params_k", "학습 파라미터 (K)", "{:.1f}"),
        ("routing_gap_pct", "oracle→hard RMSE 저하 (%)", "{:.1f}"),
        ("avg_cost", "평균 HI 비용", "{:.1f}"),
        ("random_seg_rmse", "Random-seg E2 RMSE", "{:.4f}"),
    ]
    # 스칼라 지표는 애초에 시나리오와 무관(전역값)이므로 metric_n_cols와 별개로 자체 열 수를 갖는다
    # — cross_axis로 metric_n_cols=1이 되어도 스칼라 행이 세로로 7줄씩 늘어지지 않게 한다.
    grid_n_cols = max(metric_n_cols, min(len(scalar_defs_all), 4))
    n_scalar_rows = -(-len(scalar_defs_all) // grid_n_cols)  # ceil div

    # 행 구성: RMSE, MAE, MAPE(3) + 스칼라(n_scalar_rows) [+ Jaccard(1) + weighted Jaccard(1)] [+ Jacobian(1)]
    # Jaccard/자코비안 행은 시나리오×시나리오 구조에 의존하므로 축이 다르면(cross_axis) 생략한다.
    row_plan = ["rmse", "mae", "mape"] + ["scalar"] * n_scalar_rows
    if not cross_axis:
        row_plan += ["jaccard", "jaccard_weighted"]
    if with_jacobian and not cross_axis:
        row_plan.append("jacobian")
    elif with_jacobian and cross_axis:
        print("[viz] 축 간 비교 모드에서는 시나리오별 자코비안 히트맵을 생략합니다(--with-jacobian 무시).")
    total_rows = len(row_plan)

    fig = plt.figure(figsize=(2.7 * max(grid_n_cols, 7), 2.5 * total_rows))
    gs = gridspec.GridSpec(total_rows, grid_n_cols, figure=fig, hspace=0.9, wspace=0.5)

    cap_cache: dict[str, dict[str, dict[str, float]]] = {b.label: _compute_capacity_panels(b) for b in bundles}

    def _get(metric):
        def _fn(b, col):
            return cap_cache[b.label][col][metric]
        return _fn

    metric_row_defs = {"rmse": ("RMSE — ", "RMSE", "{:.4f}"),
                        "mae": ("MAE — ", "MAE", "{:.4f}"),
                        "mape": ("MAPE — ", "MAPE(%)", "{:.2f}")}

    scalar_cache = {b.label: _scalar_metrics(b) for b in bundles}

    row_i = 0
    scalar_row_i = 0
    im1 = im1b = im2 = None
    for kind in row_plan:
        if kind in metric_row_defs:
            prefix, ylabel, fmt = metric_row_defs[kind]
            _bar_row(fig, [gs[row_i, c] for c in range(metric_n_cols)], col_labels, bundles,
                     _get(kind), prefix, ylabel, fmt=fmt)
        elif kind == "scalar":
            chunk = scalar_defs_all[scalar_row_i * grid_n_cols:(scalar_row_i + 1) * grid_n_cols]
            _scalar_bar_row(fig, [gs[row_i, c] for c in range(len(chunk))], chunk, bundles, scalar_cache)
            scalar_row_i += 1
        elif kind == "jaccard":
            im1 = _scenario_heatmap_row(fig, gs[row_i, :], bundles, jaccard_labels, "",
                                        matrix_fn=_scenario_sim_matrix)
        elif kind == "jaccard_weighted":
            im1b = _scenario_heatmap_row(fig, gs[row_i, :], bundles, jaccard_labels, "weighted ",
                                         matrix_fn=_scenario_sim_matrix_weighted)
        elif kind == "jacobian":
            im2 = _heatmap_row(fig, [gs[row_i, c] for c in range(metric_n_cols)], col_labels, bundles,
                                "jacobian", "Jacobian cos — ")
        row_i += 1

    for im, row_key in ((im1, "jaccard"), (im1b, "jaccard_weighted"), (im2, "jacobian")):
        if im is not None:
            row_idx = row_plan.index(row_key)
            top = 1 - row_idx / total_rows
            fig.colorbar(im, cax=fig.add_axes([0.92, top - 1 / total_rows + 0.02, 0.012, 1 / total_rows - 0.04]))

    # ── 공유 범례 (run 라벨 ↔ 색상) ──────────────────────────────────────
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_COLORS[i % len(_COLORS)])
                       for i in range(len(bundles))]
    legend_texts = [f"{b.label}" for b in bundles]
    fig.legend(legend_handles, legend_texts, loc="upper center",
               ncol=min(len(bundles), 5), bbox_to_anchor=(0.5, 1.0), fontsize=9)

    if cross_axis:
        axis_desc = "axis=여러 개(축 간 비교 — 시나리오별 서브플롯/히트맵 생략)"
    else:
        axis_desc = f"axis={bundles[0].spec.axis}, n_scenarios={n_scen}"
    fig.suptitle(
        f"SCR 모델 비교 — {axis_desc}  "
        f"(RMSE/MAE/MAPE: hard routing, test_predictions.csv 기준)",
        fontsize=12, y=1.03,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] 저장 → {out_path}")

    return cap_cache, scalar_cache


# =============================================================================
# 용량 곡선 비교 (MIT/HUST 대표 셀, 시나리오별 서브플롯)
# =============================================================================

_MIT_CELL_RE  = re.compile(r"^b\d+c\d+$")     # 예: b1c0, b2c32
_HUST_CELL_RE = re.compile(r"^\d+-\d+$")      # 예: 1-7, 10-8


def _pick_rep_cells(bundles: list[RunBundle], pattern: "re.Pattern", n: int = 3) -> list[str]:
    """모든 run의 test_predictions.csv에 공통으로 존재하는 셀 중, 패턴에 맞는 것을
    최대 n개 고른다(알파벳순) — 비교 대상 run마다 test 분할이 살짝 달라도(축·파라미터
    차이로 인한 min_pts 탈락 등) 모든 run에 실제로 존재하는 셀이어야 공정한 비교가 된다."""
    common: set[str] | None = None
    for b in bundles:
        ids = {r["cell_id"] for r in b.pred_rows if pattern.match(r["cell_id"])}
        common = ids if common is None else (common & ids)
        if not common:
            return []
    return sorted(common)[:n] if common else []


def _plot_capacity_curve_comparison(
    bundles: list[RunBundle], cell_id: str, out_path: Path, cross_axis: bool = False,
) -> None:
    """지정한 셀 하나에 대해, (시나리오 × run) 조합마다 독립된 서브플롯(x=사이클,
    y=용량 Ah)을 그린다 — 행=시나리오, 열=run. 각 칸엔 실측 용량(회색) + 그 run
    하나만의 예측 곡선(해당 run 색상)만 들어간다.

    이전 버전은 한 시나리오 서브플롯에 모든 run의 예측을 겹쳐 그렸는데, 곡선끼리
    겹치면 어느 run이 어떻게 다른지 구분이 잘 안 됐다 — run당 칸을 분리해서
    "같은 셀·시나리오에서 이 run은 정확히 어떻게 예측했는가"를 서로 간섭 없이 보게 한다.

    cross_axis=True(비교 대상 run들의 시나리오 축이 서로 다름)면 시나리오 이름이
    run마다 대응되지 않으므로, 시나리오별 행 대신 셀당 1행(같은 사이클에 여러
    세그먼트가 있으면 평균)으로 축소해 "시나리오 평균" 곡선만 비교한다.
    """
    spec = bundles[0].spec

    if cross_axis:
        row_names: list[tuple[str, str | None]] = [("", None)]
    else:
        seg_names = spec.scenario_names
        # 방향별로 묶어 행 순서를 정함(충전 시나리오들 먼저, 방전 시나리오들 다음)
        chg_names = [seg_names[s] for s in spec.charge_scenario_ids]
        dis_names = [seg_names[s] for s in spec.discharge_scenario_ids]
        row_names = [("Charge", n) for n in chg_names] + [("Discharge", n) for n in dis_names]
    n_rows = len(row_names)
    n_cols = len(bundles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.6 * n_rows), squeeze=False)

    for r, (dir_label, sname) in enumerate(row_names):
        for c, b in enumerate(bundles):
            ax = axes[r][c]
            if sname is None:
                rows = [row for row in b.pred_rows if row["cell_id"] == cell_id]
            else:
                rows = [row for row in b.pred_rows
                        if row["cell_id"] == cell_id and row["seg_name"] == sname]
            if not rows:
                ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=8)
            else:
                if sname is None:
                    # 시나리오 평균: 같은 사이클에 세그먼트(시나리오)가 여러 개면 평균낸다
                    by_cycle: dict[int, list] = {}
                    for row in rows:
                        by_cycle.setdefault(row["cycle"], []).append(row)
                    cyc      = sorted(by_cycle)
                    cap_true = [float(np.mean([rr["cap_true_Ah"] for rr in by_cycle[cy]])) for cy in cyc]
                    cap_pred = [float(np.mean([rr["cap_pred_Ah"] for rr in by_cycle[cy]])) for cy in cyc]
                else:
                    rows.sort(key=lambda row: row["cycle"])
                    cyc      = [row["cycle"] for row in rows]
                    cap_true = [row["cap_true_Ah"] for row in rows]
                    cap_pred = [row["cap_pred_Ah"] for row in rows]
                ax.plot(cyc, cap_true, color="0.3", lw=1.8, ls="-", zorder=1, label="실측(true)")
                ax.plot(cyc, cap_pred, color=_COLORS[c % len(_COLORS)], lw=1.3,
                        alpha=0.9, zorder=2, label=_short_label(b.label))

            if r == 0:
                ax.set_title(_short_label(b.label, maxlen=20), fontsize=8.5)
            if c == 0:
                row_label = "시나리오 평균" if sname is None else f"{dir_label}\n{sname}"
                ax.set_ylabel(f"{row_label}\nCapacity [Ah]", fontsize=7.5)
            if r == n_rows - 1:
                ax.set_xlabel("Cycle", fontsize=8)
            ax.tick_params(labelsize=6.5)
            ax.grid(alpha=0.3)

    # 공유 범례 (run 라벨 + 실측) — 전체 서브플롯 핸들 취합
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels:
                    handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, 1.0 + 0.5 / (2.6 * n_rows)), fontsize=8.5)

    axis_desc = "축 간 비교 — 시나리오 평균" if cross_axis else f"axis={spec.axis}"
    fig.suptitle(f"SOH 예측 곡선 비교 — {cell_id}  ({axis_desc})", fontsize=12,
                 y=1.0 + 1.0 / (2.6 * n_rows))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] 저장 → {out_path}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    print(f"[viz] device={device}")

    if len(args.runs) < 2:
        raise ValueError("--runs 는 2개 이상 지정해야 비교가 의미 있습니다.")

    bundles = _load_bundles(args.runs, args.labels)
    cross_axis = _check_cross_axis(bundles)

    for b in bundles:
        _build_model_for_run(b, device, args.checkpoint_name)
        _benchmark_inference(b, device, args.infer_batch_size, args.infer_warmup, args.infer_reps)
        if args.with_jacobian and not cross_axis:
            _compute_jacobian_profiles(b, device, args.jacobian_max_samples)

    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = OUT_ROOT / (args.out_name or f"{ts}_{args.title or 'result_comparison'}")
    out_path = out_dir / "result_comparison.png"

    cap_cache, scalar_cache = _plot_all(bundles, args.with_jacobian, out_path, cross_axis=cross_axis)

    # ── 대표 셀 SOH 예측 곡선 비교 (시나리오×run 그리드) ──────────────────────────
    if args.rep_cells:
        # 사용자가 직접 지정한 셀만 정확히 그린다 (데이터셋 자동판별/개수제한 없음).
        for cell in args.rep_cells:
            _plot_capacity_curve_comparison(
                bundles, cell, out_dir / f"capacity_curve_compare_{cell}.png",
                cross_axis=cross_axis,
            )
    else:
        for ds_name, pattern in (("MIT", _MIT_CELL_RE), ("HUST", _HUST_CELL_RE)):
            cells = _pick_rep_cells(bundles, pattern, n=3)
            if not cells:
                print(f"[viz] {ds_name} 대표 셀 없음(모든 run에 공통인 {ds_name} 셀 미발견) — 곡선 비교 생략")
                continue
            for cell in cells:
                _plot_capacity_curve_comparison(
                    bundles, cell, out_dir / f"capacity_curve_compare_{ds_name}_{cell}.png",
                    cross_axis=cross_axis,
                )

    summary = {
        "cross_axis": cross_axis,
        "axis": bundles[0].spec.axis if not cross_axis else [b.spec.axis for b in bundles],
        "scenario_names": bundles[0].spec.scenario_names if not cross_axis else {
            b.label: b.spec.scenario_names for b in bundles
        },
        "runs": {
            b.label: {
                "run_dir": str(b.run_dir),
                "capacity": cap_cache[b.label],
                "scalar": scalar_cache[b.label],
            }
            for b in bundles
        },
    }
    if args.with_jacobian:
        summary["jacobian_available"] = True
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[viz] 요약 JSON → {out_dir / 'summary.json'}")
    print(f"\n[viz] 완료 — {out_dir}")


if __name__ == "__main__":
    main()
