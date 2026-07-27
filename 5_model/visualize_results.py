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
  Row 5   : 선정 HI 자카드(Jaccard) 유사도 히트맵 — run별로 1개씩, probe(방향게이트) +
            시나리오 간 (scenario × scenario) 행렬. 한 run 안에서 시나리오마다 얼마나
            다른 HI 서브셋을 선정했는지 보기 위함 (대각선=1.0, 낮을수록 시나리오 간
            HI 선택이 서로 다름). routing_table.csv의 이진 마스크만 사용(모델 재로딩 불요).
  Row 6   : (--with-jacobian 지정 시) 선정 HI 자코비안(gradient) 코사인 유사도
            히트맵 — Overall + 시나리오별. 실제 테스트 샘플에서
            d(cap_pred)/d(x_hi) 를 시나리오별로 평균한 벡터끼리 비교 —
            "같은 HI를 골랐어도 실제로 비슷하게 쓰는가"까지 검증.

전제: 비교 대상 run들은 동일한 시나리오 축(scenario_spec.json의 axis / n_scenarios /
      scenario_names)을 공유해야 한다. 다르면 명확한 에러로 중단한다.

결과 저장 위치:
  _5_data_model_scr/comparison/<MMDD_HHMM>_result_comparison/
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

from utils.hi_schema import N_HI
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
                   help="결과 폴더명 오버라이드 (기본 <MMDD_HHMM>_result_comparison)")
    return p.parse_args()


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


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
        self.has_clf = "hard" in self.metrics.get("test", {})

        self.pred_rows = self._load_predictions()
        self.routing_sets = self._load_routing_table()

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
                    "seg_name": r["seg_name"],
                    "soh_true": float(r["soh_true"]),
                    "soh_pred": float(r["soh_pred"]),
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


def _validate_same_axis(bundles: list[RunBundle]) -> None:
    ref = bundles[0].spec
    for b in bundles[1:]:
        if b.spec.axis != ref.axis or b.spec.scenario_names != ref.scenario_names:
            raise ValueError(
                "비교 대상 run들의 시나리오 축이 다릅니다 — 같은 축(axis)으로 학습된 "
                "run들만 비교할 수 있습니다.\n"
                f"  [{ref.axis}] {bundles[0].label}: {ref.scenario_names}\n"
                f"  [{b.spec.axis}] {b.label}: {b.spec.scenario_names}"
            )


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
        return f"q_frac_wide/n1-{n1}%_n2-{n2}%_N-{ns}"
    return spec.axis


def _build_test_dataset_for_run(b: RunBundle):
    from datasets.segment_dataset import build_datasets, SegmentNormalizer

    cfg = json.loads(json.dumps(b.cfg))  # deep copy (json roundtrip은 이 cfg가 순수 dict/list/str/number라 안전)
    data_cfg = cfg.setdefault("data", {})
    axis_dir = _axis_dir_from_spec(b.spec)
    if not data_cfg.get("seg_data_dir"):
        data_cfg["seg_data_dir"] = f"_4_data_hi/{axis_dir}/seg"
    if not data_cfg.get("data_dir"):
        data_cfg["data_dir"] = f"_4_data_hi/{axis_dir}/cycle"

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
    """단일 run 내부에서 라벨(probe/시나리오)끼리 선정 HI 자카드 유사도.
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
                           labels: list[str], title_prefix: str):
    """run별로 하나씩(열=run) labels×labels(probe+시나리오) 자카드 행렬을 그린다.

    _heatmap_row(kind="jaccard")는 "같은 시나리오를 run끼리 비교"(run×run)했지만,
    이 함수는 "한 run 안에서 시나리오끼리 비교"(scenario×scenario) — 시나리오별로
    실제로 다른 HI를 선정했는지 보기 위한 본래 의도에 맞춘 버전.
    """
    n_runs = len(bundles)
    inner = gridspec.GridSpecFromSubplotSpec(1, n_runs, subplot_spec=gs_row_spec, wspace=0.7)
    n_lab = len(labels)
    im = None
    for c, b in enumerate(bundles):
        ax = fig.add_subplot(inner[0, c])
        m = _scenario_sim_matrix(b, labels)
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


def _plot_all(bundles: list[RunBundle], with_jacobian: bool, out_path: Path):
    scen_names = bundles[0].spec.scenario_names
    n_scen = len(scen_names)
    n_cols = n_scen + 1
    col_labels = ["Overall"] + list(scen_names)
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
    n_scalar_rows = -(-len(scalar_defs_all) // n_cols)  # ceil div — 열이 부족한 축이면 자동으로 여러 줄에 배치

    # 행 구성: RMSE, MAE, MAPE(3) + 스칼라(n_scalar_rows) + Jaccard(1) [+ Jacobian(1)]
    row_plan = ["rmse", "mae", "mape"] + ["scalar"] * n_scalar_rows + ["jaccard"]
    if with_jacobian:
        row_plan.append("jacobian")
    total_rows = len(row_plan)

    fig = plt.figure(figsize=(2.7 * max(n_cols, 7), 2.5 * total_rows))
    gs = gridspec.GridSpec(total_rows, n_cols, figure=fig, hspace=0.9, wspace=0.5)

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
    im1 = im2 = None
    for kind in row_plan:
        if kind in metric_row_defs:
            prefix, ylabel, fmt = metric_row_defs[kind]
            _bar_row(fig, [gs[row_i, c] for c in range(n_cols)], col_labels, bundles,
                     _get(kind), prefix, ylabel, fmt=fmt)
        elif kind == "scalar":
            chunk = scalar_defs_all[scalar_row_i * n_cols:(scalar_row_i + 1) * n_cols]
            _scalar_bar_row(fig, [gs[row_i, c] for c in range(len(chunk))], chunk, bundles, scalar_cache)
            scalar_row_i += 1
        elif kind == "jaccard":
            im1 = _scenario_heatmap_row(fig, gs[row_i, :], bundles, jaccard_labels, "")
        elif kind == "jacobian":
            im2 = _heatmap_row(fig, [gs[row_i, c] for c in range(n_cols)], col_labels, bundles,
                                "jacobian", "Jacobian cos — ")
        row_i += 1

    if im1 is not None:
        jaccard_row = row_plan.index("jaccard")
        top = 1 - jaccard_row / total_rows
        fig.colorbar(im1, cax=fig.add_axes([0.92, top - 1 / total_rows + 0.02, 0.012, 1 / total_rows - 0.04]))
    if im2 is not None:
        jacobian_row = row_plan.index("jacobian")
        top = 1 - jacobian_row / total_rows
        fig.colorbar(im2, cax=fig.add_axes([0.92, top - 1 / total_rows + 0.02, 0.012, 1 / total_rows - 0.04]))

    # ── 공유 범례 (run 라벨 ↔ 색상) ──────────────────────────────────────
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_COLORS[i % len(_COLORS)])
                       for i in range(len(bundles))]
    legend_texts = [f"{b.label}" for b in bundles]
    fig.legend(legend_handles, legend_texts, loc="upper center",
               ncol=min(len(bundles), 5), bbox_to_anchor=(0.5, 1.0), fontsize=9)

    fig.suptitle(
        f"SCR 모델 비교 — axis={bundles[0].spec.axis}, n_scenarios={n_scen}  "
        f"(RMSE/MAE/MAPE: hard routing, test_predictions.csv 기준)",
        fontsize=12, y=1.03,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] 저장 → {out_path}")

    return cap_cache, scalar_cache


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
    _validate_same_axis(bundles)

    for b in bundles:
        _build_model_for_run(b, device, args.checkpoint_name)
        _benchmark_inference(b, device, args.infer_batch_size, args.infer_warmup, args.infer_reps)
        if args.with_jacobian:
            _compute_jacobian_profiles(b, device, args.jacobian_max_samples)

    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = OUT_ROOT / (args.out_name or f"{ts}_result_comparison")
    out_path = out_dir / "result_comparison.png"

    cap_cache, scalar_cache = _plot_all(bundles, args.with_jacobian, out_path)

    summary = {
        "axis": bundles[0].spec.axis,
        "scenario_names": bundles[0].spec.scenario_names,
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
