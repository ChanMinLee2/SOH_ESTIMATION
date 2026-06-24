import os
import sys
import time
from pathlib import Path
import pickle
import copy

# 1. 'src' 모듈을 찾을 수 있도록 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

try:
    from thop import profile
except ImportError:
    profile = None

# NumPy 버전 호환성 패치
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

from src.models import get_model, PhysicsInformedWrapper
from src.train import get_dataloaders, ConfigNamespace, seed_everything
from src.hyperparams import HYPERPARAMS, MODEL_MAP


# ==========================================
# 1. Inference Engine
# ==========================================
def evaluate_model(model, test_loader, config, is_dl=True):
    results = {"targets": [], "preds": [], "time": [], "cell_id": [], "scenario": []}
    total_inference_time = 0.0
    print(
        f"[Info] Starting Inference on Test Set ({len(test_loader.dataset)} samples)..."
    )

    if is_dl:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
                # x, m, s, y, cell_id, time(cycle) 반환 (src/train.py BatterySOHDataset 참조)
                x = batch[0].to(config.device)
                m = batch[1].to(config.device)
                s = batch[2].to(config.device)
                y = batch[3].to(config.device)
                cids = batch[4]
                t = batch[5].to(config.device)

                if getattr(config, "add_seq_dim", False) and len(x.shape) == 2:
                    x = x.unsqueeze(1)

                start_t = time.perf_counter()

                # PhysicsInformedWrapper 또는 일반 모델 호출 (t 인자 제거)
                if isinstance(model, PhysicsInformedWrapper):
                    preds = model(x, mode=m, return_pde=False).squeeze(-1)
                else:
                    preds = model(x).squeeze(-1)

                batch_time = time.perf_counter() - start_t
                total_inference_time += batch_time

                results["targets"].extend(y.cpu().numpy())
                results["preds"].extend(preds.cpu().numpy())
                
                # 데이터셋에서 가져온 실제 사이클 정보 사용 (Trend Consistency 및 시각화용)
                results["time"].extend(t.cpu().numpy().flatten())
                results["cell_id"].extend(cids)
                results["scenario"].extend(s.cpu().numpy().flatten())
    else:
        # ML 모델 추론 (BatterySOHDataset 속성 사용)
        X_test = test_loader.dataset.x.numpy()
        y_test = test_loader.dataset.y.numpy()
        s_test = test_loader.dataset.s.numpy()
        c_test = test_loader.dataset.cell_ids
        t_test = test_loader.dataset.t.numpy()

        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)

        start_t = time.perf_counter()
        preds = model.predict(X_test)
        total_inference_time = time.perf_counter() - start_t

        results["targets"].extend(y_test)
        results["preds"].extend(preds)
        results["time"].extend(t_test.flatten()) # 실제 타임라인
        results["cell_id"].extend(c_test)
        results["scenario"].extend(s_test)

    avg_inference_time_ms = (total_inference_time / len(test_loader.dataset)) * 1000
    return pd.DataFrame(results), avg_inference_time_ms


# ==========================================
# 2. 성능 지표 계산 및 시각화
# ==========================================
def plot_individual_cells(df_results, config):
    """
    특정 셀(최대 3개)을 선택하여 사이클 흐름에 따른 정답(Line)과 예측값(Scatter)을 시각화합니다.
    """
    save_dir = config.exp_dir / "evaluation" / "cell_plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    unique_cells = df_results["cell_id"].unique()
    # 평가 데이터에 있는 셀 중 최대 3개 선택
    target_cells = unique_cells[:3]

    sns.set_theme(style="whitegrid")
    
    for cell_id in target_cells:
        # 파일 경로로 사용할 수 없는 문자 제거 (Windows 예약 문자 및 괄호 등)
        clean_cell_id = str(cell_id).strip()
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')', "'", ' ']:
            clean_cell_id = clean_cell_id.replace(char, '_')
        
        cell_data = df_results[df_results["cell_id"] == cell_id].copy()

        # 'time' (실제 사이클 인덱스) 기준으로 정렬
        cell_data = cell_data.sort_values("time")

        plt.figure(figsize=(12, 6))

        # 정답: 검은색 선 그래프
        plt.plot(
            cell_data["time"], 
            cell_data["targets"], 
            color="black", 
            label="Actual (Ground Truth)", 
            linewidth=2,
            alpha=0.8
        )

        # 예측값: 파란색 산점도
        plt.scatter(
            cell_data["time"], 
            cell_data["preds"], 
            color="#1f77b4", 
            label="Predicted", 
            s=15, 
            alpha=0.6
        )

        plt.title(f"Degradation Trend: Cell {cell_id} ({config.model_name})")
        plt.xlabel("Cycle Number (from Dataset)")
        plt.ylabel("Capacity (SOH)")
        plt.legend()
        plt.tight_layout()
        
        save_path = save_dir / f"cell_{clean_cell_id}_trend.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"[Info] Individual cell plots saved to '{save_dir}'.")


# ==========================================
# 2-1. 시나리오별 열화 트렌드 시각화 (모드×SOC별 Capacity 추적)
# ==========================================
def plot_scenario_degradation_trends(model, config, is_dl=True, target_cell_ids=None):
    """
    특정 셀(최대 3개)에 대해 각 모드-시나리오(Charge/Discharge × High/Mid/Low)별로
    사이클 흐름에 따른 정답 Capacity(검은 실선)와 예측 Capacity(파란 점선)를 시각화합니다.

    각 (cell, cycle, scenario) 조합에서 length_p가 가장 긴 세그먼트만 사용하여
    하나의 깨끗한 선으로 표현합니다.

    Args:
        model: 학습된 모델 (DL 또는 ML)
        config: ConfigNamespace 객체
        is_dl: True이면 DL 모델, False이면 ML 모델
        target_cell_ids: 시각화할 셀 ID 리스트 (None이면 자동 선택)
    """
    save_dir = config.exp_dir / "evaluation" / "scenario_trends"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 전체 데이터 풀 로드 ---
    root_path = Path(config.processed_data_root)
    full_pool = []
    for d_type in config.dataset_types:
        data_path = (
            root_path
            / f"case_{HYPERPARAMS['major_version']}"
            / f"{d_type}_optimized_tensors.pkl"
        )
        if not data_path.exists():
            print(f"[Warning] Data not found: {data_path}")
            continue
        with open(data_path, "rb") as f:
            pool = pickle.load(f)
            full_pool.extend(pool)

    if not full_pool:
        print("[Warning] No data loaded for scenario trend plots. Skipping.")
        return

    # --- 2. 시나리오 정의 ---
    # mode_label: 1=Charge, 0=Discharge
    # soc_label: -2=High, -1=Mid, 0=Low
    scenario_defs = {
        "Charge-High":     {"mode_label": 1, "soc_label": -2},
        "Charge-Mid":      {"mode_label": 1, "soc_label": -1},
        "Charge-Low":      {"mode_label": 1, "soc_label": 0},
        "Discharge-High":  {"mode_label": 0, "soc_label": -2},
        "Discharge-Mid":   {"mode_label": 0, "soc_label": -1},
        "Discharge-Low":   {"mode_label": 0, "soc_label": 0},
    }

    # --- 3. 타겟 셀 선택 ---
    if target_cell_ids is None:
        all_cells_in_pool = list(set(item["cell"] for item in full_pool))
        all_cells_in_pool.sort()
        target_cell_ids = all_cells_in_pool[:3]

    print(f"[Info] Scenario degradation trend targets: {target_cell_ids}")

    # --- 4. 셀별 데이터 필터링 및 추론 ---
    for cell_id in target_cell_ids:
        # 해당 셀의 모든 아이템 추출
        cell_items = [item for item in full_pool if item["cell"] == cell_id]
        if not cell_items:
            print(f"[Warning] No data for cell {cell_id}. Skipping.")
            continue

        # 파일 이름용 셀 ID 정리
        clean_cell_id = str(cell_id).strip()
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')', "'", ' ']:
            clean_cell_id = clean_cell_id.replace(char, '_')

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f"Scenario Degradation Trends: Cell {cell_id} ({config.model_name})",
            fontsize=16,
            fontweight="bold",
        )

        for idx, (scen_name, scen_filter) in enumerate(scenario_defs.items()):
            ax = axes[idx // 3, idx % 3]
            target_mode = scen_filter["mode_label"]
            target_soc = scen_filter["soc_label"]

            # 해당 시나리오의 아이템 필터링
            scen_items = [
                item for item in cell_items
                if item.get("mode_label") == target_mode
                and item.get("soc_label") == target_soc
            ]

            if not scen_items:
                ax.set_title(f"{scen_name} (No Data)", fontsize=11)
                ax.text(0.5, 0.5, "No segments", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
                continue

            # 각 사이클에서 가장 긴 length_p의 세그먼트만 유지
            best_per_cycle = {}
            for item in scen_items:
                cyc = item["cyc"]
                length_p = item.get("length_p", 0)
                if cyc not in best_per_cycle or length_p > best_per_cycle[cyc].get("length_p", 0):
                    best_per_cycle[cyc] = item

            # 사이클 순으로 정렬
            sorted_cycs = sorted(best_per_cycle.keys(), key=lambda x: int(x))
            selected_items = [best_per_cycle[cyc] for cyc in sorted_cycs]

            # Ground truth capacity 추출
            cycles = np.array([int(item["cyc"]) for item in selected_items])
            actual_caps = np.array([item["capacity"] for item in selected_items])

            # 모델 추론
            x_batch = np.array([item["x"] for item in selected_items])
            x_batch = np.nan_to_num(x_batch, nan=0.0)

            if is_dl:
                model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(config.device)
                    if getattr(config, "add_seq_dim", False) and len(x_tensor.shape) == 2:
                        x_tensor = x_tensor.unsqueeze(1)

                    if isinstance(model, PhysicsInformedWrapper):
                        m_tensor = torch.full(
                            (len(selected_items), 1),
                            float(target_mode),
                            dtype=torch.float32,
                        ).to(config.device)
                        pred_caps = model(x_tensor, mode=m_tensor, return_pde=False).squeeze(-1).cpu().numpy()
                    else:
                        pred_caps = model(x_tensor).squeeze(-1).cpu().numpy()
            else:
                # ML 모델
                if len(x_batch.shape) == 3:
                    x_batch = x_batch.reshape(x_batch.shape[0], -1)
                pred_caps = model.predict(x_batch)

            pred_caps = np.array(pred_caps).flatten()

            # 플롯: 검은 실선 (정답), 파란 점선 (예측)
            ax.plot(
                cycles, actual_caps,
                color="black", linewidth=2.0, linestyle="-",
                label="Actual", alpha=0.9,
            )
            ax.plot(
                cycles, pred_caps,
                color="#1f77b4", linewidth=1.8, linestyle="--",
                label="Predicted", alpha=0.85,
            )

            # 시나리오별 에러 계산
            scen_mae = np.mean(np.abs(actual_caps - pred_caps))
            scen_rmse = np.sqrt(np.mean((actual_caps - pred_caps) ** 2))

            ax.set_title(
                f"{scen_name}\n(MAE={scen_mae:.4f}, RMSE={scen_rmse:.4f})",
                fontsize=11,
            )
            ax.set_xlabel("Cycle", fontsize=10)
            ax.set_ylabel("Capacity", fontsize=10)
            ax.legend(fontsize=9, loc="best")
            ax.tick_params(labelsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = save_dir / f"cell_{clean_cell_id}_scenario_trends.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {save_path}")

    print(f"[Info] Scenario degradation trend plots saved to '{save_dir}'.")


def calculate_and_plot_metrics(df_results, model, config, avg_inf_time_ms, is_dl=True):
    save_dir = config.exp_dir / "evaluation"
    save_dir.mkdir(parents=True, exist_ok=True)

    targets = df_results["targets"].values
    preds = df_results["preds"].values

    metrics = {}

    # [A] 기본 회귀 지표
    metrics["1. MAE (%)"] = mean_absolute_error(targets, preds)
    metrics["2. RMSE (%)"] = np.sqrt(mean_squared_error(targets, preds))
    metrics["3. MAPE (%)"] = np.mean(np.abs((targets - preds) / (targets + 1e-6))) * 100
    metrics["4. R2 Score"] = r2_score(targets, preds)
    metrics["5. Max Error (%)"] = np.max(np.abs(targets - preds))

    # [B] 불확실성 지표
    residuals = targets - preds
    std_res = np.std(residuals)
    ci_upper = preds + 1.96 * std_res
    ci_lower = preds - 1.96 * std_res

    metrics["6. 95% CI Range (%)"] = np.mean(ci_upper - ci_lower)
    within_ci = np.logical_and(targets >= ci_lower, targets <= ci_upper)
    metrics["7. PICP (%)"] = np.mean(within_ci) * 100

    # [C] 단조 감소성 (Trend Consistency)
    monotonic_scores = []
    for cell, group in df_results.groupby("cell_id"):
        group = group.sort_values("time")
        diffs = np.diff(group["preds"].values)
        mono_score = np.sum(diffs <= 0.001) / len(diffs) if len(diffs) > 0 else 1.0
        monotonic_scores.append(mono_score)
    metrics["8. Trend Consistency"] = np.mean(monotonic_scores)

    # [D] Inference Time & Complexity
    metrics["9. Inference Time (ms/sample)"] = avg_inf_time_ms

    if is_dl:
        metrics["10. Parameter Count"] = sum(p.numel() for p in model.parameters())
        if profile is not None:
            try:
                dummy_x = torch.randn(1, config.input_dim).to(config.device)
                if getattr(config, "add_seq_dim", False):
                    dummy_x = dummy_x.unsqueeze(1)

                if config.use_pi or isinstance(model, PhysicsInformedWrapper):
                    macs, _ = profile(
                        model,
                        inputs=(
                            dummy_x,
                            torch.randn(1, 1).to(config.device),  # mode
                            False,  # return_pde (Boolean)
                        ),
                        verbose=False,
                    )
                else:
                    macs, _ = profile(model, inputs=(dummy_x,), verbose=False)
                metrics["11. FLOPs"] = macs * 2
            except:
                metrics["11. FLOPs"] = "Profiler Error"
        else:
            metrics["11. FLOPs"] = "thop not installed"
    else:
        metrics["10. Parameter Count"] = "N/A (ML Model)"
        metrics["11. FLOPs"] = "N/A (ML Model)"

    # 콘솔 출력
    print("\n" + "=" * 50)
    print(f"[{config.model_name}] Performance Metrics Summary")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<35}: {v:.6f}")
        else:
            print(f"{k:<35}: {v}")
    print("=" * 50)

    # 결과 리포트 저장
    with open(save_dir / f"{config.model_name}_metrics_report.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # 시각화 1: 예측 성능
    sns.set_theme(style="whitegrid")
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=targets, y=preds, alpha=0.3, ax=axes1[0], color="#1f77b4")
    axes1[0].plot(
        [targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2
    )
    axes1[0].set_title(f"Actual vs Predicted (RMSE: {metrics['2. RMSE (%)']:.4f})")
    axes1[0].set_xlabel("Actual SOH")
    axes1[0].set_ylabel("Predicted SOH")

    sns.histplot(residuals, bins=50, kde=True, ax=axes1[1], color="#2ca02c")
    axes1[1].axvline(0, color="red", linestyle="--")
    axes1[1].set_title(f"Error Distribution")
    axes1[1].set_xlabel("Error (Actual - Predicted)")
    plt.tight_layout()
    fig1.savefig(save_dir / f"{config.model_name}_1_Regression.png", dpi=300)
    plt.close(fig1)

    # 개별 셀 시각화 추가
    plot_individual_cells(df_results, config)

    print(f"[Info] Evaluation completed. Results saved to '{save_dir}'.")


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    MODELS_TO_EVALUATE = [
        "MLP",
        "TABNET",
        "ITRANSFORMER",
        "XGBOOST",
        "LIGHTGBM",
        "RF",
        "SVR",
        "GPR",
    ]
    # MODELS_TO_EVALUATE = ["MLP", "TABNET", "ITRANSFORMER", "XGBOOST", "LIGHTGBM", "RF"]

    seed_everything(HYPERPARAMS["seed"])
    base_config = ConfigNamespace(HYPERPARAMS)
    base_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {base_config.device}")
    print(f"Loading Test DataLoader...")
    _, _, test_loader = get_dataloaders(base_config)

    for model_name in MODELS_TO_EVALUATE:
        model_name_upper = model_name.upper()
        print(f"\n" + "=" * 50)
        print(f"Evaluating Model: {model_name_upper}")
        print("=" * 50)

        current_params = copy.deepcopy(HYPERPARAMS)
        current_params["model_name"] = model_name_upper
        is_dl = model_name_upper in ["MLP", "TABNET", "ITRANSFORMER"]
        current_params["add_seq_dim"] = model_name_upper in ["ITRANSFORMER"]

        cfg = ConfigNamespace(current_params)
        cfg.device = base_config.device

        # 모델의 버전에 맞는 디렉토리 설정
        exp_dir = cfg.setup_experiment_dir()

        # DL 모델 로드
        if is_dl:
            specific_params = getattr(cfg, f"{model_name.lower()}_params", {})
            model = get_model(
                model_name_upper,
                use_pi=cfg.use_pi,
                feature_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
                **specific_params,
            )

            if not cfg.checkpoint_path.exists():
                print(
                    f"[Warning] Checkpoint not found: {cfg.checkpoint_path}. Skipping."
                )
                continue

            print(f"[Info] Loading weights from: {cfg.checkpoint_path}")
            model.load_state_dict(
                torch.load(cfg.checkpoint_path, map_location=cfg.device)
            )
            model = model.to(cfg.device)

            df_results, avg_inf_time = evaluate_model(
                model, test_loader, cfg, is_dl=True
            )
            calculate_and_plot_metrics(df_results, model, cfg, avg_inf_time, is_dl=True)

            # 시나리오별 열화 트렌드 시각화
            plot_scenario_degradation_trends(model, cfg, is_dl=True)

        # ML 모델 로드
        else:
            model_path = cfg.checkpoint_path.with_suffix(".pkl")
            if not model_path.exists():
                print(f"[Warning] ML Model not found: {model_path}. Skipping.")
                continue

            print(f"[Info] Loading model from: {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            df_results, avg_inf_time = evaluate_model(
                model, test_loader, cfg, is_dl=False
            )
            calculate_and_plot_metrics(
                df_results, model, cfg, avg_inf_time, is_dl=False
            )

            # 시나리오별 열화 트렌드 시각화
            plot_scenario_degradation_trends(model, cfg, is_dl=False)
