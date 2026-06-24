import os
import sys
from pathlib import Path

# 'src' 모듈을 찾을 수 있도록 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
import time

# NumPy 버전 호환성 패치 (pickle load 오류 방지)
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

import matplotlib.pyplot as plt
from src.models import get_model, PhysicsInformedWrapper
from src.hyperparams import HYPERPARAMS, MODEL_MAP


# 딕셔너리를 객체(Object)처럼 점(.)으로 접근하기 위한 래퍼 클래스
class ConfigNamespace:
    def __init__(self, d):
        self.__dict__.update(d)
        self.project_root = Path(__file__).resolve().parent.parent

    def get_version_str(self):
        minor = MODEL_MAP.get(self.model_name.upper(), 0)
        return f"{self.major_version}.{minor}.{self.patch_version}"

    def setup_experiment_dir(self):
        v_str = self.get_version_str()
        self.exp_dir = self.project_root / "experiments" / v_str
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config.json
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            serializable_config = {}
            for k, v in self.__dict__.items():
                if k == "project_root":
                    continue
                if isinstance(v, (Path, torch.device)):
                    serializable_config[k] = str(v)
                else:
                    serializable_config[k] = v

            json.dump(serializable_config, f, indent=4, ensure_ascii=False)

        # Update internal paths to point to exp_dir
        self.checkpoint_path = self.exp_dir / "best_model.pth"
        self.log_path = self.exp_dir / "train_log.txt"
        return self.exp_dir


# ==========================================
# 1. Utilities (Seeding)
# ==========================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 2. Dataset & DataLoader (Data Loading)
# ==========================================
class BatterySOHDataset(Dataset):
    def __init__(self, data_list, target_col="capacity", add_seq_dim=False):
        # x: (N, 45) [45 HI]
        x_raw = np.array([item["x"] for item in data_list])
        y_raw = np.array([item[target_col] for item in data_list])

        self.cell_ids = [item.get("cell", "unknown") for item in data_list]

        # mode: (N, 1) - 충전(1)/방전(0) 라벨
        m_raw = np.array([item.get("mode_label", 1) for item in data_list])

        # cycle: (N, 1) - 사이클 인덱스 (추후 정렬용)
        t_raw = np.array([item.get("cyc", 0) for item in data_list])

        # scenario: (N, 1) - 시나리오를 정수 인덱스로 변환
        scenario_map = {
            "charge-high": 0,
            "charge-mid": 1,
            "charge-low": 2,
            "discharge-high": 3,
            "discharge-mid": 4,
            "discharge-low": 5,
        }

        s_raw = []
        for item in data_list:
            scen_str = item.get("scenario", "")
            if not scen_str:
                mode_str = "charge" if item.get("mode_label", 1) == 1 else "discharge"
                soc_val = item.get("soc_label", 0)
                soc_str = "high" if soc_val == -2 else "mid" if soc_val == -1 else "low"
                scen_str = f"{mode_str}-{soc_str}"
            s_raw.append(scenario_map.get(scen_str, 0))

        s_raw = np.array(s_raw)

        # NaN 체크 및 처리
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        y_raw = np.nan_to_num(y_raw, nan=0.0)

        self.x = torch.tensor(x_raw, dtype=torch.float32)
        self.m = torch.tensor(m_raw, dtype=torch.float32).unsqueeze(-1)
        self.s = torch.tensor(s_raw, dtype=torch.long)
        self.t = torch.tensor(t_raw, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y_raw, dtype=torch.float32)

        if add_seq_dim:
            self.x = self.x.unsqueeze(1)  # (N, 45) -> (N, 1, 45)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # x, mode, scenario, y, cell_id, time(cycle) 반환
        return (
            self.x[idx],
            self.m[idx],
            self.s[idx],
            self.y[idx],
            self.cell_ids[idx],
            self.t[idx],
        )


def get_dataloaders(config):
    full_pool = []
    root_path = Path(config.processed_data_root)
    print(f"[Info] Searching for data in: {root_path}")
    for d_type in config.dataset_types:
        data_path = (
            root_path
            / f"case_{HYPERPARAMS['major_version']}"
            / f"{d_type}_optimized_tensors.pkl"
        )
        if not data_path.exists():
            print(f"[Warning] Data not found for {d_type} at {data_path}")
            continue

        print(f"[Info] Loading {d_type} data... (this may take a minute)")
        try:
            with open(data_path, "rb") as f:
                pool = pickle.load(f)
                for item in pool:
                    item["source"] = d_type
                full_pool.extend(pool)
                print(f"[Success] Loaded {len(pool)} samples from {d_type}")
        except Exception as e:
            print(f"[Error] Failed to load {d_type}: {e}")

    if not full_pool:
        print(
            "[Critical] No data was loaded. Please check if the .pkl files exist in D:/ drive."
        )
        raise FileNotFoundError("No data files found to load.")

    # Cell-wise Split to prevent data leakage
    print("[Info] Splitting data into Train/Val/Test (6:2:2)...")
    unique_cells = list(set([(item["source"], item["cell"]) for item in full_pool]))
    np.random.shuffle(unique_cells)

    n_cells = len(unique_cells)
    test_idx = int(n_cells * (1 - config.test_ratio))
    val_idx = int(test_idx * (1 - config.val_ratio / (1 - config.test_ratio)))

    train_cells = set(unique_cells[:val_idx])
    val_cells = set(unique_cells[val_idx:test_idx])
    test_cells = set(unique_cells[test_idx:])

    train_data = [
        item for item in full_pool if (item["source"], item["cell"]) in train_cells
    ]
    val_data = [
        item for item in full_pool if (item["source"], item["cell"]) in val_cells
    ]
    test_data = [
        item for item in full_pool if (item["source"], item["cell"]) in test_cells
    ]

    print(f"Dataset Split (6:2:2) Summary:")
    print(f"  Train: {len(train_cells)} cells, {len(train_data)} samples")
    print(f"  Val  : {len(val_cells)} cells, {len(val_data)} samples")
    print(f"  Test : {len(test_cells)} cells, {len(test_data)} samples")

    train_loader = DataLoader(
        BatterySOHDataset(train_data, config.target_col, config.add_seq_dim),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        BatterySOHDataset(val_data, config.target_col, config.add_seq_dim),
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        BatterySOHDataset(test_data, config.target_col, config.add_seq_dim),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# ==========================================
# 3. Utilities (Early Stopping & Engine)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, path="checkpoint.pth"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_epoch(model, dataloader, criterion, optimizer, config):
    model.train()
    running_loss = 0.0
    running_data_loss = 0.0  # [추가] Data Loss 누적
    running_pde_loss = 0.0  # [추가] PDE Loss 누적
    processed_samples = 0

    for i, (x, m, s, y, cid, t) in enumerate(
        tqdm(dataloader, desc="  Training Batch", leave=False)
    ):
        x, m, s, y = (
            x.to(config.device),
            m.to(config.device),
            s.to(config.device),
            y.to(config.device),
        )

        # [수정] LSTM, iTransformer 등 시퀀스 차원이 필요한 모델을 위해 동적으로 차원 추가
        if getattr(config, "add_seq_dim", False) and len(x.shape) == 2:
            x = x.unsqueeze(1)

        # 입력값 유효성 체크
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            print(f"\n[Error] Non-finite values detected in input batch {i}!")
            continue

        optimizer.zero_grad()

        # PI 옵션에 따른 분기 처리
        if config.use_pi:
            # CuDNN RNN의 Double Backward 미지원 에러 우회
            with torch.backends.cudnn.flags(enabled=False):
                preds, pde_residual, u_hi = model(x, mode=m, return_pde=True)
            preds = preds.squeeze(-1)

            loss_data = criterion(preds, y)
            loss_pde = torch.mean(pde_residual**2)

            # Monotonicity Loss: 선택한 타겟 피처(HI)에 대한 용량 변화율(u_hi)의 방향을 제약
            # 물리적 지식에 따라, 해당 피처가 증가할 때 SOH가 감소해야 한다면 양수 기울기에 페널티 부여
            loss_mono = torch.mean(torch.nn.functional.relu(u_hi) ** 2)

            # Adaptive Weighting: 학습 가능한 파라미터(log_var_*)를 사용하여 각 손실 함수의 스케일을 자동 조절
            # 참고: IEEE TIV 논문 식 (22)
            loss_data_w = (
                torch.exp(-model.log_var_data) * loss_data + model.log_var_data
            )
            loss_pde_w = torch.exp(-model.log_var_pde) * loss_pde + model.log_var_pde
            loss_mono_w = (
                torch.exp(-model.log_var_mono) * loss_mono + model.log_var_mono
            )

            # 통합 Loss (하이퍼파라미터 alpha, beta 대신 학습된 가중치 사용)
            loss = loss_data_w + loss_pde_w + loss_mono_w

            if not torch.isfinite(loss):
                print(f"\n[Error] Loss is {loss.item()} at batch {i}!")
                print(
                    f"  loss_data: {loss_data.item()}, loss_pde: {loss_pde.item()}, loss_mono: {loss_mono.item()}"
                )
                optimizer.zero_grad()
                continue

            batch_data_loss = loss_data.item()
            # 로깅의 편의성을 위해 물리 제약 기반 Loss(PDE + Mono)를 하나로 묶음
            batch_pde_loss = loss_pde.item() + loss_mono.item()
        else:
            if isinstance(model, PhysicsInformedWrapper):
                preds = model(x, mode=m).squeeze(-1)
            else:
                preds = model(x).squeeze(-1)

            loss = criterion(preds, y)

            if not torch.isfinite(loss):
                print(f"\n[Error] Loss is {loss.item()} at batch {i}!")
                optimizer.zero_grad()
                continue

            batch_data_loss = loss.item()
            batch_pde_loss = 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 가중치 유효성 체크
        is_weights_ok = True
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"\n[Error] Parameter {name} became non-finite at batch {i}!")
                is_weights_ok = False
                break

        if not is_weights_ok:
            break

        # [수정] 각각의 로스 성분을 분리하여 누적합산
        running_loss += loss.item() * x.size(0)
        running_data_loss += batch_data_loss * x.size(0)
        running_pde_loss += batch_pde_loss * x.size(0)
        processed_samples += x.size(0)

    if processed_samples == 0:
        return float("nan"), float("nan"), float("nan")

    # 반환값 변경: Total Loss, Data Loss, PDE Loss
    return (
        running_loss / processed_samples,
        running_data_loss / processed_samples,
        running_pde_loss / processed_samples,
    )


def validate_epoch(model, dataloader, criterion, config):
    model.eval()
    running_loss = 0.0
    processed_samples = 0
    with torch.no_grad():
        for i, (x, m, s, y, cid, t) in enumerate(dataloader):
            x, m, s, y = (
                x.to(config.device),
                m.to(config.device),
                s.to(config.device),
                y.to(config.device),
            )

            # [수정] LSTM, iTransformer 등 시퀀스 차원이 필요한 모델을 위해 동적으로 차원 추가
            if getattr(config, "add_seq_dim", False) and len(x.shape) == 2:
                x = x.unsqueeze(1)

            if config.use_pi:
                preds = model(x, mode=m, return_pde=False).squeeze(-1)
            else:
                if isinstance(model, PhysicsInformedWrapper):
                    preds = model(x, mode=m).squeeze(-1)
                else:
                    preds = model(x).squeeze(-1)

            loss = criterion(preds, y)

            if torch.isfinite(loss):
                running_loss += loss.item() * x.size(0)
                processed_samples += x.size(0)

    if processed_samples == 0:
        return float("nan"), float("nan"), float("nan")

    val_loss = running_loss / processed_samples
    # Evaluation에서는 PI Loss가 계산되지 않으므로 (Total, Data, PDE) 포맷에 맞춰 반환
    return val_loss, val_loss, 0.0


def plot_loss_curve(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        history["train_loss"], label="Train Total Loss", color="black", linewidth=2
    )
    plt.plot(history["val_loss"], label="Val Loss", color="red", linewidth=2)

    # [추가] PI Loss가 존재하고 0보다 큰 경우에 한해 추가 로스들도 시각화
    if "train_pde_loss" in history and sum(history["train_pde_loss"]) > 0:
        plt.plot(
            history["train_data_loss"],
            label="Train Data Loss",
            linestyle="--",
            alpha=0.7,
        )
        plt.plot(
            history["train_pde_loss"], label="Train PI Loss", linestyle="-.", alpha=0.7
        )

    plt.yscale("log")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


# ==========================================
# 4. Training Engine
# ==========================================
def fit(model, train_loader, val_loader, config):
    model_name = config.model_name
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.factor,
        patience=15,
        min_lr=config.min_lr,
        verbose=False,
    )

    checkpoint_path = config.checkpoint_path
    log_path = config.log_path

    early_stopping = EarlyStopping(
        patience=config.patience, verbose=True, path=checkpoint_path
    )

    # [수정] 히스토리에 Data Loss, PI Loss 기록 공간 추가
    history = {
        "train_loss": [],
        "train_data_loss": [],
        "train_pde_loss": [],
        "val_loss": [],
        "epoch_time": [],
    }

    print(
        f"\n>>> Starting Experiment: {config.get_version_str()} (Model: {model_name} | PI: {config.use_pi})"
    )

    # [수정] 훈련 시작 전 로그 파일 헤더에 Data Loss, PI Loss 컬럼 추가
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Epoch\tTrain_Total\tTrain_Data\tTrain_PI\tVal_Loss\tEpoch_Time(s)\n")

    for epoch in tqdm(range(1, config.epochs + 1), desc="Epochs"):
        start_time = time.time()

        # [수정] train_epoch 및 validate_epoch에서 3개의 로스 반환값을 받음
        train_total, train_data, train_pde = train_epoch(
            model, train_loader, criterion, optimizer, config
        )
        val_total, _, _ = validate_epoch(model, val_loader, criterion, config)

        epoch_time = time.time() - start_time

        # [수정] 로그 파일에 세부 내역 저장
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch:04d}\t{train_total:.6f}\t{train_data:.6f}\t{train_pde:.6f}\t{val_total:.6f}\t{epoch_time:.4f}\n"
            )

        # [수정] 콘솔 출력 시 PI 여부에 따라 포맷 변경
        if config.use_pi:
            print(
                f"  Epoch [{epoch:03d}/{config.epochs}] | Train: {train_total:.6f} (Data: {train_data:.6f}, PI: {train_pde:.6f}) | "
                f"Val: {val_total:.6f} | Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            print(
                f"  Epoch [{epoch:03d}/{config.epochs}] | Train: {train_total:.6f} | "
                f"Val: {val_total:.6f} | Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        history["train_loss"].append(train_total)
        history["train_data_loss"].append(train_data)
        history["train_pde_loss"].append(train_pde)
        history["val_loss"].append(val_total)
        history["epoch_time"].append(epoch_time)

        scheduler.step(val_total)
        early_stopping(val_total, model)

        if early_stopping.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 가중치 로드 및 시각화
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))

    plot_loss_curve(history, config.exp_dir / "loss_curve.png")

    return model, history, early_stopping.val_loss_min


# ==========================================
# 5. Main Execution (Automated Pipeline)
# ==========================================
if __name__ == "__main__":
    MODELS_TO_RUN = [
        "MLP",
        "ITRANSFORMER",
        "TABNET",
        "XGBOOST",
        "LIGHTGBM",
        "RF",
        "SVR",
        "GPR",
    ]

    seed_everything(HYPERPARAMS["seed"])

    base_config = ConfigNamespace(HYPERPARAMS)
    base_config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {base_config.device}")
    print(f"Experiment Version: {base_config.get_version_str()}")
    print(
        f"Models to run: {MODELS_TO_RUN}"
        f"\n\nStarting training pipeline...\n"
        f"{'='*60}"
    )

    train_loader, val_loader, test_loader = get_dataloaders(base_config)

    results_summary = []

    for model_name in MODELS_TO_RUN:

        model_name_upper = model_name.upper()

        # if model_name_upper in ["MLP", "ITRANSFORMER", "TABNET"]:
        #     continue
        current_params = copy.deepcopy(HYPERPARAMS)
        current_params["model_name"] = model_name_upper

        is_dl = model_name_upper in ["MLP", "TABNET", "ITRANSFORMER"]
        is_ml = model_name_upper in ["RF", "SVR", "GPR", "XGBOOST", "LIGHTGBM"]

        current_params["add_seq_dim"] = model_name_upper in ["ITRANSFORMER"]

        cfg = ConfigNamespace(current_params)
        cfg.device = base_config.device

        exp_dir = cfg.setup_experiment_dir()

        specific_params = getattr(cfg, f"{model_name.lower()}_params", {})
        model = get_model(
            model_name_upper,
            use_pi=cfg.use_pi if is_dl else False,
            feature_dim=cfg.input_dim,
            pi_target_idx=getattr(cfg, "pi_target_idx", 0),
            output_dim=cfg.output_dim,
            **specific_params,
        )

        print(
            f"\n>>> Starting Experiment: {cfg.get_version_str()} (Model: {model_name_upper})"
        )

        if is_dl:
            model = model.to(cfg.device)
            best_model, hist, best_val = fit(model, train_loader, val_loader, cfg)
            test_loss, _, _ = validate_epoch(best_model, test_loader, nn.MSELoss(), cfg)
        else:
            print(f"  [ML] Preparing NumPy data for {model_name_upper}...")
            X_train = train_loader.dataset.x.numpy()
            y_train = train_loader.dataset.y.numpy()
            X_test = test_loader.dataset.x.numpy()
            y_test = test_loader.dataset.y.numpy()

            if len(X_train.shape) == 3:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

            # [추가] GPR 메모리 부족(OOM) 방지를 위한 훈련 데이터 서브샘플링
            if model_name_upper == "GPR":
                max_gpr_samples = 5000
                if len(X_train) > max_gpr_samples:
                    print(
                        f"  [Warning] GPR memory limit: Subsampling training data from {len(X_train)} to {max_gpr_samples}."
                    )
                    idx = np.random.choice(len(X_train), max_gpr_samples, replace=False)
                    X_train = X_train[idx]
                    y_train = y_train[idx]

            print(f"  [ML] Fitting {model_name_upper}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            test_loss = np.mean((y_test - y_pred) ** 2)
            best_val = 0.0

            with open(cfg.checkpoint_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(model, f)
            print(f"  [ML] Model saved to {cfg.checkpoint_path.with_suffix('.pkl')}")

        rmse = np.sqrt(test_loss)
        print(f"  [Result] Test MSE: {test_loss:.6f} | RMSE: {rmse:.6f}")

        results_summary.append(
            {
                "version": cfg.get_version_str(),
                "model": model_name_upper,
                "best_val_mse": best_val,
                "test_mse": test_loss,
                "test_rmse": rmse,
            }
        )

    print("\n" + "=" * 50)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 50)
    summary_df = pd.DataFrame(results_summary)
    print(summary_df)

    summary_path = (
        base_config.project_root
        / "experiments"
        / f"summary_major_{base_config.major_version}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Summary] Saved to: {summary_path}")
