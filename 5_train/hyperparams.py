# ==========================================
# hyperparams.py
# ==========================================

# 모델별 마이너 번호 매핑
MODEL_MAP = {
    "MLP": 1,
    "ITRANSFORMER": 2,
    "TABNET": 3,
    "XGBOOST": 4,
    "LIGHTGBM": 5,
    "RF": 6,
    "SVR": 7,
    "GPR": 8,
}

HYPERPARAMS = {
    # 1. Experiment & Tracking
    "major_version": 14,  # Parameter Set Index
    "patch_version": 1,  # Fixed for now
    "seed": 42,
    # 2. Data Options
    "dataset_types": ["mit"],  # List of datasets to use
    "processed_data_root": "D:/chanminLee/data_store/LFP_SOH_estimation",
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "input_dim": 78,
    # "input_dim": 48,
    # "input_dim": 33,
    "target_col": "capacity",
    "add_seq_dim": False,
    # 3. Model General
    "model_name": "MLP",
    "output_dim": 1,
    # 4. Physics-Informed (PI) Options
    "use_pi": True,
    "pi_target_idx": 0,  # [추가] PINN 편미분에 사용할 타겟 HI 피처의 인덱스 (예: 0번은 mean_v)
    "alpha": 100.0,
    "beta": 0.1,
    # 5. Deep Learning Training Parameters
    "batch_size": 512,
    "epochs": 300,
    "learning_rate": 5e-4,  # 0.0005
    "weight_decay": 1e-4,
    "patience": 8,
    "factor": 0.5,
    "min_lr": 1e-7,
    # 6. Model-Specific Parameters
    "mlp_params": {"hidden_dims": [128, 64], "dropout": 0.2},
    "itransformer_params": {"seq_len": 1, "d_model": 64, "n_heads": 4, "e_layers": 2},
    "tabnet_params": {"hidden_dim": 64},
    "xgboost_params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    "lightgbm_params": {
        "n_estimators": 200,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
    },
    "rf_params": {"n_estimators": 200, "max_depth": 15},
    "svr_params": {"C": 100, "gamma": "scale", "kernel": "rbf"},
    "gpr_params": {"length_scale": 1.0, "noise_level": 1.0},
}
