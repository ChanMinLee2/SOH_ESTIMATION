import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# 1. 딥러닝 모델 (PyTorch 기반)
# ==========================================


class TabularAttentionNet(nn.Module):
    """
    정형 데이터(Tabular Data)의 특성을 고려하여,
    각 Feature의 중요도를 스스로 학습하는 간소화된 Attention 기반 MLP (TabNet 영감)
    """

    def __init__(self, input_dim=40, output_dim=1, hidden_dim=64):
        super(TabularAttentionNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Softmax(dim=-1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, output_dim)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        attn = self.attention(x)
        out = feat * attn  # Feature별 가중치 곱 (Sparse Attention 효과)
        return self.fc(out)


class InvertedTransformer(nn.Module):
    def __init__(
        self,
        num_variates=40,
        seq_len=1,
        d_model=64,
        n_heads=4,
        e_layers=2,
        output_dim=1,
    ):
        super(InvertedTransformer, self).__init__()
        self.num_variates = num_variates
        self.project = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_variates * d_model, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x_inv = x.transpose(1, 2)
        x_emb = self.project(x_inv)
        out = self.transformer(x_emb)
        return self.fc(out)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=40, output_dim=1, hidden_dims=[128, 64], dropout=0.2):
        super(SimpleMLP, self).__init__()
        layers = [nn.Flatten()]

        in_features = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h_dim

        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. Physics-Informed (PI) Wrapper
# ==========================================


class PhysicsInformedWrapper(nn.Module):
    """
    기존 딥러닝 모델(Solution Network)을 감싸서
    물리 정보 기반 신경망(Dynamics Network) 연산을 추가하는 래퍼 클래스입니다.
    """

    def __init__(self, base_model, feature_dim=40, pi_target_idx=0):
        super(PhysicsInformedWrapper, self).__init__()
        self.solution_net = base_model  # F(x)
        self.feature_dim = feature_dim
        self.pi_target_idx = pi_target_idx

        # Dynamics Network G(x, u, u_hi, u_x)
        # 입력 차원 계산: x(feature_dim) + u(1) + u_hi(1) + u_x(feature_dim)
        g_input_dim = feature_dim + 1 + 1 + feature_dim
        self.dynamics_net = nn.Sequential(
            nn.Linear(g_input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),  # 예측된 열화율(decay rate) 반환
        )

        # [추가] Adaptive Weighting을 위한 학습 가능한 파라미터 (Log Variance)
        # IEEE 논문의 수식 (22)에 기반하여 각 손실항의 가중치를 자동 조절
        self.log_var_data = nn.Parameter(torch.zeros(1))
        self.log_var_pde = nn.Parameter(torch.zeros(1))
        self.log_var_mono = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode=None, return_pde=False):
        """
        x: (B, D) or (B, L, D)
        mode: (B, 1) - 1: Charge, 0: Discharge
        """
        # [추가] 피처 마스킹 로직
        # 45 HI 구조: [0:15] 충전, [15:30] 방전, [30:45] 공통/IC
        if mode is not None:
            # x가 (B, D)인 경우와 (B, L, D)인 경우를 모두 처리
            mask = torch.ones_like(x)

            # 충전 모드(1)인 경우: 방전 피처(15:30)를 0으로 마스킹
            # 방전 모드(0)인 경우: 충전 피처(0:15)를 0으로 마스킹
            for i in range(x.size(0)):
                if mode[i] > 0.5:  # Charge
                    if len(x.shape) == 3:
                        mask[i, :, 15:30] = 0
                    else:
                        mask[i, 15:30] = 0
                else:  # Discharge
                    if len(x.shape) == 3:
                        mask[i, :, 0:15] = 0
                    else:
                        mask[i, 0:15] = 0
            x = x * mask

        if return_pde:
            x.requires_grad_(True)

        if not return_pde:
            return self.solution_net(x)

        # 1. Solution Network 예측: u = F(x)
        u = self.solution_net(x)

        # 2. 편미분 계산 (u_x) - t가 없으므로 x 전체에 대해 편미분
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        # 3. 형태 맞추기 (Flatten)
        x_flat = x.view(x.size(0), -1)  # (B, feature_dim)
        u_x_flat = u_x.view(u_x.size(0), -1)  # (B, feature_dim)
        
        # 4. 타겟 지표(HI)에 대한 편미분 추출 (이전의 u_t 역할을 대신함)
        # 시퀀스 모델인 경우 마지막 시점의 특징을 기준으로 삼거나, 2D 구조인 경우 바로 인덱싱
        if len(u_x.shape) == 3:
             u_hi = u_x[:, -1, self.pi_target_idx].unsqueeze(-1) # (B, 1)
        else:
             u_hi = u_x[:, self.pi_target_idx].unsqueeze(-1) # (B, 1)

        # 5. Dynamics Network 입력 생성 및 추론
        # 입력 차원: x(D) + u(1) + u_hi(1) + u_x(D)
        g_input = torch.cat([x_flat, u, u_hi, u_x_flat], dim=1)  # (B, g_input_dim)
        g_out = self.dynamics_net(g_input)

        # 6. PDE Residual 계산: H = u_hi - G(x, u, u_hi, u_x)
        pde_residual = u_hi - g_out

        return u, pde_residual, u_hi


# ==========================================
# 3. Machine Learning Models
# ==========================================


def get_rf_model():
    return RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )


def get_svr_model():
    return SVR(kernel="rbf", C=100, gamma="scale")


def get_gpr_model():
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, random_state=42
    )


def get_xgboost_model(**kwargs):
    try:
        return XGBRegressor(random_state=42, n_jobs=-1, **kwargs)
    except ImportError:
        raise ImportError(
            "xgboost is not installed. Please install it using 'pip install xgboost'."
        )


def get_lightgbm_model(**kwargs):
    try:
        return LGBMRegressor(random_state=42, n_jobs=-1, **kwargs)
    except ImportError:
        raise ImportError(
            "lightgbm is not installed. Please install it using 'pip install lightgbm'."
        )


# ==========================================
# 4. Model Factory
# ==========================================


def get_model(model_name, use_pi=False, feature_dim=40, pi_target_idx=0, **kwargs):
    model_name = model_name.upper()

    # [수정] Cycle-Agnostic PINN이므로 시간(t)가 없어 입력 차원은 항상 feature_dim과 동일
    actual_input_dim = feature_dim

    # --- Deep Learning Models ---
    if model_name in ["ITRANSFORMER", "TABNET", "MLP"]:
        if model_name == "ITRANSFORMER":
            base_model = InvertedTransformer(num_variates=actual_input_dim, **kwargs)
        elif model_name == "TABNET":
            base_model = TabularAttentionNet(input_dim=actual_input_dim, **kwargs)
        elif model_name == "MLP":
            base_model = SimpleMLP(input_dim=actual_input_dim, **kwargs)

        # PI 옵션 활성화 시 Wrapper로 감싸서 반환
        if use_pi:
            return PhysicsInformedWrapper(base_model, feature_dim=feature_dim, pi_target_idx=pi_target_idx)
        else:
            return base_model

    # --- Machine Learning Models ---
    elif model_name in ["RF", "SVR", "GPR", "XGBOOST", "LIGHTGBM"]:
        if use_pi:
            print(
                "Warning: Machine Learning 모델은 미분(Autograd)이 불가능하여 PI 모듈을 사용할 수 없습니다. 일반 모델을 반환합니다."
            )

        if model_name == "RF":
            return get_rf_model()
        elif model_name == "SVR":
            return get_svr_model()
        elif model_name == "GPR":
            return get_gpr_model()
        elif model_name == "XGBOOST":
            return get_xgboost_model(**kwargs)
        elif model_name == "LIGHTGBM":
            return get_lightgbm_model(**kwargs)

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
