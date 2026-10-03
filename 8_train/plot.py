"""
8_train/plot.py — Step 8(train.py) 전용 시각화 모음.

_plot_loss_curves: train_log_v2.csv의 손실 항(mse/ce/l0)을 에폭별로 시각화.
train.py가 로그에 tr_mse/tr_ce/tr_l0/lambda_scen 컬럼을 쓰기 시작한 뒤(2026-10-02)
추가 — "epoch 1엔 L0가 얼마, MSE가 얼마였다가 epoch 300엔 어떻게 바뀌는지"를
한눈에 보기 위함.

_plot_gate_probs: 2026-10-02에 model_lib/utils/gate_io.py에서 이동 — gate_io.py는
이제 JSON 저장/로드 전용(6_synergy/plot.py·7_kernel/plot.py처럼 시각화는 각 스텝
폴더 로컬 plot.py에 두는 관례와 맞춤, 동작 변화 없는 순수 이동). 원래 출처는
model_lib/legacy/train_scr.py(Stage0, 삭제됨).

train.py가 `from plot import _plot_loss_curves, _plot_gate_probs`로 가져다 쓴다.
"""

from __future__ import annotations

from pathlib import Path

from models.scr_model import SCRModel


def _plot_loss_curves(log_path: Path, output_path: Path) -> None:
    """
    2개 서브플롯:
      좌) 가중된 기여도(stackplot) — mse + lambda_scen*ce + lambda_l0*l0 쌓은 면적의
          맨 위 선이 곧 total loss와 같다(SCRLoss.forward의 total 정의 그대로).
      우) 가중치를 곱하기 전 raw 값(log-scale) — 람다가 변하는 것과 무관하게 각 항
          자체가 줄어들고 있는지 보기 위함.
    train_log_v2.csv에 tr_mse/tr_ce/tr_l0 컬럼이 없는 과거 run(이 기능 추가 이전)은
    조용히 건너뛴다.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[train] matplotlib/pandas 미설치 — loss_curves.png 생략")
        return

    df = pd.read_csv(log_path)
    required = {"tr_mse", "tr_ce", "tr_l0", "lambda_l0", "lambda_scen"}
    if not required.issubset(df.columns):
        print(f"[train] {log_path.name}에 손실 항 컬럼이 없어 loss_curves.png 생략 "
              f"(옛 run이거나 2026-10-02 이전 로그)")
        return

    epoch = df["epoch"]
    mse = df["tr_mse"]
    ce_w = df["lambda_scen"] * df["tr_ce"]
    l0_w = df["lambda_l0"] * df["tr_l0"]

    fig, (ax_w, ax_raw) = plt.subplots(1, 2, figsize=(14, 5))

    ax_w.stackplot(epoch, mse, ce_w, l0_w,
                   labels=["mse", "lambda_scen*ce", "lambda_l0*l0"],
                   colors=["steelblue", "darkorange", "seagreen"], alpha=0.85)
    ax_w.set_title("가중된 기여도 (쌓은 높이 = total loss)", fontsize=10, fontweight="bold")
    ax_w.set_xlabel("epoch")
    ax_w.set_ylabel("loss 기여분")
    ax_w.legend(fontsize=8, loc="upper right")

    ax_raw.plot(epoch, mse, label="mse (raw)", color="steelblue")
    ax_raw.plot(epoch, df["tr_ce"], label="ce (raw)", color="darkorange")
    ax_raw.plot(epoch, df["tr_l0"], label="l0 (raw)", color="seagreen")
    ax_raw.set_yscale("log")
    ax_raw.set_title("람다 곱하기 전 raw 값 (log scale)", fontsize=10, fontweight="bold")
    ax_raw.set_xlabel("epoch")
    ax_raw.set_ylabel("loss (log)")
    ax_raw.legend(fontsize=8)

    fig.suptitle("Phase 1 — Loss term breakdown by epoch", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Saved loss curve plot → {output_path}")


def _plot_gate_probs(
    model: SCRModel,
    output_path: Path,
    hi_cols_ref: list[str],
    charge_m: int,
    discharge_m: int,
    scen_k: int,
) -> None:
    """
    8개 서브플롯: charge probe / discharge probe / 6 scen gates
    x축: HI 인덱스, y축: gate_prob. threshold(m/k) 기준선 표시.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[train] matplotlib 미설치 — gate_probs.png 생략")
        return

    gates_info = [
        ("Charge Probe",    model.charge_probe_gate,    charge_m,    "steelblue"),
        ("Discharge Probe", model.discharge_probe_gate, discharge_m, "darkorange"),
    ]
    seg_names = model.spec.scenario_names
    for s in range(model.n_scenarios):
        gates_info.append((f"Scen: {seg_names[s]}", model.scen_gates[s], scen_k, "seagreen"))

    n_plots = len(gates_info)   # 8
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    axes = axes.flatten()

    is_grouped = any(hasattr(gate, "group_index") for _, gate, _, _ in gates_info)

    for ax, (title, gate, threshold, color) in zip(axes, gates_info):
        prob = gate.gate_prob().detach().cpu().numpy()
        idx  = np.arange(len(prob))
        sorted_idx = np.argsort(prob)[::-1]
        sorted_prob = prob[sorted_idx]

        if hasattr(gate, "group_index"):
            # 그룹 계층 게이트 — 막대를 그룹별 색으로 칠해서 같은 그룹 멤버가 랭킹에서
            # 뭉쳐 있는지(=그룹이 실제로 같이 움직인다) 한눈에 보이게 하고, 그룹 자체의
            # 게이트 확률(멤버 오프셋 제외)을 점선으로 겹쳐 그린다.
            group_idx = gate.group_index.detach().cpu().numpy()
            sorted_groups = group_idx[sorted_idx]
            cmap = plt.cm.tab20(np.linspace(0, 1, max(gate.n_groups, 1)))
            bar_colors = cmap[sorted_groups % 20]
            ax.bar(range(len(sorted_prob)), sorted_prob, color=bar_colors, alpha=0.85)
            group_prob = gate.group_gate_prob().detach().cpu().numpy()
            ax.plot(range(len(sorted_prob)), group_prob[sorted_groups],
                    color="black", linestyle=":", linewidth=1.0, alpha=0.7,
                    label=f"group_gate_prob ({gate.n_groups} groups)")
        else:
            ax.bar(range(len(sorted_prob)), sorted_prob, color=color, alpha=0.7)

        if threshold <= len(sorted_prob):
            cutoff = float(sorted_prob[threshold - 1]) if threshold > 0 else 1.0
            ax.axvline(x=threshold - 0.5, color="red", linestyle="--", linewidth=1.2,
                       label=f"top-{threshold} cutoff")
            ax.axhline(y=cutoff, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("HI rank", fontsize=8)
        ax.set_ylabel("gate_prob", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        # 상위 5개 이름 표시
        for rank in range(min(5, len(sorted_idx))):
            hi_name = hi_cols_ref[sorted_idx[rank]]
            short   = hi_name.split("_dis_hi")[0].split("_chg_lo")[0]
            ax.text(rank, sorted_prob[rank] + 0.01, short,
                    rotation=90, fontsize=5, ha="center", va="bottom")

    _suptitle = "Phase 1 — Gate Probability by HI (sorted desc)"
    if is_grouped:
        _suptitle += "  [scen gates: grouped — bar color=synergy group, dotted=group_gate_prob]"
    fig.suptitle(_suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Saved gate_prob plot → {output_path}")
