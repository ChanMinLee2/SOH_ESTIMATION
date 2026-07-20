"""
standalone_overfit_test.py
지정한 시나리오 축별 gate 경로 + 모델 변형(variant) 목록으로 과적합 테스트 실행.

사용법:
  python tmp_standalone_overfit.py                           # 기본 (S-size 변형만)
  python tmp_standalone_overfit.py --model-size-variant true # S+M+L 전체

결과:    OUTPUT_DIR/{axis}_{variant_name}.csv  +  overfit_comparison.png
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# [사용자 설정] 이 섹션만 수정
# ─────────────────────────────────────────────────────────────────────────────

# 테스트할 시나리오 축과 해당 Phase 1 결과 폴더
# None 으로 설정하면 해당 축은 실행하지 않음
AXES: dict[str, str | None] = {
    "qfrac":    None,                          # Phase 1 완료 후 경로 입력
    "protocol": "_5_data_model_scr/0715_1015",
    "vwindow":  "_5_data_model_scr/0715_1040",
    "rcs":      "_5_data_model_scr/0715_1149",
    "cluster":  None,                          # Phase 1 완료 후 경로 입력
}

# ── 기본 변형 (항상 실행) — 각 family의 S-size 1개 ────────────────────────
# family: "mlp" | "transformer" | "i_transformer" | "resnet_tab" | "ft_transformer"
# model_cfg 키:
#   MLP            → mlp_hidden_dims
#   Transformer    → tr_n_heads, tr_n_layers, tr_d_ff
#   iTransformer   → tr_n_heads, tr_n_layers, tr_d_ff
#   ResNet-Tab     → resnet_n_blocks, resnet_d_hidden_factor
#   FT-Transformer → tr_n_heads, tr_n_layers, tr_d_ff
VARIANTS_BASE: list[dict] = [
    # ── MLP ──────────────────────────────────────────────────────────────────
    {   # S (default): 130→128→64→1  (~25K params)
        "name": "MLP-S", "family": "mlp", "d_head": 128,
        "model_cfg": {},
    },

    # ── Transformer ──────────────────────────────────────────────────────────
    {   # S: d_model=128, 2-layer, 4-head  (~282K params)
        "name": "Tr-S", "family": "transformer", "d_head": 128,
        "model_cfg": {"tr_n_heads": 4, "tr_n_layers": 2, "tr_d_ff": 256},
    },

    # ── iTransformer ─────────────────────────────────────────────────────────
    {
        "name": "iTr-S", "family": "i_transformer", "d_head": 128,
        "model_cfg": {"tr_n_heads": 4, "tr_n_layers": 2, "tr_d_ff": 256},
    },

    # ── ResNet-Tab (Gorishniy et al., NeurIPS 2021) ───────────────────────────
    {   # S: d=128, 3 blocks  (~75K params)
        "name": "ResNet-S", "family": "resnet_tab", "d_head": 128,
        "model_cfg": {"resnet_n_blocks": 3, "resnet_d_hidden_factor": 2.0},
    },

    # ── FT-Transformer (Gorishniy et al., NeurIPS 2021) ──────────────────────
    # {   # S: d_model=64, 2-layer, 4-head  (~130 tokens)
    #     "name": "FTTr-S", "family": "ft_transformer", "d_head": 64,
    #     "model_cfg": {"tr_n_heads": 4, "tr_n_layers": 2, "tr_d_ff": 128},
    # },
]

# ── 사이즈 변형 (--model-size-variant true 시에만 실행) ──────────────────────
VARIANTS_SIZE: list[dict] = [
    # ── MLP ──────────────────────────────────────────────────────────────────
    {   # M: 130→256→128→64→1  (~74K params)
        "name": "MLP-M", "family": "mlp", "d_head": 256,
        "model_cfg": {"mlp_hidden_dims": [256, 128, 64]},
    },
    {   # L: 130→512→256→128→64→1  (~239K params)
        "name": "MLP-L", "family": "mlp", "d_head": 512,
        "model_cfg": {"mlp_hidden_dims": [512, 256, 128, 64]},
    },

    # ── Transformer ──────────────────────────────────────────────────────────
    {   # M: d_model=256, 3-layer, 8-head  (~1.1M params)
        "name": "Tr-M", "family": "transformer", "d_head": 256,
        "model_cfg": {"tr_n_heads": 8, "tr_n_layers": 3, "tr_d_ff": 512},
    },
    {   # L: d_model=512, 4-layer, 8-head  (~4.4M params)
        "name": "Tr-L", "family": "transformer", "d_head": 512,
        "model_cfg": {"tr_n_heads": 8, "tr_n_layers": 4, "tr_d_ff": 1024},
    },

    # ── iTransformer ─────────────────────────────────────────────────────────
    {
        "name": "iTr-M", "family": "i_transformer", "d_head": 256,
        "model_cfg": {"tr_n_heads": 8, "tr_n_layers": 3, "tr_d_ff": 512},
    },
    {
        "name": "iTr-L", "family": "i_transformer", "d_head": 512,
        "model_cfg": {"tr_n_heads": 8, "tr_n_layers": 4, "tr_d_ff": 1024},
    },

    # ── ResNet-Tab ────────────────────────────────────────────────────────────
    {   # M: d=256, 4 blocks  (~261K params)
        "name": "ResNet-M", "family": "resnet_tab", "d_head": 256,
        "model_cfg": {"resnet_n_blocks": 4, "resnet_d_hidden_factor": 2.0},
    },
    {   # L: d=512, 6 blocks  (~1.6M params)
        "name": "ResNet-L", "family": "resnet_tab", "d_head": 512,
        "model_cfg": {"resnet_n_blocks": 6, "resnet_d_hidden_factor": 2.0},
    },

    # # ── FT-Transformer ────────────────────────────────────────────────────────
    # {   # M: d_model=128, 3-layer, 4-head
    #     "name": "FTTr-M", "family": "ft_transformer", "d_head": 128,
    #     "model_cfg": {"tr_n_heads": 4, "tr_n_layers": 3, "tr_d_ff": 256},
    # },
    # {   # L: d_model=256, 4-layer, 8-head
    #     "name": "FTTr-L", "family": "ft_transformer", "d_head": 256,
    #     "model_cfg": {"tr_n_heads": 8, "tr_n_layers": 4, "tr_d_ff": 512},
    # },
]

N_SAMPLES  = 2048
MAX_EPOCHS = 3000
LR         = 1e-3
LOG_EVERY  = 20

DEVICE_STR = "auto"
OUTPUT_DIR = Path("tmp_overfit_variants")

# 게이트 마스크 설정
CHARGE_PROBE_M    = 3
DISCHARGE_PROBE_M = 2
SCEN_K_COUNT      = 55
D_PROBE           = 64
DROPOUT           = 0.1

DATA_CFG_BASE = {
    "datasets":                  ["MIT", "HUST"],
    "is_cross_dataset_evaluate": True,
    "train_ratio":               0.6,
    "val_ratio":                 0.2,
    "test_ratio":                0.2,
    "split_seed":                42,
    "min_cycles_per_cell":       10,
    "io_workers":                4,
    "use_initial_capacity":      True,
    "nominal_capacities":        {"MIT": 1.1, "HUST": 1.2},
}

# ─────────────────────────────────────────────────────────────────────────────
# 시각화 스타일 (family × size 인덱스)
# ─────────────────────────────────────────────────────────────────────────────
_FAMILY_COLORS = {
    "mlp":            ["#1a4b8c", "#3a7fd5", "#89b8f5"],   # 파랑  (S→M→L 밝아짐)
    "transformer":    ["#8b2500", "#d95f00", "#f5a06e"],   # 주황
    "i_transformer":  ["#1b5e20", "#388e3c", "#81c784"],   # 초록
    "resnet_tab":     ["#4a0080", "#8e24aa", "#ce93d8"],   # 보라
    "ft_transformer": ["#795548", "#a1887f", "#d7ccc8"],   # 갈색
}
_SIZE_LINESTYLES = ["solid", "dashed", "dotted"]
_SIZE_MARKERS    = ["o", "s", "^"]
_SIZE_HATCHES    = {"solid": "", "dashed": "//", "dotted": "xx"}


def _family_size_map(variants: list[dict]) -> dict[str, list[str]]:
    """family → [variant_name ...] 순서 맵 (size index 계산용)."""
    fmap: dict[str, list[str]] = {}
    for v in variants:
        fmap.setdefault(v["family"], []).append(v["name"])
    return fmap


def _get_style(variant: dict, fmap: dict[str, list[str]]) -> dict:
    family   = variant["family"]
    palette  = _FAMILY_COLORS.get(family, ["#555", "#888", "#bbb"])
    names    = fmap.get(family, [variant["name"]])
    idx      = names.index(variant["name"]) if variant["name"] in names else 0
    idx      = min(idx, len(palette) - 1)
    ls       = _SIZE_LINESTYLES[idx % len(_SIZE_LINESTYLES)]
    return {
        "color":     palette[idx],
        "linestyle": ls,
        "marker":    _SIZE_MARKERS[idx % len(_SIZE_MARKERS)],
        "hatch":     _SIZE_HATCHES[ls],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 경로 · Import 설정
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.compat import install_numpy2_shim
    install_numpy2_shim()
except ImportError:
    pass

from utils.hi_schema import N_HI
from utils.metrics import r2
from models.scr_model import SCRModel
from datasets.segment_dataset import build_datasets, collate_fn
from common.scenario import get_segmenter as _get_segmenter
from common.scenario.base import ScenarioSpec


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_device() -> torch.device:
    if DEVICE_STR == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE_STR)


def _to_bool_mask(indices: list[int], top_n: int) -> torch.Tensor:
    mask = torch.zeros(N_HI, dtype=torch.bool)
    for i in indices[:top_n]:
        if 0 <= i < N_HI:
            mask[i] = True
    return mask


def _load_probe_masks(gates_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    for name in ("classification_HIs.json", "scenario_classification_HIs.json"):
        p = gates_dir / name
        if p.exists():
            data = json.loads(p.read_text())
            break
    else:
        raise FileNotFoundError(f"probe JSON not found in {gates_dir}")
    if "charge_ranked" in data:
        return (_to_bool_mask(data["charge_ranked"],    CHARGE_PROBE_M),
                _to_bool_mask(data["discharge_ranked"], DISCHARGE_PROBE_M))
    ranked = data.get("ranked_indices", [])
    return _to_bool_mask(ranked, CHARGE_PROBE_M), _to_bool_mask(ranked, DISCHARGE_PROBE_M)


def _load_scen_masks(gates_dir: Path, n_scenarios: int) -> torch.Tensor:
    for name in ("regression_HIs.json", "scenario_regression_HIs.json"):
        p = gates_dir / name
        if p.exists():
            data = json.loads(p.read_text())
            break
    else:
        raise FileNotFoundError(f"scen JSON not found in {gates_dir}")
    masks = torch.zeros(n_scenarios, N_HI, dtype=torch.bool)
    for s in range(n_scenarios):
        for i in data.get(f"seg_{s}_ranked", data.get(f"seg_{s}", []))[:SCEN_K_COUNT]:
            if 0 <= i < N_HI:
                masks[s, i] = True
    return masks


def _collect_tiny(loader: DataLoader, n_samples: int, device: torch.device) -> dict:
    buf: dict[str, list] = {}
    n = 0
    for batch in loader:
        for k, v in batch.items():
            buf.setdefault(k, []).append(v)
        n += next(iter(batch.values())).size(0)
        if n >= n_samples:
            break
    return {k: torch.cat(vs)[:n_samples].to(device) for k, vs in buf.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 과적합 테스트
# ─────────────────────────────────────────────────────────────────────────────
def run_overfit(
    model: SCRModel,
    tiny_dev: dict,
    max_epochs: int,
    lr: float,
    csv_path: Path,
) -> dict:
    test_model = copy.deepcopy(model)
    for m in test_model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    test_model.train()

    opt = torch.optim.AdamW(test_model.parameters(), lr=lr, weight_decay=0.0)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "loss", "r2"])

    max_r2  = float("-inf")
    ep_max  = 0
    r2_val  = float("nan")

    for epoch in range(max_epochs):
        opt.zero_grad()
        out  = test_model(tiny_dev)
        loss = nn.functional.mse_loss(out["cap_pred"], tiny_dev["target"])
        loss.backward()
        opt.step()

        r2_val = float(r2(tiny_dev["target"].cpu().numpy(),
                          out["cap_pred"].detach().cpu().numpy()))
        if r2_val > max_r2:
            max_r2 = r2_val
            ep_max = epoch + 1

        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch + 1, f"{loss.item():.6f}", f"{r2_val:.6f}"])

    print(f"    max R2={max_r2:.4f} @ ep{ep_max:4d} | final R2={r2_val:.4f}")
    return {"max_r2": max_r2, "ep_max": ep_max, "final_r2": r2_val}


# ─────────────────────────────────────────────────────────────────────────────
# 플롯 생성
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(
    results:    dict[tuple, dict],
    all_dfs:    dict[tuple, pd.DataFrame],
    active_axes: list[str],
    variants:   list[dict],
    out_dir:    Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fmap  = _family_size_map(variants)
    n_ax  = len(active_axes)
    n_cols = max(n_ax, 2)
    CLIP  = -0.5

    fig, grid = plt.subplots(2, n_cols, figsize=(6.5 * n_cols, 11))
    if n_cols == 1:
        grid = grid.reshape(2, 1)

    # ── 상단: 축별 R² 수렴 곡선 ──────────────────────────────────────────────
    for col, axis in enumerate(active_axes):
        ax = grid[0, col]
        for v in variants:
            df  = all_dfs[(axis, v["name"])]
            st  = _get_style(v, fmap)
            r2c = df["r2"].clip(lower=CLIP)
            ax.plot(df["epoch"], r2c,
                    label=v["name"],
                    color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=3, linewidth=1.8,
                    markevery=max(1, MAX_EPOCHS // 80))
            idx = df["r2"].idxmax()
            y_peak = float(df.loc[idx, "r2"])
            ax.scatter(int(df.loc[idx, "epoch"]), max(y_peak, CLIP),
                       color=st["color"], s=55, zorder=5)

        ax.axhline(0,   color="gray", linestyle="--", lw=0.8, alpha=0.5)
        ax.axhline(0.7, color="red",  linestyle=":",  lw=1.2, label="R2=0.7")
        ax.set_title(f"axis = {axis}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"R2  (clip >= {CLIP})")
        ax.set_ylim(CLIP - 0.05, 1.05)
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.2)

    for col in range(n_ax, n_cols):
        grid[0, col].axis("off")

    # ── 하단 좌: Max R² 막대 ──────────────────────────────────────────────────
    ax_bar = grid[1, 0]
    x      = np.arange(n_ax)
    n_v    = len(variants)
    w      = 0.8 / n_v
    offset = -(n_v - 1) / 2 * w

    for i, v in enumerate(variants):
        st   = _get_style(v, fmap)
        vals = [results[(ax_n, v["name"])]["max_r2"] for ax_n in active_axes]
        bars = ax_bar.bar(x + offset + i * w, vals, w,
                          label=v["name"],
                          color=st["color"], alpha=0.88,
                          hatch=st["hatch"])
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax_bar.axhline(0.7, color="red", linestyle=":", lw=1.2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(active_axes, fontsize=11)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Max R2")
    ax_bar.set_title(f"Max R2  (N={N_SAMPLES}, ep={MAX_EPOCHS})", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7, ncol=2)
    ax_bar.grid(axis="y", alpha=0.2)

    # ── 하단 우: 요약 텍스트 ──────────────────────────────────────────────────
    ax_txt = grid[1, 1] if n_cols > 1 else grid[1, 0]
    ax_txt.axis("off")

    col_w  = 8
    header = f"{'Variant':<12}" + "".join(f"{a:>{col_w}}" for a in active_axes) + f"{'avg':>{col_w}}"
    lines  = [f"Max R2  (ep={MAX_EPOCHS})", "=" * len(header), header, "-" * len(header)]

    prev_fam = None
    for v in variants:
        if v["family"] != prev_fam and prev_fam is not None:
            lines.append("")
        prev_fam = v["family"]
        vals = [results[(ax_n, v["name"])]["max_r2"] for ax_n in active_axes]
        row  = f"{v['name']:<12}" + "".join(f"{val:>{col_w}.4f}" for val in vals)
        row += f"{np.mean(vals):>{col_w}.4f}"
        lines.append(row)

    lines.append("=" * len(header))
    best_key = max(results, key=lambda k: results[k]["max_r2"])
    lines.append(f"Best: {best_key[1]} / {best_key[0]}  R2={results[best_key]['max_r2']:.4f}")

    ax_txt.text(0.02, 0.97, "\n".join(lines),
                transform=ax_txt.transAxes, fontsize=8.5,
                va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    for col in range(2, n_cols):
        grid[1, col].axis("off")

    fig.suptitle(
        f"Overfit Test  (dropout=0, weight_decay=0, lr={LR})\n"
        f"N_samples={N_SAMPLES}, max_epochs={MAX_EPOCHS}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = out_dir / "overfit_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone overfit test for SCR model variants")
    parser.add_argument(
        "--model-size-variant",
        type=lambda x: x.lower() == "true",
        default=False,
        metavar="true|false",
        help="사이즈 변형(M/L) 포함 여부 (default: false — S-size만 실행)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    active_variants = VARIANTS_BASE + (VARIANTS_SIZE if args.model_size_variant else [])

    device = _resolve_device()
    active_axes = [ax for ax, p in AXES.items() if p is not None]

    print(f"Device: {device}")
    print(f"Model size variants: {'enabled (S+M+L)' if args.model_size_variant else 'disabled (S only)'}")
    print(f"Axes    ({len(active_axes)}/{len(AXES)}): {active_axes}")
    print(f"Variants ({len(active_variants)}): {[v['name'] for v in active_variants]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: dict[tuple, pd.DataFrame] = {}
    results: dict[tuple, dict]         = {}

    for axis, gate_from_raw in AXES.items():
        if gate_from_raw is None:
            print(f"\n[SKIP] {axis}: gate 경로 미지정 (None)")
            continue

        print(f"\n{'='*65}")
        print(f"Axis: {axis}  |  gates: {gate_from_raw}")
        print(f"{'='*65}")

        gate_from = PROJECT_ROOT / gate_from_raw
        gates_dir = gate_from / "gates" if (gate_from / "gates").exists() else gate_from

        spec_path = gate_from / "scenario_spec.json"
        spec = ScenarioSpec.load(spec_path) if spec_path.exists() \
               else _get_segmenter(axis, {axis: {}}).get_spec()
        print(f"  spec: n_scenarios={spec.n_scenarios}, n_classes={spec.n_classes}")

        ch_mask, dis_mask = _load_probe_masks(gates_dir)
        scen_masks        = _load_scen_masks(gates_dir, spec.n_scenarios)
        print(f"  probe mask: charge={ch_mask.sum().item()}, discharge={dis_mask.sum().item()}")
        print(f"  scen mask:  avg {scen_masks.float().sum(dim=1).mean().item():.1f} HI/scenario")

        cfg = {
            "data": {**DATA_CFG_BASE,
                     "seg_data_dir": f"_4_data_hi/{axis}/seg",
                     "data_dir":     f"_4_data_hi/{axis}/cycle"},
            "loss":       {"leak_cols": ["stat_q_abs", "stat_energy_seg"]},
            "regression": {"scen_k_count": SCEN_K_COUNT},
        }
        print(f"  Loading dataset: _4_data_hi/{axis}/seg ...")
        train_ds, _, _, _ = build_datasets(cfg, spec=spec)
        loader = DataLoader(train_ds, batch_size=2048, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)

        tiny_dev = _collect_tiny(loader, N_SAMPLES, device)
        t_std = tiny_dev["target"].std().item()
        print(f"  Tiny batch: {N_SAMPLES} samples | target std={t_std:.5f}")

        for v in active_variants:
            print(f"\n  [{v['name']}]  family={v['family']}  d_head={v['d_head']}")
            model_cfg = {**v["model_cfg"], "regression_model": v["family"]}
            model = SCRModel(
                d_probe=D_PROBE, d_head=v["d_head"], dropout=DROPOUT,
                charge_probe_mask=ch_mask, discharge_probe_mask=dis_mask,
                scen_masks=scen_masks, model_cfg=model_cfg, spec=spec,
            ).to(device)
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    params: {n_p:,}")

            csv_path = OUTPUT_DIR / f"{axis}_{v['name']}.csv"
            res = run_overfit(model, tiny_dev, MAX_EPOCHS, LR, csv_path)
            results[(axis, v["name"])]  = res
            all_dfs[(axis, v["name"])]  = pd.read_csv(csv_path)

    # 요약
    print(f"\n{'='*65}")
    print(f"{'Variant':<12} {'Axis':<10} {'MaxR2':>7} {'FinalR2':>8} {'ep@max':>7}")
    print(f"{'-'*65}")
    prev_fam = None
    for v in active_variants:
        if v["family"] != prev_fam and prev_fam is not None:
            print()
        prev_fam = v["family"]
        for axis in active_axes:
            r = results[(axis, v["name"])]
            print(f"{v['name']:<12} {axis:<10} {r['max_r2']:>7.4f} {r['final_r2']:>8.4f} {r['ep_max']:>7}")

    make_plots(results, all_dfs, active_axes, active_variants, OUTPUT_DIR)
    print(f"\nDone. Results in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
