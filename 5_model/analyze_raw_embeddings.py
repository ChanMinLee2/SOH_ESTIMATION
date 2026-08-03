"""
analyze_raw_embeddings.py — raw 곡선/CNN 임베딩 근본원인 분석.

docs/260801_RESULTS.md의 4자 비교(baseline vs 방안2(a) frozen vs 방안2(b) separated vs
방안1 raw flatten)에서 왜 (a)가 가장 나쁘고 방안1이 가장 나은지, 임베딩 자체를 들여다봐
근본 원인을 찾는다. 세 가지 분석을 5개 조합(qfw+mlp/transformer/resnet_tab, vqslope+mlp,
q_abs+mlp) 전체에 대해 수행한다:

  1. 차원별 상관관계 — 각 임베딩 차원(cnn_emb 64개 / raw_flat 96개)과
     "baseline(HI만) 예측의 residual"(soh_true - baseline_soh_pred) 간 Pearson 상관.
     "이 표현이 HI 대비 순증분 정보를 실제로 담고 있는가"를 직접 측정한다.
  2. (a) vs (b) 코사인 유사도·분산 비교 — 같은 세그먼트에 대해 얼린 임베딩(a)과
     학습된 임베딩(b)이 얼마나 다른지, 그리고 각 차원의 분산이 붕괴돼 있는지 확인.
  3. PCA 2D 시각화 — 시나리오별 색칠 + residual 크기별 색칠을 나란히 그려 눈으로 확인.

baseline의 predictions.csv를 재사용해 residual을 얻고(재학습 불필요), (a)/(b)/방안1
체크포인트에서 raw_cnn/raw_flat_norm 서브모듈만 직접 호출해 임베딩을 뽑는다(전체
forward()를 안 돌려도 됨 — cnn_emb/raw_flat 계산에 probe_x/scen_x가 필요 없기 때문).

결과 저장: _5_data_model_scr/comparison/raw_embedding_analysis/
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from utils.compat import install_numpy2_shim
    install_numpy2_shim()
except ImportError:
    pass

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from visualize_results import RunBundle, _build_model_for_run, _build_test_dataset_for_run

OUT_DIR = PROJECT_ROOT / "_5_data_model_scr" / "comparison" / "raw_embedding_analysis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(0)

COMBOS = [
    dict(name="qfw_mlp",
         baseline="_5_data_model_scr/0723_1633_p2_mlp_qfw_35%_20%",
         a="_5_data_model_scr/0731_1039_p2_mlp_qfw_35%_20%",
         b="_5_data_model_scr/0731_1103_p2_mlp_qfw_35%_20%",
         flat="_5_data_model_scr/0801_1524_p2_mlp_qfw_35%_20%"),
    dict(name="qfw_transformer",
         baseline="_5_data_model_scr/0723_2356_p2_tr_qfw_35%_20%",
         a="_5_data_model_scr/0803_1240_p2_tr_qfw_35%_20%",
         b="_5_data_model_scr/0731_1216_p2_tr_qfw_35%_20%",
         flat="_5_data_model_scr/0801_1543_p2_tr_qfw_35%_20%"),
    dict(name="qfw_resnet_tab",
         baseline="_5_data_model_scr/0724_0111_p2_res_qfw_35%_20%",
         a="_5_data_model_scr/0803_1319_p2_res_qfw_35%_20%",
         b="_5_data_model_scr/0731_1320_p2_res_qfw_35%_20%",
         flat="_5_data_model_scr/0801_1612_p2_res_qfw_35%_20%"),
    dict(name="vqslope_mlp",
         baseline="_5_data_model_scr/0725_1348_p2_mlp_vqs_dva",
         a="_5_data_model_scr/0731_1349_p2_mlp_vqs_dva",
         b="_5_data_model_scr/0731_1403_p2_mlp_vqs_dva",
         flat="_5_data_model_scr/0801_1643_p2_mlp_vqs_dva"),
    dict(name="qabs_mlp",
         baseline="_5_data_model_scr/0729_1908_p2_mlp_qabs_20-50%",
         a="_5_data_model_scr/0731_0049_p2_mlp_qabs_20-50%",
         b="_5_data_model_scr/0731_0339_p2_mlp_qabs_20-50%",
         flat="_5_data_model_scr/0801_1701_p2_mlp_qabs_20-50%"),
]


# =============================================================================
# 임베딩 추출
# =============================================================================

@torch.no_grad()
def _extract_embeddings(run_dir: str, kind: str, batch_size: int = 2048) -> pd.DataFrame:
    """kind: 'a'|'b' (cnn_emb, raw_cnn 서브모듈 직접 호출) | 'flat' (raw_flat_norm 직접 호출).

    전체 forward()를 안 돌리는 이유: cnn_emb/raw_flat 계산은 x_raw만 있으면 되고
    probe_x/scen_x(HI 게이트)와 무관하다 — 서브모듈만 호출하면 동일한 값을 얻으면서
    HI 게이트 마스크 로딩 등 불필요한 절차를 생략할 수 있다.
    """
    bundle = RunBundle(Path(run_dir), label=kind)
    _build_model_for_run(bundle, DEVICE, "best.pt")
    model = bundle.model
    test_ds = _build_test_dataset_for_run(bundle)

    if kind in ("a", "b"):
        assert model.raw_cnn is not None, f"{run_dir}: with_raw_cnn 모델이 아닙니다"
        sub = model.raw_cnn
    else:
        assert model.with_raw_flat, f"{run_dir}: with_raw_flat 모델이 아닙니다"
        sub = None  # raw_flat_norm은 아래서 직접 처리(reshape 필요)

    n = len(test_ds)
    embs = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_raw = test_ds.x_raw[start:end].to(DEVICE)
        if kind in ("a", "b"):
            emb = sub(x_raw)
        else:
            flat = x_raw.reshape(x_raw.size(0), -1)
            emb = model.raw_flat_norm(flat)
        embs.append(emb.cpu().numpy())
    emb_mat = np.concatenate(embs, axis=0)  # (N, D)

    df = pd.DataFrame({
        "cell_id":  test_ds.cell_ids,
        "cycle":    test_ds.cycles,
        "seg_name": test_ds.seg_names,
    })
    for d in range(emb_mat.shape[1]):
        df[f"e{d}"] = emb_mat[:, d]
    print(f"[extract] {run_dir} ({kind}): n={n:,}, dim={emb_mat.shape[1]}")
    return df


def _load_baseline_residual(run_dir: str) -> pd.DataFrame:
    bundle = RunBundle(Path(run_dir), label="baseline")
    df = pd.DataFrame(bundle.pred_rows)
    df["residual"] = df["soh_true"] - df["soh_pred"]
    return df[["cell_id", "cycle", "seg_name", "residual", "soh_true"]]


# =============================================================================
# 분석 1 — 차원별 상관관계 (임베딩 vs baseline residual)
# =============================================================================

def _dim_correlations(emb_df: pd.DataFrame, resid_df: pd.DataFrame) -> np.ndarray:
    merged = emb_df.merge(resid_df, on=["cell_id", "cycle", "seg_name"], how="inner")
    emb_cols = [c for c in emb_df.columns if c.startswith("e")]
    E = merged[emb_cols].values.astype(np.float64)   # (N, D)
    r = merged["residual"].values.astype(np.float64)  # (N,)
    E_c = E - E.mean(axis=0, keepdims=True)
    r_c = r - r.mean()
    cov = (E_c * r_c[:, None]).mean(axis=0)
    denom = E.std(axis=0) * r.std() + 1e-12
    corr = cov / denom
    print(f"  merged n={len(merged):,}/{len(emb_df):,}  |corr| mean={np.abs(corr).mean():.4f}  "
          f"max={np.abs(corr).max():.4f}")
    return corr


# =============================================================================
# 분석 2 — (a) vs (b) 코사인 유사도 / 분산 비교
# =============================================================================

def _compare_ab(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    merged = df_a.merge(df_b, on=["cell_id", "cycle", "seg_name"], suffixes=("_a", "_b"))
    cols_a = [c for c in df_a.columns if c.startswith("e")]
    cols_b = [c for c in df_b.columns if c.startswith("e")]
    A = merged[[f"{c}_a" for c in cols_a]].values.astype(np.float64)
    B = merged[[f"{c}_b" for c in cols_b]].values.astype(np.float64)
    cos = (A * B).sum(axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    var_a = A.var(axis=0)
    var_b = B.var(axis=0)
    print(f"  n_matched={len(merged):,}  cosine sim mean={cos.mean():.4f} std={cos.std():.4f}  "
          f"var(a) mean={var_a.mean():.4f}  var(b) mean={var_b.mean():.4f}")
    return {
        "n_matched": len(merged),
        "cos_mean": float(cos.mean()), "cos_std": float(cos.std()),
        "var_a_mean": float(var_a.mean()), "var_b_mean": float(var_b.mean()),
        "var_ratio_a_over_b": float(var_a.mean() / (var_b.mean() + 1e-12)),
    }


# =============================================================================
# 분석 3 — PCA 2D 시각화
# =============================================================================

def _pca_plot(emb_df: pd.DataFrame, resid_df: pd.DataFrame, title: str, out_path: Path,
              n_sample: int = 3000) -> None:
    from sklearn.decomposition import PCA

    merged = emb_df.merge(resid_df, on=["cell_id", "cycle", "seg_name"], how="inner")
    emb_cols = [c for c in emb_df.columns if c.startswith("e")]
    if len(merged) > n_sample:
        idx = RNG.choice(len(merged), size=n_sample, replace=False)
        merged = merged.iloc[idx].reset_index(drop=True)
    E = merged[emb_cols].values.astype(np.float64)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(E)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    seg_names = sorted(merged["seg_name"].unique())
    cmap = plt.get_cmap("tab10")
    for i, sname in enumerate(seg_names):
        sel = merged["seg_name"] == sname
        axes[0].scatter(Z[sel, 0], Z[sel, 1], s=4, alpha=0.5, color=cmap(i % 10), label=sname)
    axes[0].legend(fontsize=7, markerscale=2, loc="best")
    axes[0].set_title("시나리오별 색칠", fontsize=10)

    sc = axes[1].scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.6, c=merged["residual"].values,
                          cmap="RdBu_r", vmin=-np.abs(merged["residual"]).quantile(0.95),
                          vmax=np.abs(merged["residual"]).quantile(0.95))
    plt.colorbar(sc, ax=axes[1], label="residual (soh_true - baseline_pred)")
    axes[1].set_title("residual 크기별 색칠", fontsize=10)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 → {out_path}")


# =============================================================================
# main
# =============================================================================

def main() -> None:
    summary_rows = []

    for combo in COMBOS:
        name = combo["name"]
        print(f"\n{'='*70}\n  {name}\n{'='*70}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        resid_df = _load_baseline_residual(combo["baseline"])

        emb = {}
        for kind in ("a", "b", "flat"):
            emb[kind] = _extract_embeddings(combo[kind], kind)

        print("[분석1] 차원별 상관관계 (embedding vs baseline residual)")
        corr_summary = {}
        for kind in ("a", "b", "flat"):
            print(f" - {kind}:")
            corr = _dim_correlations(emb[kind], resid_df)
            corr_summary[kind] = corr
            # 상관계수 분포 히스토그램
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(corr, bins=30, color="steelblue", alpha=0.85)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{name} — {kind} 임베딩 차원별 residual 상관계수\n"
                         f"mean|corr|={np.abs(corr).mean():.4f}", fontsize=9)
            ax.set_xlabel("Pearson corr(embedding_dim, residual)")
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{name}_corr_hist_{kind}.png", dpi=140)
            plt.close(fig)

        print("[분석2] (a) vs (b) 코사인 유사도 / 분산 비교")
        ab_stats = _compare_ab(emb["a"], emb["b"])

        print("[분석3] PCA 2D 시각화")
        for kind in ("a", "b", "flat"):
            _pca_plot(emb[kind], resid_df, f"{name} — {kind} 임베딩 PCA",
                      OUT_DIR / f"{name}_pca_{kind}.png")

        summary_rows.append({
            "combo": name,
            "mean_abs_corr_a": float(np.abs(corr_summary["a"]).mean()),
            "mean_abs_corr_b": float(np.abs(corr_summary["b"]).mean()),
            "mean_abs_corr_flat": float(np.abs(corr_summary["flat"]).mean()),
            "max_abs_corr_a": float(np.abs(corr_summary["a"]).max()),
            "max_abs_corr_b": float(np.abs(corr_summary["b"]).max()),
            "max_abs_corr_flat": float(np.abs(corr_summary["flat"]).max()),
            "ab_cosine_mean": ab_stats["cos_mean"],
            "ab_cosine_std": ab_stats["cos_std"],
            "ab_var_a_mean": ab_stats["var_a_mean"],
            "ab_var_b_mean": ab_stats["var_b_mean"],
            "ab_var_ratio_a_over_b": ab_stats["var_ratio_a_over_b"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*70}\n요약 저장 → {summary_path}\n{'='*70}")
    print(summary_df.to_string(index=False))

    # 요약 막대그래프 — combo별 mean|corr| 3방식 비교
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(summary_df))
    w = 0.25
    ax.bar(x - w, summary_df["mean_abs_corr_a"], width=w, label="(a) frozen", color="#d62728")
    ax.bar(x,     summary_df["mean_abs_corr_b"], width=w, label="(b) separated", color="#2ca02c")
    ax.bar(x + w, summary_df["mean_abs_corr_flat"], width=w, label="방안1 flat", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["combo"], rotation=20, ha="right")
    ax.set_ylabel("mean |corr(embedding_dim, baseline residual)|")
    ax.set_title("임베딩이 baseline 대비 residual을 얼마나 설명하는가 (조합별)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_mean_abs_corr.png", dpi=150)
    plt.close(fig)
    print(f"저장 → {OUT_DIR / 'summary_mean_abs_corr.png'}")


if __name__ == "__main__":
    main()
