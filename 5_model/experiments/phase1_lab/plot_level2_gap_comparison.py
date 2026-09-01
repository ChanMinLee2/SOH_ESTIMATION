"""
5_model/experiments/phase1_lab/plot_level2_gap_comparison.py

v1/v2/v3(또는 임의 개수) kernel_group_features_*.pkl의 Level2 gap(커널R²−같은 멤버의
선형R²)을 나란히 비교한다. plot_synergy_levels.py의 Level2 패널이 "버전 하나"만 보여주는
것과 달리, 이 스크립트는 "버전 간 비교 + 왜 달라졌는지"를 겨냥한 2패널 그림 하나를 그린다.

배경: v3(전역 다중공선성 제거) 첫 재비교에서 평균 gap이 v1(0.304)/v2(0.263)보다 v3(0.213)가
낮게 나왔는데, 동시에 v3의 평균 선형R²(0.150)도 v2(0.124)보다 뚜렷이 올라 있었다 —
당시 build_kernel_group_features.py가 연결요소 dedup의 "사후 편입(attached)" HI까지
커널 입력에 다 넣어서, 그룹 크기가 2~29개로 들쭉날쭉해진 게 원인이라는 가설이 나왔다.
실측(이 스크립트로 확인)으로 확인 후 attached를 커널 입력에서 빼도록 되돌렸다(다중공선성
배제 라벨로만 남김) — 이후 재실행하면 n_members가 v1/v2와 동일하게 1~4로 균일해지고,
v3 선형R²도 v2와 비슷한 수준(0.123)으로 돌아온다. 좌 패널은 "달라졌다"를, 우 패널은
"그룹 크기가 교란변수가 아님"을 같은 그림에서 확인한다.

좌(분포 비교): 버전별 gap 분포를 박스플롯+지터 산점(개별 피처)으로 — 평균만이 아니라
전체 분포 이동을 본다.
우(원인 규명): x=커널 입력 raw HI 개수(n_members), y=gap, 색=버전 — 그룹이 커질수록
gap이 기계적으로 줄어드는지 직접 확인.

phase1_trainer_v2.py와 분리된 독립 스크립트 — kernel pkl들과 raw 데이터만 읽고 학습은 안
한다. 데이터는 한 번만 로드해서 모든 pkl에 재사용한다(pkl마다 재로딩하면 ~1~2분씩 낭비).

사용 예(--seg-axis/--axis-config/--data-dir/--seg-data-dir은 표준 조합이면 생략 가능 —
기본값 자동 적용):
  python 5_model/experiments/phase1_lab/plot_level2_gap_comparison.py \
      --model-config 5_model/config/main_qfref_S.yaml --split-seed 42 \
      --kernel-pkl v1=5_model/experiments/phase1_lab/results/outputs/kernel_group_features_k25_full_N2_kernel.pkl \
      --kernel-pkl v2=5_model/experiments/phase1_lab/results/outputs/kernel_group_features_k25_full_N2_kernel_v2.pkl \
      --kernel-pkl v3=5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v3.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from plot_synergy_levels import _level2_gap_for_pkl  # noqa: E402 (중복 구현 금지 — 재사용)

# 루트는 data_directories.py의 DATA_4_HI_ROOT_STR에서 가져온다(build_synergy_groups.py/
# lambda_sweep.py와 동일 이유 — PC마다 실제 드라이브가 다름).
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

# seg-axis/axis-config도 이 세션 전체에서 한 번도 안 바뀐 고정값 — 위 데이터 경로와 세트로
# 묶인 값이라(다른 조합이면 데이터 경로도 같이 바뀌어야 함) 다른 조합을 쓰려면 셋 다
# 함께 오버라이드해야 한다.
DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Level2(커널R²−선형R² gap) 버전 간 비교 — 분포 + 그룹크기 원인 분석")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR)
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--kernel-pkl", action="append", required=True, dest="kernel_pkls",
                    help="label=path 형식(반복 지정), 예: --kernel-pkl v1=... --kernel-pkl v3=...")
    p.add_argument("--min-gap", type=float, default=0.02, dest="min_gap",
                    help="'실질적 gap' 문턱(Cohen 작음 R² 증분, 기본 0.02) — plot_synergy_levels.py와 동일 기준")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 미설치 - 종료")
        return
    for _font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = _font
            break
    plt.rcParams["axes.unicode_minus"] = False

    from build_kernel_group_features import _load_train_split

    x_all, y_all, seg_idx_all, spec, _ = _load_train_split(args)

    labels: list[str] = []
    by_label: dict[str, list[tuple]] = {}
    for item in args.kernel_pkls:
        label, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--kernel-pkl은 label=path 형식이어야 합니다: {item!r}")
        recs = _level2_gap_for_pkl(path, x_all, y_all, seg_idx_all, spec)
        by_label[label] = recs
        labels.append(label)
        gaps = np.array([r[1] - r[2] for r in recs])
        print(f"[plot] {label}: n={len(recs)} 평균kernelR2={np.mean([r[1] for r in recs]):.4f} "
              f"평균선형R2={np.mean([r[2] for r in recs]):.4f} 평균gap={gaps.mean():.4f} "
              f"gap>={args.min_gap} 통과율={100*(gaps >= args.min_gap).mean():.1f}%")

    fig, (ax_dist, ax_size) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.get_cmap("tab10")

    # ---- 좌: 분포 비교(박스플롯 + 지터 산점) ----
    box_data = [[r[1] - r[2] for r in by_label[lb]] for lb in labels]
    bp = ax_dist.boxplot(box_data, labels=labels, showmeans=True, widths=0.5,
                          patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors(i))
        box.set_alpha(0.35)
    rng = np.random.RandomState(0)
    for i, lb in enumerate(labels):
        gaps = box_data[i]
        jitter = rng.uniform(-0.12, 0.12, size=len(gaps))
        ax_dist.scatter(np.full(len(gaps), i + 1) + jitter, gaps, s=14, alpha=0.6,
                         color=colors(i), edgecolors="none", zorder=3)
    ax_dist.axhline(args.min_gap, color="black", linestyle="--", linewidth=1,
                     label=f"'실질적 gap' 문턱({args.min_gap})")
    ax_dist.set_ylabel("gap = 커널R² − 선형R²(같은 멤버)")
    ax_dist.set_title("Level2 gap 분포 비교(피처별 점 = 지터)")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, axis="y", alpha=0.3)

    # ---- 우: gap vs 그룹 크기(원인 규명) ----
    for i, lb in enumerate(labels):
        recs = by_label[lb]
        n_members = np.array([r[3] for r in recs])
        gaps = np.array([r[1] - r[2] for r in recs])
        ax_size.scatter(n_members, gaps, s=28, alpha=0.7, color=colors(i), label=lb)
        if len(set(n_members.tolist())) > 1:
            coef = np.polyfit(n_members, gaps, 1)
            xs = np.linspace(n_members.min(), n_members.max(), 50)
            ax_size.plot(xs, np.polyval(coef, xs), color=colors(i), linewidth=1.5, alpha=0.8)
    ax_size.axhline(args.min_gap, color="black", linestyle="--", linewidth=1)
    ax_size.set_xlabel("커널 입력 raw HI 개수(그룹 성장 멤버, v3의 사후귀속 attached는 "
                        "커널 입력에서 제외 — build_kernel_group_features.py 참고)")
    ax_size.set_ylabel("gap")
    ax_size.set_title("gap vs 그룹 크기 — 큰 그룹일수록 gap이 줄어드는가?")
    ax_size.legend(fontsize=8)
    ax_size.grid(True, alpha=0.3)

    fig.suptitle("Level2(비선형 시너지) gap 재비교 — 버전 간 분포 + 그룹크기 원인 분석",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else (_HERE / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"level2_gap_comparison_{'_'.join(labels)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    summary = {
        lb: {
            "n": len(by_label[lb]),
            "mean_kernel_r2": float(np.mean([r[1] for r in by_label[lb]])),
            "mean_linear_r2": float(np.mean([r[2] for r in by_label[lb]])),
            "mean_gap": float(np.mean([r[1] - r[2] for r in by_label[lb]])),
        }
        for lb in labels
    }
    summary_path = out_dir / f"level2_gap_comparison_{'_'.join(labels)}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[plot] 요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
