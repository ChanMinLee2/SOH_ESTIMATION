"""
5_model/experiments/phase1_lab/plot_synergy_levels.py

시너지 정량화 Level1(선형)/Level2(비선형)/Level3(순수 이득, v-ctrl 대조군) 3개 패널을
한 그림에 그린다. phase1_trainer_v2.py와 분리된 독립 스크립트 — 이미 저장된 산출물만
읽는다(학습 없음).

Level1: synergy_groups json의 group_scores(그룹에 "추가된" 멤버의 편상관계수) 분포.
Level2: kernel_group_features pkl의 그룹별 train_r2 vs 같은 멤버로 재적합한 선형 R².
Level3: --results로 준 {버전 라벨: val_r2} 목록을 막대로 비교(v-ctrl 대조군용) —
  실제 학습 결과가 아직 없으면 이 패널은 "데이터 없음" 안내만 표시.

사용 예(Level2를 쓰려면 --model-config는 반드시 지정 — 이게 Level2 활성화 스위치.
--seg-axis/--axis-config/--data-dir/--seg-data-dir은 표준 조합이면 생략 가능):
  python 5_model/experiments/phase1_lab/plot_synergy_levels.py \
      --synergy-groups 5_model/experiments/phase1_lab/results/synergy_groups_k25_full_N2_groups_noleak.json \
      --model-config 5_model/config/main_qfref_S.yaml --split-seed 42 \
      --kernel-pkl 5_model/experiments/phase1_lab/results/kernel_group_features_k25_full_N2_kernel_v2.pkl \
      --results "v2=0.9478" --results "v-ctrl=0.91" --results "v3=(미실시)"
"""

from __future__ import annotations

import argparse
import json
import pickle
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

# 루트는 data_directories.py의 DATA_4_HI_ROOT_STR에서 가져온다(build_synergy_groups.py/
# lambda_sweep.py와 동일 이유 — PC마다 실제 드라이브가 다름). model-config/seg-axis/
# axis-config는 실험마다 바뀌므로 여전히 None 기본값 유지(Level2 활성화 여부 판단에 씀).
from data_directories import DATA_4_HI_ROOT_STR  # noqa: E402

_DATA_ROOT = f"{DATA_4_HI_ROOT_STR}/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

# seg-axis/axis-config는 이 세션 전체에서 한 번도 안 바뀐 고정값이라 기본값을 준다(다른
# 조합이면 오버라이드). model-config는 여전히 None으로 남겨 Level2 활성화 스위치로 쓴다
# (--model-config를 안 주면 Level2 패널 생략 — _level2_gaps의 존재 여부 검사 참고).
DEFAULT_SEG_AXIS = "q_frac_ref"
DEFAULT_AXIS_CONFIG = json.dumps({
    "n1": 0.35, "n2": 0.20, "ref_lag": 0, "noise_amp": 0.03,
    "noise_mode": "ou", "noise_period_cycles": 200, "n_samples": 2,
})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="시너지 Level1/2/3 통합 시각화")
    p.add_argument("--synergy-groups", required=True, dest="synergy_groups",
                    help="Level1용 — build_synergy_groups.py 산출물")
    p.add_argument("--min-effect-size", type=float, default=0.1, dest="min_effect_size",
                    help="Level1 '진짜 시너지' 문턱(Cohen 작음, 기본 0.1)")
    p.add_argument("--kernel-pkl", default=None, dest="kernel_pkl",
                    help="Level2용 — build_kernel_group_features.py 산출물(선택, 없으면 패널 생략)")
    p.add_argument("--model-config", default=None,
                    help="Level2 활성화 스위치 — 주면 Level2 계산, 안 주면 생략")
    p.add_argument("--seg-axis", default=DEFAULT_SEG_AXIS)
    p.add_argument("--axis-config", default=DEFAULT_AXIS_CONFIG)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR)
    p.add_argument("--datasets", nargs="+", default=["MIT", "HUST"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--min-gap", type=float, default=0.02, dest="min_gap",
                    help="Level2 '실질적 gap' 문턱(Cohen 작음 R² 증분, 기본 0.02)")
    p.add_argument("--results", action="append", default=[],
                    help="Level3용 — label=val_r2 형식(반복 지정), 예: --results v2=0.9478 "
                         "--results v-ctrl=0.91. 숫자로 안 읽히면 '데이터 없음'으로 표시.")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _level1_scores(groups_data: dict) -> list[float]:
    scores = []
    for k in groups_data:
        if k.endswith("_group_scores"):
            for g in groups_data[k]:
                scores.extend(abs(s) for s in g[1:])  # scores[0]=시드 단순상관, 제외
    return scores


def _level2_gap_for_pkl(kernel_pkl: str, x_all, y_all, seg_idx_all, spec) -> list[tuple]:
    """이미 로드된 (x_all, y_all, seg_idx_all, spec)로 kernel_pkl 하나의 Level2 gap을 계산.
    데이터 로딩(_load_train_split, ~1~2분)을 pkl 개수만큼 반복하지 않도록 _level2_gaps에서
    분리해뒀다 — plot_level2_gap_comparison.py(v1/v2/v3 여러 pkl 비교)가 이 함수를 재사용
    (중복 구현 금지)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    artifact = pickle.load(open(kernel_pkl, "rb"))
    out = []
    for f in artifact["features"]:
        scen_idx = f["scenario"] if isinstance(f["scenario"], int) else spec.scenario_names.index(f["scenario"])
        mask = seg_idx_all == scen_idx
        x_group = x_all[mask][:, f["members"]]
        y = y_all[mask]
        lin = LinearRegression().fit(x_group, y)
        lin_r2 = r2_score(y, lin.predict(x_group))
        out.append((f["name"], f["train_r2"], lin_r2, len(f["members"]), f["scenario"]))
    return out


def _level2_gaps(args) -> list[tuple] | None:
    if not args.kernel_pkl:
        return None
    if not (args.model_config and args.seg_axis and args.axis_config
            and args.data_dir and args.seg_data_dir):
        print("[plot] Level2: --kernel-pkl은 줬지만 데이터 로딩 인자가 부족해 생략합니다 "
              "(--model-config/--seg-axis/--axis-config/--data-dir/--seg-data-dir 필요).")
        return None
    from build_kernel_group_features import _load_train_split

    x_all, y_all, seg_idx_all, spec, _ = _load_train_split(args)
    return _level2_gap_for_pkl(args.kernel_pkl, x_all, y_all, seg_idx_all, spec)


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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

    # ---- Level 1 ----
    groups_data = json.loads(Path(args.synergy_groups).read_text(encoding="utf-8"))
    scores = _level1_scores(groups_data)
    ax1.hist(scores, bins=30, range=(0, 1), color="#2166ac", alpha=0.85)
    ax1.axvline(args.min_effect_size, color="black", linestyle="--", linewidth=1,
                label=f"'진짜 시너지' 문턱({args.min_effect_size})")
    n_pass = sum(1 for s in scores if s >= args.min_effect_size)
    ax1.set_title(f"Level 1(선형) — |편상관| 분포\n{n_pass}/{len(scores)}개 멤버가 문턱 통과")
    ax1.set_xlabel("|편상관계수|(그룹에 추가된 멤버)")
    ax1.set_ylabel("멤버 수")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- Level 2 ----
    l2 = _level2_gaps(args)
    if l2:
        lin_r2s = [x[2] for x in l2]
        ker_r2s = [x[1] for x in l2]
        gaps = [k - l for l, k in zip(lin_r2s, ker_r2s)]
        colors = ["#2166ac" if g >= args.min_gap else "#b2182b" for g in gaps]
        ax2.scatter(lin_r2s, ker_r2s, c=colors, alpha=0.7, s=25)
        lo, hi = 0, 1
        ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x(gap=0)")
        n_pass2 = sum(1 for g in gaps if g >= args.min_gap)
        ax2.set_title(f"Level 2(비선형) — 커널R² vs 선형R²\n{n_pass2}/{len(l2)}개 그룹이 "
                       f"gap≥{args.min_gap} 통과(파랑)")
        ax2.set_xlabel("선형(OLS) R²(같은 멤버)")
        ax2.set_ylabel("커널(Nystroem+Ridge) R²")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Level 2: --kernel-pkl 미지정\n(데이터 없음)",
                  ha="center", va="center", fontsize=11)
        ax2.set_xticks([]); ax2.set_yticks([])

    # ---- Level 3 ----
    parsed = []
    for item in args.results:
        label, _, val = item.partition("=")
        try:
            parsed.append((label, float(val)))
        except ValueError:
            parsed.append((label, None))
    if parsed:
        labels = [p[0] for p in parsed]
        vals = [p[1] if p[1] is not None else 0.0 for p in parsed]
        has_data = [p[1] is not None for p in parsed]
        colors3 = ["#2166ac" if h else "#cccccc" for h in has_data]
        ax3.bar(labels, vals, color=colors3, alpha=0.85)
        for i, (v, h) in enumerate(zip(vals, has_data)):
            ax3.text(i, v + 0.01, f"{v:.4f}" if h else "미실시", ha="center", fontsize=8)
        ax3.set_title("Level 3(순수 이득) — v-ctrl 대조군 비교")
        ax3.set_ylabel("val R²")
        ax3.grid(True, axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Level 3: --results 미지정\n(v-ctrl 학습 결과 나오면 채울 것)",
                  ha="center", va="center", fontsize=11)
        ax3.set_xticks([]); ax3.set_yticks([])

    fig.suptitle("시너지 정량화 Level 1/2/3", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.synergy_groups).parent
    out_path = out_dir / "synergy_levels.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")


if __name__ == "__main__":
    main()
