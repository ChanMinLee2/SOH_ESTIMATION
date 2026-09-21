"""
5_model/experiments/phase1_lab/plot_kernel_group_recipe.py

각 커널 HI가 "어떤 raw HI들을 어떤 순서로 편입시키며 자랐는지"를 시나리오별로 보여주는
플랏 — plot_hi_selection_matrix.py의 "구성비"(카테고리 % 스택바, 순서 정보 없음)와
plot_kernel_rejected.py(탈락한 후보만 다룸, 삭제됨)가 못 보여주는 두 정보를 채운다:
  1. 그룹이 자랄 때 raw HI가 편입된 순서(seed -> 2번째 -> 3번째 -> ...)
  2. 그 순서에서 각 raw HI가 실제로 기여한 점수 — seed는 타깃과의 단순 상관계수(부호 있음),
     이후 멤버는 그 시점 그룹 전체로 조건화한 편상관계수(build_synergy_groups.py._partial_corr)

두 산출물이 모두 필요하다:
  --synergy-groups-json : build_synergy_groups.py 산출물(seg_s_groups/group_names/
                           group_scores) — 성장 순서와 점수의 유일한 출처.
  --kernel-features-pkl : build_kernel_group_features.py 산출물(features[i].name이
                           "kernel_{seg}_g{gi}" 형식) — 그 그룹이 실제로 최종 커널 HI로
                           살아남았는지(다중공선성 배제/쿼터 통과)와 train_r2를 준다.
  gi가 두 파일을 잇는 열쇠다: candidates가 groups_data[f"seg_{s}_groups"]를
  enumerate(groups)로 그대로 순회하며 이름에 그 인덱스를 박아넣으므로
  (build_kernel_group_features.py), "kernel_{seg}_g3"는 항상
  synergy_groups.json의 seg_{s}_group_names[3]/group_scores[3]과 같은 그룹을 가리킨다.
  phase1_trainer_v2.py의 p1v2_summary.json에 남는 synergy_groups_json은 이 값이 아니다
  (그건 --kernel-features-pkl과 상호배타인 구버전 그룹-게이팅 경로 전용).
  2026-09-19부터 run_pipeline.py의 Step 6~9 산출물이 전부 results/p1v2_runs/
  <p1-tag>_seed<seed>/ 한 폴더에 모이므로, --run-dir로 그 폴더만 주면 두 파일을
  글롭(synergy_groups_*.json / kernel_group_features_*.pkl)으로 자동 찾는다.

사용 예:
  python 5_model/experiments/phase1_lab/plot_kernel_group_recipe.py \
      --run-dir 5_model/experiments/phase1_lab/results/p1v2_runs/p1v4_scen_lag1zone_seed42

  # 또는 두 파일을 직접 지정(다른 run의 파일을 섞어 쓰고 싶을 때):
  python 5_model/experiments/phase1_lab/plot_kernel_group_recipe.py \
      --synergy-groups-json .../synergy_groups_....json \
      --kernel-features-pkl .../kernel_group_features_....pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

_CATEGORY_COLORS = {
    "stat":  "#1f77b4",
    "diff":  "#ff7f0e",
    "lfp":   "#2ca02c",
    "morph": "#9467bd",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="커널 HI 그룹의 raw HI 편입 순서(+ 편입 당시 점수)를 시나리오별로 그림"
    )
    p.add_argument("--run-dir", default=None, dest="run_dir",
                    help="2026-09-19부터 run_pipeline.py가 Step 6~9 산출물을 전부 "
                         "results/p1v2_runs/<p1-tag>_seed<seed>/ 안에 모아두므로, 이 폴더 "
                         "하나만 주면 --synergy-groups-json/--kernel-features-pkl을 "
                         "각각 synergy_groups_*.json / kernel_group_features_*.pkl 글롭으로 "
                         "자동 찾는다(파일이 정확히 하나씩이어야 함). 둘 다 명시하면 이 값은 무시.")
    p.add_argument("--synergy-groups-json", default=None, dest="synergy_groups_json",
                    help="build_synergy_groups.py 산출물(seg_s_groups/group_names/group_scores). "
                         "--run-dir 없이 쓸 땐 필수.")
    p.add_argument("--kernel-features-pkl", default=None, dest="kernel_features_pkl",
                    help="build_kernel_group_features.py 산출물 — 최종 커널 HI 목록(name에 "
                         "'g{gi}'로 그룹 인덱스가 박혀 있음)과 train_r2 정렬 기준. "
                         "--run-dir 없이 쓸 땐 필수.")
    p.add_argument("--sort", choices=["r2", "seed"], default="r2",
                    help="행(커널 HI) 정렬: r2=train_r2 내림차순(기본), "
                         "seed=시너지 json의 시드 강도 순(=원래 그룹 인덱스 순)")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                    help="기본: --kernel-features-pkl과 같은 디렉터리")
    args = p.parse_args()

    if args.run_dir and not (args.synergy_groups_json and args.kernel_features_pkl):
        run_dir = Path(args.run_dir)
        if not args.synergy_groups_json:
            hits = sorted(run_dir.glob("synergy_groups_*.json"))
            if len(hits) != 1:
                p.error(f"--run-dir에서 synergy_groups_*.json을 정확히 1개 찾아야 하는데 "
                        f"{len(hits)}개 발견({run_dir}) — --synergy-groups-json으로 직접 지정하세요.")
            args.synergy_groups_json = str(hits[0])
        if not args.kernel_features_pkl:
            hits = sorted(f for f in run_dir.glob("kernel_group_features_*.pkl"))
            if len(hits) != 1:
                p.error(f"--run-dir에서 kernel_group_features_*.pkl을 정확히 1개 찾아야 하는데 "
                        f"{len(hits)}개 발견({run_dir}) — --kernel-features-pkl로 직접 지정하세요.")
            args.kernel_features_pkl = str(hits[0])
    elif not (args.synergy_groups_json and args.kernel_features_pkl):
        p.error("--run-dir를 안 주면 --synergy-groups-json과 --kernel-features-pkl을 "
                "둘 다 직접 지정해야 합니다.")
    return args


def _strip_seg_suffix(name: str, seg_name: str) -> str:
    suffix = f"_{seg_name}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _category_of(base_name: str) -> str:
    return base_name.split("_", 1)[0] if "_" in base_name else base_name


def _load_synergy(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_kernel_features(path: Path) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            artifact = pickle.load(f)
    return artifact["features"]


def _gi_from_name(name: str) -> int | None:
    """'kernel_dis_hi_g3' -> 3. 이름 규칙이 바뀌면(build_kernel_group_features.py) 여기도 맞춰야 함."""
    if "_g" not in name:
        return None
    try:
        return int(name.rsplit("_g", 1)[1])
    except ValueError:
        return None


def _draw_recipe_panel(ax, rows: list[dict], seg_name: str, max_len: int):
    """rows[i] = {"kernel_name", "train_r2", "members": [(name, score), ...]}
    (members[0]의 score = 시드 단순상관, members[1:]의 score = 편입 당시 편상관)."""
    import matplotlib.patches as mpatches

    n = len(rows)
    step = 1.5     # 칩 중심 간 x 간격 — 긴 raw HI 이름이 칩 박스 밖으로 삐져나와도
                    # 옆 칩/R² 배지와 안 겹치게 여유를 둔다(칩 폭 자체보다 넉넉하게).
    chip_w = 1.32
    fontsize = 8.0 if n <= 12 else (6.6 if n <= 22 else 5.6)
    score_fs = max(fontsize - 1.5, 5.0)

    for row_i, row in enumerate(rows):
        y = n - 1 - row_i  # 위에서부터 그림(첫 행이 맨 위)
        members = row["members"]
        for pos, (name, score) in enumerate(members):
            x = pos * step
            cat = _category_of(name)
            color = _CATEGORY_COLORS.get(cat, "#888888")
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - chip_w / 2, y - 0.34), chip_w, 0.68,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=0.9, edgecolor=color, facecolor=color + "22",
                )
            )
            ax.text(x, y + 0.05, name, ha="center", va="center",
                     fontsize=fontsize, color="#1a1a1a", clip_on=False)
            score_lbl = f"seed r={score:+.2f}" if pos == 0 else f"pcorr={score:+.2f}"
            ax.text(x, y - 0.46, score_lbl, ha="center", va="center",
                     fontsize=score_fs, color=color)
            if pos > 0:
                ax.annotate("", xy=(x - chip_w / 2, y), xytext=(x - step + chip_w / 2, y),
                            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9))

        # 오른쪽 끝에 train_r2 배지 — 항상 max_len 칸 뒤 고정 위치라 칩 개수와 무관하게 안 겹침
        ax.text((max_len - 1) * step + chip_w / 2 + 0.3, y, f"R²={row['train_r2']:.2f}",
                 ha="left", va="center", fontsize=fontsize, color="#333333", fontweight="bold")

    ax.set_xlim(-chip_w, (max_len - 1) * step + chip_w * 2.4)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_yticks([])
    ax.set_xticks([i * step for i in range(max_len)])
    ax.set_xticklabels([f"편입 {i+1}" if i > 0 else "시드" for i in range(max_len)], fontsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"{seg_name}  (N={n})", fontsize=10.5, fontweight="bold")


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

    synergy_path = Path(args.synergy_groups_json)
    kernel_path = Path(args.kernel_features_pkl)
    if not synergy_path.exists():
        print(f"[plot] {synergy_path} 없음")
        return
    if not kernel_path.exists():
        print(f"[plot] {kernel_path} 없음")
        return

    synergy = _load_synergy(synergy_path)
    features = _load_kernel_features(kernel_path)

    n_scen = 0
    while f"seg_{n_scen}_seg_name" in synergy:
        n_scen += 1
    if n_scen == 0:
        print("[plot] synergy-groups-json에서 시나리오를 찾을 수 없습니다(seg_s_seg_name 없음)")
        return

    feats_by_scen: dict[int, list[dict]] = {s: [] for s in range(n_scen)}
    seg_name_to_idx = {synergy[f"seg_{s}_seg_name"]: s for s in range(n_scen)}
    n_unresolved = 0
    for feat in features:
        gi = _gi_from_name(feat["name"])
        s = seg_name_to_idx.get(feat["scenario"])
        if gi is None or s is None:
            n_unresolved += 1
            continue
        feats_by_scen[s].append({**feat, "_gi": gi})
    if n_unresolved:
        print(f"[plot] 경고: 이름에서 그룹 인덱스를 못 읽은 커널 {n_unresolved}개는 제외")

    panel_rows: dict[int, list[dict]] = {}
    max_len = 2
    for s in range(n_scen):
        seg_name = synergy[f"seg_{s}_seg_name"]
        group_names = synergy[f"seg_{s}_group_names"]
        group_scores = synergy[f"seg_{s}_group_scores"]
        rows = []
        for feat in feats_by_scen[s]:
            gi = feat["_gi"]
            if gi >= len(group_names):
                continue
            names = [_strip_seg_suffix(n, seg_name) for n in group_names[gi]]
            scores = group_scores[gi]
            rows.append({
                "kernel_name": feat["name"],
                "train_r2": feat["train_r2"],
                "members": list(zip(names, scores)),
                "_seed_rank": gi,
            })
            max_len = max(max_len, len(names))
        if args.sort == "r2":
            rows.sort(key=lambda r: -r["train_r2"])
        else:
            rows.sort(key=lambda r: r["_seed_rank"])
        panel_rows[s] = rows

    n_cols = 3 if n_scen > 2 else n_scen
    n_rows_grid = (n_scen + n_cols - 1) // n_cols
    max_rows_in_panel = max((len(r) for r in panel_rows.values()), default=1)
    fig_w = 4.6 * n_cols
    fig_h = max(3.0, max_rows_in_panel * 0.62) * n_rows_grid
    fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for s in range(n_scen):
        r, c = divmod(s, n_cols)
        seg_name = synergy[f"seg_{s}_seg_name"]
        rows = panel_rows[s]
        if not rows:
            axes[r][c].axis("off")
            axes[r][c].set_title(f"{seg_name}  (커널 없음)", fontsize=10.5)
            continue
        _draw_recipe_panel(axes[r][c], rows, seg_name, max_len)
    for s in range(n_scen, n_rows_grid * n_cols):
        r, c = divmod(s, n_cols)
        axes[r][c].axis("off")

    cat_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=col,
                               markersize=9, label=cat)
                   for cat, col in _CATEGORY_COLORS.items()]
    fig.legend(handles=cat_handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.01), title="raw HI 카테고리 (칩 색)")
    fig.suptitle(f"커널 HI 조합 — {kernel_path.stem}", fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path(args.out_dir) if args.out_dir else kernel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"kernel_group_recipe_{kernel_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 저장: {out_path}")

    print("\n[요약] 시나리오별 커널 HI 수 / 평균 그룹 크기:")
    for s in range(n_scen):
        rows = panel_rows[s]
        if not rows:
            print(f"  {synergy[f'seg_{s}_seg_name']:<8} 커널 0개")
            continue
        avg_size = sum(len(r["members"]) for r in rows) / len(rows)
        print(f"  {synergy[f'seg_{s}_seg_name']:<8} 커널 {len(rows):>3}개, 평균 멤버 {avg_size:.2f}개")


if __name__ == "__main__":
    main()
