"""Figure 3 -- 진단 플랏 3종을 1x3으로 합성(2026-10-12).

model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42/figures/ 안의 이미 렌더링된
PNG 3개를 그대로 가로로 이어붙인다(원본 플랏을 재생성하지 않음 -- 단순 이미지 합성).

  (a) scatter_test_hard.png                              -- pred vs true SOH 산점도
  (b) error_heatmap_capacity_cycle_pct_test_hard.png      -- |오차|(%) 히트맵, cycle x capacity
  (c) error_heatmap_capacity_scenario_pct_test_hard.png   -- |오차|(%) 히트맵, scenario x capacity

(c)는 사용자가 원래 붙여넣은 경로 중 (b)와 중복된 자리를 대신함 -- "시나리오별" 오차
분해가 논문의 시나리오 조건화 스토리와 바로 연결돼 (b)(cycle 기준)와 상호보완적이라
추정으로 골랐음(확인 필요하면 바꿀 것).

실행: python docs/figures/fig3_diagnostics.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "model_lib/results/p1v2_runs/1009_1342_p1v2_baseline_seed42/figures"

PANELS = [
    ("scatter_test_hard.png", "(a)"),
    ("error_heatmap_capacity_cycle_pct_test_hard.png", "(b)"),
    ("error_heatmap_capacity_scenario_pct_test_hard.png", "(c)"),
]


def build_composite() -> Image.Image:
    imgs = [Image.open(SRC_DIR / name).convert("RGB") for name, _ in PANELS]
    target_h = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.height != target_h:
            w = int(im.width * target_h / im.height)
            im = im.resize((w, target_h), Image.LANCZOS)
        resized.append(im)

    gap = 24
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, target_h), "white")
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


if __name__ == "__main__":
    canvas = build_composite()
    out_png = Path(__file__).resolve().parent / "fig3_diagnostics.png"
    canvas.save(out_png, dpi=(300, 300))
    print(f"saved: {out_png}")
    for name, letter in PANELS:
        print(f"  {letter} <- {name}")
