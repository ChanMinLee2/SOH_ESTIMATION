"""
plot_scenario_verification_comparison.py — 검증 시나리오(baseline/구(a)/구(b)/방안1/
신규(a))별 SOH 예측 곡선을, 세그멘테이션 시나리오 축(chg_lo/mid/hi, dis_hi/mid/lo)
단위를 유지한 채 비교한다.

`docs/260804_RESULTS.md` §1/§6이 요약 지표(R²/RMSE/MAPE)로 5개 검증 시나리오를
비교했다면, 이 스크립트는 그 이면의 실제 예측 곡선(대표 셀 1개, 사이클에 따른
Capacity[Ah])을 시나리오별로 쪼개서 눈으로 직접 비교하게 해준다.

기존 `visualize_results.py`의 `RunBundle`(predictions.csv만 읽고 모델은 로딩하지
않음)과 `_plot_capacity_curve_comparison`(행=시나리오, 열=run 그리드)을 그대로
재사용한다 — 모델을 안 띄우므로 baseline(HI만)과 신규 CNN(3채널, 3D 출력)처럼
아키텍처가 서로 다른 run들을 섞어도 안전하다.

검증 대상 5개 조합(qfw+mlp/transformer/resnet_tab, vqslope+mlp, q_abs+mlp) 각각에
대해 별도 출력 폴더를 만들고, 그 안에 대표 셀마다 1장씩
"capacity_curve_verification_{cell}.png"(행=6개 시나리오, 열=5개 검증방식)를 저장한다.

RUN_TABLE의 각 항목은 `docs/260801_RESULTS.md` §0 / `docs/260804_RESULTS.md` §0,§6의
run_dir 표를 그대로 옮긴 것이다 — baseline 5개 전부 §6.1에서 지적한
scen_k_count=55 문제를 basefix(k=15 통제 재학습)로 해결한 최종 run_dir을 쓴다.

사용:
  python 5_model/plot_scenario_verification_comparison.py
  python 5_model/plot_scenario_verification_comparison.py --combos qfw_mlp vqslope_mlp
  python 5_model/plot_scenario_verification_comparison.py --n-cells 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualize_results import (          # noqa: E402  (경로 설정 뒤 import)
    RunBundle,
    _load_bundles,
    _check_cross_axis,
    _pick_rep_cells,
    _plot_capacity_curve_comparison,
    _MIT_CELL_RE,
    _HUST_CELL_RE,
)

# ─────────────────────────────────────────────────────────────────────────────
# 검증 대상 5개 조합 × 검증 시나리오 5개(baseline/구a/구b/방안1/신규a) run_dir 표
# 열 순서 = 딕셔너리 삽입 순서 그대로 플롯 열 순서가 된다.
# ─────────────────────────────────────────────────────────────────────────────
RUN_TABLE: dict[str, dict[str, str]] = {
    "qfw_mlp": {
        "baseline": "_5_data_model_scr/0804_1001_p2_mlp_qfw_35%_20%",   # basefix(k=15,lr=5e-4) 완료
        "구(a)":    "_5_data_model_scr/0731_1039_p2_mlp_qfw_35%_20%",
        "구(b)":    "_5_data_model_scr/0731_1103_p2_mlp_qfw_35%_20%",
        "방안1":    "_5_data_model_scr/0801_1524_p2_mlp_qfw_35%_20%",
        "신규(a)":  "_5_data_model_scr/0804_0303_p2_mlp_qfw_35%_20%",
    },
    "qfw_transformer": {
        "baseline": "_5_data_model_scr/0804_1024_p2_tr_qfw_35%_20%",    # basefix(k=15) 완료
        "구(a)":    "_5_data_model_scr/0803_1240_p2_tr_qfw_35%_20%",
        "구(b)":    "_5_data_model_scr/0731_1216_p2_tr_qfw_35%_20%",
        "방안1":    "_5_data_model_scr/0801_1543_p2_tr_qfw_35%_20%",
        "신규(a)":  "_5_data_model_scr/0804_0331_p2_tr_qfw_35%_20%",
    },
    "qfw_resnet_tab": {
        "baseline": "_5_data_model_scr/0804_1102_p2_res_qfw_35%_20%",   # basefix(k=15) 완료
        "구(a)":    "_5_data_model_scr/0803_1319_p2_res_qfw_35%_20%",
        "구(b)":    "_5_data_model_scr/0731_1320_p2_res_qfw_35%_20%",
        "방안1":    "_5_data_model_scr/0801_1612_p2_res_qfw_35%_20%",
        "신규(a)":  "_5_data_model_scr/0804_0429_p2_res_qfw_35%_20%",
    },
    "vqslope_mlp": {
        "baseline": "_5_data_model_scr/0804_1148_p2_mlp_vqs_dva",       # basefix(k=15) 완료
        "구(a)":    "_5_data_model_scr/0731_1349_p2_mlp_vqs_dva",
        "구(b)":    "_5_data_model_scr/0731_1403_p2_mlp_vqs_dva",
        "방안1":    "_5_data_model_scr/0801_1643_p2_mlp_vqs_dva",
        "신규(a)":  "_5_data_model_scr/0804_0500_p2_mlp_vqs_dva",
    },
    "qabs_mlp": {
        "baseline": "_5_data_model_scr/0729_1908_p2_mlp_qabs_20-50%",   # 원래부터 k=15 — basefix 불필요
        "구(a)":    "_5_data_model_scr/0731_0049_p2_mlp_qabs_20-50%",
        "구(b)":    "_5_data_model_scr/0731_0339_p2_mlp_qabs_20-50%",
        "방안1":    "_5_data_model_scr/0801_1701_p2_mlp_qabs_20-50%",
        "신규(a)":  "_5_data_model_scr/0804_0546_p2_mlp_qabs_20-50%",
    },
}

OUT_ROOT = PROJECT_ROOT / "_5_data_model_scr" / "comparison" / "scenario_verification"

# ─────────────────────────────────────────────────────────────────────────────
# chg_lo/chg_hi 라벨 스왑 보정 — SOC 정합성 버그(2026-07-30 수정, q_frac_wide.py/
# vqslope.py 커밋 a9c410b)가 코드에는 반영됐지만, 그 코드로 실제 HI pkl이 다시
# 뽑힌 건 2026-08-04 CNN raw curve 작업(01:20/02:10) 때가 처음이었다. 그 사이
# (7/30 코드 수정 ~ 8/4 데이터 재추출) 학습된 구(a)/구(b)/방안1 run들은 여전히
# 버그 있는 pkl(충전 chg_lo↔chg_hi 이름이 반대)로 학습됐을 것으로 추정된다.
#
# RMSE로 직접 검증: baseline(8/4 재추출 이후 재학습된 basefix)과 신규(a)(8/4
# 이후 학습) 둘 다 "chg_hi RMSE < chg_lo RMSE"로 일관되는데, qfw_mlp/transformer/
# resnet_tab·vqslope_mlp 4개 조합의 구(a)/구(b)/방안1 12개 run 전부 정반대
# ("chg_lo RMSE < chg_hi RMSE")로 나타나 — 라벨이 뒤바뀐 상태로 통계적으로
# 뚜렷하게(12/12) 확인됨. 그래서 이 4개 조합의 구(a)/구(b)/방안1에 한해 플롯
# 직전 chg_lo↔chg_hi seg_name 을 서로 바꿔친다.
#
# q_abs(discharge dis_hi/dis_lo)는 같은 방식으로 재검증한 결과 baseline/구(b)/
# 방안1은 오히려 신규(a)와 같은(정상) 패턴을 보였고 구(a) 하나만 반대로 나와 —
# 표본이 약하고(1/3) 원인이 불분명해 여기서는 스왑을 적용하지 않는다.
_CHG_SWAP_AFFECTED: set[tuple[str, str]] = {
    (combo, label)
    for combo in ("qfw_mlp", "qfw_transformer", "qfw_resnet_tab", "vqslope_mlp")
    for label in ("구(a)", "구(b)", "방안1")
}


def _swap_chg_lo_hi(bundle: "RunBundle") -> None:
    """bundle.pred_rows 의 seg_name 에서 'chg_lo'<->'chg_hi' 를 맞바꾼다(제자리)."""
    _swap = {"chg_lo": "chg_hi", "chg_hi": "chg_lo"}
    for row in bundle.pred_rows:
        sn = row.get("seg_name")
        if sn in _swap:
            row["seg_name"] = _swap[sn]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combos", nargs="+", default=None, choices=list(RUN_TABLE.keys()),
                   help="비교할 조합만 지정 (기본: RUN_TABLE 전체 5개)")
    p.add_argument("--n-cells", type=int, default=5,
                   help="MIT/HUST 각각에서 뽑을 대표 셀 수 (기본 5 → 조합당 최대 10장)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    combos = args.combos or list(RUN_TABLE.keys())

    for combo in combos:
        scen_map = RUN_TABLE[combo]
        labels = list(scen_map.keys())
        run_dirs = list(scen_map.values())

        print(f"\n=== [{combo}] 검증 시나리오 {len(labels)}개 로딩: {labels} ===")
        try:
            bundles: list[RunBundle] = _load_bundles(run_dirs, labels)
        except FileNotFoundError as e:
            print(f"[skip] {combo}: {e}")
            continue

        for label, bundle in zip(labels, bundles):
            if (combo, label) in _CHG_SWAP_AFFECTED:
                _swap_chg_lo_hi(bundle)
                print(f"[fix] {combo}/{label}: chg_lo<->chg_hi seg_name 스왑 적용 "
                      f"(SOC 정합성 버그 — docs 참조)")

        cross_axis = _check_cross_axis(bundles)
        if cross_axis:
            print(f"[warn] {combo}: 검증 시나리오 간 축이 서로 다름 감지 — "
                  f"시나리오별 행 대신 평균 1행으로 축소됩니다(원래는 같은 축이어야 함, 확인 필요).")

        out_dir = OUT_ROOT / combo
        out_dir.mkdir(parents=True, exist_ok=True)

        cells: list[str] = []
        for pattern in (_MIT_CELL_RE, _HUST_CELL_RE):
            cells += _pick_rep_cells(bundles, pattern, n=args.n_cells)
        if not cells:
            print(f"[skip] {combo}: 5개 검증 시나리오 전부에 공통으로 존재하는 대표 셀 없음")
            continue
        print(f"[{combo}] 대표 셀: {cells}")

        for cell in cells:
            _plot_capacity_curve_comparison(
                bundles, cell, out_dir / f"capacity_curve_verification_{cell}.png",
                cross_axis=cross_axis,
            )

    print(f"\n[완료] 결과 위치: {OUT_ROOT}")


if __name__ == "__main__":
    main()
