"""
1_convert/convert_calce.py — CALCE CS2/CX2(LCO) 데이터셋 전용 변환기.

MIT/HUST(convert_unified.py)와 완전히 분리된 파일 — 공용 유틸만 convert_unified에서
가져다 쓰고, CALCE 고유 파싱 로직은 전부 이 파일 안에 둔다.

출처: CALCE(Center for Advanced Life Cycle Engineering, Univ. of Maryland),
https://calce.umd.edu/battery-data — CS2(1.1Ah)/CX2(1.35Ah), 둘 다 LCO, 0.5C CC
4.2V까지 충전 후 CV(0.05A 컷오프), 2.7V 방전 컷오프. 인용:
  He, W., Williard, N., Osterman, M., Pecht, M. (2011). Journal of Power Sources,
  196(23), 10314-10321.

**셀 하나 = 폴더 하나(예: ncm/CS2_8/), 그 안에 시험 날짜별 파일이 여러 개**
(예: CS2_8_1_19_10.txt = 2010-01-19). 실측 확인 결과(2026-09-05) 파일 형식이
셀·날짜에 따라 둘로 갈린다:

  (A) Arbin 표준 export(.xlsx, "Channel_*" 시트) — CALCE_META.txt가 설명하는 포맷.
      컬럼: Cycle_Index, Test_Time(s), Current(A), Voltage(V), Discharge_Capacity(Ah), ...
      Current(A)는 이미 부호 있음(+충전/-방전). **Cycle_Index는 파일마다 1로 리셋된다**
      (실측 확인: CS2_33의 연속된 두 주간 파일이 둘 다 Cycle_Index 1~50).
      **Discharge_Capacity(Ah)도 사이클별로 안 리셋되고 파일(시험 세션) 전체에 걸쳐
      누적된다**(실측 확인: cycle=2의 min이 cycle=1의 max와 정확히 일치하는 식으로
      계속 이어짐) — CALCE_META.txt의 "사이클별 최댓값이 그 사이클의 실제 용량"은
      이 포맷에는 안 맞아, 사이클 내 (최댓값-최솟값)으로 그 사이클만의 방전량을
      따로 구한다.

  (B) 원시 채널 덤프(.txt, tab-구분) — CALCE_META.txt에 설명 없는, 별도로 발견한 포맷.
      컬럼: Time, Status code, Status category, ..., mV, mA, ..., Capacity, ...
      `Cycle_Index` 자체가 없다. 대신 mV/mA로 직접 전압/전류를 읽고, mA 부호(+충전/
      -방전, 실측 확인됨 — CS2_8 한 파일 안에서 충전 CC(mV 4005->4204, mA+550)
      다음 방전 CC(mV 3606->3587, mA-549)로 정확히 전환됨)로 phase를 판정한 뒤,
      "방전(또는 그 이전) -> 충전"으로 넘어가는 지점을 사이클 경계로 재구성한다
      (CALCE_META.txt 조언: "Step_Index 전이 패턴이나 전류 부호 변화로 사이클을
      재구성하는 경우가 많다"). 이 Capacity 컬럼은 사이클마다 0으로 리셋된다(실측
      확인, min=0). 단위는 문서화가 없는데, mAh(/1000)로 가정하면 CS2(정격 1.1Ah)
      최댓값이 항상 ~0.1Ah로 나와 정격의 9%밖에 안 돼 명백히 틀렸다 — /100(=0.01Ah
      단위로 가정)이면 ~1.0Ah로 정격과 합리적으로 맞아 이 값으로 정정했다(2026-09-05).
      공식 문서가 없는 추정치이므로 실제 학습 투입 전 재검증 권장.

      **재검증 결과(2026-09-05, 4_hi_analysis/hi_correlation.py Step4 통합 중 발견)**:
      순수 (B)-포맷 셀(CS2_8, CS2_21, CX2_31 — xlsx 전혀 없음)은 Step4 HI 추출의
      "완전 사이클" 게이트(사이클 내 전류·시간 적분으로 실측한 방전량 q_local ≥
      capacity_Ah × 0.30)를 셀당 100% 탈락한다. 실측: 이 3개 셀은 q_local/capacity_Ah
      비율이 항상 1.8~2.3% 수준 — capacity_Ah(raw Capacity/100)는 ~1.0Ah 근처로
      맞게 나오지만, (B)-포맷의 로컬 사이클 재구성(전류부호 전이 기반)이 실제로는
      완전한 충방전 사이클이 아니라 수 분짜리 조각을 "사이클 1개"로 인식하고 있다는
      뜻(예: CS2_8 cycle=1은 총 270초, 완전 사이클이라면 0.5C×1.1Ah 기준 ~2시간
      필요). xlsx가 섞인 셀(CX2_16, CX2_33)이나 순수 xlsx 셀은 이 문제가 없음 —
      Cycle_Index를 파일이 직접 제공하기 때문. 근본 수정에는 (B) 포맷의 공식 문서가
      필요해 보류 — 현재는 이 3개 셀이 Step4 HI 데이터셋에서 조용히 빠진 채
      13/16 셀로 진행 중(1_convert~3_integrity 산출물 자체는 정상, Step4 게이트에서만
      전량 탈락).

  같은 셀 폴더 안에 (A)/(B)가 섞여 있는 경우도 있다(CX2_16, CX2_33) — 파일명의
  날짜(예: "..._1_19_10.txt" = 2010-01-19)로 정렬해 시간순으로 잇고, 각 파일의
  "로컬 사이클 번호"에 누적 오프셋을 더해 셀 전체 기준 전역 사이클 번호를 만든다
  (두 포맷 다 로컬 사이클 번호는 항상 1부터 시작하므로 이 오프셋 로직 하나로 공통
  처리 가능).

사용:
  python convert_calce.py --workers 4
  python convert_calce.py --cell CS2_8
"""

from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from convert_unified import (  # noqa: E402
    PROJECT_ROOT, RAW_OUTPUT_ROOT, OUTPUT_ROOT,
    assign_phase, save_cell, _make_time_relative, _fix_time_monotonicity,
    _remove_zero_current_rest, _remove_outlier_cycles, _run,
)
import sys as _sys  # noqa: E402
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))
from data_directories import EXTERNAL_DATA_ROOT  # noqa: E402

# 2026-09-05: 디스크 공간 때문에 D 드라이브로 이전(data_directories.py의 기존
# _4_data_hi 관례와 동일 원칙 — 이 저장소가 그 유일한 진입점이므로 여기서도 재사용).
CALCE_ROOT = EXTERNAL_DATA_ROOT / "_0_data_raw" / "CALCE"
# 이 폴더 밑에는 CALCE 아닌 것(data_transfer.py, excel/ 등)도 섞여 있으므로 이 접두사만 취급.
_CELL_DIR_RE = re.compile(r"^(CS2|CX2)_\d+$")

NOMINAL_CAPACITY_AH = {"CS2": 1.1, "CX2": 1.35}  # calce.umd.edu/battery-data 명시값


def _cell_chemistry_group(cell_name: str) -> str:
    return "CS2" if cell_name.startswith("CS2") else "CX2"


def _calce_date_key(path: Path, cell_name: str) -> tuple[int, int, int]:
    """파일명 "{cell_name}_{M}_{D}_{YY}" -> (year, month, day) 정렬 키.

    .txt 원본 파일 일부가 셀 폴더명과 대소문자가 다르다(실측 확인: CX2_16 폴더의
    .xlsx는 "CX2_16_...", .txt는 "cx2_16_..." — 대문자/소문자 혼재). 대소문자
    무시하고 접두사를 벗겨낸다. 파일명 끝에 "_self discharge test" 같은 부가
    설명이 붙어도 앞의 월/일/연도 3토큰만 쓰므로 영향 없다.
    """
    stem = path.stem
    prefix = cell_name + "_"
    if stem.lower().startswith(prefix.lower()):
        remainder = stem[len(prefix):]
    else:
        remainder = stem
    parts = remainder.split("_")

    def _leading_int(token: str) -> int:
        # 실측 확인: "CS2_21_7_9b_10.txt"(day="9b"), "cx2_33_7_6_10second.txt"(yy="10second")
        # 처럼 오탈자/부가문자가 숫자 뒤에 바로 붙는 경우가 있음 -> 선행 숫자만 취함.
        m = re.match(r"\d+", token)
        if not m:
            raise ValueError(f"날짜 토큰에서 숫자를 찾을 수 없음: {token!r} (path={path})")
        return int(m.group())

    month, day, yy = _leading_int(parts[0]), _leading_int(parts[1]), _leading_int(parts[2])
    year = 2000 + yy if yy < 100 else yy
    return (year, month, day)


def _parse_calce_xlsx(path: Path) -> pd.DataFrame:
    """Arbin 표준 export. 로컬 cycle(그 파일 안 1부터 시작)로 반환 — 오프셋은 호출부 책임."""
    xl = pd.ExcelFile(path)
    sheet_name = next((s for s in xl.sheet_names if s.startswith("Channel")), None)
    if sheet_name is None:
        return pd.DataFrame()
    raw = xl.parse(sheet_name)
    if raw.empty or "Cycle_Index" not in raw.columns:
        return pd.DataFrame()

    cycle      = raw["Cycle_Index"].astype(float).round().astype(int).values
    time_s     = raw["Test_Time(s)"].values.astype(float)
    voltage_V  = raw["Voltage(V)"].values.astype(float)
    current_A  = raw["Current(A)"].values.astype(float)
    dis_cap_Ah = raw["Discharge_Capacity(Ah)"].values.astype(float)

    phase = assign_phase(current_A)
    df = pd.DataFrame({
        "cycle": cycle, "time_s": time_s, "voltage_V": voltage_V,
        "current_A": current_A, "phase": phase, "_dis_cap_Ah": dis_cap_Ah,
    })
    # Discharge_Capacity(Ah)는 사이클마다 리셋되는 게 아니라 "파일 전체(그 시험 세션)
    # 누적값"이다(실측 확인, 2026-09-05 — CS2_33_1_10_11.xlsx에서 cycle=2 min이
    # cycle=1 max와 정확히 일치, cycle=3 min이 cycle=2 max와 일치하는 식으로 계속
    # 누적됨). CALCE_META.txt의 "사이클별 최댓값이 그 사이클의 실제 용량"은 이
    # 파일들에는 안 맞는 설명 — 그 사이클 안에서의 최댓값-최솟값(=그 사이클 동안
    # 실제로 흘러나간 방전량)을 써야 한다.
    grp = df.groupby("cycle")["_dis_cap_Ah"]
    df["capacity_Ah"] = grp.transform("max") - grp.transform("min")
    return df.drop(columns=["_dis_cap_Ah"])


_MIN_LOCAL_CYCLE_ROWS = 100  # 실측 CS2_8 기준 진짜 사이클은 200~500행대, 노이즈성
                             # 스퓨리어스 사이클은 대부분 10행 미만(2026-09-05 실측)


def _merge_short_local_cycles(cycle_arr: np.ndarray, min_rows: int) -> np.ndarray:
    """행 수가 min_rows 미만인 로컬 사이클을 바로 앞 사이클로 병합.

    _parse_calce_txt의 전류부호 기반 사이클 재구성은 REST 구간의 노이즈가 순간적으로
    +0.01A(충전 판정 임계값)를 넘으면 그걸 새 사이클 시작으로 잘못 인식한다(실측
    확인: CS2_8에서 1410개 "사이클" 중 78개가 10행 미만 — 진짜 사이클은 200~500행대에
    몰려 있음). 이 함수가 그런 스퓨리어스 조각을 직전 사이클로 흡수해 없앤다. 맨 앞
    사이클이 너무 짧으면(병합할 "직전"이 없음) 그대로 둔다 — 이후 단계
    (_remove_outlier_cycles 등)에서 걸러질 수 있다.
    """
    if len(cycle_arr) == 0:
        return cycle_arr
    unique_cycles = sorted(set(cycle_arr.tolist()))
    counts = {c: int((cycle_arr == c).sum()) for c in unique_cycles}
    remap: dict[int, int] = {}
    prev_kept = None
    for c in unique_cycles:
        if counts[c] < min_rows and prev_kept is not None:
            remap[c] = prev_kept
        else:
            remap[c] = c
            prev_kept = c
    merged = np.array([remap[c] for c in cycle_arr])
    # 병합으로 생긴 번호 갭 없이 1부터 연속으로 재부여.
    renumber = {c: i + 1 for i, c in enumerate(sorted(set(merged.tolist())))}
    return np.array([renumber[c] for c in merged])


def _parse_calce_txt(path: Path) -> pd.DataFrame:
    """원시 채널 덤프(CALCE_META.txt에 없는 포맷). 로컬 cycle을 전류부호 전이로 재구성."""
    raw = pd.read_csv(path, sep="\t")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    if raw.empty or "mV" not in raw.columns or "mA" not in raw.columns:
        return pd.DataFrame()

    time_s    = raw["Time"].values.astype(float)
    voltage_V = raw["mV"].values.astype(float) / 1000.0
    current_A = raw["mA"].values.astype(float) / 1000.0
    cap_raw   = raw["Capacity"].values.astype(float) if "Capacity" in raw.columns else np.zeros(len(raw))

    phase = np.array(assign_phase(current_A))
    # 사이클 경계 = "충전이 아니던 상태 -> 충전"으로 전이하는 지점(모듈 docstring 참고).
    is_new_cycle = (phase == "charge") & (np.roll(phase, 1) != "charge")
    if len(phase) > 0:
        is_new_cycle[0] = (phase[0] == "charge")
    cycle_local = np.cumsum(is_new_cycle)
    cycle_local[cycle_local == 0] = 1  # 첫 충전 이전 행(있다면) cycle 1로 편입
    cycle_local = _merge_short_local_cycles(cycle_local, _MIN_LOCAL_CYCLE_ROWS)

    df = pd.DataFrame({
        "cycle": cycle_local, "time_s": time_s, "voltage_V": voltage_V,
        "current_A": current_A, "phase": phase, "_cap_raw": cap_raw,
    })
    # Capacity 컬럼 단위 미문서화. 최초엔 mAh(=/1000)로 가정했으나 실측 결과
    # CS2_8(정격 1.1Ah)의 사이클별 최댓값이 항상 ~100이라 /1000이면 0.1Ah로
    # 나와(정격의 9%) 명백히 틀렸다. /100(=0.01Ah 단위)으로 가정하면 ~1.0Ah로
    # 정격과 합리적으로 맞는다(2026-09-05 실측 재조정) — 다만 공식 문서가 없는
    # 추정치이므로 실제 학습 투입 전 재검증 권장(모듈 docstring 참고).
    df["capacity_Ah"] = df.groupby("cycle")["_cap_raw"].transform("max") / 100.0
    return df.drop(columns=["_cap_raw"])


def convert_calce_cell(cell_dir: Path, out_dir: Path, raw_out_dir: Path) -> dict:
    """변환 후 stats dict 반환. 스킵 시 빈 dict."""
    cell_name = cell_dir.name  # 예: "CS2_8"
    files = [p for p in cell_dir.iterdir() if p.suffix.lower() in (".txt", ".xlsx")]
    if not files:
        return {}
    files.sort(key=lambda p: _calce_date_key(p, cell_name))

    global_offset = 0
    parts: list[pd.DataFrame] = []
    n_files_txt = n_files_xlsx = n_files_failed = 0
    for f in files:
        try:
            if f.suffix.lower() == ".xlsx":
                df_local = _parse_calce_xlsx(f)
                n_files_xlsx += 1
            else:
                df_local = _parse_calce_txt(f)
                n_files_txt += 1
        except Exception:
            n_files_failed += 1
            print(f"  [경고] {f} 파싱 실패, 스킵:\n{traceback.format_exc()}")
            continue
        if df_local.empty:
            continue
        df_local = df_local.copy()
        df_local["cycle"] = df_local["cycle"] + global_offset
        global_offset = int(df_local["cycle"].max())
        parts.append(df_local)

    if not parts:
        return {}
    df = pd.concat(parts, ignore_index=True)

    # ── raw 저장 (이상치 제거 없음) ──────────────────────────────────────────
    save_cell(raw_out_dir, cell_name, {
        "cell_id": cell_name, "dataset": "CALCE",
        "chemistry_group": _cell_chemistry_group(cell_name),
        "nominal_capacity_Ah": NOMINAL_CAPACITY_AH[_cell_chemistry_group(cell_name)],
        "n_cycles": df["cycle"].nunique(),
        "n_files_txt": n_files_txt, "n_files_xlsx": n_files_xlsx,
    }, df)

    # ── 이상치 제거 후 _1_data_unified에 저장 (MIT/HUST와 동일 파이프라인 재사용) ──
    df = _make_time_relative(df)
    df = _fix_time_monotonicity(df)
    df, n_rest_removed = _remove_zero_current_rest(df)
    df, n_outliers = _remove_outlier_cycles(df)
    if df.empty:
        return {}

    dis_caps = (df[df["phase"] == "discharge"]
                .groupby("cycle")["capacity_Ah"].first()
                .dropna().sort_index())
    init_cap  = float(dis_caps.iloc[0])  if len(dis_caps) > 0 else np.nan
    final_cap = float(dis_caps.iloc[-1]) if len(dis_caps) > 0 else np.nan

    meta = {
        "cell_id":            cell_name,
        "dataset":            "CALCE",
        "chemistry_group":    _cell_chemistry_group(cell_name),
        "n_cycles":           df["cycle"].nunique(),
        "n_rest_removed":     n_rest_removed,
        "n_outliers_removed": n_outliers,
    }
    save_cell(out_dir, cell_name, meta, df)

    return {
        "cell_id":       cell_name,
        "chemistry":     _cell_chemistry_group(cell_name),
        "n_files_txt":   n_files_txt,
        "n_files_xlsx":  n_files_xlsx,
        "n_files_failed": n_files_failed,
        "total_cycles":  df["cycle"].nunique(),
        "total_rows":    len(df),
        "init_cap_Ah":   round(init_cap,  4) if np.isfinite(init_cap)  else "",
        "final_cap_Ah":  round(final_cap, 4) if np.isfinite(final_cap) else "",
    }


def _calce_worker(args):
    """top-level 함수 — Windows ProcessPoolExecutor 필수."""
    cell_dir_str, out_dir_str, raw_out_dir_str = args
    try:
        stats = convert_calce_cell(Path(cell_dir_str), Path(out_dir_str), Path(raw_out_dir_str))
        return ("ok", stats)
    except Exception:
        return ("err", f"{Path(cell_dir_str).name}:\n{traceback.format_exc()}")


def convert_calce(out_root: Path, target_cell: str | None = None, n_workers: int = 4) -> None:
    out_dir     = out_root / "CALCE"
    raw_out_dir = RAW_OUTPUT_ROOT / "CALCE"

    cell_dirs = sorted(
        p for p in CALCE_ROOT.iterdir() if p.is_dir() and _CELL_DIR_RE.match(p.name)
    )
    if not cell_dirs:
        print(f"[CALCE] 셀 폴더 없음: {CALCE_ROOT}")
        return
    if target_cell:
        cell_dirs = [p for p in cell_dirs if p.name == target_cell]
        if not cell_dirs:
            print(f"[CALCE] cell '{target_cell}' 없음")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[CALCE] {len(cell_dirs)}개 셀 변환  (workers={n_workers})")
    print(f"  raw     → {raw_out_dir}")
    print(f"  unified → {out_dir}")

    args_list = [(str(p), str(out_dir), str(raw_out_dir)) for p in cell_dirs]
    records, errors = _run(args_list, _calce_worker, "CALCE cells", n_workers)

    print(f"  완료: {len(records)} 성공, {errors} 실패")
    if records:
        docs_dir = PROJECT_ROOT / "docs"
        docs_dir.mkdir(exist_ok=True)
        csv_path = docs_dir / "calce_conversion_summary.csv"
        pd.DataFrame(records).sort_values("cell_id").to_csv(csv_path, index=False)
        print(f"  요약 CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CALCE(CS2/CX2, LCO) → 통일 포맷 변환")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cell", default=None, help="단일 셀만 변환 (예: 'CS2_8')")
    args = parser.parse_args()
    convert_calce(Path(args.output_root), target_cell=args.cell, n_workers=args.workers)


if __name__ == "__main__":
    main()
