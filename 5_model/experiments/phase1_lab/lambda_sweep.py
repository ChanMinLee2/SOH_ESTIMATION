"""
5_model/experiments/phase1_lab/lambda_sweep.py

lambda_l0 정규화 경로(regularization path) 스윕 — 일회성 검증 스크립트.
docs/260824_RESULTS.md 최상단 "최우선 작업" 참고.

phase1_trainer_v2.py를 lambda_l0 값을 바꿔가며 여러 번(subprocess) 호출하고,
각 실행의 (활성 HI 개수, val_rmse/val_r2, 게이트 포화도)를 모아 CSV로 저장한다.
목적은 "성능-개수 곡선의 무릎(knee)"을 찾아 lambda_l0_auto 공식
(sqrt(10/m * 10/k))을 대체할 값을 역산하는 것.

phase1_trainer_v2.py/train_scr.py/scr_trainer.py 등 기존 파일은 전혀 건드리지
않는다(phase1_trainer_v2.py에는 --lambda-l0-override 플래그 하나만 추가함 —
기본값 None이라 기존 동작과 100% 동일). 이 스크립트는 그 위에서 subprocess로
반복 실행만 담당하는 별도 오케스트레이터다 — 일회성 실험이라 phase1_trainer_v2.py
본체에 스윕 로직을 넣지 않고 분리했다.

기준 버전: Stage1+2 아키텍처 고정(--synergy-groups-json/--kernel-features-pkl
없음 — v0/v1/v2와의 교란 변수 제거). seed/split-seed는 기본 42 고정.

axis-config의 ref_lag/noise_amp/noise_mode/noise_period_cycles는 이 검증 전체에서
한 번도 안 바뀐 값이라 스크립트 내부에 고정(FIXED_AXIS_PARAMS)해뒀다 — CLI에서는
실험마다 실제로 바뀌는 n1/n2/n_samples만 받는다. 부수 효과로, 껍데기 JSON
문자열을 통째로 셸에 따옴표째 넘길 일이 없어져 Windows PowerShell이 내장 큰따옴표를
지워버리는 인자 전달 버그도 같이 회피된다(축-config는 이제 이 스크립트 내부
subprocess.run() 호출에서만 만들어지고 조립되므로 Python이 알아서 이스케이프한다).

사용 예 (dry-run으로 먼저 커맨드만 확인):
  python 5_model/experiments/phase1_lab/lambda_sweep.py \
      --model-config 5_model/config/main_qfref_S.yaml \
      --seg-axis q_frac_ref \
      --n1 0.35 --n2 0.20 --n-samples 2 \
      --scen-k 25 --seed 42 --split-seed 42 --beta-min 0.1 \
      --lambda-min 0.0001 --lambda-max 0.1 --n-points 9 \
      --max-epochs 250 --patience 60 --dry-run

실제 실행은 --dry-run만 빼면 된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows 콘솔이 cp949일 때 한글/특수문자 print가 UnicodeEncodeError로 죽는 문제 방지
# (긴 스윕 도중 실패 로그 한 줄 때문에 전체 스윕이 죽으면 안 됨).
for _stream in (sys.stdout, sys.stderr):
    if getattr(_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "5_model"))

from utils.tqdm_utils import tqdm, write as tqdm_write  # noqa: E402

TRAINER = Path(__file__).resolve().parent / "phase1_trainer_v2.py"
SWEEP_ROOT = Path(__file__).resolve().parent / "results" / "lambda_sweep"

# axis-config 중 이 검증(Stage0~v4 전체)에서 한 번도 바뀐 적 없는 필드 — main_qfref_S.yaml
# 계열 실행 예제(run_all_stages.py 등)와 동일 값으로 고정. 실험마다 바뀌는 n1/n2/n_samples만
# CLI로 받고 나머지는 여기서 채운다.
FIXED_AXIS_PARAMS = {
    "ref_lag": 0,
    "noise_amp": 0.03,
    "noise_mode": "ou",
    "noise_period_cycles": 200,
}

# phase1_trainer_v2.py는 train_scr.py의 자동 경로계산(_axis_dir)을 재사용하지 않으므로
# --data-dir/--seg-data-dir을 직접 받아야 한다(안 주면 즉시 RuntimeError). 아래 경로는
# n1=0.35/n2=0.20/n_samples=2 + 위 FIXED_AXIS_PARAMS 조합으로 train_scr.py가 실제로
# 만드는 경로를 기존 성공 런(config.yaml)에서 그대로 가져온 값 — 이 조합을 계산하는
# 포매팅 로직을 여기 재구현하지 않는 이유는 train_scr.py의 _axis_dir 규칙과 몰래
# 어긋날 위험 때문(중복 구현 금지 원칙). --n1/--n2/--n-samples를 표준값(0.35/0.20/2)에서
# 바꾸면 이 기본값도 더는 맞지 않으니 --data-dir/--seg-data-dir을 직접 넘겨야 한다.
_DATA_ROOT = "D:/chanminLee/LFP_SOH_prediction_v2/_4_data_hi/q_frac_ref/n1-35%_n2-20%_N-2_lag-0_noise-3%_ou-200"
DEFAULT_DATA_DIR = f"{_DATA_ROOT}/cycle"
DEFAULT_SEG_DATA_DIR = f"{_DATA_ROOT}/seg"

CSV_FIELDS = [
    "lambda_l0", "tag", "run_dir", "status", "selected_epoch", "gate_saturation",
    "val_rmse", "val_r2", "active_reg_avg", "active_cls_avg", "elapsed_sec",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lambda_l0 정규화 경로 스윕 (Stage1+2 고정)")
    p.add_argument("--model-config", required=True)
    p.add_argument("--seg-axis", required=True)
    p.add_argument("--n1", type=float, default=0.35, help="axis-config.n1 (charge 분할 비율)")
    p.add_argument("--n2", type=float, default=0.20, help="axis-config.n2 (discharge 분할 비율)")
    p.add_argument("--n-samples", type=int, default=2, dest="n_samples",
                   help="axis-config.n_samples (noise 시나리오 샘플 수, 기본 2 = N2 계열과 동일)")
    p.add_argument("--scen-k", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None,
                   help="기본값은 --seed와 동일")
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="기본값은 n1=0.35/n2=0.20/n_samples=2 표준 조합의 실제 경로. "
                        "--n1/--n2/--n-samples를 표준값에서 바꾸면 반드시 직접 지정할 것")
    p.add_argument("--seg-data-dir", default=DEFAULT_SEG_DATA_DIR,
                   help="기본값 근거는 --data-dir와 동일")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=None)

    p.add_argument("--lambdas", default=None,
                   help="쉼표구분 명시적 lambda 리스트(예: '0.0001,0.001,0.01'). "
                        "주어지면 --lambda-min/--lambda-max/--n-points 무시.")
    p.add_argument("--lambda-min", type=float, default=0.0001)
    p.add_argument("--lambda-max", type=float, default=0.1)
    p.add_argument("--n-points", type=int, default=9,
                   help="로그스케일 스윕 지점 개수(양끝 포함)")

    p.add_argument("--tag-prefix", default="lsweep")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 학습 없이 각 lambda별 커맨드만 출력")
    args = p.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    return args


def _log_spaced(lo: float, hi: float, n: int) -> list[float]:
    import math
    if n <= 1:
        return [lo]
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    return [round(10 ** (log_lo + (log_hi - log_lo) * i / (n - 1)), 6) for i in range(n)]


def _fmt_lambda(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _axis_config_json(args: argparse.Namespace) -> str:
    axis_cfg = {"n1": args.n1, "n2": args.n2, **FIXED_AXIS_PARAMS, "n_samples": args.n_samples}
    return json.dumps(axis_cfg, separators=(",", ":"))


def _build_cmd(args: argparse.Namespace, lam: float, tag: str) -> list[str]:
    cmd = [
        sys.executable, str(TRAINER),
        "--model-config", args.model_config,
        "--seg-axis", args.seg_axis,
        "--axis-config", _axis_config_json(args),
        "--scen-k", str(args.scen_k),
        "--seed", str(args.seed),
        "--split-seed", str(args.split_seed),
        "--beta-min", str(args.beta_min),
        "--patience", str(args.patience),
        "--lambda-l0-override", str(lam),
        "--tag", tag,
    ]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if args.seg_data_dir:
        cmd += ["--seg-data-dir", args.seg_data_dir]
    if args.max_epochs is not None:
        cmd += ["--max-epochs", str(args.max_epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    return cmd


def _run_dir_from_log(log_path: Path) -> Path | None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in reversed(text.splitlines()):
        if line.startswith("[p1v2] run dir:"):
            return Path(line.split("[p1v2] run dir:", 1)[1].strip())
    return None


def _avg_active(gates_json: Path, prob_keys: list[str]) -> float | None:
    if not gates_json.exists():
        return None
    d = json.loads(gates_json.read_text(encoding="utf-8"))
    counts = []
    for key in prob_keys:
        for probs_key in [k for k in d if k.endswith(key)]:
            probs = d[probs_key]
            counts.append(sum(1 for x in probs if x > 0.5))
    return sum(counts) / len(counts) if counts else None


def _stream_subprocess(cmd: list[str], log_path: Path, cwd: str) -> int:
    """child의 stdout/stderr를 실시간으로 콘솔 + 로그 파일에 동시 기록(tee)하고 종료코드를
    돌려준다. child 안의 tqdm epoch bar는 기본적으로 stderr에 '\\r'로 갱신되므로
    (stderr=STDOUT으로 합쳐서 받음), 라인 단위(readline)가 아니라 청크 단위로 그대로
    흘려보내야 실시간성이 유지된다 — 줄바꿈이 없는 '\\r' 갱신을 줄 단위로 읽으면
    다음 개행이 올 때까지 안 보이고 한꺼번에 쏟아진다."""
    sys.stdout.flush()
    with open(log_path, "wb") as logf, subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, bufsize=0,
    ) as proc:
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            logf.write(chunk)
        return proc.wait()


def _selected_row(train_log_csv: Path) -> dict | None:
    if not train_log_csv.exists():
        return None
    rows = list(csv.DictReader(train_log_csv.open(encoding="utf-8")))
    selected = [r for r in rows if r.get("is_selected") in ("1", "True")]
    return selected[-1] if selected else (rows[-1] if rows else None)


def _collect_result(lam: float, tag: str, run_dir: Path | None, elapsed: float, status: str) -> dict:
    row = {
        "lambda_l0": lam, "tag": tag, "run_dir": str(run_dir) if run_dir else "",
        "status": status, "selected_epoch": "", "gate_saturation": "",
        "val_rmse": "", "val_r2": "", "active_reg_avg": "", "active_cls_avg": "",
        "elapsed_sec": round(elapsed, 1),
    }
    if run_dir is None or not run_dir.exists():
        return row
    summary_path = run_dir / "p1v2_summary.json"
    if summary_path.exists():
        summ = json.loads(summary_path.read_text(encoding="utf-8"))
        row["selected_epoch"] = summ.get("selected_epoch", "")
        row["gate_saturation"] = summ.get("gate_saturation", "")
    sel = _selected_row(run_dir / "logs" / "train_log_v2.csv")
    if sel:
        row["val_rmse"] = sel.get("val_rmse", "")
        row["val_r2"] = sel.get("val_r2", "")
        if row["gate_saturation"] == "":
            row["gate_saturation"] = sel.get("gate_saturation", "")
    reg_avg = _avg_active(run_dir / "gates" / "regression_HIs.json", ["_probs"])
    cls_avg = _avg_active(run_dir / "gates" / "classification_HIs.json", ["_probs"])
    row["active_reg_avg"] = reg_avg if reg_avg is not None else ""
    row["active_cls_avg"] = cls_avg if cls_avg is not None else ""
    return row


def main() -> None:
    args = _parse_args()
    lambdas = ([float(x) for x in args.lambdas.split(",")] if args.lambdas
               else _log_spaced(args.lambda_min, args.lambda_max, args.n_points))

    timestamp = datetime.now().strftime("%m%d_%H%M")
    sweep_dir = SWEEP_ROOT / f"{timestamp}_{args.tag_prefix}"
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "summary.csv"

    print(f"[lambda_sweep] {len(lambdas)}개 지점: {lambdas}")
    print(f"[lambda_sweep] 결과 디렉토리: {sweep_dir}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    pbar = tqdm(lambdas, desc="lambda sweep", unit="run")
    for i, lam in enumerate(pbar):
        tag = f"{args.tag_prefix}_l{_fmt_lambda(lam)}"
        cmd = _build_cmd(args, lam, tag)
        pbar.set_description(f"[{i+1}/{len(lambdas)}] lambda={lam:g}")
        tqdm_write(f"\n[{i+1}/{len(lambdas)}] lambda_l0={lam} tag={tag}")
        tqdm_write("  " + " ".join(cmd))

        if args.dry_run:
            continue

        log_path = logs_dir / f"{tag}.log"
        t0 = time.time()
        # child 안의 epoch tqdm 바가 같은 줄을 실시간으로 갱신하는 동안 바깥쪽(스윕) 바가
        # 같은 화면 줄을 두고 다투지 않도록, 스트리밍하는 동안만 바깥 바를 지워둔다.
        pbar.clear()
        returncode = _stream_subprocess(cmd, log_path, str(PROJECT_ROOT))
        pbar.refresh()
        elapsed = time.time() - t0
        status = "ok" if returncode == 0 else f"fail(rc={returncode})"

        run_dir = _run_dir_from_log(log_path) if returncode == 0 else None
        row = _collect_result(lam, tag, run_dir, elapsed, status)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        tqdm_write(f"  -> status={status} elapsed={elapsed:.0f}s "
              f"active_reg={row['active_reg_avg']} active_cls={row['active_cls_avg']} "
              f"val_r2={row['val_r2']} sat={row['gate_saturation']}")
        if status != "ok":
            tqdm_write(f"  !! 실패 - 로그 확인: {log_path}")
        pbar.set_postfix(status=status, val_r2=row["val_r2"], active=row["active_reg_avg"])

    if args.dry_run:
        tqdm_write("\n[lambda_sweep] --dry-run 종료 (실제 학습 없음)")
    else:
        tqdm_write(f"\n[lambda_sweep] 완료. 요약 CSV: {csv_path}")


if __name__ == "__main__":
    main()
