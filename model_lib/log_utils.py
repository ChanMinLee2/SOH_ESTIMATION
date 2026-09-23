"""
5_model/experiments/phase1_lab/log_utils.py

docs/phase1_lab/RESULTS_LOG.md에 실험 기록을 자동으로 append하는 공용 유틸.
analyze_convergence.py / analyze_hi_synergy.py / materialize_ensemble_gates.py /
run_all_stages.py가 각자 실행 끝에 append_log_entry()를 호출해 로그를 남긴다 —
사람이 수동으로 복사-붙여넣기 할 필요 없음(기존에 만든 RESULTS_LOG.md 템플릿과
동일한 형식으로 append만 함, 기존 내용은 절대 덮어쓰지 않음).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOG_PATH = PROJECT_ROOT / "docs" / "phase1_lab" / "RESULTS_LOG.md"


def current_command_str() -> str:
    """지금 실행 중인 명령어를 로그용 문자열로 재구성."""
    return sys.executable + " " + " ".join(sys.argv)


def append_log_entry(
    tag: str,
    purpose: str,
    command: str,
    result_files: list[str],
    key_metrics: str,
    interpretation: str,
) -> None:
    """RESULTS_LOG.md 끝에 실험 기록 한 섹션을 append (기존 내용 보존, append만)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Phase1 독립검증 실험 로그\n\n", encoding="utf-8")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    result_files_md = "\n".join(f"  - `{f}`" for f in result_files) if result_files else "  - (없음)"
    entry = f"""
### {ts} — {tag}

- **목적**: {purpose}
- **명령어**:
  ```powershell
  {command}
  ```
- **결과 파일**:
{result_files_md}
- **핵심 수치**: {key_metrics}
- **해석 / 다음 액션**: {interpretation}

---
"""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[log] 실험 기록 자동 추가: {LOG_PATH}  (태그={tag})")
