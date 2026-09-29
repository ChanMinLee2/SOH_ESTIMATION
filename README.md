# SOH_ESTIMATION
LFP SOH ESTIMATION with partial cycles

## 설치 (최초 1회)

```bash
pip install -e .
```

저장소를 editable install해서 `data_directories`/`parameters`/`common`/`models`/
`datasets` 등을 실행 위치와 무관하게 import할 수 있게 한다(`pyproject.toml`).
이 단계를 건너뛰면 각 스텝 스크립트가 `ModuleNotFoundError`로 죽는다.

## 실행

```bash
python run_pipeline.py                 # 전체 파이프라인 (Step 1~9)
python run_pipeline.py 8               # 학습+평가만 (Step 5~7 산출물 재사용)
```

## 문서

| 문서 | 내용 |
|---|---|
| `docs/PIPELINE.md` | 데이터 변환부터 SOH 출력까지 9단계 전체 흐름 — 처음 보는 사람용 |
| `docs/PARAMETERS.md` | `parameters.py` 구조 + 스크립트별 CLI 옵션 전체 목록 |
| `docs/REFACTORING.md` | 리팩토링 진행 이력(살아있는 문서) |
