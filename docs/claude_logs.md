# Claude 작업 로그

| 날짜 | Q | A |
|------|---|---|
| 2026-07-16 | VII 세부실험계획서 재작성 | 3축평가(E1상한선/E2배포시나리오/E3라우팅) 추가, R시리즈 실험절차·단계별 실행순서 재편 |
| 2026-07-16 | dvdq q_frac 정규화·분류기 재활성화·test_rs 생성 확인 | hi_correlation.py /q_tot 2줄, lambda_scen 0→0.5, HUST 0개 확인→재실행 |
| 2026-07-16 | test_rs 재생성 요청(--force, MIT+HUST 전체) | workers=6, 200셀 64분 완료, 세그 297만/사이클 244k, 오류 0건 |
| 2026-07-16 | seg_diagnose.py 자동화 + test_rs NaN 많은 이유 조사 | hi_features*.pkl 자동탐색 추가; NaN 원인=CC구간 ΔI≈0·짧은세그 Q범위부족·플래토미달 |
| 2026-07-16 | claude_logs.md 매 답변 후 100자 이내 요약 업데이트 규칙 | 규칙 적용 시작, feedback 메모리 저장, 현 세션 소급 정리 |
| 2026-07-16 | 로그 상세도 향상 요청 | A 컬럼에 파일명·수치·원인 등 구체 내용 추가 |
| 2026-07-16 | run_pipeline.py 모델학습까지 확장 | Step 6~8(Phase1/2/평가) 추가, 디렉터리 자동감지, --to-step/--model-config 인자 신설 |
| 2026-07-16 | run_pipeline.py yaml·gates_from 자동주입 확인 | CLI --gates-from > yaml 순위, scr.yaml 기본, P1→P2 dir 자동감지로 yaml 하드코딩 무시됨 |
| 2026-07-16 | 학습 시 자동 과적합 테스트 제거 | scr.yaml run_overfit_test: true→false, 별도 tmp_standalone_overfit.py 사용 |
| 2026-07-16 | scen_k_count:64 + is_auto_mk=false 시 probe-m 재사용 되나? | probe(분류기, 3/2개)와 scen(회귀헤드, 64개)은 별도 마스크로 독립 동작. 재사용 정상. |
| 2026-07-16 | run dir에 phase/model/axis 식별자 추가 | train_scr.py timestamp에 _p1_prot / _p2_mlp_prot 형태 suffix 추가 |
| 2026-07-16 | hard/soft routing 현재 가능한지 확인 | 불가. probe_mlp 제거됨, lambda_scen 코드에서 0.0 고정, routing_mode TypeError fallback → 항상 none 동작 |
| 2026-07-16 | B안(별도분류기) 구현 + PIPELINE.md 갱신 | train_classifier.py 신설, evaluator routing 추가, run_pipeline Step8분류기/Step9평가, PIPELINE.md 전면 갱신 |
| 2026-07-16 | Phase1 이중목적(CE+MSE) 구현 | scr_model probe_mlp 추가(probe_x→CE), scr_loss CE 활성화, train_scr with_probe_mlp 전달, train_classifier probe마스크로 입력 제한, evaluator probe_x로 routing |
| 2026-07-17 | 리뷰 이슈 1+2+3 수정 | λ_scen 0.5→0.01(CE-MSE 균형), scen_k 64→55(L0유효화), probe_x에 direction concat(충방전 구분) — scr_model/train_classifier/evaluator 수정 |
| 2026-07-17 | random_seg_test scatter+셀플랏 추가 | evaluator.plot_for_dataset() 추가, _pick_rs_rep_cells() 추가, _run_random_segment_test에 figures 저장 (데이터셋별 3셀) |
| 2026-07-17 | routing 텐서 생성 오류 수정 | spec.routing jagged list → torch.tensor() 실패; 패딩 후 수동 채움으로 수정 (scr_evaluator.py:103) |
| 2026-07-18 | PIPELINE.md 갱신 + demo plot 추가 | scen_k 55·λ_scen 0.01·probe_mlp·CE항·분류기65D·evaluator routing 6곳 갱신; tmp_make_test_rs에 _plot_demo_cycle() 추가 (demo_segments.png) |
| 2026-07-18 | random_seg_test 플랏 빈 시나리오 원인 수정 | ①batch["seg_idx"]→batch_d["seg_idx"] 저장버그 수정 ②routing=none시 test_rs seg_idx(0/1)↔모델 시나리오 ID 불일치→direction_routing으로 방향별 첫 시나리오로 보정 |
| 2026-07-18 | 3축 비교 스크립트 작성 + 성능 평가 | compare_runs.py: E1/E2 성능표·HI Jaccard/τ·달성률 리포트, 그래프 2종 저장. 현 달성률 prot36%·vwin26%·rcs-80%; E2 R²<0이 최우선 해결 과제 |
| 2026-07-18 | demo_segments.png 충전 미표시 원인 수정 + 3셀/데이터셋 표시 | _cv_start: median→argmax(전반50%피크)로 ramp-up 오감지 제거; q_cc<0.05 폴백 추가; _plot_demo_cycle: n_cells_per_ds=3, 3행×4열 레이아웃. MIT충전=RNG(캐시 없음·--force 필요), HUST=캐시 |
| 2026-07-18 | NaN 감소 수정 구현 (morph+fallback) | tmp_make_test_rs: fastdtw+_load_ref_curves+_compute_morph 추가(cycle1 기준 morph 6개), min_len 0.05→0.15; hi_correlation: dv_di_seg·corr_vi/qi→0 폴백, valley→0 폴백, peak_w 단측추정, plateau_v_slope/v_ent_plateau 임계 5·10→3 |
| 2026-07-19 | morph 100% NaN 원인 수정 | Windows spawn 멀티프로세스에서 모듈레벨 _fastdtw 바인딩이 worker globals에 미등록 → _compute_morph 내부에서 local import(from fastdtw import fastdtw as _fdt)로 변경; dist lambda 제거(기본 Euclidean, 동일결과+빠름) |
| 2026-07-20 | 4논문 vs 프로젝트 차별성 + 강화 아이디어 | Shi:충전만/NMC, iMOE:이완30분필수/궤적예측, Su:완전사이클/전이학습, Ke:전압클러스터링; 차별점=LFP특화·충방전임의세그·CE+MSE공동라우팅·E3평가 |
| 2026-07-20 | rcs 축 세그먼트 분리 방식 설명 | n_samples=6·window=0.3; 중심q_frac 1/3등분→lo/mid/hi; routing[[0,1,2],[5,4,3]]→6시나리오; 방전전체·충전CC만; 결정론적seed |
| 2026-07-21 | hi_compute.py 신설 + hi_correlation 위임 | 5_model/hi_compute.py: @hi레지스트리+W캐시클래스+66HI(현재이름 유지); hi_correlation.py 말미에 from hi_compute import _seg_* 덮어쓰기; 66개 이름·순서 hi_schema.py 완전일치 확인 |
| 2026-07-21 | q_frac_wide 시나리오 축 신설 + 문서 추가 | common/scenario/q_frac_wide.py: 3구간(hi/mid/lo)×방향=6시나리오, linspace 균등격자, n1/n2/n_samples 파라미터; __init__.py 등록, train_scr.py _AXIS_SHORT 추가; SCENARIO_STRATEGY.md 섹션4·5·6 갱신 |
