# 배포 자료 검증

확인일: 2026-11-04. 이 기록은 LaTeX 문서 개정 및 공개 배포 자료에 대한 검사이며, 전체 신경망 학습의 완료를 뜻하지 않습니다.

| 항목 | 확인 결과 |
|---|---|
| 이번 개정의 파이프라인·완료 판정 시험 | 29 passed (Windows CPU) |
| FEniCS 통합 및 저자 자료 회귀 시험 | 4 passed (Ubuntu DOLFIN) |
| 이전 전체 CPU 회귀 시험 | 126 passed, 9 skipped; 이번 문서 개정에서는 수치 해법을 변경하지 않음 |
| 완료된 신경망 | 13개; 설정·예정 횟수·최종 가중치·예측 배열 검증 |
| FEniCS 기준해 | 19개; 설정·배열·잔차·출처 해시 확인 |
| FEM 추가 검증 | 균질 해석해 6개 및 외부 FEAP 비교를 실제 재계산 |
| Jupyter Notebook | 코드 셀 15개 순서대로 재실행, 오류 출력 0개 |
| 전체 보고서 | 39쪽; LaTeX 조판 경고 0개, 모든 페이지 자동 검사 및 시각 검토 |
| 별도 FEM 보고서 | 17쪽; LaTeX 조판 경고 0개, 모든 페이지 자동 검사 및 시각 검토 |
| 출처 자료 | 배포 자료 7개 해시 일치; 미배포 원문 PDF와 발췌 7개는 미검증으로 명시 |

이번 시험 명령은 `python -m pytest tests/test_finish_full_study.py tests/test_full_study.py -q --tb=short`와 Ubuntu의 `python3 -m pytest tests/test_fem.py tests/test_author_reference.py -q`입니다. Windows에서는 `CUDA_VISIBLE_DEVICES=-1`을 적용하여 진행 중인 GPU 학습과 분리했습니다.

노트북 기본값은 `RUN_FULL_STUDY=False`, `RUN_SMOKE=False`이며, 문서 검증에서 장기 학습을 중복 실행하지 않았습니다. 공개 저장소의 13개 완료 결과와 로컬에서 계속 계산 중인 실행은 구분합니다.

PDF 검증은 `FULL_REPORT_LAYOUT_QA.json`과 `FEM_REPORT_LAYOUT_QA.json`, 조판 기록은 각 `*_TEX_BUILD.json`, 수치 검증은 `results/fem_report_verification.json`에 있습니다. 자동 검사는 수치 정확도의 증명이 아니며 별도의 FEM·외부 자료 검증과 함께 해석해야 합니다.
