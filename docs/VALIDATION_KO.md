# 배포 자료 검증

확인일: 2026-11-04. 이 기록은 공개 배포 폴더에서 수행한 검사에 관한 것이며, 전체 신경망 실험의 완료를 뜻하지 않는다.

| 항목 | 확인 결과 |
|---|---|
| CPU 테스트 | 126 passed, 9 skipped. GPU 전용·WSL DOLFIN 환경이 필요한 항목은 Windows CPU 실행에서 건너뜀 |
| 완료된 신경망 | 11개: 설정·예정 횟수·최종 가중치·예측 배열 검증 |
| FEniCS 기준해 | 19개: 배열·설정·유한성·선형 잔차 확인 |
| Jupyter Notebook | 코드 셀 15개 순서대로 실행, 오류 출력 0개 |
| 한국어 보고서 | 38쪽 전체 렌더링 및 자동 검사 통과; 조판 시각 검토 |
| 출처 검사 | 배포 자료 7개 해시 일치; 미배포 PDF와 발췌 7개는 미검증으로 명시 |

테스트 명령은 공개 폴더에서 `CUDA_VISIBLE_DEVICES=-1`로 GPU를 제외하고 `python -m pytest -q --tb=short`를 실행한 것이다. 코드 셀 기본값은 `RUN_FULL_STUDY=False`, `RUN_SMOKE=False`이다. 따라서 노트북 검증에서 장기 학습을 중복 실행하지 않았다.

PDF 검사 결과: `FULL_REPORT_LAYOUT_QA.json`. 수치 결과의 원시 배열과 해시는 `FULL_STUDY_RESULTS.json` 및 각 실행 폴더에서 확인할 수 있다.
