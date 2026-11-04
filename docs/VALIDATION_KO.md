# 배포 자료 검증

확인일: 2026-11-04. 이 기록은 LaTeX 문서 개정 및 공개 배포 자료에 대한 검사이며, 전체 신경망 학습의 완료를 뜻하지 않습니다.

| 항목 | 확인 결과 |
|---|---|
| 이번 수식 수정의 문서·파이프라인 시험 | 35 passed (수식 회귀 6개 + 완료 판정 29개, Windows CPU) |
| 이전 개정의 FEniCS 통합 및 저자 자료 회귀 시험 | 4 passed (Ubuntu DOLFIN) |
| 이전 전체 CPU 회귀 시험 | 126 passed, 9 skipped; 이번 문서 개정에서는 수치 해법을 변경하지 않음 |
| 완료된 신경망 | 13개; 설정·예정 횟수·최종 가중치·예측 배열 검증 |
| FEniCS 기준해 | 19개; 설정·배열·잔차·출처 해시 확인 |
| FEM 추가 검증 | 균질 해석해 6개 및 외부 FEAP 비교를 실제 재계산 |
| Jupyter Notebook | 이전 실행의 코드 셀 15개·출력 동일성 확인; 오류 출력 0개, 안내 링크만 수정 |
| 전체 보고서 | 39쪽 자동 재검사; 변경 4쪽 시각 검토, 나머지 35쪽은 이전 검토본과 픽셀 동일 |
| 별도 FEM 보고서 | 17쪽 자동 재검사; 전 쪽이 이전 검토본과 픽셀 동일, 약형·구성식 페이지 추가 시각 확인 |
| 출처 자료 | 배포 자료 7개 해시 일치; 미배포 원문 PDF와 발췌 7개는 미검증으로 명시 |

이번 시험 명령은 `python -m pytest tests/test_latex_report.py tests/test_finish_full_study.py tests/test_full_study.py -q --tb=short`입니다. Ubuntu의 FEM 회귀 시험은 이전 개정에서 수행했습니다. Windows에서는 `CUDA_VISIBLE_DEVICES=-1`을 적용하여 진행 중인 GPU 학습과 분리했습니다.

노트북 기본값은 `RUN_FULL_STUDY=False`, `RUN_SMOKE=False`이며, 문서 검증에서 장기 학습을 중복 실행하지 않았습니다. 공개 저장소의 13개 완료 결과와 로컬에서 계속 계산 중인 실행은 구분합니다.

PDF 검증은 `validation/FULL_REPORT_LAYOUT_QA.json`과 `validation/FEM_REPORT_LAYOUT_QA.json`, 조판 기록은 각 `*_TEX_BUILD.json`, 수치 검증은 `results/fem_report_verification.json`에 있습니다. 자동 검사는 수치 정확도의 증명이 아니며 별도의 FEM·외부 자료 검증과 함께 해석해야 합니다.

## GitHub 수식 표시 수정

GitHub Markdown 변환 과정에서 보호되지 않은 TeX 명령의 역슬래시·첨자·별표가 변경되는 문제를 재현했습니다. 블록 수식은 `math` 코드 블록, 본문 수식은 보호된 인라인 구문으로 바꾸었습니다. 부등호의 HTML 이중 이스케이프와 표 안의 절댓값·노름 구분자 충돌도 제거했습니다.

실제 GitHub Markdown API에서 전체 보고서 254개, FEM 보고서 83개, 저자 표기 문서 10개, 총 **347개 수식의 변환 전후 완전 일치**를 확인했습니다. 별도의 MathJax 3.2.2 SVG 변환에서도 **347개 중 오류 0개**였습니다. 이 검사는 브라우저 스크린샷 확인을 의미하지 않습니다. 검사 결과는 `validation/REPORT_MATH_QA.json`에 있습니다.

구버전 한국어 보고서·그림집·노트북과 이전 배포 압축본, 중복 README를 작업 폴더에서 제거했습니다. 최신 전체/FEM 보고서의 PDF·TeX·Markdown은 유지하며, 이전 커밋은 삭제하지 않습니다.

## 원형 단면 비교의 출처 검증

보고서 그림 3의 예측 NPZ 6개와 FEniCS50 NPZ 1개의 SHA-256이 기존 그림 생성 기록과 일치함을 확인했습니다. 내보낸 CSV 707행을 별도로 다시 읽어 원본 배열의 x=0.5 단면과 모든 값이 정확히 일치함을 확인했습니다. `tests/test_full_figures.py` 9개와 수식 시험 6개, 합계 15개가 통과했습니다. PDF 전체 39쪽을 재검사하고 갱신한 캡션과 앞뒤 페이지를 검토했습니다. 자체 계산 자료와 Harandi et al. (2024), Fig. 11의 벤치마크 출처를 구분해 표기했습니다.

## 2026-11-05 문서 디렉터리 정리

보고서·노트북·안내·검증 기록을 각각 `reports/`, `notebooks/`, `docs/`, `validation/`으로 이동했습니다. 관련 검사 39개와 노트북 코드 셀 15개가 통과했으며 GitHub 수식 347개도 재확인했습니다. 기존 PDF 3개와 수치 결과는 보존했습니다. 새 위치의 LaTeX를 별도 조판한 결과 전체 보고서 39쪽, FEM 보고서 17쪽의 페이지별 본문이 동일하며 경고는 0개입니다.

[파일·링크·노트북 검증](../validation/DOCUMENT_LAYOUT_VALIDATION.json), [LaTeX 이동 검증](../validation/DOCUMENT_TEX_RELOCATION.json)에 근거를 기록했습니다. 공개 수치 자료는 기존 13/27개 완료 시점의 자료이며, 이번 변경은 학습 결과 추가 업로드가 아닙니다.
