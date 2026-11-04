# 재현 절차

## 1. 계산 환경

현재 검증 환경은 Windows Python 3.11, PyTorch 2.11 및 NVIDIA RTX 3090이며, 유한요소 해석에는 WSL Ubuntu의 DOLFIN 2019.2 계열을 사용하였다. 설치 버전은 요구사항 파일과 작업별 `environment.json`에 기록되어 있다. 신경망의 기본 정밀도는 float64이고, 전체 실험 명세는 CUDA 그래프를 사용한다. CPU 환경에서 결과를 읽을 수 있으나 전체 학습에는 별도 CPU 설정이 필요하다.

Windows Python과 WSL Ubuntu를 준비한 후 저장소 루트에서 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_notebook.ps1
```

설치 스크립트는 Windows의 `.venv`와 WSL의 FEniCS 환경을 구분한다. WSL 배포판 자체는 설치하지 않는다. PDF를 다른 운영체제에서 생성할 경우 `FONT_KO`, `FONT_KO_BOLD`에 한국어 TrueType 글꼴 경로를 지정한다.

## 2. 저장된 결과의 검토

`Full_Paper_Reproduction_KO.ipynb`의 `RUN_FULL_STUDY=False`, `RUN_SMOKE=False`를 유지하고 모든 셀을 실행한다. 이 과정은 완료된 예측과 FEM 배열에서 오차를 다시 계산한다. 직접 혼합 출력과 구성식·Fourier 법칙으로 얻은 출력을 선택할 수 있다. 그림집의 기준 FEM 격자와 본문 오차 표의 FEM200 기준은 각 캡션을 확인한다.

## 3. 전체 수치실험

```powershell
.\run.ps1 -Task full
```

위 명령은 FEM, 신경망 실험, 단독 시간 측정, 그림, 노트북, 보고서 및 검증을 순서대로 수행한다. 각 장 100,000회 학습과 별도의 1,000,000회 연속 실행을 포함하므로 소요 시간은 장비에 따라 수 시간 이상이다. 실행 중 동일한 큐를 중복 시작하지 않는다.

단일 실험을 재개하려면 다음과 같이 실행한다.

```powershell
.\.venv\Scripts\python.exe -m harandi.full_study --job ablation_circle_noenergy
```

모델·최적화기·진행 위치를 포함하는 체크포인트가 있으면 같은 조건으로 이어서 계산한다. 완료 판정은 예정된 갱신 횟수, 설정, 가중치와 예측의 유한성, 출력 좌표 및 모델 구조를 확인한다.

## 4. 그림과 보고서만 재생성

```powershell
.\.venv\Scripts\python.exe -m harandi.full_figures --partial
.\.venv\Scripts\python.exe scripts/build_full_korean_report.py --partial --pinn-python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts/check_full_report_layout.py
```

`--partial`은 검증된 일부 결과로 진행 보고서를 생성한다. 모든 실험이 끝난 뒤 이 옵션을 생략하면 전체 완료와 자료의 일치 여부를 검사한다. PDF 검사는 모든 페이지를 렌더링하여 누락 글리프·경계 초과 등을 검사하며, 문서의 시각적 검토는 별도 단계이다.

## 5. 결과 해석

논문의 색 범위를 맞춘 그림과 수치 정확도의 일치는 별도의 문제다. 색 범위를 벗어나는 값은 포화 표시로 확인하고, 평균오차와 최대오차를 함께 검토한다. 원문의 미기재 조건, 추가 DEM 경계 실험 및 전이학습 가중치 0.5의 재구성 선택은 `FULL_STUDY_PROTOCOL.json`에 명시되어 있다.

실행 가능한 코드와 완료된 수치실험의 범위도 구분한다. 공개 스냅샷에 포함되지 않은 실험의 성능은 아직 결론으로 사용할 수 없다.
