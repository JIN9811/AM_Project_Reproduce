# 논문 전체 실험: 한국어 실행 안내

이 문서는 Harandi 등의 [열탄성 혼합 PINN 논문](https://doi.org/10.1002/nme.7388)에 대해 식별 가능한 실험군을 계산·검증·비교하는 전체 연구의 안내서다. 원문의 미기재 설정과 모순은 재구성 선택으로 공개한다. 이전의 짧은 학습 연구를 원문 전체 예산의 완료 결과로 간주하지 않는다.

## 먼저 열 파일

- [Full_Paper_Reproduction_KO.ipynb](Full_Paper_Reproduction_KO.ipynb): 실행 설정, 완료·대기 상태, 실제 FEM/PINN 검증, NPZ 기반 오차 재계산, 편집 가능한 비교 그림.
- `PAPER_EXPERIMENT_INVENTORY.json`: 원문 실험·그림·표·출처 충돌 목록.
- `FULL_STUDY_PROTOCOL.json`: 전체 연구의 실제 선택과 예산.
- `FULL_STUDY_RESULTS.json`: 그림집의 검증 완료 실행, 누락 실행·기준해, 비교 결과, 색 범위 감사.
- `FULL_STUDY_GALLERY_KO.html`, `FULL_STUDY_FIGURES.pdf`: 해당 파일이 생성된 시점의 전체 또는 부분 그림집. 제목만으로 완료를 판단하지 말고 JSON의 상태를 함께 확인한다.

노트북 기본값은 `RUN_FULL_STUDY=False`다. **검증 완료 실행만 읽고 누락·진행 중 작업을 대기 표에 남긴다.** 기본 미리보기는 장기 학습이나 시간 측정을 시작하지 않는다. 각 오차는 실제 `prediction.npz`와 `solution.npz`에서 다시 계산하며 저장된 보고서 수치를 그대로 가져오지 않는다.

## 환경과 시작

프로젝트 폴더 전체를 유지한다. 노트북만 복사하면 모듈·기준해·가중치를 찾을 수 없다. Windows에서는 기존 `setup.ps1`로 Python/WSL 환경을 준비한 뒤 다음을 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_notebook.ps1
```

JupyterLab은 기본으로 새 `Full_Paper_Reproduction_KO.ipynb`를 연다. **Harandi CUDA Python 3.11** 커널을 고른다. FEniCS는 WSL Ubuntu Python, PyTorch는 현재 커널 Python을 쓴다. Linux의 별도 FEM Python은 `FENICS_PYTHON` 환경변수로 지정한다. 전체 작업 목록은 CUDA 그래프 항목을 포함하므로 CUDA가 없는 환경은 별도의 CPU 구성이 필요하다.

## 한 번에 전체 계산

환경 설치 후 PowerShell에서 다음 명령으로 FEM부터 신경망·시간 측정·그림·노트북·보고서·새 ZIP까지 순서대로 실행할 수 있다. 이미 GPU 작업이 진행 중이면 중복 실행하지 않는다. 실행기는 중복 큐를 차단한다.

```powershell
.\run.ps1 -Task full
```

완료 검증과 문서 생성은 `scripts/finish_full_study.py`가 담당한다. PDF용 Python을 별도로 쓰려면 `-DocumentPython '경로/python.exe'`를 지정한다. 진행 현황은 `FULL_STUDY_PROGRESS.html`, 세부 상태는 `full_pipeline_status.json`에 기록한다. 기존 압축파일을 보존하고 새 전체 연구는 상위 폴더의 `harandi_full_study.zip`에 저장한다. 모든 계산·검증·문서 생성이 끝나기 전에는 완료 ZIP을 만들지 않는다.

설정 셀에서 다음을 지정하고 **Run All Cells**를 실행한다.

```python
RUN_FULL_STUDY = True
NEURAL_WORKERS = 1        # 1 또는 2
RUN_SMOKE = False         # True이면 별도 소규모 실제 검증도 수행
```

노트북은 누락된 두 기하의 FEM 기준해와 전이학습용 비율 1…10 및 15의 기준해를 준비한다. 그 후 현재 `harandi.full_study.catalogue()`의 신경망 작업을 실행한다. 모든 신경망 작업이 끝난 다음 시간 연구를 단일 작업으로 실행한다. 완료된 작업은 재사용하고 같은 설정의 체크포인트는 이어서 계산한다. 실행 개수는 catalogue에서 동적으로 읽으며 감사 기록 수와 같다고 가정하지 않는다.

한 작업의 실패는 해당 폴더의 `console.log`에서 확인한다. 원인을 수정한 뒤 같은 설정으로 다시 실행하면 완료 작업을 다시 학습하지 않는다. 한 폴더를 둘 이상의 작업이 동시에 쓰지 않게 한다. 시간 연구 중에는 외부 GPU 작업도 함께 실행하지 않아야 비교가 의미 있다.

Windows PowerShell에서 현재 프로젝트용 Python을 쓰는 명령행 경로는 다음과 같다.

```powershell
$studyPython = '.venv/Scripts/python.exe'
& $studyPython -m harandi.full_study --list
& $studyPython -m harandi.full_study --fem-only
& $studyPython -m harandi.full_study --groups mixed baseline transfer --workers 1
& $studyPython -m harandi.full_study --groups timing --workers 1
& $studyPython -m harandi.full_figures
```

위 경로는 `setup.ps1`이 생성하는 가상환경을 기준으로 한다. 노트북은 하위 신경망 프로세스에 `sys.executable`을 사용하여 현재 커널과 동일한 Python으로 실행한다. FEM은 의도적으로 DOLFIN 환경을 사용한다.

## 재시작과 상태 확인

노트북의 단일 재시작 예:

```python
RUN_FULL_STUDY = False
RESUME_JOB_ID = 'transfer_w0.1'
```

명령행:

```powershell
& $studyPython -m harandi.full_study --job transfer_w0.1
```

현재 상태는 `results/full_study/queue_status.json`, 작업별 `run_status.json` 또는 `experiment.json`, `progress.json`, `history.csv`, `console.log`에 남는다. **체크포인트나 예측 파일이 있는 것만으로 완료가 아니다.** `job_complete`는 설정·요구 예산·실제 업데이트·최종 상태를 검증한다. 미완료 상태의 비교를 전체 연구 결과로 표시하지 않는다.

개별 실행에서 설정을 바꾸면 기존 체크포인트와 불일치하여 재시작이 거부될 수 있다. 다른 물리·네트워크·가중치 실험은 별도 폴더로 분리한다. Remark 7의 100만 회 추가 실험처럼 의도한 예산 확장은 전체 실행기의 명시적 계보와 검증을 따른다.

## 포함하는 실험과 해석

**학습점 분포의 차이:** 원형은 51×51 직교 격자와 시드가 고정된 내부 균등 난수 2,399점을 합친 5,000점이다. 원문 그림 5의 계면·경계 보강 좌표는 공개되지 않아 그 분포를 그대로 재현하지 않았다. 같은 점 수가 같은 배치·정확도를 뜻하지 않으며, 동일 점 가중치도 면적에 정확한 적분과 구분한다.

**동시 DEM의 온도 기울기 처리:** DEM의 역학 에너지는 온도를 처방값으로 취급하고 해당 항에서 온도망으로 향하는 기울기를 차단한다. 동시 표준·혼합 PINN의 역학 손실은 온도망으로 기울기를 전달한다. 이는 단방향 열탄성 물리를 위한 변분 재구성이지만 원문 식 14의 u,T 공동 최소화와 문자 그대로 같지는 않다. 방법 비교에는 이 최적화 경로 차이가 포함된다. 두 선택은 `FULL_STUDY_PROTOCOL.json`의 `circle_sampling`, `dem_thermal_stop_gradient`에도 기록한다.

| 실험군 | 구현 범위와 읽는 방법 |
|---|---|
| 혼합 PINN | 원형·사각형, 동시·순차. Table 1 기준 동시 100k, 순차 각 장 100k/총 200k. |
| 표준 PINN·DEM | 원형에서 두 학습 방식. 큰 오차와 실패도 공개한다. DEM 하드 경계 등 추가 공정성 연구는 기본 비교와 분리한다. |
| 경계·에너지 | 사각형 왼쪽 약 1,000점 보강, 원형 에너지 항 제외, 실제 경계 오차 재계산. |
| 하드 경계·최적화 | 하드 Adam 10k와 하드 L-BFGS 10k. 소프트 Adam 100k와는 예산도 달라 단일 요인 비교가 아니다. |
| 구조 정확도 | 깊이 1,2,4,5,6에서 너비 40; 너비 5,10,20,40,60에서 깊이 5. 5×40 중복 실행 재사용. |
| 시간 | 25개 깊이×너비 조합의 각 100 L-BFGS 업데이트. 25개 추가 정확도 수렴 연구가 아니다. |
| Remark 7 | 본문 1M Adam 문구를 Table 1의 100k와 분리한 추가 실행. 원문의 예산 충돌을 남긴다. |
| 전이학습 | 비율 1…10을 각각 Adam 10k. 가중치 0,0.1,0.5,1. 비율 15는 최종 테스트 전용. 0.5는 미기재 네 번째 값을 위한 보완 선택. |
| FEM 수렴 | 50·100·200 요소/변. 200을 비교 기준으로 변화량을 계산하되 해석적 정답이라고 부르지 않는다. |

순차 학습, 더 큰 망, 물리 손실이 언제나 우세하다는 결론을 전제하지 않는다. 원문에도 성분별 예외·최대오차 악화·비단조성이 있다. 평균 및 최대 오차와 8개 성분을 함께 확인한다.

## 오차와 편집 가능한 그림

설정 셀의 `OUTPUT_KIND`는 `direct_mixed_outputs` 또는 `constitutive_outputs`다. 전자는 혼합 신경망 직접 출력, 후자는 변위·온도 미분에서 구성식과 Fourier 법칙으로 계산한 응력·열유속이다. 두 결과의 차이를 숨기지 않는다.

`REFERENCE_SUFFIX='n200'`은 200×200 FEM을 정량 비교 기준으로 쓴다. `n50`, `n100`, `n100_nodal`로 바꿀 수 있으며 요소 수와 재료장 모드를 표에 표시한다. 전이학습은 항상 비율 15의 별도 기준해를 사용한다.

상대 L2, 평균 절대오차/기준해 최대 절댓값, 최대 절대오차/기준해 최대 절댓값은 서로 다른 지표다. 전단응력·횡열유속처럼 기준값이 작은 성분의 상대오차에는 특히 주의한다. 절대오차도 재계산 자료에 남긴다. 새 계산은 `.notebook_work/full_study/recomputed_errors.json`에 저장되어 기존 결과 파일을 수정하지 않는다.

`FIELD`와 전체장 셀의 `selected_run`을 바꾸면 8성분 전체장을 선택할 수 있다. 방법 비교, 경계 검사, 하드 최적화 이력, 깊이·너비 연구, 네 가중치 전이학습 단면은 모두 노트북 안의 Matplotlib 코드로 편집할 수 있다. 그림 25 단면은 패널·본문에 따라 x=0.35를 쓰며 캡션 x=0.5와의 충돌을 기록한다.

## 작은 검증과 그림집

`RUN_SMOKE=True`는 별도 작업 폴더에서 작은 균질 FEM과 실제 Adam/L-BFGS 업데이트를 실행한다. 기본값에서는 이전 저장 검증을 찾아 출처를 명시하고, 자료가 없으면 대기로 둔다. 작은 검증은 수치 구현 동작을 확인할 뿐 장기 실험 완료나 논문 수치 일치를 증명하지 않는다.

미완료 연구의 부분 그림집:

```powershell
& $studyPython -m harandi.full_figures --partial
```

노트북에서 `BUILD_FIGURE_PACK=True`를 선택해도 된다. 실제 완료 실행만 사용하며 `FULL_STUDY_RESULTS.json`에 pending 목록을 남긴다. 원문 인쇄 색 범위 밖 실제 값은 감사표에 공개하고 배열을 색 범위에 맞게 바꾸지 않는다.

## 노트북 원본과 검증

`python scripts/build_full_notebook.py`는 새 노트북만 생성하며 기존 노트북·보고서 생성기를 수정하지 않는다. 생성 시 nbformat 스키마와 모든 코드 셀의 컴파일을 확인한다. 재생성은 저장된 실행 출력을 초기화하므로 일상적인 갱신은 노트북을 직접 실행한다.

```powershell
& $studyPython scripts/build_full_notebook.py
& $studyPython -m jupyter nbconvert --execute --to notebook --inplace Full_Paper_Reproduction_KO.ipynb --ExecutePreprocessor.timeout=300
```

두 번째 명령은 노트북에 저장된 설정을 실행한다. 기본 미리보기에서는 장기 학습을 시작하지 않지만, RUN_FULL_STUDY를 True로 저장했다면 전체 계산을 수행한다. 장기 실행은 적절한 셀 시간 제한 또는 JupyterLab에서 실행한다.

검증된 사실은 실제 배열·설정·예산·이력·출처와 연결한다. 원문과 차이가 있으면 재구성 선택과 수치 차이를 그대로 보고하며, 논문의 결론에 맞추기 위해 결과를 조정하지 않는다.
