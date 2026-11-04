# LaTeX 보고서 편집 및 재생성

전체 연구 보고서와 독립 FEM 보고서는 같은 XeLaTeX 조판기를 사용합니다. 본문은 국문 공학 논문의 문제 정의–정식화–검증–고찰 순서를 따르며, 수식은 LaTeX 원본으로 작성하여 번호를 붙입니다. PDF에는 검색 가능한 벡터 수식과 글꼴을 포함합니다.

| 보고서 | PDF | 편집 원본 | 생성기 |
|---|---|---|---|
| 전체 FEM/PINN 연구 | [PDF](../FULL_PROJECT_REPORT_KO.pdf) | [LaTeX](../FULL_PROJECT_REPORT_KO.tex), [Markdown](../FULL_PROJECT_REPORT_KO.md) | `scripts/build_full_korean_report.py` |
| FEM 정식화 및 검증 | [PDF](../FEM_REPORT_KO.pdf) | [LaTeX](../FEM_REPORT_KO.tex), [Markdown](../FEM_REPORT_KO.md) | `scripts/build_fem_report.py` |

FEM 보고서의 완료는 신경망 전체 학습의 완료를 뜻하지 않습니다. 전체 보고서는 검증된 실행만 수록하며, `--partial`로 생성하면 표지와 꼬리말에 진행 상태를 표시합니다.

## 1. 설치

Windows에서는 프로젝트의 FEniCS 환경과 같은 Ubuntu WSL을 사용합니다. Ubuntu 터미널에서 다음 시스템 패키지를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-lang-korean texlive-fonts-recommended fontconfig
```

문서 생성과 PDF 검증에 사용할 Python에는 다음 의존성을 설치합니다.

```bash
python -m pip install -r requirements-documents.txt
```

기본 글꼴은 국문 UnBatang·UnDotum, 영문 Latin Modern, 수학 Latin Modern Math입니다. 이 컴퓨터에서는 TeX Live 2022의 XeLaTeX를 실제 설치하여 조판했습니다. Windows 생성기는 로컬 `xelatex`가 없으면 `wsl -d Ubuntu`를 호출합니다. 다른 배포판 이름을 쓰면 `HARANDI_WSL_DISTRO` 환경변수를 지정합니다. Linux에서는 PATH에 있는 `xelatex`와 `fc-list`를 사용합니다.

## 2. 전체 보고서

프로젝트 루트에서 문서용 Python을 실행합니다. `--pinn-python`에는 PyTorch가 설치된 프로젝트 Python의 실제 경로를 넣습니다. Python 환경을 하나로 구성했다면 같은 실행파일을 지정해도 됩니다.

```powershell
python scripts/build_full_korean_report.py --partial --pinn-python .venv/Scripts/python.exe
python scripts/check_full_report_layout.py
```

전체 등록 작업과 그림 생성이 모두 끝난 경우에만 `--partial`을 제거합니다. 생성기는 완료 판정을 독립적으로 확인하고 예측 NPZ에서 오차를 다시 계산합니다. 이 명령은 신경망을 학습시키지 않습니다. 최종 학습 파이프라인도 같은 생성기를 사용하므로 이후 보고서 역시 LaTeX로 작성됩니다.

## 3. 독립 FEM 보고서

먼저 Ubuntu에서 FEniCS 검증을 실행합니다. 프로젝트 루트로 이동한 상태에서 다음 명령을 사용합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/verify_fem_report.py
python3 -m pytest tests/test_fem.py tests/test_author_reference.py -q
```

검증기는 저장된 19개 기준해를 검사하고, 균질 문제의 두 재료 모드·세 격자에 대한 6개 계산과 저자 FEAP 자료 비교를 새로 수행합니다. 기존 기준해와 신경망 가중치는 수정하지 않습니다. 이어서 문서용 Python으로 아래 명령을 실행합니다.

```powershell
python scripts/build_fem_report.py
python scripts/check_full_report_layout.py --pdf FEM_REPORT_KO.pdf --output FEM_REPORT_LAYOUT_QA.json
```

보고서 생성기는 `results/fem_report_verification.json`에 기록된 NPZ 해시와 현재 파일이 다르면 중단합니다. 표·그림은 저장된 배열에서 계산합니다. 신경망 라이브러리나 학습 완료 결과는 필요하지 않습니다.

## 4. 수동 LaTeX 편집

생성된 `.tex`를 직접 편집할 수 있습니다. 이후 Python 생성기를 실행하면 편집 내용은 생성기 내용으로 덮어써지므로, 지속적으로 유지할 본문 변경은 해당 Python 파일에 반영합니다. 프로젝트 루트에서 다음처럼 직접 조판할 수도 있습니다.

```bash
xelatex -interaction=nonstopmode -halt-on-error FEM_REPORT_KO.tex
xelatex -interaction=nonstopmode -halt-on-error FEM_REPORT_KO.tex
xelatex -interaction=nonstopmode -halt-on-error FEM_REPORT_KO.tex
```

그림의 상대 경로를 보존해야 합니다. 전체 보고서는 같은 명령의 파일명을 `FULL_PROJECT_REPORT_KO.tex`로 바꿉니다. 일반 생성기는 중간 파일을 `tmp/latex/`에 두고 검증된 PDF만 루트에 복사합니다. Python 생성기는 누락 글리프를 발견하면 실패로 처리하고, 조판 경고와 원본·PDF 해시를 `*_TEX_BUILD.json`에 저장합니다.

## 5. 검증 범위

자동 검사는 PDF 모든 페이지를 이미지로 렌더링하고, 텍스트가 페이지 밖으로 나가는지, 미해결 문자나 대체 문자가 있는지, 실제 TeX/OpenType 글꼴에 해당 문자가 있는지 확인합니다. 수식 글꼴을 한글 글꼴의 문자표로 잘못 검사하지 않습니다. 자동 통과는 시각 검토나 과학적 정확성의 증명이 아니므로 실제 페이지 이미지도 검토해야 합니다.

그림의 색 범위 일치와 수치해의 정확도는 별개입니다. FEM 그림은 논문에서 기록한 팔레트와 범위를 사용하며, 범위 밖 값의 비율을 `figures/fem_report/color_scale_audit.json`에 보존합니다. 값을 잘라 저장하거나 저자의 변위 진폭에 맞추어 보정하지 않습니다.

원저자 구현과의 차이는 [표기 및 구현 대조](AUTHOR_NOTATION_KO.md), 국문 편집의 참고 자료는 [편집 참고문헌](EDITORIAL_REFERENCES_KO.md)을 참조합니다.
