# 원저자 구현과 보고서 표기의 대응

대조 대상은 원저자 공개 저장소 [phyml4e/PINNs](https://github.com/phyml4e/PINNs)의 고정 커밋 `5b426dc8eb2255b1bb3b8630619228b36cdb464e`입니다. [Thermomechanic/Thermo_mechanic.py](https://github.com/phyml4e/PINNs/blob/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic/Thermo_mechanic.py)를 2026-11-04에 다시 확인했습니다. 저장소의 MIT 라이선스와 포함 자료의 출처는 기존 `data/AUTHORS_LICENSE` 및 `docs/SOURCE_PROVENANCE.json`에 보존합니다.

## 독립 출력과 구성식 출력을 구분하는 표기

| 저자 코드 | 보고서 | 의미 |
|---|---|---|
| `sig_x_O`, `sig_y_O`, `sig_xy_O` | $`\widehat\sigma_{xx},\widehat\sigma_{yy},\widehat\sigma_{xy}`$ | 독립 응력망 출력 |
| `sigma_x`, `sigma_y`, `sigma_xy` | $`\boldsymbol\sigma(\boldsymbol u,T)`$ | 변위 미분·온도로 계산한 응력 |
| `q_x_O`, `q_y_O` | $`\widehat q_x,\widehat q_y`$ | 독립 열유속망 출력 |
| `q_x`, `q_y` | $`\boldsymbol q(T)=-k\nabla T`$ | Fourier 법칙으로 계산한 열유속 |
| `EQ*_m`, `EQ*_t` | $`\mathcal L_{\mathrm{link},\sigma},\mathcal L_{\mathrm{link},q}`$ | 독립 출력과 구성식 출력의 연결 |
| `PDE*_m`, `PDE1_t` | $`\nabla\cdot\widehat{\boldsymbol\sigma},\nabla\cdot\widehat{\boldsymbol q}`$ | 역학 평형·열보존 잔차 |

이 표기는 “응력”이라는 같은 이름 아래 서로 다른 수치량이 비교되는 일을 막기 위한 것입니다. 전체 보고서의 직접 출력 표와 구성식/Fourier 출력 표도 이 구분을 유지합니다. FEM의 응력과 열유속은 주 미지수로 풀지 않고 변위와 온도로부터 계산합니다.

## 전단응력의 계수 차이

저자 파일의 [47–55행](https://github.com/phyml4e/PINNs/blob/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic/Thermo_mechanic.py#L47-L55)은 텐서 전단변형률을 다음과 같이 정의합니다.

```math
\varepsilon_{xy}=\frac12(u_{x,y}+u_{y,x}),\qquad \mu=\frac{E}{2(1+\nu)}.
```

공개 코드의 응력 계수는 이 정의에서 $`\mu\varepsilon_{xy}`$가 됩니다. 표준 등방성 평면변형률 구성식은 다음과 같습니다.

```math
\sigma_{xy}=2\mu\varepsilon_{xy}=\mu(u_{x,y}+u_{y,x}).
```

본 프로젝트의 FEM과 PINN은 후자의 식을 공통으로 사용합니다. 보고서에는 이를 수정 사항으로 명시하며 원저자 코드를 그대로 복제했다고 표현하지 않습니다. 이번 문서 수정에서 계산된 수치나 기존 구성식을 새로 바꾸지는 않았습니다.

## 공개 코드와 논문 전체 실험의 구분

공개 파일은 SciANN/TensorFlow, 8개 독립 함수, 5개 은닉층·40개 뉴런, 하드 경계 출력 변환 및 특정 학습 실행을 포함합니다. 이 파일 하나가 논문의 모든 소프트/하드 경계, 순차 학습, 구조 연구와 전이학습 설정을 완전히 공개한 것은 아닙니다. 전체 보고서는 공개 파일의 확인 가능한 사실과 논문의 조건, 재구성 선택을 구분합니다.

공개 코드의 일 기반 보조 항도 온도를 고정한 열탄성 퍼텐셜 에너지와 동일하게 부르지 않습니다. 내부 일·외부 일 차이와 구성식 연결·평형·경계 손실을 각각 명시합니다.

## 공개 FEAP 자료와의 차이

독립 FEniCS50 계산을 저자 51×51 자료에 대조한 결과, 온도의 상대 $`L^2`$ 차이는 약 0.00045%, 변위 벡터는 약 2.04%입니다. 최소제곱 진폭비는 저자/FEniCS 약 0.9800003입니다. 이 비율은 관측된 관계이며 원인은 공개 자료만으로 확정하지 못했습니다. 결과 진폭을 맞추는 보정은 적용하지 않았습니다. 수치 근거는 [FEM 검증 기록](../results/fem_report_verification.json)과 [독립 FEM 보고서](../reports/FEM_REPORT_KO.pdf)에 있습니다.
