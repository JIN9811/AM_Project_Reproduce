# 원형 단면 비교 그림의 데이터 출처

대상: 전체 보고서 그림 3, `fig11_circle_methods.png`.

**이 그림은 본 프로젝트에서 학습한 여섯 모델의 예측 결과와 본 프로젝트의 FEniCS 유한요소해를 직접 그린 것이다.** Harandi 등 논문의 Fig. 11은 벤치마크와 비교 항목의 출처이며, 원문 그림을 복사하거나 원문의 곡선을 디지타이징하지 않았다. 이 그림에 원저자의 사전학습 가중치 예측이나 저자의 FEAP 해를 섞지 않았다.

## 곡선별 수치 자료

| 곡선 | 표시 | 실제 Adam 갱신 | 원본 예측 배열 |
|---|---|---:|---|
| Standard coupled | 빨강 파선 | 100,000 | [NPZ](../results/full_study/baseline_circle_standard_coupled/prediction.npz) |
| Standard sequential | 빨강 점선 | 200,000 | [NPZ](../results/full_study/baseline_circle_standard_sequential/prediction.npz) |
| DEM coupled | 초록 파선 | 100,000 | [NPZ](../results/full_study/baseline_circle_dem_coupled/prediction.npz) |
| DEM sequential | 초록 점선 | 200,000 | [NPZ](../results/full_study/baseline_circle_dem_sequential/prediction.npz) |
| Mixed coupled | 파랑 파선 | 100,000 | [NPZ](../results/full_study/mixed_circle_coupled/prediction.npz) |
| Mixed sequential | 파랑 점선 | 200,000 | [NPZ](../results/full_study/mixed_circle_sequential/prediction.npz) |
| FEM50 | 검정 실선 | 해당 없음 | [NPZ](../results/fem_circle_n50/solution.npz) |

파선은 동시(coupled), 점선은 순차(sequential) 학습이다. 동시 학습은 100,000번의 공동 갱신, 순차 학습은 열·역학 각각 100,000번으로 합계 200,000번이다. 따라서 양쪽의 전체 갱신 수를 모두 100,000번으로 해석하면 안 된다. 완료 여부와 예측 파일 해시는 내보내기 시 실제 자료와 대조한다.

검정선의 FEniCS50은 50×50 Q1 사각형 요소, 2,500요소·2,601절점의 자체 계산이다. 보고서 일반 정량 표의 FEM200과는 별도의 비교 기준이다.

## 추출과 후처리

- 이 데이터는 학습에 입력한 정답 배열이 아니라 **학습이 끝난 모델의 평가 예측값**이다.
- 각 원본의 101×101 평가 좌표에서 x=0.5에 해당하는 101개 점을 그대로 선택한다. 단면 보간·평활화·진폭 보정은 적용하지 않는다.
- 그림의 네 성분은 수평변위, 수평 정상응력, 온도, 수평 열유속이다.
- 신경망 응력은 `constitutive_stress`, 열유속은 `fourier_flux`를 사용한다. 혼합망의 독립 응력·열유속 출력과 구분한다. FEM 응력·열유속은 FEM 결과의 복원 장이다.
- 원자료: [단면 CSV · 707행](../results/figure11_sections.csv), [파일별 SHA-256과 출처 JSON](../results/figure11_provenance.json).
- 생성 코드: [원본 배열 읽기·단면 추출·그리기](../harandi/full_figures.py), [출처·단면 CSV 내보내기](../scripts/export_figure11_provenance.py).

## 논문 레퍼런스와 캡션

Harandi, A., Moeineddin, A., Kaliske, M., Reese, S., & Rezaei, S. (2024). *Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains*. International Journal for Numerical Methods in Engineering, 125(4), e7388. [DOI: 10.1002/nme.7388, Fig. 11](https://doi.org/10.1002/nme.7388). 온라인 선게재: 2023년 11월 8일.

권장 캡션: **본 프로젝트의 학습 결과와 독립 FEniCS 해석에 의한 원형 개재물 x=0.5 단면 비교. 벤치마크와 비교 구성은 Harandi et al. (2024), Fig. 11을 참조하였다. 모든 곡선은 자체 계산 자료로 재작성하였으며, 원문 그림의 전재 또는 디지타이징이 아니다.**

재생성: 프로젝트 루트에서 `python scripts/export_figure11_provenance.py`를 실행한다. 신경망 재학습이나 FEM 재계산은 수행하지 않는다.
