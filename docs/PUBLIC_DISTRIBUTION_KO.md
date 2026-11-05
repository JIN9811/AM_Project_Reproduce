# 공개 자료의 범위와 출처

공개 저장소에는 자체 구현, 재현 절차, 한국어 보고서, 직접 계산한 그림, 검증된 FEniCS 기준해 및 완료된 신경망 실행을 수록한다. 진행 중인 작업의 부분 가중치·운영 로그·가상환경은 포함하지 않는다. 결과의 완료 목록과 배포 시점은 `validation/PUBLIC_SNAPSHOT.json`을 기준으로 한다.

## 원문 및 외부 자료

기준 논문의 전체 PDF는 공개 저장소에 포함하지 않는다. Word 비교 보고서에는 원문 데이터 그림 23개를 자체 계산 그림과 구분하여 수록하며, 해당 원본 그림은 `figures/paper_reference/paper_fig*.png`에 있다. 원본의 축·범례·색 막대를 보존하였고 CC BY-NC-ND 4.0 출처와 이용조건은 같은 폴더의 `provenance.json`과 `ATTRIBUTION.txt`에 명시한다. 프로젝트 코드의 MIT 라이선스는 이 원문 그림에 적용되지 않는다.

DOI와 PDF의 원본 SHA-256은 `docs/SOURCE_PROVENANCE.json`에 보존한다. 저자 공개 데이터와 가중치의 라이선스는 `data/AUTHORS_LICENSE`에 있다. 원문에서 추출한 수치 색 범위와 직접 계산한 결과 그림은 출처를 명시하여 제공한다.

공개 배포 설정에서는 포함된 자료와 원문 그림 23개의 SHA-256을 검사한다. 미배포 전체 PDF와 이전 `original_fig*.png` 발췌는 `not_distributed_not_verified`로 기록하며, 이 표시는 검증 성공을 뜻하지 않는다. `external_paper_and_crops_included=false`는 이 이전 자료 묶음의 호환성 필드이고, 현재 배포하는 `paper_fig*.png`는 `bundled_artifacts`에서 별도로 검증한다. 외부 PDF 없이 FEniCS/PINN 계산, 결과 비교, 노트북 및 세 보고서의 재생성이 가능하다. 원문 그림을 다시 추출할 때에는 지정된 SHA-256과 일치하는 전체 PDF가 필요하다.

## 스냅샷과 장기 학습

GitHub의 결과는 특정 시점에 검증된 자료이며 원래 컴퓨터의 실시간 대시보드가 아니다. `validation/PUBLIC_SNAPSHOT.json`에는 등록된 52개 작업의 완료 상태와 자료 확정 시각이 기록되어 있다. 완료된 실행의 모델과, 1M 연속 실행에 필요한 100k 선행 체크포인트를 포함한다. 새로운 학습은 복제한 환경의 `results/full_study/`에 저장되며 GitHub에 자동 업로드되지 않는다.

기존 단축 학습 실험은 공개본의 주 결과에 포함하지 않는다. 최신 전체·FEM·Word 비교 보고서와 노트북은 동일한 검증 완료 목록을 사용하며, 이전 보고서 내용은 Git 이력에서 확인한다.
