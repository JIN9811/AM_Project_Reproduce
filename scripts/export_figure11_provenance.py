"""Export the seven plotted curves and their numerical sources, without training."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from harandi.full_figures import extract_section, load_prediction
from harandi.full_study import catalogue, job_complete

METHODS = (
    ('baseline_circle_standard_coupled', 'Standard coupled', '빨강 파선'),
    ('baseline_circle_standard_sequential', 'Standard sequential', '빨강 점선'),
    ('baseline_circle_dem_coupled', 'DEM coupled', '초록 파선'),
    ('baseline_circle_dem_sequential', 'DEM sequential', '초록 점선'),
    ('mixed_circle_coupled', 'Mixed coupled', '파랑 파선'),
    ('mixed_circle_sequential', 'Mixed sequential', '파랑 점선'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT/'FULL_STUDY_RESULTS.json').read_text(encoding='utf-8'))
    jobs = {j['id']: j for j in catalogue()}
    records, samples = [], []
    for name, label, style in METHODS:
        assert job_complete(jobs[name], ROOT/'results/full_study'), name
        path = ROOT/'results/full_study'/name/'prediction.npz'
        assert sha(path) == manifest['jobs'][name]['prediction_sha256'], name
        data = load_prediction(path)
        y, fields, extraction = extract_section(data['xy'], data['derived'], .5)
        status = manifest['jobs'][name]['status_metadata']
        updates = status['completed_updates'] if 'completed_updates' in status else sum(status['completed'].values())
        records.append({'label': label, 'style': style, 'source': path.relative_to(ROOT).as_posix(),
                        'sha256': sha(path), 'job': name, 'actual_updates': updates,
                        'samples': len(y), 'extraction': extraction})
        samples.extend([[label, .5, float(yi), *map(float, row[[0,2,5,6]])] for yi, row in zip(y, fields)])
    path = ROOT/'results/fem_circle_n50/solution.npz'
    assert sha(path) == manifest['references']['circle_n50']['sha256']
    with np.load(path, allow_pickle=False) as fem:
        y, fields, extraction = extract_section(fem['xy'], fem['fields'], .5)
    records.append({'label': 'FEM50', 'style': '검정 실선', 'source': path.relative_to(ROOT).as_posix(),
                    'sha256': sha(path), 'samples': len(y), 'extraction': extraction})
    samples.extend([['FEM50', .5, float(yi), *map(float, row[[0,2,5,6]])] for yi, row in zip(y, fields)])
    assert len(samples) == 707 and all(r['extraction']['method'] == 'exact grid line' for r in records)
    target = ROOT/'results/figure11_sections.csv'
    with target.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['curve','x','y','ux','sxx_constitutive','T','qx_fourier'])
        writer.writerows(samples)
    provenance = {'figure': 'figures/full_study/fig11_circle_methods.png',
                  'figure_sha256': sha(ROOT/'figures/full_study/fig11_circle_methods.png'),
                  'data_origin': 'Project-trained models and independently computed FEniCS reference',
                  'paper_role': 'Benchmark and comparison layout, Figure 11; no digitized curves or copied pixels',
                  'paper_doi': '10.1002/nme.7388', 'curves': records,
                  'csv': target.relative_to(ROOT).as_posix(), 'csv_sha256': sha(target),
                  'total_rows': len(samples), 'numerical_sources_verified': True}
    (ROOT/'results/figure11_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    rows = ['| 곡선 | 표시 | 실제 Adam 갱신 | 원본 예측 배열 |', '|---|---|---:|---|']
    for r in records:
        rows.append(f"| {r['label']} | {r['style']} | {r.get('actual_updates','해당 없음'):,} | [NPZ](../{r['source']}) |" if 'actual_updates' in r else
                    f"| {r['label']} | {r['style']} | 해당 없음 | [NPZ](../{r['source']}) |")
    text = '''# 원형 단면 비교 그림의 데이터 출처

대상: 전체 보고서 그림 3, `fig11_circle_methods.png`.

**이 그림은 본 프로젝트에서 학습한 여섯 모델의 예측 결과와 본 프로젝트의 FEniCS 유한요소해를 직접 그린 것이다.** Harandi 등 논문의 Fig. 11은 벤치마크와 비교 항목의 출처이며, 원문 그림을 복사하거나 원문의 곡선을 디지타이징하지 않았다. 이 그림에 원저자의 사전학습 가중치 예측이나 저자의 FEAP 해를 섞지 않았다.

## 곡선별 수치 자료

'''+ '\n'.join(rows)+'''

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
'''
    (ROOT/'docs/FIGURE_11_PROVENANCE_KO.md').write_text(text, encoding='utf-8', newline='\n')
    print(f'Verified {len(records)} numerical sources; exported {len(samples)} rows.')


if __name__ == '__main__':
    main()
