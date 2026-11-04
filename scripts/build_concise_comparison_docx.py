"""Build the concise Korean Word comparison from a verified result snapshot.

No training, checkpoint selection, figure digitisation or prediction rescaling.
Requires python-docx, numpy and Pillow. Render the resulting DOCX before release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/CONCISE_COMPARISON_KO.docx'
FIELDS = ('ux', 'uy', 'sxx', 'syy', 'sxy', 'T', 'qx', 'qy')
LABEL = ('uₓ', 'uᵧ', 'σₓₓ', 'σᵧᵧ', 'σₓᵧ', 'T', 'qₓ', 'qᵧ')
SNAP = json.loads((ROOT / 'validation/FULL_STUDY_RESULTS.json').read_text('utf-8'))
LIMITS = json.loads((ROOT / 'data/paper_color_limits.json').read_text('utf-8'))
EVIDENCE = {'result_snapshot_sha256': hashlib.sha256((ROOT / 'validation/FULL_STUDY_RESULTS.json').read_bytes()).hexdigest(),
            'completed_jobs': SNAP['completed_jobs'], 'figures': [], 'metrics_verified': [],
            'excluded_conceptual_figures': ['1', '2', '3', '4', '22'],
            'source_data_cutoff_kst': '2026-11-05 03:13 KST',
            'comparison_reference': 'FEM50; mesh convergence uses FEM200',
            'prediction_representation': 'constitutive stress and Fourier flux derived from u,T'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_data():
    """Independently recompute reported FEM50 metrics from saved NPZ arrays."""
    assert len(SNAP['completed_jobs']) == 26, 'Review the prose and budgets before using another snapshot.'
    for name in SNAP['completed_jobs']:
        job = SNAP['jobs'][name]
        pred_path = ROOT / 'results/full_study' / name / 'prediction.npz'
        assert digest(pred_path) == job['prediction_sha256'], name
        if name.startswith('transfer_'):
            ref_id, cmp_id = 'ratio15_n50', 'FEM_ratio15_n50'
        else:
            ref_id, cmp_id = job['config']['geometry'] + '_n50', 'FEM_n50'
        ref_spec = SNAP['references'][ref_id]
        ref_path = ROOT / ref_spec['source'].replace('\\', '/')
        assert digest(ref_path) == ref_spec['sha256'], ref_id
        with np.load(pred_path) as p, np.load(ref_path) as r:
            pi = np.lexsort((p['xy'][:, 0], p['xy'][:, 1]))
            ri = np.lexsort((r['xy'][:, 0], r['xy'][:, 1]))
            assert np.allclose(p['xy'][pi], r['xy'][ri], atol=1e-12, rtol=0)
            values = p['fields'].copy()
            if 'constitutive_stress' in p:
                values[:, 2:5] = p['constitutive_stress']
                values[:, 6:8] = p['fourier_flux']
            values = values[pi]
            truth = r['fields'][ri]
            for k, field in enumerate(FIELDS):
                err = np.abs(values[:, k] - truth[:, k])
                recorded = job['comparisons'][cmp_id]['derived'][field]
                calculated = {'mean_absolute_error': err.mean(), 'max_absolute_error': err.max(),
                              'relative_l2': np.linalg.norm(err) / np.linalg.norm(truth[:, k]),
                              'mean_absolute_error_over_reference_peak_percent': 100 * err.mean() / np.abs(truth[:, k]).max()}
                for metric, value in calculated.items():
                    assert np.isclose(value, recorded[metric], rtol=1e-10, atol=1e-12), (name, field, metric)
        EVIDENCE['metrics_verified'].append({'job': name, 'prediction_sha256': digest(pred_path),
                                            'reference_sha256': digest(ref_path), 'checks': 32})


def m(job, field, metric='mean_absolute_error_over_reference_peak_percent'):
    key = 'FEM_ratio15_n50' if job.startswith('transfer_') else 'FEM_n50'
    return SNAP['jobs'][job]['comparisons'][key]['derived'][field][metric]


def val(number):
    return f'{number:.4g}'


def peak(job, field):
    return m(job, field, 'max_absolute_error')


def run_font(run, size=None, bold=None):
    run.font.name = 'Malgun Gothic'
    run.font.color.rgb = RGBColor(0, 0, 0)
    rp = run._element.get_or_add_rPr()
    rf = rp.rFonts
    if rf is None:
        rf = OxmlElement('w:rFonts'); rp.insert(0, rf)
    for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        rf.set(qn('w:' + attr), 'Malgun Gothic')
    if size: run.font.size = Pt(size)
    if bold is not None: run.bold = bold
    return run


DOC = Document()
SEC = DOC.sections[0]
SEC.page_width, SEC.page_height = Cm(21), Cm(29.7)
SEC.top_margin, SEC.bottom_margin = Cm(1.55), Cm(1.55)
SEC.left_margin, SEC.right_margin = Cm(1.65), Cm(1.65)
SEC.header_distance, SEC.footer_distance = Cm(.65), Cm(.65)
for name, size in [('Normal', 10), ('Title', 20), ('Heading 1', 14), ('Heading 2', 11.5), ('Caption', 8.5)]:
    st = DOC.styles[name]; st.font.name = 'Malgun Gothic'; st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0, 0, 0)
    rf = st.element.rPr.rFonts
    for attr in list(rf.attrib):
        if attr.endswith('Theme'):
            del rf.attrib[attr]
    for attr in ('ascii', 'hAnsi', 'eastAsia'):
        rf.set(qn('w:' + attr), 'Malgun Gothic')
    st.paragraph_format.space_after = Pt(5)
    st.paragraph_format.line_spacing = 1.12
    if name.startswith('Heading'):
        st.paragraph_format.space_before = Pt(0)
        st.paragraph_format.keep_with_next = True
for border in list(DOC.styles['Title'].element.xpath('.//w:pBdr')):
    border.getparent().remove(border)
footer = SEC.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_font(footer.add_run('Harandi 열탄성 재현 비교  ·  '), 8)
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
DOC.core_properties.title = 'Harandi 열탄성 FEM PINN 실험 조건과 결과 비교'
DOC.core_properties.subject = '전체 조건 비교와 논문 데이터 그림 23개의 대응 및 의견'
DOC.core_properties.author = 'AM Project'
DOC.core_properties.keywords = 'FEniCS, PINN, Harandi, 재현, 비교'


def p(text='', bold=False, size=None, style=None):
    para = DOC.add_paragraph(style=style)
    run_font(para.add_run(text), size, bold)
    para.paragraph_format.widow_control = True
    return para


def page(title):
    DOC.add_page_break()
    DOC.add_paragraph(title, 'Heading 1')


def table(headers, rows, widths, size=9.1):
    t = DOC.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    pr = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement('w:' + edge); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'D9D9D9'); borders.append(e)
    pr.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for edge, value in [('top', '65'), ('bottom', '65'), ('left', '95'), ('right', '95')]:
        e = OxmlElement('w:' + edge); e.set(qn('w:w'), value); e.set(qn('w:type'), 'dxa'); margins.append(e)
    pr.append(margins)
    for col, width in zip(t.columns, widths): col.width = Cm(width)
    for i, values in enumerate([headers, *rows]):
        row = t.rows[0] if i == 0 else t.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if i == 0: trpr.append(OxmlElement('w:tblHeader'))
        for k, (cell, text, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Cm(width); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pp = cell.paragraphs[0]; pp.paragraph_format.space_after = Pt(0)
            pp.paragraph_format.line_spacing = 1.08
            if i == 0: pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_font(pp.add_run(str(text)), size, i == 0)
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E5EBEF' if i == 0 else ('F7F8FA' if i % 2 == 0 else 'FFFFFF')); cell._tc.get_or_add_tcPr().append(shade)
    DOC.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


def figure(fid, filename, title, paper_page, width=17.4, max_height=None):
    path = ROOT / 'figures/full_study' / filename
    with Image.open(path) as im: ratio = im.height / im.width
    if max_height: width = min(width, max_height / ratio)
    para = DOC.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.keep_with_next = True
    pic = para.add_run().add_picture(str(path), width=Cm(width))
    pic._inline.docPr.set('descr', f'자체 계산 결과. 논문 그림 {fid} 대응. {title}')
    cap = p(f'자체 계산 · 논문 Fig. {fid} 대응  |  {title}  [1, p. {paper_page}]', style='Caption')
    EVIDENCE['figures'].append({'paper_figure': str(fid), 'paper_page': paper_page,
        'path': str(path.relative_to(ROOT)).replace('\\','/'), 'sha256': digest(path), 'embedded_width_cm': width})


def comparison(paper, ours, opinion):
    ours = ours.removeprefix('본 실험 ')
    para = DOC.add_paragraph(); para.paragraph_format.space_after = Pt(5)
    for head, text in [('논문 ', paper), ('  본 실험 ', ours), ('  의견 ', opinion)]:
        run_font(para.add_run(head), 9.4, True)
        run_font(para.add_run(text), 9.4)


def all_field_table(job, geometry):
    rows = []
    for i, f in enumerate(FIELDS):
        rows.append([LABEL[i], val(LIMITS[geometry][f]['error'][1]), val(peak(job,f)), f'{m(job,f):.2f}'])
    table(['성분', '논문 색 막대 상한', '본 실험 최대오차', '본 실험 평균오차 %'], rows, [2.2,5,5,5.5], 8.9)
    p('논문 열은 인쇄된 오차 색 막대 상한이며 실제 최댓값으로 단정하지 않는다. 수치는 그림과 같은 FEM50 기준이다.', size=8.4)


def main():
    verify_data()
    DOC.add_paragraph('Harandi 열탄성 재현 비교', 'Title')
    p('FEM과 PINN의 전체 실험 조건 및 데이터 그림별 의견', size=12)
    p('혼합 PINN의 개선 경향은 확인되지만, 모든 반례와 방법 순위가 논문과 일치하지 않는다. 아래 그림은 모두 자체 학습·FEniCS 계산 결과이며, 논문 그림을 가져온 결과가 아니다.', size=10)
    p('수치 자료 기준 2026-11-05 03:13 KST  ·  신경망 26/27개 완료  ·  별도 시간 측정 0/25개', size=8.7)
    DOC.add_paragraph('1 물리 조건과 FEM 비교', 'Heading 1')
    table(['항목', '논문 및 저자 자료', '본 실험 FEM과 PINN'], [
        ['해석 영역', '정상 평면변형률 열탄성, [0,1]²', '동일 영역과 단방향 열→역학 연성'],
        ['길이와 물리 단위', 'Fig. 2의 μm와 본문 mm가 충돌', '수치 단위 영역; mm·GPa·K·W/(m K)는 원문 표기. 별도 SI 환산 없음'],
        ['원형 개재물', '직경 0.30, 중심 미기재', '중심 (0.60,0.35), 반지름 0.15'],
        ['사각형 개재물', '4개; 위치 미기재. 좌상단 높이 0.20', '저자 재료 자료를 사용. 좌상단 높이 0.22'],
        ['원형 재료 E와 k', '기지 0.3 / 개재물 1.0', 'FEM과 PINN 공통으로 동일'],
        ['사각형 재료 E와 k', '기지 1.0 / 개재물 0.5', 'FEM과 PINN 공통으로 동일'],
        ['ν와 열팽창', 'ν=0.3; α와 T₀의 수치 미기재', 'ν=0.3, α=1, T₀=0. α는 벤치마크 값'],
        ['변위 경계', '좌우 uₓ=0, 상하 uᵧ=0', '동일 롤러 구속 및 접선력 0. 전면 완전고정 아님'],
        ['열 경계', '좌 T=1, 우 T=0, 상하 qᵧ=0', '동일 온도 및 단열 조건'],
        ['FEM 해법', 'FE 기준해; 공개 저장소에 FEAP 자료', 'FEniCS DOLFIN 2019.2, 사각형 Q1. T 해석 후 u 해석'],
        ['FEM 격자', 'Fig. A1에서 변당 50, 100, 200', '원형·사각형 각각 50², 100², 200² 요소'],
        ['FEM 재료 및 복원', '세부 적분·복원 조건 불충분', '요소 중심의 불연속 재료. 요소 응력·열유속을 절점으로 복원'],
        ['PINN 재료 표현', '저자 재료 자료·보간 활용', '50² 재료 격자에서 절점 평균 후 쌍선형 보간. FEM의 불연속 계수와 차이'],
    ], [3,6.7,8], 8.9)
    p('사각형 실제 좌표 (x구간; y구간): (0.10–0.34; 0.12–0.44), (0.50–0.92; 0.18–0.42), (0.42–0.94; 0.52–0.94), (0.06–0.34; 0.60–0.82). [1–3]', size=8.5)

    page('2 PINN 공통 설정과 구현 차이')
    table(['항목', '논문 조건', '본 실험 조건'], [
        ['혼합 신경망', '8개 독립 스칼라 망; 5층×40, tanh', '동일. 출력 uₓ,uᵧ,σₓₓ,σᵧᵧ,σₓᵧ,T,qₓ,qᵧ'],
        ['표준 PINN 및 DEM', '주 변수 u,T; 기본 5층×40, tanh', 'uₓ,uᵧ,T의 3개 독립 망. 응력·열유속은 미분 계산'],
        ['원형 학습점', '약 5,000개, 계면·경계 보강', '51² 격자+내부 균등 난수 2,399개=5,000. 원문 배치와 다름'],
        ['사각형 학습과 평가', '학습 2,601개, 평가 10,201개', '51² 학습, 모든 문제 101² 평가. 보강 실험은 1,000개 추가'],
        ['정밀도와 초기값', '시드·정밀도 미기재', 'float64, seed=42. 단일 시드'],
        ['Adam', '학습률 0.001, full batch', '동일 학습률·배치. 동시 100,000회'],
        ['순차 학습', '열/역학 각 100,000회, 20,000회 교대', '총 200,000회. 비활성 망 동결; 새 역학 블록에서 Adam 상태 초기화'],
        ['혼합 손실', '평형·연결·경계 MSE, 에너지 MAE', '저자 코드의 내부 일−외부 일 항, 에너지 가중치 1. 전이 w는 전체 물리 손실 가중치'],
        ['일반 경계 처리', '기본 soft, 별도 hard 연구', '기본 단위 페널티 soft. DEM hard 2개는 추가 민감도 실험'],
        ['hard 출력 변환', '식 62–64', 'uₓ=x(x−1)Nₓ, uᵧ=y(y−1)Nᵧ, T=1−x+x(x−1)Nₜ'],
        ['L-BFGS 세부값', '10,000회; 선 탐색 등 미기재', 'lr=1, strong-Wolfe, history=80, max_iter=1, max_eval=25; 허용오차 10⁻¹⁰/10⁻¹⁴'],
        ['전단 구성식', '저자 코드의 텐서 전단에 μ를 곱함', 'FEM·PINN 모두 σₓᵧ=2μεₓᵧ로 통일. 문자 그대로 복제한 구현은 아님'],
        ['경계와 에너지 해석', '식 56의 역학 NBC=0과 손실 그림 충돌', '롤러의 접선력 손실 사용. 일 항을 0이어야 하는 물리 잔차로 해석하지 않음'],
        ['DEM 온도 처리', '공동 최소화 세부 경로 미기재', '역학 에너지에서 T 기울기 차단. 표준·혼합 동시 학습과 경로가 다름'],
        ['표준 PINN 미분', '미분·재료 보간 세부 미기재', '보간 결절을 피한 점과 계수 미분. tanh 해석적 공간 미분 사용'],
        ['하드웨어와 가속', '기본 CPU Xeon Platinum 8160; 시간 연구 M2 Pro', 'RTX 3090, PyTorch 2.11 CUDA; Adam CUDA graph. 기기별 시간 직접 등치 불가'],
        ['저장과 결과 선택', '상세 정책 미기재', '1,000 업데이트마다 checkpoint 덮어쓰기. 비교는 예정 마지막 업데이트; 중간 최적값 선택 없음'],
    ], [3,6.5,8.2], 8.6)
    p('L-BFGS 한 업데이트에는 여러 손실·기울기 평가가 들어갈 수 있다. 동일 반복 수는 동일 계산비용을 뜻하지 않는다. [1–3]', size=8.5)

    page('3 전체 실험 구성과 비교 기준')
    table(['실험군', '논문 조건', '본 실험 및 완료 수'], [
        ['원형 기본 6개', '혼합·표준·DEM × 동시·순차; 5×40', 'Adam 동시 100k / 순차 200k, soft. 6/6'],
        ['사각형 기본 2개', '혼합 동시·순차; 5×40', 'Adam 동시 100k / 순차 200k, soft. 2/2'],
        ['좌측 경계 보강', '약 1,000점 추가; 별도 예산 미기재', '100k 동시 Adam, 3,601점, 새 학습. 1/1'],
        ['에너지 제외', '제외하면 악화한다는 정성 관측만 제시', '원형 동시 Adam 100k, 양 장의 에너지 항 제외. 1/1'],
        ['hard Adam', '사각형, 2,601점, 10k', '5×40, 동일 반복 수. 1/1'],
        ['hard L-BFGS 구조', '깊이 1,2,4,5,6 × 너비40; 5층 × 너비5,10,20,40,60', '각 10k; 중복 5×40 제외 9/9. Fig. 17–18·B1–B2도 5×40 재사용'],
        ['전이 4개', '재료비 1…10, 각 10k; 시험15. w=0,0.1 명시, 상한1', 'Adam 총100k씩, w=0,0.1,0.5,1. 4/4. 0.5는 재구성 선택'],
        ['추가 DEM hard', '별도 보고 없음', '원형 동시100k·순차200k. 2/2'],
        ['Remark 7', '사각형 Adam 1M. 기본100k와 관계 불명확', 'soft 100k 모델에서 900k 추가. 자료 확정 시 학습 중, 0/1'],
        ['Table 4 시간', '5깊이 × 5너비, 각100 L-BFGS 평균', '25조합 모두 등록. 독립 측정 미수행 0/25'],
        ['FEM 참조 19개', '격자 수렴 및 전이용 FEM 데이터', '기본6+절점 재료 민감도2+전이 비율11=19개 완료'],
    ], [3,7,7.7], 9)
    DOC.add_paragraph('전이학습의 추가 조건', 'Heading 2')
    p('Eᵣ=kᵣ=E기지/E개재물=k기지/k개재물을 1부터 10까지 순서대로 학습한다. 기지 E=k=1, 개재물 E=k=1/r. 입력은 x,y,log₁₀Eᵣ,log₁₀kᵣ이며 51²점의 FEM50 8성분 비정규화 MSE를 사용한다. soft 경계, 비율 전환 시 가중치만 전달하고 Adam은 초기화한다. 비율15는 학습·모델 선택에 사용하지 않는다.', size=9.4)
    DOC.add_paragraph('그림을 읽는 기준', 'Heading 2')
    p('이 문서의 평균오차(%)는 100×평균|PINN−FEM|/최대|FEM|이다. 별도 표기가 없으면 FEM50의 101² 좌표에서 비교하고, 응력·열유속은 u,T를 미분한 구성식·Fourier 출력이다. 독립 혼합 출력과는 구분한다. 격자 수렴만 FEM200을 기준으로 한다.', size=9.4)
    p('논문의 평균 상대오차 정의가 충분히 명시되지 않아 서로 같은 지표라고 단정하지 않는다. 색 범위를 맞춘 그림에서도 범위 밖 값은 삼각형으로 표시하며, 오차 수치를 잘라내지 않는다. 개념도 Fig. 1–4, 22를 제외한 데이터 그림 23개를 모두 비교한다.', size=9.4)

    page('4 배치점과 변형 형상')
    figure(5,'fig05_circle_points_mesh.png','원형 학습점과 평가점 및 FEM 격자',13,width=14.1)
    comparison('계면·경계에 보강한 약 5,000점.', '같은 점수이지만 규칙 격자와 균등 난수 조합.', '배치가 달라 국소 계면 오차의 완전 일치는 기대하기 어렵다.')
    figure(7,'fig07_circle_deformation.png','원형 변형 형상',15,width=14.1)
    comparison('원형의 미변형·PINN·FE 변형 형상.', '저장 변위로 계산한 변형 형상이며 별도 배율 보정 없음.', '눈으로 보이는 유사성은 변위 오차 지표와 함께 판단한다.')
    figure(12,'fig12_rectangles_points_mesh.png','사각형 학습점과 평가점 및 FEM 격자',20,width=14.1)
    comparison('2,601개 학습점과 10,201개 평가점.', '51²/101²로 점수는 일치.', '좌상단 개재물 치수와 원문 FE 도해의 위치 차이는 남는다.')

    page('5 원형 동시 및 순차 학습 이력')
    figure(6,'fig06_circle_coupled_losses.png','동시 Adam 100000회',14,width=11.4)
    j=SNAP['jobs']['mixed_circle_coupled']['history']['rows'][-1]
    comparison('에너지 항의 정체와 나머지 항의 감소.', f"최종 총손실 {j['total']:.3g}; 에너지 M={j['M_energy']:.3g}, T={j['T_energy']:.3g}.", '손실 감소만으로 FEM 오차 감소나 에너지 항의 필요성을 단정할 수 없다.')
    figure(10,'fig10_circle_sequential_losses.png','각 장 Adam 100000회와 총 200000회',18,width=11.4)
    comparison('열·역학을 20,000회씩 교대하며 비활성 망 동결.', '각 장의 실제 활성 업데이트만 표시.', '역학 학습만으로 동결된 정상 열전도 망이 변하는 현상을 인위적으로 만들지 않았다.')

    page('6 원형 역학 전체장')
    comparison('Fig. 8의 응력 오차가 계면 부근에 집중.', f"본 실험 σₓₓ 최대오차 {peak('mixed_circle_coupled','sxx'):.4f}; 원문 색 막대 상한 0.0632.", '전단 구성식과 재료 표현 차이가 있어 같은 색 분포를 정확도 일치로 해석하지 않는다.')
    figure(8,'fig08_circle_coupled_mechanical.png','PINN과 FEM50 및 5개 역학 성분의 절대오차',16,width=14.4)

    page('7 원형 열 전체장과 성분별 오차')
    comparison('Fig. 9의 T 오차 색 막대 상한 0.0109.', f"본 실험 T 최대오차 {peak('mixed_circle_coupled','T'):.4f}.", '온도보다 미분량인 열유속에서 상대오차가 더 크다.')
    figure(9,'fig09_circle_coupled_thermal.png','PINN과 FEM50 및 3개 열 성분의 절대오차',17,width=16)
    all_field_table('mixed_circle_coupled','circle')

    page('8 원형 방법별 단면 비교')
    comparison('혼합 PINN이 표준 PINN·DEM보다 우수한 경향.', '혼합 방식의 평균오차가 비교한 주요 4성분에서 작음.', '순차가 항상 우수하지는 않으며 기본 DEM은 경계 페널티의 영향을 받는다.')
    figure(11,'fig11_circle_methods.png','직접 학습한 6개 모델과 FEniCS50의 x=0.5 단면',19,width=17.4)
    names=['mixed_circle_coupled','mixed_circle_sequential','baseline_circle_standard_coupled','baseline_circle_standard_sequential','baseline_circle_dem_coupled','baseline_circle_dem_sequential']
    labels=['혼합 동시','혼합 순차','표준 동시','표준 순차','DEM 동시','DEM 순차']
    table(['방법','uₓ 평균 %','σₓₓ 평균 %','T 평균 %','qₓ 평균 %'],[[lab]+[f'{m(name,f):.2f}' for f in ('ux','sxx','T','qx')] for name,lab in zip(names,labels)],[4.5,3.3,3.3,3.3,3.3])
    p(f"추가 DEM hard 제어에서 동시 uₓ 평균오차는 {m('baseline_circle_dem_coupled','ux'):.2f}%→{m('sensitivity_circle_dem_hard_coupled','ux'):.2f}%로 감소했지만 qₓ 최대오차는 {peak('baseline_circle_dem_coupled','qx'):.3g}→{peak('sensitivity_circle_dem_hard_coupled','qx'):.3g}로 증가했다. 경계 처리 개선이 모든 장의 개선을 보장하지 않는다.",size=9.3)

    page('9 사각형 역학 전체장')
    comparison('Fig. 13에서 soft 경계 부근 오차를 관측.', f"본 실험 uₓ 최대오차 {peak('mixed_rectangles_coupled','ux'):.4f}; 원문 색 막대 상한 0.0289.", '국소 오차의 위치와 함께 FEM 재료 경계·복원 차이를 고려해야 한다.')
    figure(13,'fig13_rectangles_coupled_mechanical.png','PINN과 FEM50 및 5개 역학 성분의 절대오차',21,width=14.4)

    page('10 사각형 열 전체장과 성분별 오차')
    comparison('Fig. 14의 온도·열유속 분포와 계면 오차.', f"본 실험 T 평균오차 {m('mixed_rectangles_coupled','T'):.2f}%, qₓ {m('mixed_rectangles_coupled','qx'):.2f}%.", '온도의 낮은 오차가 열유속의 같은 수준 정확도를 뜻하지 않는다.')
    figure(14,'fig14_rectangles_coupled_thermal.png','PINN과 FEM50 및 3개 열 성분의 절대오차',22,width=16)
    all_field_table('mixed_rectangles_coupled','rectangles')

    page('11 경계점 보강과 사각형 단면 비교')
    figure(15,'fig15_left_refinement.png','기본 배치와 왼쪽 1000점 추가',23,width=14.5)
    ur=peak('mixed_rectangles_coupled','ux')/peak('refine_rectangles_left1000','ux')
    tr=peak('mixed_rectangles_coupled','T')/peak('refine_rectangles_left1000','T')
    comparison('최대오차 감소 배수 uₓ=1.6, T=3.0.', f'uₓ={ur:.2f}배, T={tr:.2f}배. 1 미만은 악화.', '변위 개선만 재현되며 온도 개선은 재현되지 않았다.')
    figure(16,'fig16_rectangles_sections.png','동시 및 순차 학습의 x=0.5 단면',23,width=14.5)
    comparison('주요 단면에서 순차 학습의 개선을 논의.', f"순차 uₓ 평균오차 {m('mixed_rectangles_sequential','ux'):.2f}%는 동시 {m('mixed_rectangles_coupled','ux'):.2f}%보다 작지만 σₓₓ는 {m('mixed_rectangles_coupled','sxx'):.2f}%→{m('mixed_rectangles_sequential','sxx'):.2f}%로 증가.", '학습 순위는 성분별로 다르다.')

    page('12 강제 경계조건과 최적화기')
    figure(17,'fig17_boundary_optimizer.png','soft Adam과 hard Adam 및 hard L-BFGS',24,width=16.2)
    comparison('hard Adam과 hard L-BFGS를 각각 10k 비교.', f"hard uₓ 평균오차는 Adam {m('hard_rectangles_adam10k','ux'):.2f}%, L-BFGS {m('arch_l5_w40','ux'):.2f}%.", '본 실험은 Adam이 더 작다. 그림의 soft 100k는 보조 조건이며 동일 예산 비교가 아니다.')
    figure(18,'fig18_hard_objective.png','기록된 실제 최적화 업데이트와 총손실',24,width=16.2)
    comparison('L-BFGS의 빠른 손실 감소.', f"최종 총손실 Adam {SNAP['jobs']['hard_rectangles_adam10k']['history']['rows'][-1]['total']:.3f}, L-BFGS {SNAP['jobs']['arch_l5_w40']['history']['rows'][-1]['total']:.3f}.", '원문 순위가 재현되지 않았다. 반복 수와 closure 평가 수를 혼동하지 않는다.')

    page('13 선택한 구조의 오차 분포와 에너지 제외')
    comparison('Fig. 19는 2×40, 5×20, 5×40, 6×40을 비교.', f"본 실험 uₓ 평균오차는 차례로 {', '.join(f'{m(n,"ux"):.2f}%' for n in ['arch_l2_w40','arch_l5_w20','arch_l5_w40','arch_l6_w40'])}.", '큰 망이 항상 더 정확하지 않으며 5×20의 오차가 특히 크다.')
    figure(19,'fig19_architectures.png','동일 10000 L-BFGS 예산의 구조별 절대오차',25,width=16)
    DOC.add_paragraph('원문의 에너지 제외 관측도 별도로 확인', 'Heading 2')
    table(['조건','uₓ 평균 %','σₓₓ 평균 %','T 평균 %','qₓ 평균 %'],[[label]+[f'{m(n,f):.2f}' for f in ('ux','sxx','T','qx')] for n,label in [('mixed_circle_coupled','에너지 포함'),('ablation_circle_noenergy','에너지 제외')]],[4.5,3.3,3.3,3.3,3.3])
    p(f"논문 p.13은 에너지 제외 시 악화를 서술하지만 수치·독립 그림을 주지 않는다. 본 실험에서는 주요 평균오차가 감소했고, σₓₓ 최대오차는 {peak('mixed_circle_coupled','sxx'):.3f}→{peak('ablation_circle_noenergy','sxx'):.3f}로 증가했다. 원문의 일반적 악화 주장은 같은 형태로 재현되지 않았다.",size=9.3)

    page('14 은닉층 수에 따른 오차')
    comparison('Fig. 20은 너비40, 깊이 1·2·4·5·6 연구이며 캡션 오기 존재.', f"σₓₓ 최대오차 4층={peak('arch_l4_w40','sxx'):.3f}, 5층={peak('arch_l5_w40','sxx'):.3f}, 6층={peak('arch_l6_w40','sxx'):.3f}.", '5층의 응력 최대오차 악화 예외는 재현되지 않았다. 전체 경향도 단조롭지 않다.')
    figure(20,'fig20_depth_derived.png','너비40에서 8개 성분의 평균 및 최대 절대오차',26,width=17)
    p('원문 Fig. 20의 축과 본문을 따라 깊이를 비교한다. 존재하지 않는 3층 실험을 보간하지 않았다.',size=9.1)

    page('15 은닉층 너비와 시간 측정 범위')
    comparison('Fig. 21은 5층에서 너비 5·10·20·40·60 비교.', f"uₓ 평균오차 너비20={m('arch_l5_w20','ux'):.2f}%, 40={m('arch_l5_w40','ux'):.2f}%, 60={m('arch_l5_w60','ux'):.2f}%.", '너비 증가가 항상 개선을 주지 않는 점은 일치한다. 단일 시드로 최적 너비를 일반화할 수 없다.')
    figure(21,'fig21_width_derived.png','5개 은닉층에서 8개 성분의 평균 및 최대 절대오차',27,width=16.2)
    p('Table 4 전체 범위: 깊이 {1,2,4,5,6} × 너비 {5,10,20,40,60}, 총25조합. 논문은 M2 Pro에서 각100회 평균을 보고했으나 본 컴퓨터의 독립 측정은 25개 모두 미수행이다. 수렴 학습 중의 시간을 동일 벤치마크 값으로 대체하지 않았다.',size=9.3)
    p('Remark 7의 Adam 100만 회 추가 실험도 자료 확정 시 미완료이므로 최종 성능을 제시하지 않는다.',size=9.3)

    page('16 미학습 재료비에서 물리와 데이터의 결합')
    comparison('Fig. 23은 비율15, w=0.1의 4개 주요 성분.', f"본 실험 평균오차 uₓ={m('transfer_w0.1','ux'):.2f}%, σₓₓ={m('transfer_w0.1','sxx'):.2f}%, T={m('transfer_w0.1','T'):.2f}%, qₓ={m('transfer_w0.1','qx'):.2f}%.", '응력의 큰 오차가 남으므로 정확한 외삽 재현으로 볼 수 없다.')
    figure(23,'fig23_transfer.png','FEM50 비율15 기준의 w=0.1 예측과 절대오차',28,width=16.8)
    p('원문 색 범위 밖의 값은 색 막대 삼각형으로 표시한다. 비율15는 학습이나 최종 체크포인트 선택에 쓰지 않았다.',size=9.1)

    page('17 미학습 재료비에서 데이터만 사용한 경우')
    comparison('Fig. 24는 같은 비율15, w=0의 데이터 전용 모델.', f"본 실험 평균오차 uₓ={m('transfer_w0','ux'):.2f}%, σₓₓ={m('transfer_w0','sxx'):.2f}%, T={m('transfer_w0','T'):.2f}%, qₓ={m('transfer_w0','qx'):.2f}%.", 'w=0.1과 비교하면 변위·응력·qₓ는 나쁘지만 T는 더 좋다.')
    figure(24,'fig24_transfer.png','FEM50 비율15 기준의 w=0 예측과 절대오차',29,width=16.8)
    p('추가 가중치의 평균오차 (uₓ, σₓₓ, T, qₓ): '+ '; '.join(f'w={w}: '+', '.join(f'{m("transfer_w"+w,f):.2f}%' for f in ('ux','sxx','T','qx')) for w in ['0.5','1'])+'. w=0.5는 원문에 명시되지 않은 추가 선택이다.',size=9.1)

    page('18 전이학습 단면과 악화 사례')
    comparison('Fig. 25의 패널·본문은 x=0.35, 캡션은 x=0.5.', '패널·본문의 x=0.35를 사용.', '논문의 qₓ 최대오차 악화와 본 실험의 T·qᵧ 악화를 구분해야 한다.')
    figure(25,'fig25_transfer_sections.png','비율15에서 데이터 전용과 물리 결합의 단면',30,width=16.5)
    table(['성분','w=0 평균 %','w=0.1 평균 %','w=0 최대','w=0.1 최대'],[[LABEL[i],f'{m("transfer_w0",f):.2f}',f'{m("transfer_w0.1",f):.2f}',val(peak('transfer_w0',f)),val(peak('transfer_w0.1',f))] for i,f in enumerate(FIELDS)],[2.1,3.9,3.9,3.9,3.9],8.7)
    p('8성분 중 6개는 평균·최대오차가 감소했지만 T와 qᵧ는 악화했다. 원문 qₓ 오차 색 막대 상한은 0.2573→0.2787이나, 본 실험 qₓ 최대오차는 '+f"{peak('transfer_w0','qx'):.4f}→{peak('transfer_w0.1','qx'):.4f}"+'로 감소했다.',size=9.1)
    p('기록된 w=0과 0.1 학습 시간은 각각 1,258.3초와 3,713.9초로 약66.1% 절감이었다. 논문의 약82%와 다르며, 독립·동일 하드웨어 시간 벤치마크는 아니다.',size=9.1)

    page('19 FEM 격자 수렴')
    comparison('Fig. A1은 변당100개에서 거의 수렴한다고 정성 평가.', '100→200에서 uₓ·T 오차는 작으나 응력·열유속은 더 큰 차이.', '100격자를 무오차 기준해로 간주하지 않는다.')
    figure('A1','figA1_mesh_convergence.png','사각형 FEM50과 FEM100 및 FEM200의 x=0.5 단면',35,width=16.5)
    table(['성분','FEM50 대 FEM200 L² %','FEM100 대 FEM200 L² %'],[[LABEL[i]]+[f"{100*SNAP['fem_mesh_convergence'][key][f]['relative_l2']:.4f}" for key in ('FEM50_vs_FEM200','FEM100_vs_FEM200')] for i,f in enumerate(FIELDS)],[3.3,7.2,7.2],8.8)
    p('독립 FEM 검증: 19개 기준해, 균질 해석해 검증 및 열수지 확인. 저자 FEAP와 비교한 온도 상대오차는 약4.50×10⁻⁶, 변위는 약2.04%이며 변위 진폭비 약0.98의 원인은 확정되지 않았다. 일치를 위해 크기를 보정하지 않았다. [3]',size=9.2)

    page('20 강제 경계 L BFGS 역학 전체장')
    comparison('Fig. B1은 hard L-BFGS 10k의 5개 역학 성분.', f"본 실험 uₓ 평균오차 {m('arch_l5_w40','ux'):.2f}%, σₓₓ {m('arch_l5_w40','sxx'):.2f}%.", 'Fig. 17과 같은 5×40 실행을 재사용한다. 원문 두 그림의 실행 동일성은 명확하지 않다.')
    figure('B1','figB1_hard_fullfields.png','hard L-BFGS 10000회 후 역학 5성분',36,width=14.4)

    page('21 강제 경계 L BFGS 열 전체장과 출처')
    comparison('Fig. B2는 hard L-BFGS 10k의 열 3성분.', f"본 실험 T 평균오차 {m('arch_l5_w40','T'):.2f}%, qₓ {m('arch_l5_w40','qx'):.2f}%.", 'Dirichlet 조건을 정확히 만족해도 내부 열유속의 정확도가 보장되지는 않는다.')
    figure('B2','figB2_hard_fullfields.png','hard L-BFGS 10000회 후 열 3성분',37,width=16)
    p('부록 B 대응 그림은 계산값에 따른 자체 색 범위를 사용한다. Fig. 8·9·13·14·23·24의 원문 색 범위 고정 그림과 구분한다.',size=8.7)
    DOC.add_paragraph('참고문헌 및 계산 자료', 'Heading 2')
    p('[1] Harandi, A. et al. Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering 125(4), e7388 (2024; 온라인 2023). DOI: https://doi.org/10.1002/nme.7388',size=8.6)
    p('[2] 저자 공개 코드 phyml4e/PINNs, Thermomechanic. 검토 커밋 5b426dc8eb2255b1bb3b8630619228b36cdb464e. https://github.com/phyml4e/PINNs/tree/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic',size=8.4)
    p('[3] AM_Project_Reproduce. 원자료 results/full_study 및 results/fem_*; 그림 figures/full_study; 원문 조건 docs/PAPER_EXPERIMENT_INVENTORY.md; 수치 출처 validation/FULL_STUDY_RESULTS.json. https://github.com/JIN9811/AM_Project_Reproduce',size=8.6)
    p('원문 그림의 번호는 비교 위치를 지칭한다. 본 문서의 23개 그림은 모두 해당 논문 구성을 참고하여 자체 계산 자료로 생성한 그림이다. 논문 수치의 디지타이징, 저자 가중치 사용, 그림 전재는 없다.',size=8.6)

    expected = {str(i) for i in range(5,22)} | {'23','24','25','A1','B1','B2'}
    assert {x['paper_figure'] for x in EVIDENCE['figures']} == expected
    assert len(EVIDENCE['figures']) == 23
    OUT.parent.mkdir(exist_ok=True)
    DOC.save(OUT)
    EVIDENCE['docx_sha256'] = digest(OUT)
    EVIDENCE['table_count'] = len(DOC.tables)
    EVIDENCE['embedded_figure_count'] = len(DOC.inline_shapes)
    EVIDENCE['status'] = 'authored; rendering and visual review required'
    (ROOT/'validation/CONCISE_COMPARISON_QA.json').write_text(json.dumps(EVIDENCE, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'figures':len(DOC.inline_shapes),'tables':len(DOC.tables),'verified_metric_values':26*32},ensure_ascii=False))


if __name__ == '__main__':
    main()
