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
PAPER = json.loads((ROOT / 'figures/paper_reference/provenance.json').read_text('utf-8'))
ORIGINALS = {x['paper_figure']: x for x in PAPER['figures']}
EVIDENCE = {'result_snapshot_sha256': hashlib.sha256((ROOT / 'validation/FULL_STUDY_RESULTS.json').read_bytes()).hexdigest(),
            'completed_jobs': SNAP['completed_jobs'], 'figures': [], 'original_figures': [], 'figure_pairs': [], 'metrics_verified': [],
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
    run.font.name = 'Times New Roman'
    run.font.color.rgb = RGBColor(0, 0, 0)
    rp = run._element.get_or_add_rPr()
    rf = rp.rFonts
    if rf is None:
        rf = OxmlElement('w:rFonts'); rp.insert(0, rf)
    for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        rf.set(qn('w:' + attr), 'Batang' if attr == 'eastAsia' else 'Times New Roman')
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
    st = DOC.styles[name]; st.font.name = 'Times New Roman'; st.font.size = Pt(size)
    st.font.color.rgb = RGBColor(0, 0, 0)
    rf = st.element.rPr.rFonts
    for attr in list(rf.attrib):
        if attr.endswith('Theme'):
            del rf.attrib[attr]
    for attr in ('ascii', 'hAnsi', 'eastAsia'):
        rf.set(qn('w:' + attr), 'Batang' if attr == 'eastAsia' else 'Times New Roman')
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
    para = DOC.add_paragraph(title, 'Heading 1')
    para.paragraph_format.page_break_before = True


def table(headers, rows, widths, size=9.1):
    t = DOC.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    pr = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement('w:' + edge)
        e.set(qn('w:val'), 'single' if edge in ('top', 'bottom') else 'nil')
        e.set(qn('w:sz'), '8'); e.set(qn('w:color'), '000000'); borders.append(e)
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
            elif (len(headers) > 3 or headers[0] == '성분') and k > 0:
                pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run_font(pp.add_run(str(text)), size, i == 0)
            if i == 0:
                cb = OxmlElement('w:tcBorders'); edge = OxmlElement('w:bottom')
                edge.set(qn('w:val'), 'single'); edge.set(qn('w:sz'), '5'); edge.set(qn('w:color'), '000000')
                cb.append(edge); cell._tc.get_or_add_tcPr().append(cb)
    DOC.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


def image_block(path, width, max_height, description):
    path = Path(path)
    with Image.open(path) as im: ratio = im.height / im.width
    if max_height: width = min(width, max_height / ratio)
    para = DOC.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.keep_with_next = True
    pic = para.add_run().add_picture(str(path), width=Cm(width))
    pic._inline.docPr.set('descr', description)
    return width


def pair(fid, filename, title, comment, own_height=19.3):
    """One complete original figure page followed by one computed figure page."""
    fid = str(fid); original = ORIGINALS[fid]
    pn = original['paper_page']; seq = len(EVIDENCE['figure_pairs'])
    path = ROOT / original['path']
    assert digest(path) == original['sha256'], path
    page(f'그림 {fid} 논문 원본')
    p(title, size=11)
    w = image_block(path, 17.7, 21.4, f'논문 원본 Fig. {fid}, p. {pn}. {title}. Harandi et al., DOI 10.1002/nme.7388.')
    p(f'Fig. {fid}. {title}. Harandi et al. [1, p. {pn}].', style='Caption')
    p('출처: https://doi.org/10.1002/nme.7388  ·  원본 PDF에서 그림 영역을 추출하였다.', size=8.5)
    p('© 2023 The Authors. John Wiley & Sons Ltd.  CC BY-NC-ND 4.0. 그림의 내용과 색상은 원본과 동일하다.', size=8.2)
    EVIDENCE['original_figures'].append({**original, 'embedded_width_cm': w})

    page(f'그림 {fid} 본 실험')
    p(title, size=11)
    path = ROOT / 'figures/full_study' / filename
    w = image_block(path, 17.7, own_height, f'본 실험 계산 결과. 논문 Fig. {fid} 대응. {title}. 자체 학습 PINN 및 FEniCS 자료.')
    p(f'Fig. {fid} 대응 결과. 본 실험에서 계산한 FEM·PINN 자료 [3].', style='Caption')
    p(comment, size=10)
    EVIDENCE['figures'].append({'paper_figure': fid, 'paper_page': pn,
        'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'embedded_width_cm': w})
    EVIDENCE['figure_pairs'].append({'paper_figure': fid, 'original_report_page': 4 + 2 * seq,
                                     'computed_report_page': 5 + 2 * seq})


def all_field_table(job, geometry):
    rows = []
    for i, f in enumerate(FIELDS):
        rows.append([LABEL[i], val(LIMITS[geometry][f]['error'][1]), val(peak(job,f)), f'{m(job,f):.2f}'])
    table(['성분', '논문 색 막대 상한', '본 실험 최대오차', '본 실험 평균오차 %'], rows, [2.2,5,5,5.5], 8.9)
    p('논문에는 오차 색 막대의 상한을, 본 실험에는 FEM50 대비 최대 절대오차를 제시하였다. 두 값은 정의가 다르므로 직접적인 우열 판단에는 주의가 필요하다.', size=8.4)


def main():
    verify_data()
    DOC.add_paragraph('Harandi 열탄성 재현 비교', 'Title')
    p('FEM과 PINN의 해석 조건 및 수치 결과 비교', size=12)
    p('본 보고서는 Harandi 등의 열탄성 해석을 재현하고, FEM 기준해에 대한 PINN의 정확도를 검토한다. 원형 개재물 문제에서는 혼합 PINN의 개선 경향을 확인하였으나, 경계점 보강과 최적화기의 효과는 논문과 일치하지 않았다. 따라서 해석 조건의 차이를 밝히고, 각 결과의 일치 범위와 한계를 함께 논의하였다.', size=10)
    p('해석 조건을 먼저 비교한 뒤, 논문의 데이터 그림 23개와 본 실험의 대응 결과를 연속된 두 쪽에 제시하였다. 논문 원본은 문헌 [1]에서, 자체 계산 자료는 [3]에서 확인할 수 있다.', size=9.4)
    p('분석에는 2026년 11월 5일 03:13 KST까지 학습을 마친 26개 모델을 사용하였다. 장기 학습과 별도의 계산시간 비교는 분석 범위에서 제외하였다.', size=8.7)
    DOC.add_paragraph('1 물리 조건과 FEM 비교', 'Heading 1')
    table(['항목', '논문 및 저자 자료', '본 실험 FEM과 PINN'], [
        ['해석 영역', '정상 평면변형률 열탄성, [0,1]²', '동일 영역에서 열장이 역학장에 영향을 주는 단방향 연성'],
        ['길이와 물리 단위', 'Fig. 2는 μm, 본문은 mm로 표기', '수치상 단위 영역. mm·GPa·K·W/(m K)는 원문 표기를 따르며, 별도의 SI 환산은 적용하지 않음'],
        ['원형 개재물', '직경 0.30, 중심 좌표는 제시하지 않음', '중심 (0.60,0.35), 반지름 0.15'],
        ['사각형 개재물', '4개, 좌상단 높이 0.20. 위치 좌표는 제시하지 않음', '저자 재료 자료에 따른 형상. 좌상단 높이 0.22'],
        ['원형 재료 E와 k', '기지 0.3 / 개재물 1.0', 'FEM과 PINN 공통으로 동일'],
        ['사각형 재료 E와 k', '기지 1.0 / 개재물 0.5', 'FEM과 PINN 공통으로 동일'],
        ['ν와 열팽창', 'ν=0.3; α와 T₀의 수치는 제시하지 않음', 'ν=0.3, α=1, T₀=0으로 설정'],
        ['변위 경계', '좌우 uₓ=0, 상하 uᵧ=0', '동일한 롤러 구속. 접선 방향은 자유이며 접선력은 0'],
        ['열 경계', '좌 T=1, 우 T=0, 상하 qᵧ=0', '동일 온도 및 단열 조건'],
        ['FEM 해법', '유한요소 기준해. 저자 저장소에 FEAP 자료 공개', 'FEniCS DOLFIN 2019.2의 사각형 Q1 요소. 온도 해석 후 변위 해석'],
        ['FEM 격자', 'Fig. A1에서 변당 50, 100, 200', '원형·사각형 각각 50², 100², 200² 요소'],
        ['FEM 재료 및 복원', '적분과 절점 복원의 세부 조건은 확인하기 어려움', '요소 중심에서 불연속 재료값 부여. 요소 응력과 열유속을 절점으로 복원'],
        ['PINN 재료 표현', '저자 재료 자료와 보간 사용', '50² 재료 격자의 절점 평균값을 쌍선형 보간. FEM의 불연속 재료 표현과 구별'],
    ], [3,6.7,8], 8.9)
    p('사각형 개재물의 범위는 (x구간; y구간)으로 각각 (0.10–0.34; 0.12–0.44), (0.50–0.92; 0.18–0.42), (0.42–0.94; 0.52–0.94), (0.06–0.34; 0.60–0.82)이다. [1–3]', size=8.5)

    page('2 PINN 공통 설정과 구현 차이')
    table(['항목', '논문 조건', '본 실험 조건'], [
        ['혼합 신경망', '독립 스칼라 신경망 8개, 5층×40, tanh', '동일 구조. 출력은 uₓ,uᵧ,σₓₓ,σᵧᵧ,σₓᵧ,T,qₓ,qᵧ'],
        ['표준 PINN 및 DEM', '주 변수 u,T; 기본 5층×40, tanh', 'uₓ,uᵧ,T의 3개 독립 망. 응력·열유속은 미분 계산'],
        ['원형 학습점', '약 5,000개, 계면과 경계에 추가 배치', '51² 격자점과 균등 난수점 2,399개로 구성한 5,000점'],
        ['사각형 학습과 평가', '학습 2,601개, 평가 10,201개', '51² 학습, 모든 문제 101² 평가. 보강 실험은 1,000개 추가'],
        ['정밀도와 초기값', '난수 시드와 정밀도는 제시하지 않음', '배정밀도 float64, 난수 시드 42의 단일 실험'],
        ['Adam', '학습률 0.001, full batch', '동일 학습률·배치. 동시 100,000회'],
        ['순차 학습', '열과 역학 각 100,000회, 20,000회마다 교대', '총 200,000회. 비활성 망을 고정하고 역학 학습 재개 시 Adam 상태 초기화'],
        ['혼합 손실', '평형·연결·경계 MSE, 에너지 MAE', '저자 코드의 내부 일−외부 일 항, 에너지 가중치 1. 전이 w는 전체 물리 손실 가중치'],
        ['일반 경계 처리', '기본 soft 조건과 별도의 hard 조건 비교', '단위 가중치의 soft 페널티. DEM의 hard 조건 2종을 추가 비교'],
        ['hard 출력 변환', '식 62–64', 'uₓ=x(x−1)Nₓ, uᵧ=y(y−1)Nᵧ, T=1−x+x(x−1)Nₜ'],
        ['L-BFGS 세부값', '10,000회. 선 탐색의 세부 조건은 제시하지 않음', 'lr=1, strong-Wolfe, history=80, max_iter=1, max_eval=25; 허용오차 10⁻¹⁰/10⁻¹⁴'],
        ['전단 구성식', '저자 코드는 텐서 전단변형률에 μ를 곱함', 'FEM과 PINN 모두 σₓᵧ=2μεₓᵧ 적용. 저자 구현과 계수 차이'],
        ['경계와 에너지 해석', '식 56의 역학 NBC=0과 손실 그림이 불일치', '롤러의 접선력 손실 적용. 일 항의 비영 값을 곧바로 물리 잔차로 해석하지 않음'],
        ['DEM 온도 처리', '공동 최소화의 세부 경로는 제시하지 않음', '역학 에너지에서 온도망으로의 기울기 전달을 차단'],
        ['표준 PINN 미분', '미분과 재료 보간의 세부 조건은 제시하지 않음', '보간 결절을 피한 점에서 계수 미분. tanh의 해석적 공간 미분 사용'],
        ['하드웨어와 가속', '기본 CPU Xeon Platinum 8160; 시간 비교 M2 Pro', 'RTX 3090, PyTorch 2.11 CUDA, Adam CUDA graph. 장비 차이를 고려한 시간 비교 필요'],
        ['저장과 결과 선택', '상세 정책은 제시하지 않음', '1,000회마다 최근 가중치를 덮어쓰기 저장. 중간 최적값 대신 지정된 최종 반복의 결과 사용'],
    ], [3,6.5,8.2], 8.6)
    p('L-BFGS는 한 번의 갱신에 손실과 기울기를 여러 차례 평가하므로, 반복 수가 같아도 Adam과 계산비용이 같지는 않다. [1–3]', size=8.5)

    page('3 전체 실험 구성과 비교 기준')
    table(['실험군', '논문 조건', '본 실험 조건과 비교 범위'], [
        ['원형 기본 6종', '혼합·표준·DEM의 동시 및 순차 학습, 5×40', 'soft 조건, Adam 동시 10만 회 또는 순차 20만 회'],
        ['사각형 기본 2종', '혼합 동시 및 순차 학습, 5×40', 'soft 조건, Adam 동시 10만 회 또는 순차 20만 회'],
        ['좌측 경계 보강', '약 1,000점 추가. 별도 학습 횟수는 제시하지 않음', '3,601점에서 초기화 후 Adam 동시 학습 10만 회'],
        ['에너지 제외', '정확도 저하를 정성적으로 기술', '원형에서 열과 역학의 에너지 항을 제외하고 Adam 동시 학습 10만 회'],
        ['hard Adam', '사각형, 2,601점, 1만 회', '5×40 구조에서 동일 반복 수 적용'],
        ['hard L-BFGS 구조', '너비 40에서 깊이 1,2,4,5,6; 5층에서 너비 5,10,20,40,60', '서로 다른 9개 구조를 각 1만 회 학습. 5×40 결과는 Fig. 17–18과 B1–B2에도 사용'],
        ['전이학습 4종', '재료비 1–10에서 각 1만 회, 시험 재료비 15. w=0,0.1 명시, 상한 1', 'w=0,0.1,0.5,1에서 각각 Adam 총 10만 회. w=0.5는 추가 비교 조건'],
        ['추가 DEM hard', '별도 보고 없음', '원형에서 동시 10만 회와 순차 20만 회 비교'],
        ['장기 학습', 'Remark 7의 사각형 Adam 100만 회. 기본 학습과의 관계는 불명확', 'soft 10만 회 모델에 90만 회를 추가하는 조건. 분석 기준 시점에는 학습이 끝나지 않아 제외'],
        ['계산시간 비교', 'Table 4의 깊이 5종과 너비 5종. 각 100회 L-BFGS 평균 시간', '25개 구조의 독립 시간 측정을 수행하지 않아 비교에서 제외'],
        ['FEM 기준해', '격자 수렴 및 전이학습용 FEM 자료', '기본 6개, 절점 재료 민감도 2개, 전이 재료비 11개의 총 19개'],
    ], [3,7,7.7], 9)
    DOC.add_paragraph('전이학습의 추가 조건', 'Heading 2')
    p('재료비 Eᵣ=E기지/E개재물과 kᵣ=k기지/k개재물을 같게 두고 1부터 10까지 순차 학습하였다. 기지는 E=k=1, 개재물은 E=k=1/r로 설정하였다. 입력은 x,y,log₁₀Eᵣ,log₁₀kᵣ이며, 데이터 손실은 51²점의 FEM50 8성분에 대한 비정규화 MSE이다. soft 경계를 적용하고 재료비가 바뀔 때 가중치는 전달하되 Adam은 초기화하였다. 재료비 15는 학습과 모델 선택에서 제외하였다.', size=9.4)
    DOC.add_paragraph('오차 산정과 비교 범위', 'Heading 2')
    p('평균오차(%)는 100×평균|PINN−FEM|/최대|FEM|로 정의하였다. 별도 표기가 없으면 101² 평가점에서 FEM50과 비교하였으며, 응력과 열유속은 독립 혼합 출력이 아닌 변위와 온도의 미분값으로 산정하였다. 격자 수렴에서는 FEM200을 기준해로 사용하였다.', size=9.4)
    p('논문의 평균 상대오차는 정의가 충분히 제시되지 않아 본 지표와 동일하게 취급하지 않았다. 색 범위를 맞춘 그림에서 상·하한을 벗어나는 값은 색 막대 끝의 삼각형으로 표시하였으며, 정량 오차에는 원래 값을 반영하였다. 비교 대상은 개념도 Fig. 1–4와 22를 제외한 데이터 그림 23개이다.', size=9.4)

    pair(5, 'fig05_circle_points_mesh.png', '원형 개재물의 학습점 평가점 및 FEM 격자',
         '논문은 재료 계면과 경계에 학습점을 추가한 반면, 본 실험은 51² 격자점에 균등 난수점 2,399개를 더하여 총 5,000점을 구성하였다. 전체 점 수는 유사하지만 국소적인 점 밀도가 다르므로, 계면 부근의 오차를 비교할 때 이러한 배치 차이를 고려해야 한다.')

    j = SNAP['jobs']['mixed_circle_coupled']['history']['rows'][-1]
    pair(6, 'fig06_circle_coupled_losses.png', '원형 문제의 동시 학습 손실 이력',
         f"Adam 100,000회 학습 후 총손실은 {j['total']:.3g}에 도달하였다. 역학과 열의 에너지 항은 각각 {j['M_energy']:.3g}, {j['T_energy']:.3g}이었으며, 논문과 마찬가지로 감소가 제한적이었다. 에너지 항이 해의 정확도에 미치는 영향은 그림 19에 이어 제시한 제외 실험에서 검토하였다.", own_height=18.5)

    pair(7, 'fig07_circle_deformation.png', '원형 개재물의 미변형 형상과 변형 형상',
         'PINN 예측 변위와 FEM50 변위를 각각 적용하여 변형 형상을 구하였다. 두 결과의 진폭을 맞추는 보정은 적용하지 않았다. 전체 형상만으로 국소적인 오차를 판단하기는 어려우므로, 변위와 응력의 성분별 차이를 그림 8에서, 열장의 차이를 그림 9에서 비교하였다.')

    pair(8, 'fig08_circle_coupled_mechanical.png', '원형 문제의 변위 응력 및 절대오차',
         f"재료 계면에 오차가 집중되는 공간적 경향은 논문과 유사하였다. 다만 σₓₓ의 최대오차는 {peak('mixed_circle_coupled','sxx'):.4f}이었으며, 논문의 오차 색 막대 상한 0.0632와 차이가 있었다. 전단 구성식과 재료 보간이 서로 다르므로, 이 차이를 학습 성능만으로 설명하기는 어렵다.", own_height=20)

    pair(9, 'fig09_circle_coupled_thermal.png', '원형 문제의 온도 열유속 및 절대오차',
         f"온도의 최대오차는 {peak('mixed_circle_coupled','T'):.4f}이었으며, 논문의 오차 색 막대 상한 0.0109보다 컸다. 본 실험에서는 온도보다 열유속의 평균오차가 크게 나타나, 온도장의 일치가 그 기울기의 정확도까지 보장하지는 않음을 확인하였다.", own_height=14.2)
    all_field_table('mixed_circle_coupled', 'circle')

    pair(10, 'fig10_circle_sequential_losses.png', '원형 문제의 순차 학습 손실 이력',
         '열장과 역학장을 20,000회마다 교대로 학습하여 각 장에 100,000회, 총 200,000회의 갱신을 적용하였다. 가로축은 누적 갱신 횟수이며, 각 곡선은 해당 장이 학습된 구간을 나타낸다. 역학장 학습 중에는 열전도 신경망을 고정하므로 온도 출력이 유지된다.', own_height=18.5)

    pair(11, 'fig11_circle_methods.png', '원형 문제의 방법별 단면 비교',
         'x=0.5 단면에서 혼합 PINN의 네 성분은 표준 PINN과 DEM보다 FEM50에 가까운 분포를 보였다. 혼합 정식화의 개선 경향은 논문과 일치하였으나, 동시 학습과 순차 학습의 상대적인 정확도는 성분에 따라 달랐다.', own_height=12.7)
    names = ['mixed_circle_coupled', 'mixed_circle_sequential', 'baseline_circle_standard_coupled', 'baseline_circle_standard_sequential', 'baseline_circle_dem_coupled', 'baseline_circle_dem_sequential']
    labels = ['혼합 동시', '혼합 순차', '표준 동시', '표준 순차', 'DEM 동시', 'DEM 순차']
    table(['방법', 'uₓ 평균 %', 'σₓₓ 평균 %', 'T 평균 %', 'qₓ 평균 %'],
          [[lab] + [f'{m(name,f):.2f}' for f in ('ux','sxx','T','qx')] for name,lab in zip(names,labels)], [4.5,3.3,3.3,3.3,3.3])
    p(f"DEM의 경계조건을 hard 방식으로 바꾸자 동시 학습의 uₓ 평균오차는 {m('baseline_circle_dem_coupled','ux'):.2f}%에서 {m('sensitivity_circle_dem_hard_coupled','ux'):.2f}%로 감소하였다. 그러나 qₓ 최대오차는 {peak('baseline_circle_dem_coupled','qx'):.3g}에서 {peak('sensitivity_circle_dem_hard_coupled','qx'):.3g}까지 증가하여, 변위의 개선이 열유속의 국소 정확도 향상으로 이어지지는 않았다.", size=9.4)

    pair(12, 'fig12_rectangles_points_mesh.png', '사각형 개재물의 학습점 평가점 및 FEM 격자',
         '학습점 2,601개와 평가점 10,201개는 논문의 조건과 일치한다. 개재물의 위치는 저자 재료 자료에 따랐으나, 좌상단 개재물의 높이는 0.22로 본문에 제시된 0.20과 다르며 원문의 FEM 도해와도 일부 위치 차이가 있다. 따라서 점 수가 같더라도 형상까지 완전히 동일한 비교는 아니다.')

    pair(13, 'fig13_rectangles_coupled_mechanical.png', '사각형 문제의 변위 응력 및 절대오차',
         f"uₓ의 최대오차는 {peak('mixed_rectangles_coupled','ux'):.4f}이었으며, 논문의 오차 색 막대 상한 0.0289와 가까웠다. 오차는 경계와 재료 계면에서 국소적으로 나타났다. 이 값에는 신경망의 근사 오차뿐 아니라 FEM의 불연속 재료와 PINN의 보간 재료 사이의 차이도 포함된다.", own_height=20)

    pair(14, 'fig14_rectangles_coupled_thermal.png', '사각형 문제의 온도 열유속 및 절대오차',
         f"온도와 qₓ의 평균오차는 각각 {m('mixed_rectangles_coupled','T'):.2f}%와 {m('mixed_rectangles_coupled','qx'):.2f}%였다. 온도장은 FEM50에 비교적 가까웠으나, 그 미분으로 구한 열유속에는 더 큰 차이가 남았다. 열장의 정확도는 온도 분포와 열유속을 함께 고려하여 평가할 필요가 있다.", own_height=14.2)
    all_field_table('mixed_rectangles_coupled', 'rectangles')

    ur = peak('mixed_rectangles_coupled','ux') / peak('refine_rectangles_left1000','ux')
    tr = peak('mixed_rectangles_coupled','T') / peak('refine_rectangles_left1000','T')
    pair(15, 'fig15_left_refinement.png', '왼쪽 경계점 보강에 따른 오차 변화',
         f'왼쪽 경계에 1,000점을 추가한 결과, 보강 전후 최대오차의 비는 uₓ에서 {ur:.2f}, T에서 {tr:.2f}이었다. 이 비는 보강 전 오차를 보강 후 오차로 나눈 값으로, 논문은 각각 1.6과 3.0을 보고하였다. 본 실험에서는 변위 오차만 줄고 온도 오차는 오히려 증가하여, 경계점 보강의 효과가 두 장에서 동일하게 나타나지 않았다.')

    pair(16, 'fig16_rectangles_sections.png', '사각형 문제의 동시 및 순차 학습 단면',
         f"x=0.5 단면과 함께 전체 평가 영역의 오차를 비교하였다. 순차 학습의 uₓ 평균오차는 {m('mixed_rectangles_sequential','ux'):.2f}%로 동시 학습의 {m('mixed_rectangles_coupled','ux'):.2f}%보다 작았으나, σₓₓ 평균오차는 {m('mixed_rectangles_coupled','sxx'):.2f}%에서 {m('mixed_rectangles_sequential','sxx'):.2f}%로 증가하였다. 순차 학습이 변위 정확도를 높이더라도 응력까지 일관되게 개선하지는 못하였다.")

    pair(17, 'fig17_boundary_optimizer.png', '강제 경계조건과 최적화기에 따른 장 분포',
         f"hard 경계조건에서 10,000회 학습한 uₓ 평균오차는 Adam {m('hard_rectangles_adam10k','ux'):.2f}%, L-BFGS {m('arch_l5_w40','ux'):.2f}%로, 본 실험에서는 Adam이 더 정확하였다. 함께 제시한 soft Adam 결과는 100,000회 학습한 모델이므로, hard 조건과의 차이를 경계 처리 방식만의 효과로 해석할 수는 없다.")

    pair(18, 'fig18_hard_objective.png', '강제 경계조건에서 최적화기별 손실 이력',
         f"Adam과 L-BFGS의 최종 총손실은 각각 {SNAP['jobs']['hard_rectangles_adam10k']['history']['rows'][-1]['total']:.3f}, {SNAP['jobs']['arch_l5_w40']['history']['rows'][-1]['total']:.3f}였다. 논문에서는 L-BFGS가 더 낮은 손실에 도달하였으나, 본 조건에서는 그 우위가 재현되지 않았다. 가로축은 파라미터 갱신 횟수이며 L-BFGS 내부의 반복 평가를 세지 않으므로, 이를 계산시간의 비교로 해석해서는 안 된다.")

    arch = ['arch_l2_w40','arch_l5_w20','arch_l5_w40','arch_l6_w40']
    pair(19, 'fig19_architectures.png', '신경망 구조에 따른 절대오차 분포',
         '2×40, 5×20, 5×40, 6×40 구조의 uₓ 평균오차는 각각 ' + ', '.join(f'{m(n,"ux"):.2f}%' for n in arch) + '였다. 모든 구조에 L-BFGS 10,000회를 적용하였으나, 신경망의 크기에 비례하여 정확도가 높아지지는 않았다. 정해진 학습 횟수에서는 구조의 확대만으로 오차 감소를 기대하기 어렵다.', own_height=14.2)
    DOC.add_paragraph('에너지 항 제외 실험', 'Heading 2')
    table(['조건', 'uₓ 평균 %', 'σₓₓ 평균 %', 'T 평균 %', 'qₓ 평균 %'],
          [[label] + [f'{m(n,f):.2f}' for f in ('ux','sxx','T','qx')] for n,label in [('mixed_circle_coupled','에너지 포함'),('ablation_circle_noenergy','에너지 제외')]], [4.5,3.3,3.3,3.3,3.3])
    p(f"논문은 에너지 항을 제외하면 정확도가 저하된다고 기술하였으나, 별도 그림이나 정량값은 제시하지 않았다[1, p. 13]. 본 실험에서는 표에 제시한 평균오차가 모두 감소한 반면, σₓₓ 최대오차는 {peak('mixed_circle_coupled','sxx'):.3f}에서 {peak('ablation_circle_noenergy','sxx'):.3f}까지 증가하였다. 에너지 항의 효과는 평균 정확도와 국소 최대오차를 구분하여 판단해야 한다.", size=9.4)

    pair(20, 'fig20_depth_derived.png', '은닉층 수에 따른 평균 및 최대 절대오차',
         f"은닉층 너비를 40으로 고정하고 1·2·4·5·6층을 비교하였다. σₓₓ 최대오차는 4층 {peak('arch_l4_w40','sxx'):.3f}, 5층 {peak('arch_l5_w40','sxx'):.3f}, 6층 {peak('arch_l6_w40','sxx'):.3f}이었다. 논문에서 보고한 5층의 응력 최대오차 증가는 재현되지 않았으며, 본 실험에서는 6층의 오차가 더 컸다.", own_height=18.9)
    p('원문 캡션의 neurons는 축과 본문의 hidden layers와 불일치하므로, 여기서는 은닉층 수를 기준으로 해석하였다. 수행하지 않은 3층 값은 보간하지 않았으며, 본 실험 그림에는 원문의 네 성분을 포함한 총 여덟 성분을 제시하였다.', size=9)

    pair(21, 'fig21_width_derived.png', '은닉층 너비에 따른 평균 및 최대 절대오차',
         f"은닉층 수를 5로 고정하고 너비 5·10·20·40·60을 비교하였다. uₓ 평균오차는 너비 20에서 {m('arch_l5_w20','ux'):.2f}%, 40에서 {m('arch_l5_w40','ux'):.2f}%, 60에서 {m('arch_l5_w60','ux'):.2f}%였다. 너비 40 이후에는 추가 개선이 나타나지 않았으며, 너비 증가가 지속적인 오차 감소로 이어지지 않는 경향은 논문과 같았다.", own_height=18)
    p('구조에 따른 계산비용은 별도의 검증이 필요하다. 본 연구는 Table 4의 25개 구조에 대한 독립 시간 측정을 수행하지 않았으며, Remark 7의 Adam 100만 회 결과도 분석 기준 시점에 확보되지 않아 비교에서 제외하였다.', size=9)

    pair(23, 'fig23_transfer.png', '미학습 재료비 15에서 물리와 데이터 결합 모델',
         f"학습에 사용하지 않은 재료비 15에서, w=0.1 모델의 평균오차는 uₓ {m('transfer_w0.1','ux'):.2f}%, σₓₓ {m('transfer_w0.1','sxx'):.2f}%, T {m('transfer_w0.1','T'):.2f}%, qₓ {m('transfer_w0.1','qx'):.2f}%였다. 응력의 오차가 상대적으로 커, 외삽 조건에서 성분별 정확도에 차이가 나타났다. 이 재료비는 모델 선택에도 사용하지 않았다.", own_height=19)
    p('논문의 색 범위를 적용하였으며, 범위 밖 값은 색 막대 끝의 삼각형으로 표시하였다.', size=9)

    pair(24, 'fig24_transfer.png', '미학습 재료비 15에서 데이터 전용 모델',
         f"데이터만으로 학습한 w=0 모델의 평균오차는 uₓ {m('transfer_w0','ux'):.2f}%, σₓₓ {m('transfer_w0','sxx'):.2f}%, T {m('transfer_w0','T'):.2f}%, qₓ {m('transfer_w0','qx'):.2f}%였다. w=0.1에 비해 uₓ, σₓₓ, qₓ는 부정확하였으나 온도 오차는 더 작았다. 물리 손실의 추가 효과가 모든 성분에 동일하게 나타나는 것은 아니다.", own_height=18.7)
    p('물리 손실의 가중치를 더 높였을 때 uₓ, σₓₓ, T, qₓ의 평균오차는 ' + '; '.join(f'w={w}에서 각각 ' + ', '.join(f'{m("transfer_w"+w,f):.2f}%' for f in ('ux','sxx','T','qx')) for w in ['0.5','1']) + '였다. w=0.5는 본 실험에서 추가한 비교 조건이다.', size=9)

    pair(25, 'fig25_transfer_sections.png', '전이학습 단면과 물리 손실 추가에 따른 오차 변화',
         '단면 위치는 원문의 패널과 본문에 따라 x=0.35로 정하였다. 이는 캡션의 x=0.5 표기와 다르다. 물리 손실을 추가하면 여덟 성분 중 여섯 성분의 평균오차와 최대오차가 줄었으나, T와 qᵧ에서는 두 오차가 모두 증가하였다.', own_height=11.8)
    table(['성분', 'w=0 평균 %', 'w=0.1 평균 %', 'w=0 최대', 'w=0.1 최대'],
          [[LABEL[i], f'{m("transfer_w0",f):.2f}', f'{m("transfer_w0.1",f):.2f}', val(peak('transfer_w0',f)), val(peak('transfer_w0.1',f))] for i,f in enumerate(FIELDS)], [2.1,3.9,3.9,3.9,3.9], 8.9)
    p(f"논문에서는 qₓ의 오차 색 막대 상한이 0.2573에서 0.2787로 증가하였다. 본 실험의 최대오차는 {peak('transfer_w0','qx'):.4f}에서 {peak('transfer_w0.1','qx'):.4f}까지 감소하여, 논문에 나타난 qₓ의 악화 경향은 확인되지 않았다.", size=9.2)
    p('학습 시간은 w=0에서 1,258.3초, w=0.1에서 3,713.9초로, 데이터 전용 모델에서 약 66.1% 짧았다. 논문의 약 82%와는 차이가 있으나, 독립적인 시간 측정 실험이 아니므로 정량적인 속도 비교에는 한계가 있다.', size=9.2)

    pair('A1', 'figA1_mesh_convergence.png', '사각형 문제의 FEM 격자 수렴',
         '논문은 한 변당 100개 요소에서 해가 거의 수렴한 것으로 평가하였다. 본 실험에서도 요소 수를 100에서 200으로 늘렸을 때 uₓ와 T의 변화는 작았으나, 응력과 열유속에는 상대적으로 큰 차이가 남았다. 아래 표는 FEM200을 기준으로 산정한 상대 L² 오차이다.', own_height=12.3)
    table(['성분', 'FEM50 대 FEM200 L² %', 'FEM100 대 FEM200 L² %'],
          [[LABEL[i]] + [f"{100*SNAP['fem_mesh_convergence'][key][f]['relative_l2']:.4f}" for key in ('FEM50_vs_FEM200','FEM100_vs_FEM200')] for i,f in enumerate(FIELDS)], [3.3,7.2,7.2], 9)
    p('FEM의 타당성은 19개 기준해와 균질 문제의 해석해, 열수지를 통해 검토하였다. 저자의 FEAP 자료에 대한 상대오차는 온도에서 약 4.50×10⁻⁶, 변위에서 약 2.04%였다. 변위 진폭비가 약 0.98인 원인은 규명되지 않았으며, 본 비교에는 진폭 보정을 적용하지 않았다. [3]', size=9.2)

    pair('B1', 'figB1_hard_fullfields.png', '강제 경계조건과 L BFGS의 역학장',
         f"5×40 구조에 L-BFGS 10,000회를 적용한 결과, uₓ와 σₓₓ의 평균오차는 각각 {m('arch_l5_w40','ux'):.2f}%와 {m('arch_l5_w40','sxx'):.2f}%였다. 여기서는 그림 17과 동일한 모델의 역학장을 제시하였다. 다만 논문의 두 그림이 동일한 학습 결과에 근거하는지는 확인되지 않아, 그림 사이의 차이를 직접 대응시키기는 어렵다.", own_height=20)

    pair('B2', 'figB2_hard_fullfields.png', '강제 경계조건과 L BFGS의 열장',
         f"T와 qₓ의 평균오차는 각각 {m('arch_l5_w40','T'):.2f}%와 {m('arch_l5_w40','qx'):.2f}%였다. Dirichlet 경계조건을 정확히 만족시켜도 내부 열유속에는 오차가 남으므로, 경계조건의 충족과 내부 해의 정확도는 구분해야 한다. 부록 B의 대응 그림은 자체 계산 범위에 맞춘 색 막대를 사용하였으므로 색상과 수치를 함께 비교해야 한다.", own_height=18.7)

    page('참고문헌 및 자료 출처')
    p('[1] Harandi, A., Moeineddin, A., Kaliske, M., Reese, S., and Rezaei, S. Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering, 125(4), e7388, 2024 (온라인 게재 2023). https://doi.org/10.1002/nme.7388', size=10)
    p('[2] phyml4e/PINNs. Thermomechanic. 저자 공개 코드, 버전 5b426dc8eb2255b1bb3b8630619228b36cdb464e. https://github.com/phyml4e/PINNs/tree/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic', size=9.5)
    p('[3] AM_Project_Reproduce. 자체 FEniCS 및 PINN 계산 자료. https://github.com/JIN9811/AM_Project_Reproduce', size=10)
    DOC.add_paragraph('그림의 출처와 이용조건', 'Heading 2')
    p('논문 원본 23개는 문헌 [1]의 PDF에서 추출하여 비례 축소하였다. 축, 범례, 색 막대를 포함한 그림 내용은 변경하지 않았다. 원본 그림의 저작권은 저자에게 있으며, CC BY-NC-ND 4.0의 이용조건을 따른다. https://creativecommons.org/licenses/by-nc-nd/4.0/', size=10)
    p('본 실험의 그림은 직접 학습한 PINN과 FEniCS 계산값으로 작성하였다. 논문 그림에서 수치를 추출하거나 저자의 학습 가중치를 사용하지 않았다. 원본 그림의 추출 위치와 자체 계산 결과의 수치 검증 자료는 저장소 [3]에 함께 수록하였으며, 원본 그림에는 프로젝트 코드와 별도의 이용조건이 적용된다.', size=10)
    DOC.add_paragraph('비교의 범위와 한계', 'Heading 2')
    p('비교에는 Fig. 5–21, 23–25, A1, B1, B2의 데이터 그림 23쌍을 포함하였다. 개념도 Fig. 1–4와 22는 제외하였으며, 원문에 별도 그림이 없는 에너지 항 제외 결과는 그림 19에 이어 제시하였다.', size=10)
    p('분석 대상은 2026년 11월 5일 03:13 KST까지 학습을 마친 26개 모델이다. Adam 100만 회의 장기 학습과 25개 구조의 독립 시간 측정은 포함하지 않았다. 모든 신경망은 단일 난수 시드로 학습하였으므로, 여기서 확인한 방법별 차이를 반복 실험에서의 통계적 우열로 일반화할 수는 없다.', size=10)

    expected = [str(i) for i in range(5,22)] + ['23','24','25','A1','B1','B2']
    assert [x['paper_figure'] for x in EVIDENCE['figures']] == expected
    assert [x['paper_figure'] for x in EVIDENCE['original_figures']] == expected
    assert len(DOC.inline_shapes) == 46
    OUT.parent.mkdir(exist_ok=True)
    DOC.save(OUT)
    EVIDENCE['docx_sha256'] = digest(OUT)
    EVIDENCE['source_paper_sha256'] = PAPER['source_pdf_sha256']
    EVIDENCE['original_figure_license'] = PAPER['license']
    EVIDENCE['table_count'] = len(DOC.tables)
    EVIDENCE['embedded_figure_count'] = len(DOC.inline_shapes)
    EVIDENCE['expected_pages'] = 50
    EVIDENCE['table_style'] = 'White background, black text and horizontal top/header/bottom rules; no shading or vertical rules.'
    EVIDENCE['status'] = 'authored; rendering and visual review required'
    (ROOT/'validation/CONCISE_COMPARISON_QA.json').write_text(json.dumps(EVIDENCE, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(OUT), 'figure_pairs':23, 'embedded_figures':len(DOC.inline_shapes), 'tables':len(DOC.tables), 'verified_metric_values':26*32}, ensure_ascii=False))


if __name__ == '__main__':
    main()
