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
    p(f'자료: figures/full_study/{filename}', size=8.2)
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
    p('논문 수치는 오차 색 막대의 상한이다. 본 실험의 최대오차는 FEM50과의 실제 차이로 계산하였다.', size=8.4)


def main():
    verify_data()
    DOC.add_paragraph('Harandi 열탄성 재현 비교', 'Title')
    p('FEM과 PINN의 전체 실험 조건 및 데이터 그림별 의견', size=12)
    p('Harandi 등의 열탄성 문제를 대상으로 논문과 본 실험의 해석 조건 및 계산 결과를 비교하였다. 혼합 PINN의 개선 경향은 확인되었으나, 경계점 보강과 최적화기 비교 등에서는 논문과 다른 결과가 나타났다.', size=10)
    p('조건표 뒤에 데이터 그림 23개를 논문 원본 한 쪽과 본 실험 결과 한 쪽으로 배치하였다. 각 원본의 출처는 [1]이며, 자체 계산 자료는 [3]에 수록하였다.', size=9.4)
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

    pair(5, 'fig05_circle_points_mesh.png', '원형 개재물의 학습점 평가점 및 FEM 격자',
         '논문은 계면과 경계에 점을 보강하였다. 본 실험은 51² 격자와 균등 난수 2,399개로 총 5,000점을 구성하였다. 점수는 유사하나 배치가 달라 계면 부근의 오차에 차이가 생길 수 있다.')

    j = SNAP['jobs']['mixed_circle_coupled']['history']['rows'][-1]
    pair(6, 'fig06_circle_coupled_losses.png', '원형 문제의 동시 학습 손실 이력',
         f"Adam 100,000회 후 총손실은 {j['total']:.3g}, 에너지 항은 역학 {j['M_energy']:.3g}, 열 {j['T_energy']:.3g}이다. 논문과 같이 에너지 항의 감소가 제한적이다. 이 항의 포함 여부에 따른 결과는 그림 19 뒤의 표에 정리하였다.", own_height=18.5)

    pair(7, 'fig07_circle_deformation.png', '원형 개재물의 미변형 형상과 변형 형상',
         '자체 학습 변위와 FEM50 변위로 변형 형상을 계산하였다. 변위의 크기를 맞추기 위한 보정은 하지 않았다. 형상의 유사성과 정량 오차는 구분하며, 성분별 오차는 그림 8과 9에서 비교한다.')

    pair(8, 'fig08_circle_coupled_mechanical.png', '원형 문제의 변위 응력 및 절대오차',
         f"σₓₓ 최대오차는 {peak('mixed_circle_coupled','sxx'):.4f}이며, 논문의 오차 색 막대 상한은 0.0632이다. 계면 부근에 오차가 집중되는 경향은 유사하다. 전단 구성식과 재료 보간이 달라 오차의 크기는 일치하지 않는다.", own_height=20)

    pair(9, 'fig09_circle_coupled_thermal.png', '원형 문제의 온도 열유속 및 절대오차',
         f"온도 최대오차는 {peak('mixed_circle_coupled','T'):.4f}이며, 논문의 오차 색 막대 상한은 0.0109이다. 본 실험에서는 온도보다 열유속의 평균오차가 크다.", own_height=14.2)
    all_field_table('mixed_circle_coupled', 'circle')

    pair(10, 'fig10_circle_sequential_losses.png', '원형 문제의 순차 학습 손실 이력',
         '열과 역학을 20,000회씩 교대로 학습하여 각 장에 100,000회, 총 200,000회를 적용하였다. 가로축은 전체 업데이트 수이며 해당 장을 학습한 구간만 표시하였다. 비활성 망은 동결하므로 역학 학습 중 정상 열전도 망의 출력은 변하지 않는다.', own_height=18.5)

    pair(11, 'fig11_circle_methods.png', '원형 문제의 방법별 단면 비교',
         'x=0.5에서 혼합 PINN의 주요 네 성분이 표준 PINN과 DEM보다 FEM50에 가까웠다. 이는 논문의 경향과 같다. 다만 동시·순차 학습 간 우열은 성분별로 달랐다.', own_height=12.7)
    names = ['mixed_circle_coupled', 'mixed_circle_sequential', 'baseline_circle_standard_coupled', 'baseline_circle_standard_sequential', 'baseline_circle_dem_coupled', 'baseline_circle_dem_sequential']
    labels = ['혼합 동시', '혼합 순차', '표준 동시', '표준 순차', 'DEM 동시', 'DEM 순차']
    table(['방법', 'uₓ 평균 %', 'σₓₓ 평균 %', 'T 평균 %', 'qₓ 평균 %'],
          [[lab] + [f'{m(name,f):.2f}' for f in ('ux','sxx','T','qx')] for name,lab in zip(names,labels)], [4.5,3.3,3.3,3.3,3.3])
    p(f"추가 DEM hard 실험에서 동시 uₓ 평균오차는 {m('baseline_circle_dem_coupled','ux'):.2f}%에서 {m('sensitivity_circle_dem_hard_coupled','ux'):.2f}%로 감소하였다. 반면 qₓ 최대오차는 {peak('baseline_circle_dem_coupled','qx'):.3g}에서 {peak('sensitivity_circle_dem_hard_coupled','qx'):.3g}로 증가하였다.", size=9.4)

    pair(12, 'fig12_rectangles_points_mesh.png', '사각형 개재물의 학습점 평가점 및 FEM 격자',
         '학습점 2,601개와 평가점 10,201개는 논문과 같다. 개재물 위치는 저자 재료 자료를 사용하였다. 좌상단 개재물의 높이는 0.22로, 논문 본문의 0.20과 다르다. 원문의 FEM 도해와도 일부 위치 차이가 있다.')

    pair(13, 'fig13_rectangles_coupled_mechanical.png', '사각형 문제의 변위 응력 및 절대오차',
         f"uₓ 최대오차는 {peak('mixed_rectangles_coupled','ux'):.4f}이며, 논문의 오차 색 막대 상한은 0.0289이다. 경계 및 재료 계면에서 국소 오차가 관측된다. FEM의 불연속 재료와 PINN의 보간 재료 사이의 차이가 포함된 결과이다.", own_height=20)

    pair(14, 'fig14_rectangles_coupled_thermal.png', '사각형 문제의 온도 열유속 및 절대오차',
         f"T 평균오차는 {m('mixed_rectangles_coupled','T'):.2f}%, qₓ는 {m('mixed_rectangles_coupled','qx'):.2f}%이다. 온도장은 FEM50과 비교적 가깝지만, 미분으로 구한 열유속에는 더 큰 오차가 남았다.", own_height=14.2)
    all_field_table('mixed_rectangles_coupled', 'rectangles')

    ur = peak('mixed_rectangles_coupled','ux') / peak('refine_rectangles_left1000','ux')
    tr = peak('mixed_rectangles_coupled','T') / peak('refine_rectangles_left1000','T')
    pair(15, 'fig15_left_refinement.png', '왼쪽 경계점 보강에 따른 오차 변화',
         f'왼쪽에 1,000점을 추가한 후 최대오차 감소 배수는 uₓ {ur:.2f}, T {tr:.2f}였다. 논문은 각각 1.6과 3.0을 보고하였다. 본 실험에서는 변위 오차만 감소하였고 온도 오차는 증가하였다. 감소 배수는 보강 전 오차를 보강 후 오차로 나눈 값이다.')

    pair(16, 'fig16_rectangles_sections.png', '사각형 문제의 동시 및 순차 학습 단면',
         f"그림은 x=0.5 단면이다. 전체 영역의 uₓ 평균오차는 순차 학습에서 {m('mixed_rectangles_sequential','ux'):.2f}%로, 동시 학습의 {m('mixed_rectangles_coupled','ux'):.2f}%보다 작았다. 반면 σₓₓ 평균오차는 {m('mixed_rectangles_coupled','sxx'):.2f}%에서 {m('mixed_rectangles_sequential','sxx'):.2f}%로 증가하였다. 순차 학습의 개선은 모든 성분에서 나타나지는 않았다.")

    pair(17, 'fig17_boundary_optimizer.png', '강제 경계조건과 최적화기에 따른 장 분포',
         f"hard 조건의 10,000회 학습에서 uₓ 평균오차는 Adam {m('hard_rectangles_adam10k','ux'):.2f}%, L-BFGS {m('arch_l5_w40','ux'):.2f}%였다. 본 실험에서는 Adam의 오차가 더 작았다. 본 그림에 추가된 soft Adam은 100,000회 결과로, 두 hard 조건과 학습 예산이 다르다.")

    pair(18, 'fig18_hard_objective.png', '강제 경계조건에서 최적화기별 손실 이력',
         f"최종 총손실은 Adam {SNAP['jobs']['hard_rectangles_adam10k']['history']['rows'][-1]['total']:.3f}, L-BFGS {SNAP['jobs']['arch_l5_w40']['history']['rows'][-1]['total']:.3f}였다. L-BFGS가 더 낮은 손실에 도달한 논문과 순위가 다르다. 가로축은 실제 업데이트 수이며, L-BFGS 내부의 반복 손실·기울기 평가 횟수는 포함하지 않는다.")

    arch = ['arch_l2_w40','arch_l5_w20','arch_l5_w40','arch_l6_w40']
    pair(19, 'fig19_architectures.png', '신경망 구조에 따른 절대오차 분포',
         '2×40, 5×20, 5×40, 6×40의 uₓ 평균오차는 각각 ' + ', '.join(f'{m(n,"ux"):.2f}%' for n in arch) + '였다. 모두 L-BFGS 10,000회를 적용하였으며, 망의 크기와 정확도 사이에 단조로운 관계는 나타나지 않았다.', own_height=14.2)
    DOC.add_paragraph('에너지 항 제외 실험', 'Heading 2')
    table(['조건', 'uₓ 평균 %', 'σₓₓ 평균 %', 'T 평균 %', 'qₓ 평균 %'],
          [[label] + [f'{m(n,f):.2f}' for f in ('ux','sxx','T','qx')] for n,label in [('mixed_circle_coupled','에너지 포함'),('ablation_circle_noenergy','에너지 제외')]], [4.5,3.3,3.3,3.3,3.3])
    p(f"논문 p.13은 에너지 항 제외 시 정확도가 낮아진다고 서술하나 별도 그림이나 수치는 제시하지 않았다. 본 실험에서는 위 평균오차가 감소하였지만 σₓₓ 최대오차는 {peak('mixed_circle_coupled','sxx'):.3f}에서 {peak('ablation_circle_noenergy','sxx'):.3f}로 증가하였다.", size=9.4)

    pair(20, 'fig20_depth_derived.png', '은닉층 수에 따른 평균 및 최대 절대오차',
         f"너비 40에서 1·2·4·5·6층을 비교하였다. σₓₓ 최대오차는 4층 {peak('arch_l4_w40','sxx'):.3f}, 5층 {peak('arch_l5_w40','sxx'):.3f}, 6층 {peak('arch_l6_w40','sxx'):.3f}였다. 논문에서 언급한 5층의 응력 최대오차 증가는 재현되지 않았다.", own_height=18.9)
    p('원문 캡션의 neurons 표기는 축·본문의 hidden layers와 다르다. 본 비교는 축과 본문을 따랐으며, 수행하지 않은 3층 값을 보간하지 않았다. 자체 그림은 원문의 네 성분에 나머지 네 성분을 추가하였다.', size=9)

    pair(21, 'fig21_width_derived.png', '은닉층 너비에 따른 평균 및 최대 절대오차',
         f"5층에서 너비 5·10·20·40·60을 비교하였다. uₓ 평균오차는 너비 20에서 {m('arch_l5_w20','ux'):.2f}%, 40에서 {m('arch_l5_w40','ux'):.2f}%, 60에서 {m('arch_l5_w60','ux'):.2f}%였다. 너비를 늘려도 오차가 계속 감소하지 않는 경향은 논문과 같았다.", own_height=18)
    p('원문 Table 4의 깊이 5종×너비 5종, 총 25조합에 대한 독립 시간 측정은 미수행이다. 수렴 학습 중 기록된 시간으로 대체하지 않았다. Remark 7의 Adam 100만 회 실험도 자료 기준 시점에 미완료였다.', size=9)

    pair(23, 'fig23_transfer.png', '미학습 재료비 15에서 물리와 데이터 결합 모델',
         f"w=0.1의 평균오차는 uₓ {m('transfer_w0.1','ux'):.2f}%, σₓₓ {m('transfer_w0.1','sxx'):.2f}%, T {m('transfer_w0.1','T'):.2f}%, qₓ {m('transfer_w0.1','qx'):.2f}%였다. 응력 오차가 상대적으로 크다. 재료비 15는 학습과 체크포인트 선택에 사용하지 않았다.", own_height=19)
    p('논문의 색 범위를 적용하였으며, 범위 밖 값은 색 막대 끝의 삼각형으로 표시하였다.', size=9)

    pair(24, 'fig24_transfer.png', '미학습 재료비 15에서 데이터 전용 모델',
         f"w=0의 평균오차는 uₓ {m('transfer_w0','ux'):.2f}%, σₓₓ {m('transfer_w0','sxx'):.2f}%, T {m('transfer_w0','T'):.2f}%, qₓ {m('transfer_w0','qx'):.2f}%였다. w=0.1보다 변위·응력·qₓ 오차는 컸지만 T 오차는 작았다.", own_height=18.7)
    p('추가 가중치의 평균오차 (uₓ, σₓₓ, T, qₓ): ' + '; '.join(f'w={w}: ' + ', '.join(f'{m("transfer_w"+w,f):.2f}%' for f in ('ux','sxx','T','qx')) for w in ['0.5','1']) + '. w=0.5는 본 실험에서 추가하였다.', size=9)

    pair(25, 'fig25_transfer_sections.png', '전이학습 단면과 물리 손실 추가에 따른 오차 변화',
         '원문 패널·본문의 x=0.35를 사용하였다. 캡션의 x=0.5와는 다르다. 물리 손실을 추가한 후 8성분 중 6개의 평균·최대오차가 감소하였지만, T와 qᵧ는 증가하였다.', own_height=11.8)
    table(['성분', 'w=0 평균 %', 'w=0.1 평균 %', 'w=0 최대', 'w=0.1 최대'],
          [[LABEL[i], f'{m("transfer_w0",f):.2f}', f'{m("transfer_w0.1",f):.2f}', val(peak('transfer_w0',f)), val(peak('transfer_w0.1',f))] for i,f in enumerate(FIELDS)], [2.1,3.9,3.9,3.9,3.9], 8.9)
    p(f"논문의 qₓ 오차 색 막대 상한은 0.2573에서 0.2787로 증가하였다. 본 실험의 qₓ 최대오차는 {peak('transfer_w0','qx'):.4f}에서 {peak('transfer_w0.1','qx'):.4f}로 감소하여 해당 악화 사례는 재현되지 않았다.", size=9.2)
    p('기록된 학습 시간은 w=0에서 1,258.3초, w=0.1에서 3,713.9초로 약 66.1% 차이였다. 논문의 약 82%와 다르며, 독립 실행으로 측정한 시간은 아니다.', size=9.2)

    pair('A1', 'figA1_mesh_convergence.png', '사각형 문제의 FEM 격자 수렴',
         '논문은 변당 100개 요소에서 거의 수렴한 것으로 평가하였다. 본 실험에서도 100→200 세분화에 따른 uₓ와 T의 차이는 작았다. 응력과 열유속에는 더 큰 차이가 남았다. 아래 값은 FEM200에 대한 상대 L² 오차이다.', own_height=12.3)
    table(['성분', 'FEM50 대 FEM200 L² %', 'FEM100 대 FEM200 L² %'],
          [[LABEL[i]] + [f"{100*SNAP['fem_mesh_convergence'][key][f]['relative_l2']:.4f}" for key in ('FEM50_vs_FEM200','FEM100_vs_FEM200')] for i,f in enumerate(FIELDS)], [3.3,7.2,7.2], 9)
    p('FEM 검증에는 19개 기준해, 균질 해석해 및 열수지 검사를 사용하였다. 저자 FEAP 자료와 비교한 온도 상대오차는 약 4.50×10⁻⁶, 변위는 약 2.04%였다. 변위 진폭비 약 0.98의 원인은 확정되지 않았으며 보정 없이 비교하였다. [3]', size=9.2)

    pair('B1', 'figB1_hard_fullfields.png', '강제 경계조건과 L BFGS의 역학장',
         f"5×40 구조에서 L-BFGS 10,000회 후 uₓ 평균오차는 {m('arch_l5_w40','ux'):.2f}%, σₓₓ는 {m('arch_l5_w40','sxx'):.2f}%였다. 본 실험은 그림 17과 같은 실행의 결과를 사용하였다. 논문의 두 그림이 같은 실행에서 생성되었는지는 명확하지 않다.", own_height=20)

    pair('B2', 'figB2_hard_fullfields.png', '강제 경계조건과 L BFGS의 열장',
         f"T 평균오차는 {m('arch_l5_w40','T'):.2f}%, qₓ는 {m('arch_l5_w40','qx'):.2f}%였다. Dirichlet 경계조건을 강제로 만족시킨 후에도 내부 열유속 오차가 남았다. 부록 B 대응 그림은 자체 계산값의 색 범위를 사용하므로 원본과 색 막대 수치를 함께 비교해야 한다.", own_height=18.7)

    page('참고문헌 및 자료 출처')
    p('[1] Harandi, A., Moeineddin, A., Kaliske, M., Reese, S., and Rezaei, S. Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering, 125(4), e7388, 2024 (온라인 게재 2023). https://doi.org/10.1002/nme.7388', size=10)
    p('[2] phyml4e/PINNs, Thermomechanic. 저자 공개 코드. 검토 커밋 5b426dc8eb2255b1bb3b8630619228b36cdb464e. https://github.com/phyml4e/PINNs/tree/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic', size=9.5)
    p('[3] AM_Project_Reproduce. 자체 FEniCS 및 PINN 계산 자료. https://github.com/JIN9811/AM_Project_Reproduce', size=10)
    DOC.add_paragraph('원본 그림과 자체 계산 그림의 구분', 'Heading 2')
    p('논문 원본 23개는 [1]의 원문 PDF에서 추출하였다. 축·범례·색 막대와 그림 내용은 변경하지 않았으며 페이지에 맞게 비례 축소하였다. 원본 그림의 저작권은 저자에게 있고 CC BY-NC-ND 4.0을 따른다. https://creativecommons.org/licenses/by-nc-nd/4.0/', size=10)
    p('원본 파일과 추출 좌표는 figures/paper_reference/provenance.json에 기록하였다. 이 자료는 자체 계산 결과 및 프로젝트 코드의 이용조건과 구분된다.', size=9.5)
    p('본 실험 그림 23개는 results/full_study와 results/fem_*의 저장 배열로 생성하였다. 원본 그림의 수치를 디지타이징하거나 저자 가중치를 사용하지 않았다. 그림 파일은 figures/full_study, 수치와 출처는 validation/FULL_STUDY_RESULTS.json 및 CONCISE_COMPARISON_QA.json에서 확인할 수 있다.', size=10)
    DOC.add_paragraph('비교 범위', 'Heading 2')
    p('데이터 그림 Fig. 5–21, 23–25, A1, B1, B2의 23쌍을 포함하였다. 개념도 Fig. 1–4, 22는 제외하였다. 원문에 독립 그림이 없는 에너지 항 제외 관측은 그림 19의 본 실험 페이지에 별도로 정리하였다.', size=10)
    p('수치 자료는 2026년 11월 5일 03:13 KST에 확정한 26개 완료 학습 결과이다. 자료 확정 이후의 학습 상태를 나타내지 않는다. Adam 100만 회 추가 실험과 25개 구조의 독립 시간 측정은 이 문서의 완료 결과에 포함하지 않았다.', size=10)

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
