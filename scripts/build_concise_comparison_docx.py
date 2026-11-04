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
FRONT_PAGES = 5
FEM_CHECK = json.loads((ROOT / 'results/fem_report_verification.json').read_text('utf-8'))
EVIDENCE = {'result_snapshot_sha256': hashlib.sha256((ROOT / 'validation/FULL_STUDY_RESULTS.json').read_bytes()).hexdigest(),
            'completed_jobs': SNAP['completed_jobs'], 'figures': [], 'original_figures': [], 'figure_pairs': [], 'metrics_verified': [],
            'excluded_conceptual_figures': ['1', '2', '3', '4', '22'],
            'source_data_cutoff_kst': '2026-11-05 03:13 KST',
            'comparison_reference': 'FEM50; mesh convergence uses FEM200',
            'prediction_representation': 'constitutive stress and Fourier flux derived from u,T'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_conditions():
    """Check saved configurations and FEM evidence without starting a solver."""
    records = FEM_CHECK['reference_records']
    assert len(records) == 19
    for rec in records:
        folder = ROOT / rec['folder']
        assert digest(folder / 'solution.npz') == rec['solution_sha256']
        metadata = json.loads((folder / 'metadata.json').read_text('utf-8'))
        assert metadata == rec['metadata']
        n = metadata['n']
        assert metadata['diagnostics']['elements'] == n*n
        assert metadata['diagnostics']['vertices'] == (n+1)**2
    configs = []
    for name in SNAP['completed_jobs']:
        path = ROOT / 'results/full_study' / name / 'config.json'
        c = json.loads(path.read_text('utf-8'))
        assert c['seed'] == 42 and c['dtype'] == 'float64' and c['n'] == 51
        assert c['lr'] == .001
        if name.startswith('transfer_'):
            assert c['normalization'] == 'none' and not c['hard']
            assert c['ratios'] == list(range(1, 11)) and c['steps'] == 10000
        elif name.startswith('baseline_') or name.startswith('sensitivity_'):
            assert c['boundary_weight'] == 1 and not c['dem_neumann']
            assert c['backend'] == ('jets' if c['method'] == 'standard' else 'autograd')
        else:
            assert c['energy_weight'] == 1 and c['jacobian_backend'] == 'autograd'
            assert c['reset_mechanical_adam'] == (c['mode'] == 'sequential')
        configs.append({'job': name, 'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
    paths = ['harandi/fem.py', 'harandi/physics.py', 'harandi/pinn.py', 'harandi/train.py',
             'harandi/baselines.py', 'harandi/transfer.py', 'harandi/full_study.py',
             'results/fem_report_verification.json', 'docs/SOURCE_PROVENANCE.json']
    EVIDENCE['condition_audit'] = {
        'source_files': [{'path': rel, 'sha256': digest(ROOT / rel)} for rel in paths],
        'saved_configs': configs, 'fem_references_checked': len(records),
        'mesh_dofs_before_constraints': [
            {'n': n, 'elements': n*n, 'nodes': (n+1)**2,
             'temperature': (n+1)**2, 'displacement': 2*(n+1)**2,
             'total': 3*(n+1)**2} for n in (50, 100, 200)],
        'max_algebraic_relative_residual': max(r['metadata']['diagnostics'][key]
            for r in records for key in ('thermal_relative_residual', 'mechanical_relative_residual')),
        'max_heat_balance_absolute': max(abs(r['metadata']['diagnostics']['heat_balance']) for r in records),
        'max_dirichlet_error': max(r['dirichlet_max_error'] for r in records),
        'source_code_review': 'FEM assembly/recovery, PINN objectives/batches/optimizers, author code at 5b426dc8eb2255b1bb3b8630619228b36cdb464e and paper Eqs.40,49,62-65 reviewed.',
    }


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


def sci(number):
    mantissa, exponent = f'{number:.3e}'.split('e')
    return mantissa + '×10' + str(int(exponent)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))


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

    page(f'그림 {fid} 재현 결과')
    p(title, size=11)
    path = ROOT / 'figures/full_study' / filename
    w = image_block(path, 17.7, own_height, f'재현 계산 결과. 논문 Fig. {fid} 대응. {title}. 자체 학습 PINN 및 FEniCS 자료.')
    p(f'Fig. {fid} 대응 결과. 재현 계산 자료 [3].', style='Caption')
    p(comment, size=10)
    EVIDENCE['figures'].append({'paper_figure': fid, 'paper_page': pn,
        'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path), 'embedded_width_cm': w})
    EVIDENCE['figure_pairs'].append({'paper_figure': fid, 'original_report_page': FRONT_PAGES + 1 + 2 * seq,
                                     'computed_report_page': FRONT_PAGES + 2 + 2 * seq})


def all_field_table(job, geometry):
    rows = []
    for i, f in enumerate(FIELDS):
        rows.append([LABEL[i], val(LIMITS[geometry][f]['error'][1]), val(peak(job,f)), f'{m(job,f):.2f}'])
    table(['성분', '논문 색 막대 상한', '재현 최대오차', '재현 평균오차 %'], rows, [2.2,5,5,5.5], 8.9)
    p('논문 열은 오차 색 막대의 상한이고, 재현 열은 FEM50 대비 실제 최대 절대오차이다. 색 막대 상한이 실제 최대오차와 같은지는 원문에 명시되어 있지 않다.', size=8.4)


def main():
    verify_data()
    verify_conditions()
    DOC.add_paragraph('Harandi 열탄성 재현 비교', 'Title')
    p('FEM과 PINN의 해석 조건 및 수치 결과 비교', size=12)
    DOC.add_paragraph('요약', 'Heading 2')
    p('원형 개재물의 x=0.5 단면에서 혼합 PINN은 표준 PINN과 DEM보다 FEM50에 가까웠다. 왼쪽 경계점 보강은 변위 최대오차를 줄였지만 온도 최대오차는 늘렸다. hard 조건에서는 Adam의 최종 손실이 L-BFGS보다 작아 논문과 순서가 달랐다.', size=10)
    p('논문 원본 [1]과 재현 결과 [3]를 그림별로 연속된 두 쪽에 배치하였다. 데이터 그림은 23쌍이다.', size=9.4)
    p('계산값은 2026년 11월 5일 03:13 KST까지 학습을 마친 26개 모델의 출력이다. Adam 100만 회 결과와 25개 구조의 독립 시간 측정값은 포함하지 않았다.', size=8.7)
    DOC.add_paragraph('1 공통 물리 조건', 'Heading 1')
    table(['항목', '논문 및 저자 자료', '재현 FEM과 PINN'], [
        ['해석 영역과 연성', '정상 소변형 평면변형률 열탄성, [0,1]²', '동일 영역. 온도가 역학에 영향을 주는 단방향 연성'],
        ['지배방정식과 하중', '응력 및 열유속의 발산 0. 체적력·내부 열원 0', '동일. q=−k∇T, 선형 등방성 열탄성 구성식'],
        ['길이와 물리 단위', 'Fig. 2는 μm, 본문은 mm로 표기', '수치상 단위 영역. mm·GPa·K·W/(m K)는 원문 표기를 따르며, 별도의 SI 환산은 적용하지 않음'],
        ['원형 개재물', '직경 0.30, 중심 좌표는 제시하지 않음', '중심 (0.60,0.35), 반지름 0.15'],
        ['사각형 개재물', '4개, 좌상단 높이 0.20. 위치 좌표는 제시하지 않음', '저자 재료 자료에 따른 형상. 좌상단 높이 0.22'],
        ['원형 재료 E와 k', '기지 0.3 / 개재물 1.0', 'FEM과 PINN 공통으로 동일'],
        ['사각형 재료 E와 k', '기지 1.0 / 개재물 0.5', 'FEM과 PINN 공통으로 동일'],
        ['ν와 열팽창', 'ν=0.3. α와 T₀는 본문에 수치 미기재; 저자 코드 [2]는 α=1, T₀=0', 'ν=0.3, α=1, T₀=0. 열응력 계수는 Eα/(1−2ν)'],
        ['변위 경계', '좌우 uₓ=0, 상하 uᵧ=0', '동일한 롤러 구속. 접선 방향은 자유이며 접선력은 0'],
        ['열 경계', '좌 T=1, 우 T=0, 상하 qᵧ=0', '동일 온도 및 단열 조건'],
    ], [3,6.7,8], 8.9)
    p('사각형 개재물의 범위는 (x구간; y구간)으로 각각 (0.10–0.34; 0.12–0.44), (0.50–0.92; 0.18–0.42), (0.42–0.94; 0.52–0.94), (0.06–0.34; 0.60–0.82)이다. [1–3]', size=8.5)

    page('2 FEM 해석 조건')
    table(['항목', '논문 및 저자 자료', '재현 FEM'], [
        ['프로그램과 실행', 'FEAP [1, §4.1]. 공개된 FEAP 결과 [2]', 'FEniCS DOLFIN 2019.2.0.64.dev0, Ubuntu WSL, 단일 MPI rank'],
        ['미지수와 해석 순서', '온도·변위 및 파생 응력·열유속', '온도 T를 먼저 풀고 열하중으로 전달하여 u를 계산. 연성 반복 없음'],
        ['약형과 요소 공간', '기준해를 FEM으로 계산. 세부 약형·요소 차수는 명시하지 않음', 'Galerkin 약형. T는 스칼라 Q1, u는 2성분 Q1. 응력·열유속은 주 미지수가 아님'],
        ['격자와 요소 크기', 'Fig. A1: 변당 50, 100, 200요소', '정형 4절점 사각형. n=50,100,200; h=1/n. 원형 경계는 요소 중심 판별로 근사'],
        ['재료 계수', '기지·개재물의 E와 k 분포', 'sharp: 요소 중심에서 상을 판별한 DQ0 상수. 요소 내 E와 k는 일정'],
        ['재료 민감도 조건', '별도의 대응 결과 없음', 'nodal: 50² 재료 격자의 절점 평균값을 Q1 보간. 두 기하에서 FEM100으로 별도 계산'],
        ['수치 적분', '적분 차수 미기재', '열·역학 조립과 DQ0 투영의 quadrature_degree=4'],
        ['경계조건 적용', '앞쪽 물리 조건과 동일', 'Dirichlet 조건을 선형 시스템에 직접 적용. 영 접선력·영 열유속은 자연경계조건'],
        ['계면 처리', '이질 재료의 열탄성 문제', 'Q1의 연속 T와 u 및 약형으로 완전 접합 적용. 계면 접촉저항·분리 조건 없음'],
        ['선형 해법', '해법 설정 미기재', '열·역학 시스템과 투영 모두 MUMPS 직접해법. 반복 선형해법의 수렴 허용오차는 설정하지 않음'],
        ['응력과 열유속', '저자 FEAP 자료에 절점 값 수록', 'σ(u,T), q=−k∇T를 DQ0 요소 평균으로 L² 투영. 인접 요소의 면적 가중 평균으로 절점 복원'],
        ['평가와 저장', '정규 평가점에서 PINN과 비교', '101² 좌표에서 Q1 보간. 원시 요소 평균과 절점 복원값을 따로 저장. 비교 그림은 복원값 사용'],
        ['해의 수와 기준 격자', '격자 수렴 및 전이학습용 FE 자료', '기본 6개, nodal 2개, 재료비 1–10·15의 FEM50 11개. PINN 비교는 FEM50, 격자 수렴은 FEM200'],
    ], [3,6.3,8.4], 8.6)
    DOC.add_paragraph('격자별 자유도', 'Heading 2')
    table(['변당 요소 수', '요소 수', '절점 수', 'T 자유도', 'u 자유도', '합계'],
          [[str(n), f'{n*n:,}', f'{(n+1)**2:,}', f'{(n+1)**2:,}', f'{2*(n+1)**2:,}', f'{3*(n+1)**2:,}']
           for n in (50,100,200)], [3,2.8,2.9,2.9,3,3.1], 8.7)
    p('자유도는 구속 전 값이다. Dirichlet 구속은 온도 2(n+1)개, 변위 4(n+1)개에 적용된다. T와 u는 별도의 시스템으로 풀며, 합계는 두 시스템의 자유도를 더한 값이다.', size=8.5)
    audit = EVIDENCE['condition_audit']
    p(f"저장된 19개 해의 최대 상대 대수 잔차 ‖Ax−b‖₂/‖b‖₂는 {sci(audit['max_algebraic_relative_residual'])}, 최대 |좌·우 열반력 합|은 {sci(audit['max_heat_balance_absolute'])}, Dirichlet 최대오차는 {audit['max_dirichlet_error']:g}이다. 열반력은 경계조건 적용 전 열전도 행렬에서 계산하였다. 대수 잔차와 격자 이산화 오차는 서로 다른 값이다. [3]", size=8.5)
    uniform = FEM_CHECK['uniform_analytic_checks']
    p(f"균질 해석해 비교는 n=8,16,32와 sharp·nodal의 6조건이다. 절점 u,T 최대오차는 {sci(max(r['nodal_uT_max_abs'] for r in uniform))} 이하였고, 요소 내부 uₓ 보간오차는 남았다. 저자 FEAP와의 T 및 변위 벡터 상대 L² 차이는 각각 {sci(FEM_CHECK['author_feap_check']['temperature_relative_l2'])}, {100*FEM_CHECK['author_feap_check']['displacement_relative_l2_raw']:.3f}%이다. 격자 수렴 수치는 그림 A1에 제시하였다. [2,3]", size=8.5)

    page('3 PINN 신경망과 학습 설정')
    table(['항목', '논문 및 저자 자료', '재현 PINN'], [
        ['혼합 신경망', '독립 스칼라 망 8개, 은닉층 5개×40, tanh', '동일 구조. uₓ,uᵧ,σₓₓ,σᵧᵧ,σₓᵧ,T,qₓ,qᵧ 사이 가중치 공유 없음. 출력층은 선형'],
        ['표준 PINN과 DEM', 'u,T를 주 변수로 사용', 'uₓ,uᵧ,T의 독립 망 3개, 5×40. σ와 q는 공간 미분으로 계산'],
        ['입력과 매개변수 수', '기본 입력 x,y', '좌표를 2x−1,2y−1로 변환. 기본 혼합 53,768개, 표준·DEM 20,163개; 전이 혼합 54,408개'],
        ['초기화와 정밀도', '논문에 시드·초기화 미기재. 저자 코드 [2]는 float64', 'Xavier 정규분포 gain=1, 편향 0. 시드 42, 계산 float64. 초기 가중치는 기본 float32 생성 후 float64로 변환'],
        ['원형 학습점', '약 5,000개, 계면·경계 추가 배치', '혼합·DEM: 51² 격자+균등 난수 2,399개. 내부 4,800개, 서로 다른 경계점 200개'],
        ['사각형 학습점', '2,601개', '51² 고정 격자. 내부 2,401개, 경계 200개. 각 변 51개이며 모서리는 두 변에 포함'],
        ['표준 PINN 학습점', '세부 난수 배치 미기재', '원형의 내부 4,800개를 보간 결절을 피한 균등 난수로 교체. 경계 200개 유지. 시드 73'],
        ['점 재추출과 보강', '왼쪽 경계 약 1,000점 추가', '학습 중 점 재추출 없음. 보강은 x=0, y 균등 난수 1,000개, 시드 133. 총 3,601개'],
        ['배치와 FEM 정답', '혼합 PINN full batch. §4.4에서 FE 데이터 결합', '모든 방법 full batch. 기본 PINN·DEM 학습에 FEM 정답 미사용; 전이학습에서만 사용'],
        ['Adam', '학습률 0.001', '학습률 0.001 고정. β₁=0.9, β₂=0.999, ε=10⁻⁸, weight_decay=0, AMSGrad 꺼짐'],
        ['순차 학습', '열·역학 각 100,000회, 20,000회마다 교대', '열부터 시작. 비활성 망 고정. 역학 블록 시작 시 Adam 상태 초기화; 열 블록 상태는 유지'],
        ['L-BFGS', '10,000회. 세부 선 탐색 설정 미기재', 'Adam 사전학습 없이 초기화 후 실행. lr=1, strong-Wolfe, history=80, max_iter=1, max_eval=25; grad/change 허용오차 10⁻¹⁰/10⁻¹⁴'],
        ['미분 계산', '자동미분', '혼합·DEM은 autograd. 표준 PINN은 tanh 연쇄법칙으로 1·2차 미분과 재료 구배 계산'],
        ['평가 좌표', '사각형 10,201개', '모든 결과 101² 평가. 별도의 67² 격자에서 물리 잔차 계산. 두 격자 모두 학습의 조기 종료에 미사용'],
        ['기록과 저장', '상세 정책 미기재', '기본 Adam·전이 손실 1,000회, hard Adam·L-BFGS 손실 100회마다 기록. 최근 체크포인트는 1,000회마다 갱신'],
        ['결과 선택', '반복 수를 지정', '지정된 마지막 갱신의 모델 사용. 최저 검증오차 선택·조기 종료 없음. 전이는 재료비 완료 시 별도 저장'],
        ['계산 환경', '기본 CPU Xeon Platinum 8160; 시간 비교 M2 Pro', 'Python 3.11.9, PyTorch 2.11.0+cu128, RTX 3090. Adam CUDA graph, TF32 꺼짐'],
    ], [3,6.1,8.6], 8.6)
    p('5×40은 입력·출력층을 제외한 은닉층 수와 너비이다. L-BFGS의 1회는 외부 갱신 호출이며, 내부 손실·기울기 평가 횟수와 다르다. 기본 격자의 경계점 수 200개는 모서리를 중복하지 않은 값이다. [1–3]', size=8.5)

    page('4 PINN 손실과 경계조건')
    table(['항목', '논문 및 저자 코드', '재현 구현'], [
        ['혼합 손실 구성', '열·역학의 평형·연결·경계·에너지 손실 합', '5개 역학 항과 5개 열 항의 합. 모든 항의 가중치 1; 에너지 제외 조건만 두 에너지 항 0'],
        ['평형과 연결 손실', '평형·구성식 연결의 MSE', '평형은 내부점에서만 계산. 연결은 경계를 포함한 전체 점에서 성분 제곱합의 평균'],
        ['경계 손실의 평균', '저자 코드 [2]: 전체 표본에 경계 마스크를 곱한 뒤 MSE', '각 변의 경계점만 뽑아 MSE를 구한 뒤 합산. 전체 표본 평균과 정규화 구간이 다름'],
        ['자연경계의 출력', '저자 코드 [2]: 미분 유도 출력과 독립 출력 모두에 영 전단·열유속 항 적용', '혼합은 독립 σₓᵧ의 네 변 및 독립 qᵧ의 상하 변에 페널티. 유도 출력에 대한 별도 페널티 없음'],
        ['에너지 항', '저자 코드 [2]: 내부 일 절반−외부 일의 절댓값. sqrt(abs(·))에 MSE 적용', '동일한 일 항의 절댓값을 직접 사용. 내부점 평균과 네 변별 평균으로 계산'],
        ['soft 경계', '기본 비교는 soft 조건', '법선 변위와 좌우 온도 오차의 MSE를 손실에 추가. 경계 가중치 1'],
        ['hard 경계', '식 62–64 및 [2]는 x(x−1), y(y−1) 인자', 'uₓ=x(1−x)Nₓ, uᵧ=y(1−y)Nᵧ, T=1−x+x(1−x)Nₜ. Dirichlet 조건만 출력 변환으로 만족'],
        ['전단 구성식', '식 40·49를 함께 읽으면 μεₓᵧ. 저자 코드 [2]를 정리하면 2μεₓᵧ', 'FEM과 PINN 모두 2μεₓᵧ. 저자 공개 코드와 같고 인쇄 식 49와 다름'],
        ['재료 계수', '저자 절점 재료 자료와 보간', '50² 재료 격자의 인접 요소 상을 절점 평균한 뒤 쌍선형 보간. 기본 FEM의 DQ0 계수와 다름'],
        ['표준 PINN 손실', '강형 평형식과 경계조건', '2차 공간 미분의 평형 MSE+Dirichlet·Neumann MSE. 연결·에너지 항 없음. 재료 E,k의 구배 포함'],
        ['DEM 손실', '변분 에너지와 경계조건', '평면변형률 탄성에너지+전도에너지+Dirichlet 페널티. Neumann 페널티는 꺼져 있고 자연경계로 처리'],
        ['DEM 온도 결합', '세부 기울기 경로 미기재', '역학 에너지에서는 T를 고정하여 온도망으로의 기울기 전달 차단. 면외 탄성변형률 항 포함'],
    ], [3,6.5,8.2], 8.8)
    p('에너지 항은 물리 잔차의 제곱과 같은 정의가 아니다. 경계 손실도 단위 가중치를 사용하더라도 평균 구간과 적용 출력이 다르다. 저자 공개 스크립트는 hard 조건, 각 변 400개 추가점, Adam 10,000회 설정이며 논문의 모든 그림에 대한 실행 설정을 제공하지는 않는다. [1,2]', size=9)
    p('hard 변환의 부호는 원문과 다르지만, 신경망 출력 N의 부호를 바꾸면 같은 함수 집합을 표현한다. Dirichlet 조건은 같으며 동일한 초기 가중치에서의 초기장은 다르다.', size=9)
    p('저자 코드의 전단 계수는 E/[(1+ν)(1−2ν)]×(1−2ν)=E/(1+ν)=2μ이다. 텐서 전단변형률은 εₓᵧ=(uₓ,ᵧ+uᵧ,ₓ)/2이며, FEM과 PINN 모두 이 정의를 사용한다. [2,3]', size=9)

    page('5 전체 실험 구성과 비교 기준')
    table(['실험군', '논문 조건', '재현 조건과 비교 범위'], [
        ['원형 기본 6종', '혼합·표준·DEM의 동시 및 순차 학습, 5×40', 'soft 조건, Adam 동시 10만 회 또는 순차 20만 회'],
        ['사각형 기본 2종', '혼합 동시 및 순차 학습, 5×40', 'soft 조건, Adam 동시 10만 회 또는 순차 20만 회'],
        ['좌측 경계 보강', '약 1,000점 추가. 별도 학습 횟수는 제시하지 않음', '3,601점에서 초기화 후 Adam 동시 학습 10만 회'],
        ['에너지 제외', '정확도 저하를 정성적으로 기술', '원형에서 열과 역학의 에너지 항을 제외하고 Adam 동시 학습 10만 회'],
        ['hard Adam', '사각형, 2,601점, 1만 회', '5×40 구조에서 동일 반복 수 적용'],
        ['hard L-BFGS 구조', '너비 40에서 깊이 1,2,4,5,6; 5층에서 너비 5,10,20,40,60', '서로 다른 9개 구조를 각 1만 회 학습. 5×40 결과는 Fig. 17–18과 B1–B2에도 사용'],
        ['전이학습 4종', '재료비 1–10에서 각 1만 회, 시험 재료비 15. w=0,0.1 명시, 상한 1', 'w=0,0.1,0.5,1에서 각각 Adam 총 10만 회. w=0.5는 추가 비교 조건'],
        ['추가 DEM hard', '논문에 별도 결과 없음', '원형에서 동시 10만 회와 순차 20만 회 비교'],
        ['장기 학습', 'Remark 7의 사각형 Adam 100만 회. 기본 학습과의 관계는 불명확', 'soft 10만 회 모델에 90만 회를 추가하는 조건. 분석 기준 시점에는 학습이 끝나지 않아 제외'],
        ['계산시간 비교', 'Table 4의 깊이 5종과 너비 5종. 각 100회 L-BFGS 평균 시간', '25개 구조의 독립 시간 측정을 수행하지 않아 비교에서 제외'],
        ['FEM 기준해', '격자 수렴 및 전이학습용 FEM 자료', '기본 6개, 절점 재료 민감도 2개, 전이 재료비 11개의 총 19개'],
    ], [3,7,7.7], 9)
    DOC.add_paragraph('전이학습의 추가 조건', 'Heading 2')
    p('재료비 Eᵣ=E기지/E개재물과 kᵣ=k기지/k개재물을 같게 두고 1부터 10까지 순차 학습하였다. 기지는 E=k=1, 개재물은 E=k=1/r이다. 원문 §4.4는 좌표와 국소 E(x,y),k(x,y)를 입력으로 설명한다. 재현 입력은 좌표와 전역 재료비이며 실제 망에는 2x−1,2y−1,log₁₀Eᵣ,log₁₀kᵣ를 넣었다.', size=9.2)
    p('총손실은 데이터 손실에 w배의 역학·열 물리 손실을 더한 값이다. 데이터 손실은 51²점의 FEM50 8성분과 독립 혼합 출력 간 MSE이며 성분별 정규화는 하지 않았다. soft 경계를 적용하고, 재료비가 바뀔 때 가중치는 전달하되 Adam은 초기화하였다. 재료비 15는 학습·조기 종료·모델 선택에 사용하지 않았다. [1–3]', size=9.2)
    DOC.add_paragraph('오차 산정과 비교 범위', 'Heading 2')
    p('평균오차(%)는 100×평균|PINN−FEM|/최대|FEM|로 정의하였다. 별도 표기가 없으면 101² 평가점에서 FEM50과 비교하였으며, 응력과 열유속은 독립 혼합 출력이 아닌 변위와 온도의 미분값으로 산정하였다. 격자 수렴에서는 FEM200을 기준해로 사용하였다.', size=9.4)
    p('논문의 평균 상대오차는 정의가 충분히 제시되지 않아 위 지표와 같은지 확인되지 않았다. 색 범위를 맞춘 그림에서 상·하한을 벗어나는 값은 색 막대 끝의 삼각형으로 표시하였으며, 정량 오차에는 원래 값을 반영하였다. 비교 대상은 개념도 Fig. 1–4와 22를 제외한 데이터 그림 23개이다.', size=9.4)

    pair(5, 'fig05_circle_points_mesh.png', '원형 개재물의 학습점 평가점 및 FEM 격자',
         '논문은 재료 계면과 경계에 학습점을 추가하였다. 재현에는 51² 격자점과 균등 난수점 2,399개를 합한 5,000점을 사용하였다. 전체 점 수는 유사하지만 계면과 경계의 점 밀도는 다르다.')

    j = SNAP['jobs']['mixed_circle_coupled']['history']['rows'][-1]
    pair(6, 'fig06_circle_coupled_losses.png', '원형 문제의 동시 학습 손실 이력',
         f"Adam 100,000회 후 총손실은 {j['total']:.3g}이다. 역학 에너지 항은 {j['M_energy']:.3g}, 열 에너지 항은 {j['T_energy']:.3g}이다. 에너지 항을 제외한 결과는 그림 19 아래 표에 제시하였다.", own_height=18.5)

    pair(7, 'fig07_circle_deformation.png', '원형 개재물의 미변형 형상과 변형 형상',
         'PINN과 FEM50의 변위를 각각 적용한 형상이다. 변위 확대 계수는 1이며 진폭 보정은 적용하지 않았다. 변위·응력 오차는 그림 8, 온도·열유속 오차는 그림 9에 제시하였다.')

    pair(8, 'fig08_circle_coupled_mechanical.png', '원형 문제의 변위 응력 및 절대오차',
         f"논문과 재현 결과 모두 재료 계면에 오차가 집중되었다. 재현의 σₓₓ 최대오차는 {peak('mixed_circle_coupled','sxx'):.4f}이며, 논문의 오차 색 막대 상한은 0.0632이다. 학습점 배치와 경계 손실의 평균 방식은 앞쪽 조건표와 같이 다르다.", own_height=20)

    pair(9, 'fig09_circle_coupled_thermal.png', '원형 문제의 온도 열유속 및 절대오차',
         f"온도 최대오차는 {peak('mixed_circle_coupled','T'):.4f}이며, 논문의 오차 색 막대 상한은 0.0109이다. 재현의 온도 평균오차는 열유속 평균오차보다 작았다.", own_height=14.2)
    all_field_table('mixed_circle_coupled', 'circle')

    pair(10, 'fig10_circle_sequential_losses.png', '원형 문제의 순차 학습 손실 이력',
         '열장과 역학장을 20,000회마다 교대로 학습하였다. 각 장은 100,000회, 합계 200,000회 갱신하였다. 가로축은 전체 갱신 횟수이며 각 곡선은 해당 장의 학습 구간만 표시한다. 역학장 학습 중에는 열전도 신경망의 가중치를 고정하였다.', own_height=18.5)

    pair(11, 'fig11_circle_methods.png', '원형 문제의 방법별 단면 비교',
         'x=0.5 단면에서 혼합 PINN의 uₓ, σₓₓ, T, qₓ는 표준 PINN과 DEM보다 FEM50에 가까웠다. 논문에서도 같은 경향이 나타났다. 동시·순차 학습 중 오차가 작은 방식은 성분에 따라 달랐다.', own_height=12.7)
    names = ['mixed_circle_coupled', 'mixed_circle_sequential', 'baseline_circle_standard_coupled', 'baseline_circle_standard_sequential', 'baseline_circle_dem_coupled', 'baseline_circle_dem_sequential']
    labels = ['혼합 동시', '혼합 순차', '표준 동시', '표준 순차', 'DEM 동시', 'DEM 순차']
    table(['방법', 'uₓ 평균 %', 'σₓₓ 평균 %', 'T 평균 %', 'qₓ 평균 %'],
          [[lab] + [f'{m(name,f):.2f}' for f in ('ux','sxx','T','qx')] for name,lab in zip(names,labels)], [4.5,3.3,3.3,3.3,3.3])
    p(f"DEM의 경계조건을 hard 방식으로 바꾸자 동시 학습의 uₓ 평균오차는 {m('baseline_circle_dem_coupled','ux'):.2f}%에서 {m('sensitivity_circle_dem_hard_coupled','ux'):.2f}%로 감소하였다. 그러나 qₓ 최대오차는 {peak('baseline_circle_dem_coupled','qx'):.3g}에서 {peak('sensitivity_circle_dem_hard_coupled','qx'):.3g}까지 증가하였다.", size=9.4)

    pair(12, 'fig12_rectangles_points_mesh.png', '사각형 개재물의 학습점 평가점 및 FEM 격자',
         '학습점 2,601개와 평가점 10,201개는 논문과 같다. 개재물 위치는 저자 재료 자료를 사용하였다. 좌상단 개재물의 높이는 0.22이며, 논문 본문의 0.20과 다르다. 원문의 FEM 도해와도 일부 위치 차이가 있다.')

    pair(13, 'fig13_rectangles_coupled_mechanical.png', '사각형 문제의 변위 응력 및 절대오차',
         f"uₓ 최대오차는 {peak('mixed_rectangles_coupled','ux'):.4f}이며, 논문의 오차 색 막대 상한은 0.0289이다. 경계와 재료 계면에 국소 오차가 나타났다. FEM은 불연속 재료값을 사용하고, PINN은 절점 평균값을 쌍선형 보간하였다.", own_height=20)

    pair(14, 'fig14_rectangles_coupled_thermal.png', '사각형 문제의 온도 열유속 및 절대오차',
         f"온도 평균오차는 {m('mixed_rectangles_coupled','T'):.2f}%, qₓ 평균오차는 {m('mixed_rectangles_coupled','qx'):.2f}%이다. 온도보다 그 미분으로 구한 열유속의 평균오차가 컸다.", own_height=14.2)
    all_field_table('mixed_rectangles_coupled', 'rectangles')

    ur = peak('mixed_rectangles_coupled','ux') / peak('refine_rectangles_left1000','ux')
    tr = peak('mixed_rectangles_coupled','T') / peak('refine_rectangles_left1000','T')
    pair(15, 'fig15_left_refinement.png', '왼쪽 경계점 보강에 따른 오차 변화',
         f'왼쪽 경계에 1,000점을 추가한 뒤, 최대오차의 보강 전후 비는 uₓ {ur:.2f}, T {tr:.2f}이다. 논문의 값은 각각 1.6과 3.0이다. 비는 보강 전 오차를 보강 후 오차로 나눈 값이다. 재현에서는 변위 최대오차가 감소하고 온도 최대오차가 증가하였다.')

    pair(16, 'fig16_rectangles_sections.png', '사각형 문제의 동시 및 순차 학습 단면',
         f"그림은 x=0.5 단면이다. 전체 평가 영역의 uₓ 평균오차는 동시 학습 {m('mixed_rectangles_coupled','ux'):.2f}%, 순차 학습 {m('mixed_rectangles_sequential','ux'):.2f}%이다. σₓₓ 평균오차는 각각 {m('mixed_rectangles_coupled','sxx'):.2f}%, {m('mixed_rectangles_sequential','sxx'):.2f}%이다. 순차 학습에서 uₓ 오차는 감소하고 σₓₓ 오차는 증가하였다.")

    pair(17, 'fig17_boundary_optimizer.png', '강제 경계조건과 최적화기에 따른 장 분포',
         f"hard 조건에서 10,000회 학습한 uₓ 평균오차는 Adam {m('hard_rectangles_adam10k','ux'):.2f}%, L-BFGS {m('arch_l5_w40','ux'):.2f}%이다. Adam의 오차가 더 작았다. 함께 표시한 soft Adam은 100,000회 결과로, hard 조건보다 학습 횟수가 10배 많다.")

    pair(18, 'fig18_hard_objective.png', '강제 경계조건에서 최적화기별 손실 이력',
         f"최종 총손실은 Adam {SNAP['jobs']['hard_rectangles_adam10k']['history']['rows'][-1]['total']:.3f}, L-BFGS {SNAP['jobs']['arch_l5_w40']['history']['rows'][-1]['total']:.3f}이다. 논문에서는 L-BFGS, 재현에서는 Adam의 손실이 더 작았다. 가로축은 파라미터 갱신 횟수이며 L-BFGS 내부의 반복 손실·기울기 평가 횟수는 포함하지 않는다.")

    arch = ['arch_l2_w40','arch_l5_w20','arch_l5_w40','arch_l6_w40']
    pair(19, 'fig19_architectures.png', '신경망 구조에 따른 절대오차 분포',
         '2×40, 5×20, 5×40, 6×40 구조의 uₓ 평균오차는 각각 ' + ', '.join(f'{m(n,"ux"):.2f}%' for n in arch) + '이다. 모든 구조에 L-BFGS 10,000회를 적용하였다. 신경망의 크기가 커질수록 오차가 감소하는 관계는 나타나지 않았다.', own_height=14.2)
    DOC.add_paragraph('에너지 항 제외 실험', 'Heading 2')
    table(['조건', 'uₓ 평균 %', 'σₓₓ 평균 %', 'T 평균 %', 'qₓ 평균 %'],
          [[label] + [f'{m(n,f):.2f}' for f in ('ux','sxx','T','qx')] for n,label in [('mixed_circle_coupled','에너지 포함'),('ablation_circle_noenergy','에너지 제외')]], [4.5,3.3,3.3,3.3,3.3])
    p(f"논문은 에너지 항을 제외하면 정확도가 저하된다고 기술하였으나, 별도 그림이나 정량값은 제시하지 않았다[1, p. 13]. 재현에서는 표에 제시한 네 성분의 평균오차가 모두 감소하였고, σₓₓ 최대오차는 {peak('mixed_circle_coupled','sxx'):.3f}에서 {peak('ablation_circle_noenergy','sxx'):.3f}까지 증가하였다.", size=9.4)

    pair(20, 'fig20_depth_derived.png', '은닉층 수에 따른 평균 및 최대 절대오차',
         f"너비 40에서 은닉층 1·2·4·5·6개를 비교하였다. σₓₓ 최대오차는 4층 {peak('arch_l4_w40','sxx'):.3f}, 5층 {peak('arch_l5_w40','sxx'):.3f}, 6층 {peak('arch_l6_w40','sxx'):.3f}이다. 논문에서는 4층보다 5층의 응력 최대오차가 컸다. 재현에서는 5층에서 감소하고 6층에서 증가하였다.", own_height=18.9)
    p('원문 캡션에는 neurons, 축과 본문에는 hidden layers로 적혀 있다. 비교 기준은 은닉층 수이다. 3층은 계산하지 않았으며 보간값도 넣지 않았다. 논문 그림은 네 성분, 재현 그림은 여덟 성분이다.', size=9)

    pair(21, 'fig21_width_derived.png', '은닉층 너비에 따른 평균 및 최대 절대오차',
         f"5층에서 너비 5·10·20·40·60을 비교하였다. uₓ 평균오차는 너비 20에서 {m('arch_l5_w20','ux'):.2f}%, 40에서 {m('arch_l5_w40','ux'):.2f}%, 60에서 {m('arch_l5_w60','ux'):.2f}%이다. 너비 40에서 60으로 늘리면 오차가 증가하였다. 논문에서도 너비 증가에 따라 오차가 계속 감소하지는 않았다.", own_height=18)
    p('Table 4의 25개 구조에 대한 독립 시간 측정은 수행하지 않았다. Remark 7의 Adam 100만 회 결과는 2026년 11월 5일 03:13 KST에 학습이 끝나지 않아 포함하지 않았다.', size=9)

    pair(23, 'fig23_transfer.png', '미학습 재료비 15에서 물리와 데이터 결합 모델',
         f"재료비 15에서 w=0.1의 평균오차는 uₓ {m('transfer_w0.1','ux'):.2f}%, σₓₓ {m('transfer_w0.1','sxx'):.2f}%, T {m('transfer_w0.1','T'):.2f}%, qₓ {m('transfer_w0.1','qx'):.2f}%이다. 이 네 성분 중 σₓₓ의 오차가 가장 컸다. 재료비 15는 학습과 모델 선택에 사용하지 않았다.", own_height=19)
    p('논문의 색 범위를 적용하였으며, 범위 밖 값은 색 막대 끝의 삼각형으로 표시하였다.', size=9)

    pair(24, 'fig24_transfer.png', '미학습 재료비 15에서 데이터 전용 모델',
         f"데이터만 사용한 w=0의 평균오차는 uₓ {m('transfer_w0','ux'):.2f}%, σₓₓ {m('transfer_w0','sxx'):.2f}%, T {m('transfer_w0','T'):.2f}%, qₓ {m('transfer_w0','qx'):.2f}%이다. w=0.1보다 uₓ, σₓₓ, qₓ의 오차는 크고 T의 오차는 작았다.", own_height=18.7)
    p('물리 손실의 가중치를 더 높였을 때 uₓ, σₓₓ, T, qₓ의 평균오차는 ' + '; '.join(f'w={w}에서 각각 ' + ', '.join(f'{m("transfer_w"+w,f):.2f}%' for f in ('ux','sxx','T','qx')) for w in ['0.5','1']) + '였다. w=0.5는 재현에서 추가한 비교 조건이다.', size=9)

    pair(25, 'fig25_transfer_sections.png', '전이학습 단면과 물리 손실 추가에 따른 오차 변화',
         '단면은 원문의 패널과 본문에 표시된 x=0.35이다. 원문 캡션에는 x=0.5로 적혀 있다. w=0에서 0.1로 바꾸면 여덟 성분 중 여섯 성분의 평균·최대오차가 감소하였다. T와 qᵧ의 평균·최대오차는 증가하였다.', own_height=11.8)
    table(['성분', 'w=0 평균 %', 'w=0.1 평균 %', 'w=0 최대', 'w=0.1 최대'],
          [[LABEL[i], f'{m("transfer_w0",f):.2f}', f'{m("transfer_w0.1",f):.2f}', val(peak('transfer_w0',f)), val(peak('transfer_w0.1',f))] for i,f in enumerate(FIELDS)], [2.1,3.9,3.9,3.9,3.9], 8.9)
    p(f"논문에서는 qₓ의 오차 색 막대 상한이 0.2573에서 0.2787로 증가하였다. 재현의 최대오차는 {peak('transfer_w0','qx'):.4f}에서 {peak('transfer_w0.1','qx'):.4f}까지 감소하여 논문과 반대 방향으로 변하였다.", size=9.2)
    p('학습 시간은 w=0에서 1,258.3초, w=0.1에서 3,713.9초로, 데이터 전용 모델에서 약 66.1% 짧았다. 논문에 기재된 감소율은 약 82%이다. 재현 시간은 독립 측정값이 아닌 각 학습 중 기록한 값이다.', size=9.2)

    pair('A1', 'figA1_mesh_convergence.png', '사각형 문제의 FEM 격자 수렴',
         '논문은 한 변당 100개 요소에서 해가 거의 수렴한다고 기술하였다. 재현에서는 100개에서 200개로 세분화할 때 uₓ와 T의 상대 L² 오차가 응력·열유속보다 작았다. 아래 표의 기준해는 FEM200이다.', own_height=12.3)
    table(['성분', 'FEM50 대 FEM200 L² %', 'FEM100 대 FEM200 L² %'],
          [[LABEL[i]] + [f"{100*SNAP['fem_mesh_convergence'][key][f]['relative_l2']:.4f}" for key in ('FEM50_vs_FEM200','FEM100_vs_FEM200')] for i,f in enumerate(FIELDS)], [3.3,7.2,7.2], 9)
    p('FEM 기준해는 19개이며, 균질 해석해 비교와 열수지 계산을 수행하였다. 저자의 FEAP 자료에 대한 상대 L² 차이는 온도에서 약 4.50×10⁻⁶, 변위 벡터에서 약 2.04%였다. 저자/FEM 변위 진폭비는 약 0.98이며 원인은 확인되지 않았다. 진폭 보정은 적용하지 않았다. [3]', size=9.2)

    pair('B1', 'figB1_hard_fullfields.png', '강제 경계조건과 L BFGS의 역학장',
         f"5×40 구조에 L-BFGS 10,000회를 적용한 uₓ 평균오차는 {m('arch_l5_w40','ux'):.2f}%, σₓₓ 평균오차는 {m('arch_l5_w40','sxx'):.2f}%이다. 그림 17과 동일한 모델의 출력이다. 논문에는 두 그림이 동일한 학습 결과인지 명시되어 있지 않다.", own_height=20)

    pair('B2', 'figB2_hard_fullfields.png', '강제 경계조건과 L BFGS의 열장',
         f"T 평균오차는 {m('arch_l5_w40','T'):.2f}%, qₓ 평균오차는 {m('arch_l5_w40','qx'):.2f}%이다. Dirichlet 경계조건은 출력 변환으로 만족시키며, 내부 열유속에는 오차가 남았다. 부록 B의 재현 그림은 계산값에 따라 색 범위를 정하였으므로 원본과 색 막대 범위가 다르다.", own_height=18.7)

    page('참고문헌 및 자료 출처')
    p('[1] Harandi, A., Moeineddin, A., Kaliske, M., Reese, S., and Rezaei, S. Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering, 125(4), e7388, 2024 (온라인 게재 2023). https://doi.org/10.1002/nme.7388', size=10)
    p('[2] phyml4e/PINNs. Thermomechanic. 저자 공개 코드, 버전 5b426dc8eb2255b1bb3b8630619228b36cdb464e. https://github.com/phyml4e/PINNs/tree/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic', size=9.5)
    p('[3] AM_Project_Reproduce. 자체 FEniCS 및 PINN 계산 자료. https://github.com/JIN9811/AM_Project_Reproduce', size=10)
    DOC.add_paragraph('그림의 출처와 이용조건', 'Heading 2')
    p('논문 원본 23개는 문헌 [1]의 PDF에서 추출하여 비례 축소하였다. 축, 범례, 색 막대를 포함한 그림 내용은 변경하지 않았다. 원본 그림의 저작권은 저자에게 있으며, CC BY-NC-ND 4.0의 이용조건을 따른다. https://creativecommons.org/licenses/by-nc-nd/4.0/', size=10)
    p('재현 그림은 직접 학습한 PINN과 FEniCS 계산값으로 작성하였다. 논문 그림에서 수치를 추출하거나 저자의 학습 가중치를 사용하지 않았다. 원본 그림의 추출 위치와 자체 계산 결과의 수치 검증 자료는 저장소 [3]에 함께 수록하였으며, 원본 그림에는 프로젝트 코드와 별도의 이용조건이 적용된다.', size=10)
    DOC.add_paragraph('포함된 계산값', 'Heading 2')
    p('비교에는 Fig. 5–21, 23–25, A1, B1, B2의 데이터 그림 23쌍을 포함하였다. 개념도 Fig. 1–4와 22는 제외하였으며, 원문에 별도 그림이 없는 에너지 항 제외 결과는 그림 19에 이어 제시하였다.', size=10)
    p('계산값은 2026년 11월 5일 03:13 KST까지 학습을 마친 26개 모델의 출력이다. Adam 100만 회 결과와 25개 구조의 독립 시간 측정값은 포함하지 않았다. 난수 시드는 42로 고정하였으며, 다른 시드의 반복 결과나 통계 검정값은 없다.', size=10)

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
    EVIDENCE['expected_pages'] = FRONT_PAGES + 2 * len(expected) + 1
    EVIDENCE['table_style'] = 'White background, black text and horizontal top/header/bottom rules; no shading or vertical rules.'
    EVIDENCE['status'] = 'authored; rendering and visual review required'
    (ROOT/'validation/CONCISE_COMPARISON_QA.json').write_text(json.dumps(EVIDENCE, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output':str(OUT), 'figure_pairs':23, 'embedded_figures':len(DOC.inline_shapes), 'tables':len(DOC.tables), 'verified_metric_values':26*32}, ensure_ascii=False))


if __name__ == '__main__':
    main()
