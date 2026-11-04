"""Build the Korean academic project report from verified local results.

Run with the bundled document Python, or install requirements-documents.txt.
FONT_KO and FONT_KO_BOLD may select Korean TrueType fonts on another host.
"""
from pathlib import Path
import os,json,csv,html
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
regular=Path(os.environ.get('FONT_KO','C:/Windows/Fonts/malgun.ttf'))
bold=Path(os.environ.get('FONT_KO_BOLD','C:/Windows/Fonts/malgunbd.ttf'))
if not regular.exists():raise FileNotFoundError('Set FONT_KO to an installed Korean TrueType font.')
if not bold.exists():bold=regular
pdfmetrics.registerFont(TTFont('Korean',str(regular)))
pdfmetrics.registerFont(TTFont('KoreanBold',str(bold)))
pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='KoreanBold')
styles={
    'title':ParagraphStyle('title',fontName='KoreanBold',fontSize=22,leading=32,spaceAfter=18,textColor=colors.black,wordWrap='CJK'),
    'h1':ParagraphStyle('h1',fontName='KoreanBold',fontSize=15,leading=23,spaceAfter=12,textColor=colors.black,keepWithNext=True,wordWrap='CJK'),
    'h2':ParagraphStyle('h2',fontName='KoreanBold',fontSize=11.5,leading=18,spaceBefore=8,spaceAfter=6,textColor=colors.black,keepWithNext=True,wordWrap='CJK'),
    'body':ParagraphStyle('body',fontName='Korean',fontSize=10,leading=16.8,spaceAfter=8,wordWrap='CJK'),
    'caption':ParagraphStyle('caption',fontName='Korean',fontSize=8.5,leading=13,spaceAfter=9,wordWrap='CJK'),
    'table':ParagraphStyle('table',fontName='Korean',fontSize=8.5,leading=12,wordWrap='CJK'),
    'code':ParagraphStyle('code',fontName='Courier',fontSize=8,leading=12,spaceBefore=4,spaceAfter=8),
    'equation':ParagraphStyle('equation',fontName='Korean',fontSize=11,leading=20,leftIndent=8,spaceAfter=8,wordWrap='CJK')}
story=[];markdown=[];WIDTH=170*mm

def safe_markup(text):
    # Malgun Gothic lacks U+2212 and some Unicode super/subscript glyphs.
    # Use an ASCII minus and actual PDF text positioning, never a blank glyph.
    return text.replace('−','-').replace('ᵀ','<super>T</super>').replace('₀','<sub>0</sub>')

def h(text,level=1):
    story.append(Paragraph(html.escape(text),styles['h1' if level==1 else 'h2']))
    markdown.append('#'*level+' '+text+'\n')

def p(text,style='body'):
    story.append(Paragraph(safe_markup(text),styles[style]));markdown.append(text.replace('<b>','**').replace('</b>','**').replace('<br/>','\n')+'\n')

def eq(text):p(text,'equation')

def table(headers,rows,widths=None):
    data=[[Paragraph(safe_markup(html.escape(str(v))),styles['table']) for v in row] for row in [headers]+rows]
    t=Table(data,colWidths=[WIDTH*w for w in widths] if widths else None,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7E9EC')),
          ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
          ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
          ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,10)])
    markdown.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
                    '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')

def fig(name,caption,width=WIDTH,maxheight=225*mm):
    path=ROOT/'figures/paper_style'/f'{name}.png'
    with PILImage.open(path) as im:w,hpx=im.size
    height=width*hpx/w
    if height>maxheight:width*=maxheight/height;height=maxheight
    picture=Image(str(path),width=width,height=height);picture.hAlign='CENTER'
    story.extend([picture,Spacer(1,4),Paragraph(caption,styles['caption'])])
    markdown.append(f'![{caption}](figures/paper_style/{name}.png)\n')

def code(lines):
    text='<br/>'.join(html.escape(x).replace(' ','&nbsp;') for x in lines.splitlines())
    story.append(Paragraph(text,styles['code']));markdown.append('```powershell\n'+lines+'\n```\n')

def page():story.append(PageBreak());markdown.append('\n---\n')

summary=json.loads((ROOT/'results/summary.json').read_text())
metrics=summary['PINN']
def err(run,field,group='constitutive_outputs'):
    return 100*metrics[run]['comparisons']['FEM_n200'][group][field]['relative_L2']
runlabels=[('circle_coupled','원형 동시'),('circle_sequential','원형 순차'),
           ('rectangles_coupled','사각형 동시'),('rectangles_sequential','사각형 순차')]
femmeta={g:json.loads((ROOT/f'results/fem_{g}_n200/metadata.json').read_text()) for g in ('circle','rectangles')}

story.append(Paragraph('이질 재료 열탄성 문제의<br/>혼합 PINN과 FEniCS 재현 연구',styles['title']))
markdown.append('# 이질 재료 열탄성 문제의 혼합 PINN과 FEniCS 재현 연구\n')
p('Harandi 등 논문의 수치 구현과 그림 형식 재현을 위한 프로젝트 기술 보고서<br/>작성일 2026년 11월 3일 · 기준 논문 DOI 10.1002/nme.7388','caption')
h('초록',2)
p('본 프로젝트는 원형 개재물과 네 개의 사각형 개재물이 포함된 정상 열탄성 문제를 FEniCS 유한요소법과 혼합형 물리정보 신경망으로 계산하였다. 두 기하에 대해 동시 학습과 순차 학습을 수행하고, 50×50부터 200×200까지의 유한요소 기준해를 생성하였다. 기존 결과의 시각적 차이를 줄이기 위해 원문 PDF의 팔레트, 패널 순서, 색상 범위와 과정 그림의 구성을 재현하였다.')
p('주 실험의 온도 상대 L2 오차는 0.24–0.43%, 수평 변위 오차는 1.28–1.63%이다. 전단응력과 횡방향 열유속에는 더 큰 오차가 남는다. 따라서 본 결과는 주요 경계값 문제의 실행 가능한 재현이며, 원문의 모든 학습 스케줄이나 모든 장의 수렴을 입증하는 결과로 해석하지 않는다.')
table(['실험','ux 상대 L2','T 상대 L2','σx 상대 L2','qx 상대 L2'],
      [[label]+[f'{err(run,f):.3f}%' for f in ('ux','T','sxx','qx')] for run,label in runlabels],[.28,.18,.18,.18,.18])
p('표 1. FEniCS 200×200 기준의 주 실험 오차. 응력과 열유속은 변위·온도 미분으로 계산하였다. 직접 혼합 출력의 오차는 제13절에 별도로 제시한다.','caption')
h('보고서의 범위',2)
p('독자는 이 문서와 함께 제공된 Jupyter Notebook을 이용하여 환경 확인, 실제 FEM 해석, PINN 학습, 오차 재계산 및 그림 생성을 순서대로 수행할 수 있다. 결과를 바로 열어 보는 모드와 처음부터 다시 계산하는 모드를 명확히 구분하였다. 원문 그림과의 차이는 제15절에서 설명한다.')
p('주요어  평면변형률 · 열탄성 · 이질 재료 · 혼합 PINN · FEniCS · 수치 재현성','caption')

page();h('1 문제 정의와 물리 가정')
p('계산 영역은 Ω=[0,1]²인 단위 정사각형이다. 정상 열전도를 먼저 결정하고 그 온도장을 열팽창 하중으로 사용한다. 변형에 의한 열원이나 열전도 계수 변화는 고려하지 않으므로 물리적 연성은 온도에서 변위로 향한다. 재료는 등방성 선형 탄성체이며 3차원 평면변형률 조건 εzz=0을 적용한다.')
eq('∇·σ = 0,     ∇·q = 0,     q = −k∇T')
eq('ε = (∇u + ∇uᵀ)/2,     μ = E/[2(1+ν)],<br/>λ = Eν/[(1+ν)(1−2ν)]')
eq('σ = 2με + λ tr(ε)I − Eα(T−T₀)I/(1−2ν)')
p('면외 변형률은 0이지만 면외 열변형률과 응력은 일반적으로 0이 아니다. 따라서 열응력 계수에 평면응력 식을 대신 넣을 수 없다. 전단응력은 σxy=μ(∂ux/∂y+∂uy/∂x)로 계산한다.')
table(['항목','원형 개재물','사각형 개재물'],[['기지의 E와 k','0.3','1.0'],['개재물의 E와 k','1.0','0.5'],
      ['공통 상수','ν=0.3, α=1, T₀=0','동일'],['기하 정의','중심 (0.60,0.35), 반지름 0.15','저자 제공 재료 데이터의 네 사각형']],[.31,.345,.345])
h('경계조건',2)
p('왼쪽 경계의 온도는 1, 오른쪽은 0이다. 위아래 경계는 단열이므로 qy=0이다. 좌우 경계에서 ux=0, 위아래 경계에서 uy=0으로 두며, 네 경계의 접선 방향 외력은 0이다. 이에 따라 모든 변에서 σxy=0을 적용한다.')
h('단위와 기하 해석',2)
p('그림의 mm, GPa, K, W/mK 표시는 원문에 맞춘 수치 벤치마크 표기이다. 실제 열유속의 SI 값을 얻으려면 물리적 길이와 온도 차에 따른 ΔT/L 환산을 별도로 적용해야 한다. α=1은 현실적인 열팽창 계수를 뜻하지 않는다. 원형 중심은 그림에서 추정했으며, 사각형의 좌상단 높이는 원문 치수 0.20 대신 제공 데이터의 0.22를 따랐다.')

page();h('2 유한요소 기준해와 검증')
p('FEniCS DOLFIN 2019.2.0.64.dev0을 WSL Ubuntu에 설치하여 실제 선형 시스템을 조립하고 풀었다. 온도와 변위는 Q1 사각형 요소로 근사한다. 주 기준해는 요소 중심에서 결정한 불연속 재료 계수를 사용한다. 열전도 약형은 허용 시험함수 w에 대해 다음과 같다.')
eq('∫Ω k ∇T·∇w dΩ = 0')
p('온도 해를 대입한 뒤, 허용 가상변위 v에 대한 역학 문제를 푼다. 아래 식에서 β=Eα/(1−2ν)이다.')
eq('∫Ω [2με(u):ε(v) + λ div(u) div(v)] dΩ<br/>= ∫Ω β(T−T₀) div(v) dΩ')
p('응력과 열유속의 원래 요소 평균값은 별도 XDMF/HDF5 파일로 보존한다. 신경망과의 절점 비교에는 주변 요소를 이용한 복원값을 사용한다. 이 복원 연산과 재료 경계의 불연속은 응력·열유속의 비교에 영향을 줄 수 있다.')
table(['기하','요소 수','열 잔차','역학 잔차','열수지 불균형'],
      [[label,'40,000',f"{femmeta[g]['diagnostics']['thermal_relative_residual']:.2e}",
        f"{femmeta[g]['diagnostics']['mechanical_relative_residual']:.2e}",f"{femmeta[g]['diagnostics']['heat_balance']:.2e}"]
       for g,label in [('circle','원형'),('rectangles','사각형')]],[.16,.18,.22,.22,.22])
p('표 2. 200×200 요소 해석의 선형 시스템 잔차와 반력 기반 열수지. 이는 이산 방정식의 해법 검증이며, 유한요소 공간 오차가 0이라는 뜻은 아니다.','caption')
h('독립적인 균질 해 검증',2)
eq('T=1−x,    qx=1,    uy=0,<br/>ux=(13/14)x(1−x),    σxx=−1.25')
p('균질 재료 문제의 위 해를 단위 검증에 사용하였다. Jupyter Notebook에서도 작은 유한요소 문제를 새로 풀어 온도와 변위, 잔차 및 열수지를 확인한다. 이 검증용 결과는 연구 결과 폴더와 분리하여 저장한다.')

page();h('3 혼합 PINN의 구성과 손실')
fig('fig04_network_losses','그림 1. 원문 그림 4의 구조를 따라 다시 그린 혼합 PINN 구성. 각 출력은 독립된 5×40 tanh 신경망을 사용한다.',maxheight=92*mm)
p('혼합형 정식화는 변위와 온도뿐 아니라 응력·열유속을 독립 출력으로 둔다. 평형식에 필요한 공간 미분 차수는 1차가 된다. 변위·온도에서 계산한 물리적 응력·열유속과 독립 출력 사이의 차이를 연결 손실로 제약한다.')
table(['기호','의미','본 구현의 평가'],[['SF','평형식 잔차','내부점의 div(σ), div(q)'],['CNC','연결 잔차','경계를 포함한 모든 학습점'],
      ['DBC','주어진 변위와 온도','주 실험은 출력 변환으로 정확히 만족'],['NBC','접선력과 단열 조건','해당 경계의 전단응력 및 qy'],
      ['EF','전역 항','일관된 일률·열수지 항의 제곱']],[.14,.25,.61])
eq('ux=x(1−x)Nx,   uy=y(1−y)Ny,<br/>T=1−x+x(1−x)NT')
p('위 출력 변환은 Dirichlet 조건을 학습 전부터 정확히 만족한다. 전역 항은 원문의 인쇄된 에너지 식과 다르다. 코드의 energy 옵션으로 paper, authors, variational, none도 선택할 수 있으나, 본 주 실험은 balance를 사용하였다.')

page();h('4 동시 학습과 순차 학습')
fig('fig01_training_workflow','그림 2. 원문 그림 1의 세 패널 구성을 따른 학습 절차. 동시 학습과 순차 학습은 물리 문제를 바꾸지 않고 최적화 순서만 다르게 구성한다.',maxheight=90*mm)
table(['설정','실행값'],[['네트워크','8개 독립 네트워크, 은닉층 5개, 층당 40개 tanh 노드'],
      ['정밀도와 시드','float64, seed=42'],['학습률','Adam 0.001'],['장별 예산','Adam 2,000회 + L-BFGS 3,000회'],
      ['동시 학습의 총 업데이트','5,000회'],['순차 학습의 총 업데이트','10,000회, 최대 2,000회 단위로 장 전환'],
      ['실행 장치','Windows CUDA PyTorch 2.11.0, NVIDIA RTX 3090']],[.35,.65])
p('순차 학습에서는 열 학습 중 역학 네트워크를 고정하고, 역학 학습 중 온도장을 고정한다. 장이 바뀔 때 L-BFGS의 곡률 이력을 재시작한다. L-BFGS 한 번의 업데이트는 선 탐색 때문에 여러 번의 손실 평가를 포함할 수 있으므로 epoch와 같은 값으로 해석하지 않는다.')
p('원문은 동시 학습 100,000회와 순차 학습 200,000회의 Adam을 사용한다. 이 보고서의 과정 그림은 실제 실행한 짧은 스케줄을 표시하며, 축을 늘리거나 원문의 요동을 모사하는 가짜 데이터를 추가하지 않는다.')

page();h('5 학습점과 유한요소 격자')
fig('fig05_circle_points_mesh','그림 3. 원문 그림 5 형식의 원형 문제 학습점, 평가점, 유한요소 격자.',maxheight=72*mm)
p('원형 문제는 51×51 격자와 시드가 고정된 임의 내부점을 합한 5,000개 학습점을 사용한다. 사각형 문제는 경계 부근의 연결 조건을 더 안정적으로 학습하기 위해 71×71 격자 5,041개 점을 사용하였다.')
fig('fig12_rectangles_points_mesh','그림 4. 원문 그림 12 형식의 사각형 문제. 본 실행은 원문의 2,601개보다 많은 학습점을 사용하였다.',maxheight=72*mm)
p('평가점 패널은 101×101 전체 평가 격자의 51×51 부분집합을 보여준다. PINN의 재료 계수는 50×50 재료 격자의 절점 평균을 이중선형 보간한다. 불연속 계수를 쓰는 주 FEM과의 차이를 분리하기 위해 동일 보간 계수를 사용하는 100×100 FEM 해도 제공하였다.')

page();h('6 동시 학습 과정의 항별 손실')
fig('fig06_circle_coupled_losses','그림 5. 원문 그림 6 형식의 동시 학습 손실. 열·역학을 상하 패널로 구분하고 원문의 항별 색상을 따른다. 점선은 Adam에서 L-BFGS로의 전환이다.',maxheight=183*mm)
p('추가 과정 실험은 같은 시드와 같은 수치 설정에서 손실을 10회마다 기록한다. 로그에는 가장 최근 closure 평가값이 저장되며, 중간 장은 업데이트 직후 따로 저장된다. Dirichlet 손실은 정확히 0이므로 로그 축에 가상의 작은 양수를 대신 넣지 않았다.')
p('CNC 항은 직접 출력과 구성식 사이의 불일치를 나타낸다. 손실이 작아지는 것은 필요한 조건이지만, 각 물리량의 FEM 상대오차가 동시에 작아졌음을 보장하지 않는다. 최종 정확도는 독립 평가 격자의 장별 오차로 판단한다.')

page();h('7 순차 학습 과정의 항별 손실')
fig('fig10_circle_sequential_losses','그림 6. 원문 그림 10 형식의 순차 학습 손실. 회색 배경은 다른 장을 최적화한 구간이다. 관측되지 않은 해당 장의 손실은 연결하거나 0으로 채우지 않았다.',maxheight=183*mm)
p('주 순차 실험의 이력은 100회 간격으로 기록되어 있다. 따라서 원문의 매 반복 곡선보다 성기게 보일 수 있다. 임의 보간으로 원문과 같은 진동을 만드는 대신, 관측된 기록의 해상도를 유지하였다. 열 학습 뒤 역학 문제가 보는 온도장이 달라지므로 장 전환 직후 손실 양상이 변할 수 있다.')

page();h('8 학습 중간 상태와 변형 형상')
fig('process_training_evolution','그림 7. 새로 실행한 원형 동시 학습에서 저장한 실제 중간 상태. 각 행은 동일한 색상 범위를 사용하며, 초기 장의 범위 초과는 포화될 수 있다. 원시 값은 NPZ 파일로 제공한다.',maxheight=105*mm)
p('0, 100, 500, 2,000, 3,000, 5,000회 업데이트 직후의 온도, 수평 변위와 구성식 기반 응력을 비교한다. 그림은 최종 결과를 시각적으로 보간한 것이 아니라 각 단계에서 신경망을 직접 평가하여 생성하였다.')
fig('fig07_circle_deformation','그림 8. 원문 그림 7 형식의 변형 전후 형상. 실제 변위에 배율 1을 적용하고 재료 색상은 변형 전 좌표를 따른다.',maxheight=72*mm)
p('이 벤치마크의 큰 열팽창 계수 때문에 변형이 크게 보인다. 이를 실제 재료의 대변형 해석으로 해석해서는 안 되며, 계산 모델 자체는 선형 열탄성이다.')

page();h('9 원형 문제의 역학장 결과')
fig('fig08_circle_coupled_mechanical','그림 9. 원문 그림 8 형식. 좌측은 본 PINN, 중앙은 본 FEniCS 50×50, 우측은 두 결과의 절대 차이다. 원문의 색상 범위를 그대로 사용했고 범위 밖 값은 삼각형으로 표시하였다.',maxheight=221*mm)

page();h('10 원형 문제의 열장 결과')
fig('fig09_circle_coupled_thermal','그림 10. 원문 그림 9 형식의 온도와 열유속. 원문과 같은 팔레트와 색상 범위에서 개재물 주위의 열유속 변화를 확인할 수 있다.',maxheight=164*mm)
p('온도장은 전체적으로 잘 일치하지만, 개재물 경계에서 열유속 차이가 집중된다. 미분 연산은 온도의 국소 오차를 확대하므로 온도 오차가 작다는 이유만으로 열유속의 같은 정확도를 가정할 수 없다.')
p('그림의 절대오차 색상은 원문 오차의 최댓값에 맞춰 고정했다. 그 범위 자체는 본 계산의 최대 오차를 의미하지 않는다. 본 계산의 실제 최솟값·최댓값과 범위 초과 개수는 paper_scale_audit.csv에 저장되어 있다.')

page();h('11 사각형 문제의 역학장 결과')
fig('fig13_rectangles_coupled_mechanical','그림 11. 원문 그림 13 형식. 사각형 모서리와 재료 경계의 응력 오차는 내부의 매끄러운 영역보다 크다. 원문과 재현 결과의 색상 범위를 구분 없이 자동 조절하지 않았다.',maxheight=221*mm)

page();h('12 사각형 문제의 열장 결과와 단면 해석')
fig('fig14_rectangles_coupled_thermal','그림 12. 원문 그림 14 형식. 온도, 축방향 열유속, 횡방향 열유속의 분포와 절대오차를 비교한다.',maxheight=164*mm)
p('사각형 문제에서는 기지보다 작은 전도율을 가진 개재물 때문에 열유속 분포가 층상으로 변한다. 모서리 부근의 구배가 급해지는 영역은 매끄러운 신경망 근사가 특히 어려운 부분이다. 원문 그림과 비슷한 색상 분포가 얻어져도 국소 응력과 열유속 정확도는 별도의 수치 지표로 확인해야 한다.')
p('다음 절은 모든 장의 오차를 제시한다. 그림이 유사하게 보이는지와 같은 방식으로, 직접 혼합 출력과 미분 기반 출력이 같은 값이라고 가정하지 않고 두 표현을 분리하여 평가한다.')

page();h('13 전체 물리량의 정량 검증')
fields=('ux','uy','sxx','syy','sxy','T','qx','qy')
table(['물리량']+[label for _,label in runlabels],[[f]+[f'{err(run,f):.3f}%' for run,_ in runlabels] for f in fields],[.16,.21,.21,.21,.21])
p('표 3. 구성식·Fourier 법칙으로 재계산한 장의 상대 L2 오차. 기준은 FEniCS 200×200, 평가점은 101×101이다.','caption')
table(['직접 혼합 출력']+[label for _,label in runlabels],[[f]+[f'{err(run,f,"direct_mixed_outputs"):.3f}%' for run,_ in runlabels] for f in ('sxx','syy','sxy','qx','qy')],[.16,.21,.21,.21,.21])
p('표 4. 신경망의 독립 응력·열유속 출력에 대한 상대 L2 오차. 표 3과 같은 참조 및 평가 격자를 사용하였다.','caption')
eq('상대 L2 = √[Σᵢ(aᵢ−aᵢ,FEM)²] / √[Σᵢaᵢ,FEM²]')
p('이 지표는 동일한 균일 점 집합에서 계산한 비가중 표본 노름이다. 전단응력처럼 공간적으로 작거나 국소적인 성분은 분모가 작아 상대오차가 커질 수 있다. 이것만으로 오차를 무시할 수는 없으며, 절대오차와 오차 위치를 함께 확인해야 한다. 주 보고서의 MAE, 기준 최댓값 정규화 오차와 최대 오차도 함께 제공한다.')

page();h('14 단면 비교와 유한요소 수렴')
fig('fig16_rectangles_sections','그림 13. 원문 그림 16 형식의 x=0.5 단면. 동시 학습은 파란 파선, 순차 학습은 파란 점선, FEM은 검은 실선으로 표시하였다.',maxheight=155*mm)
p('원문과 같은 단면·변수·선 종류를 사용하였다. 본 작업에서 전부 실행하지 않은 표준 PINN 및 DEM의 비교 곡선은 만들어 넣지 않았다. 사각형 기준해의 50, 100, 200 요소 수렴 그림은 전체 그림집의 figA1_rectangles_sections에 포함되어 있다.')
p('저자가 제공한 FEAP 데이터의 온도는 동일한 50×50 FEniCS 해와 약 4.5×10⁻⁶의 상대 L2 차이를 보였다. 반면 변위는 독립적으로 조립한 해의 약 0.98배였다. 자료만으로 원인을 단정할 수 없어, 이 차이를 없애기 위한 진폭 보정은 수행하지 않았다.')

page();h('15 그림의 시각적 일치와 수치적 차이')
p('기존 그림과 원문 사이의 주요 시각적 차이는 팔레트, 자동 색상 범위, 축 표시, 패널 밀도, 오차 범위 및 기준 FEM 해상도에서 발생하였다. 이번 그림은 원문 PDF에서 팔레트를 추출하고 인쇄된 범위를 변수·패널별로 기록한 뒤 재생성하였다. 원문 발췌와 재현 그림을 직접 나란히 비교할 수 있는 HTML 갤러리도 제공한다.')
limits=json.loads((ROOT/'data/paper_color_limits.json').read_text())
table(['물리량','원형 FEM 색상 범위','사각형 FEM 색상 범위'],
      [[f,' ~ '.join(f'{v:g}' for v in limits['circle'][f]['fem']),
        ' ~ '.join(f'{v:g}' for v in limits['rectangles'][f]['fem'])] for f in fields],[.20,.40,.40])
p('표 5. 원문의 색상 범위. 전단응력의 PINN 패널 하한은 FEM 패널과 다르므로 코드에 별도로 보존하였다. 오차 패널 범위도 별도로 저장되어 있다.','caption')
h('변경하지 않은 수치 정보',2)
p('원시 예측 배열, 기존 FEM 해, 기존 학습 이력과 오차 지표는 시각적 일치를 위해 수정하지 않았다. 색상 범위에 들어오도록 진폭을 조정하거나, 원문의 반복 수에 맞춰 이력을 늘리거나, 미실행 실험의 곡선을 생성하지 않았다. 색상 포화는 모든 패널에서 감사표로 추적한다.')
h('남아 있는 재현 한계',2)
p('원형 중심의 추정, 사각형 치수 표기와 제공 데이터의 불일치, 재료 보간, 구성식 해석과 경계조건 보완, 원문 에너지 항과 balance 항의 차이, 최적화 예산 차이가 남아 있다. 공개된 사각형 모델은 별도로 변환·평가했지만 해당 모델의 학습 이력은 제공되지 않았다. 따라서 색상과 배치를 맞춘 사실을 원문 수치의 완전한 복제라고 표현하지 않는다.')

page();h('16 Jupyter Notebook으로 재현하는 방법')
p('프로젝트 폴더 전체를 유지한 채 start_notebook.ps1를 실행하고 Harandi CUDA Python 3.11 커널을 선택한다. 노트북 파일만 따로 옮기면 구현 모듈과 데이터 경로가 없어 실행할 수 없다. 새 컴퓨터에서는 setup.ps1를 먼저 실행한다.')
code('powershell -NoProfile -ExecutionPolicy Bypass -File .\\setup.ps1\npowershell -NoProfile -ExecutionPolicy Bypass -File .\\start_notebook.ps1')
p('노트북의 기본 모드는 저장된 주 연구 결과를 사용한다. 그러나 실제 실행 경로를 확인하기 위해 균질 FEM 문제와 작은 PINN 학습을 새로 수행하며, 주 실험의 오차와 그림도 다시 계산한다. 실제 전체 재계산을 하려면 첫 설정 셀의 아래 세 값을 True로 바꾸고 위에서부터 모두 실행한다.')
code('RECOMPUTE_FEM = True\nRETRAIN_PINN = True\nRETRAIN_PROCESS = True')
p('RECOMPUTE_FEM은 8개 기준해를 다시 푼다. RETRAIN_PINN은 네 주 PINN을 다시 학습한다. RETRAIN_PROCESS는 중간 상태를 저장하는 추가 원형 동시 학습을 다시 수행한다. 이 옵션은 지정한 결과 폴더를 갱신하므로 기존 실험을 보존하려면 폴더를 복사하거나 출력 경로를 변경한다.')
h('명령행에서 실행하는 방법',2)
code('.\\run.ps1 -Task test\n.\\run.ps1 -Task verified -Force\n.\\run.ps1 -Task process\n.\\run.ps1 -Task paper-figures')
p('그림만 다시 만들 때는 paper-figures를 사용한다. 원문 장기 스케줄은 paper, 부록 스케줄은 appendix로 실행한다. 장기 스케줄은 계산 시간이 더 길며, 현재 보고서의 실행 결과와 구분해서 저장한다. GPU·PyTorch 버전과 선 탐색 동작 차이로 재학습 가중치가 비트 단위까지 일치하지 않을 수 있다.')

page();h('17 산출물과 자료 출처')
table(['경로','용도'],[['Harandi_Reproduction_KO.ipynb','실행 결과가 저장된 재현 노트북'],['harandi/fem.py','실제 FEniCS 약형 조립 및 해법'],
      ['harandi/pinn.py, train.py','혼합 손실, 동시·순차 학습, 체크포인트'],['harandi/process.py','실제 중간 상태 저장 실험'],
      ['harandi/paper_figures.py','논문 형식 그림과 색상 범위 감사표'],['data/paper_palette.json','원문 PDF에서 얻은 팔레트와 출처'],
      ['data/paper_color_limits.json','원문 패널별 고정 색상 범위'],['results/paper_scale_audit.csv','실제 값 범위와 색상 포화 통계'],
      ['results/process_circle_coupled','과정 학습 이력과 6개 중간 상태'],['PAPER_FIGURES_KO.html','원문 발췌와 재현 그림의 비교'],
      ['PAPER_STYLE_FIGURES.pdf','전체 논문 형식 그림집'],['scripts/build_korean_report.py','본 보고서와 Markdown 원고 생성 코드']],[.47,.53])
h('참고문헌과 근거 자료',2)
p('[1] Harandi et al. Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering. 온라인 출판 2023, 정규 권호 2024. DOI: <link href="https://doi.org/10.1002/nme.7388" color="blue">10.1002/nme.7388</link>.')
p('[2] 저자 공개 구현 및 데이터. <link href="https://github.com/phyml4e/PINNs" color="blue">github.com/phyml4e/PINNs</link>, Thermomechanic 폴더. 고정 커밋 5b426dc8eb2255b1bb3b8630619228b36cdb464e. 배포 라이선스는 data/AUTHORS_LICENSE에 보존하였다.')
p('[3] FEniCS Project. Legacy FEniCS 설치 안내. <link href="https://fenicsproject.org/download/archive/" color="blue">fenicsproject.org/download/archive</link>.')
p('[4] 본 실행의 근거 자료. SOURCE_PROVENANCE.json, EXECUTION_MANIFEST.json, 각 실행 폴더의 config.json·environment.json·metrics.json·history.csv. 원문 PDF는 사용자가 제공한 Downloads 파일이며 SHA-256으로 식별하였다.')
p('원문 발췌 이미지는 제공된 PDF와 재현 그림을 비교하기 위한 자료이다. 새 그림은 본 계산 배열로 작성하였다. 모든 수치 주장과 그림은 해당 코드와 원시 결과 파일에서 확인할 수 있도록 연결하였다.','caption')

def decorate(c,doc):
    c.saveState();c.setFont('Korean',8);c.setFillColor(colors.black)
    c.drawString(20*mm,285*mm,'열탄성 혼합 PINN과 FEniCS 재현 연구')
    c.drawRightString(190*mm,13*mm,str(doc.page))
    c.setFont('Korean',7.5);c.drawString(20*mm,13*mm,'2026년 11월 3일  |  DOI 10.1002/nme.7388  |  프로젝트 기술 보고서')
    c.restoreState()

doc=SimpleDocTemplate(str(ROOT/'PROJECT_REPORT_KO.pdf'),pagesize=A4,
    rightMargin=20*mm,leftMargin=20*mm,topMargin=22*mm,bottomMargin=21*mm,
    title='이질 재료 열탄성 문제의 혼합 PINN과 FEniCS 재현 연구',author='Reproduction project')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
(ROOT/'PROJECT_REPORT_KO.md').write_text('\n'.join(markdown),encoding='utf-8')
print('Wrote Korean academic PDF and editable Markdown source.')
