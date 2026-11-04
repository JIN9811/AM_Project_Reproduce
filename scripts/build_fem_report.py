"""Generate a separate Korean FEM study with XeLaTeX and actual saved results."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

from latex_report import LatexReport

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harandi.physics import FIELDS, RECTANGLES

SYMBOLS=[r'$u_x$',r'$u_y$',r'$\sigma_{xx}$',r'$\sigma_{yy}$',r'$\sigma_{xy}$',r'$T$',r'$q_x$',r'$q_y$']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def fmt(x): return f'{x:.4g}'
def pct(x): return f'{100*x:.4g}'
def load(root,name):
    with np.load(root/'results'/name/'solution.npz') as f: return {k:f[k].copy() for k in f.files}
def err(a,b):
    den=np.linalg.norm(b,axis=0)
    return np.divide(np.linalg.norm(a-b,axis=0),den,out=np.full_like(den,np.nan),where=den>1e-15)


def outline(ax,geom):
    if geom=='circle': ax.add_patch(Circle((.6,.35),.15,fill=False,lw=.55,ec='#222222'))
    else:
        for x0,x1,y0,y1 in RECTANGLES: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=.55,ec='#222222'))


def make_figures(root):
    dest=root/'figures/fem_report'; dest.mkdir(parents=True,exist_ok=True)
    cmap=LinearSegmentedColormap.from_list('paper',read(root/'data/paper_palette.json')['rgb'])
    limits=read(root/'data/paper_color_limits.json'); records=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'figure.dpi':130})
    for geom in ('circle','rectangles'):
        d=load(root,f'fem_{geom}_n200'); a=d['fields']; xy=d['xy']; order=np.lexsort((xy[:,0],xy[:,1]))
        for title,fields in [('mechanical',(0,1,2,3,4)),('thermal',(5,6,7))]:
            rows=2 if title=='mechanical' else 1
            fig,axes=plt.subplots(rows,3,figsize=(10,3.2*rows),layout='constrained')
            for ax,j in zip(axes.flat,fields):
                lo,hi=limits[geom][FIELDS[j]]['fem']; v=a[:,j]
                im=ax.imshow(v[order].reshape(101,101),extent=(0,1,0,1),origin='lower',cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
                outline(ax,geom); ax.set(xlabel='x',ylabel='y',title=SYMBOLS[j]); ax.set_xticks([0,.5,1]); ax.set_yticks([0,.5,1])
                fig.colorbar(im,ax=ax,shrink=.84,extend='both',format='%.3g')
                records.append({'geometry':geom,'field':FIELDS[j],'vmin':lo,'vmax':hi,
                    'data_min':float(v.min()),'data_max':float(v.max()),
                    'below_percent':float(np.mean(v<lo)*100),'above_percent':float(np.mean(v>hi)*100)})
            for ax in list(axes.flat)[len(fields):]: ax.set_visible(False)
            fig.savefig(dest/f'{geom}_{title}.png',dpi=220); fig.savefig(dest/f'{geom}_{title}.pdf'); plt.close(fig)
    # Each curve is a direct, common-grid difference against n=200, not an error estimator.
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,geom in zip(axs,('circle','rectangles')):
        ref=load(root,f'fem_{geom}_n200')['fields']
        vals=np.array([err(load(root,f'fem_{geom}_n{n}')['fields'],ref)*100 for n in (50,100)])
        for j in range(8): ax.loglog([50,100],vals[:,j],'-o',ms=4,label=SYMBOLS[j])
        ax.set(title=geom,xlabel='Elements per edge, n',ylabel='Relative difference to FEM200 (%)',xticks=[50,100])
        ax.set_xticklabels(['50','100']); ax.grid(True,which='both',alpha=.2); ax.legend(ncol=2,fontsize=8)
    fig.savefig(dest/'convergence.png',dpi=220); fig.savefig(dest/'convergence.pdf'); plt.close(fig)
    # Recovering discontinuous fields creates a visibly different interface profile.
    fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    d=load(root,'fem_circle_n100'); y=.355
    cell=np.isclose(d['cell_xy'][:,1],y); nodes=np.isclose(d['xy'][:,1],.35)
    for ax,j,cj in zip(axs,(4,6),(2,3)):
        ind=np.argsort(d['cell_xy'][cell,0]); xx=d['cell_xy'][cell,0][ind]; vv=d['cell_fields'][cell,cj][ind]
        ax.step(xx,vv,where='mid',label='Raw cell average, y=0.355',color='#0072BD')
        ind=np.argsort(d['xy'][nodes,0]); ax.plot(d['xy'][nodes,0][ind],d['fields'][nodes,j][ind],label='Recovered nodal field, y=0.350',color='#D95319')
        ax.set(xlabel='x',ylabel=SYMBOLS[j],title='circle / FEM100'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(dest/'recovery.png',dpi=220); fig.savefig(dest/'recovery.pdf'); plt.close(fig)
    (dest/'color_scale_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    return records


def build(root):
    evidence=read(root/'results/fem_report_verification.json')
    if evidence['status']!='pass' or evidence['reference_count']!=19: raise ValueError('Run scripts/verify_fem_report.py with FEniCS first.')
    for item in evidence['reference_records']:
        path=root/item['folder']/'solution.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['solution_sha256']: raise ValueError('Stale FEM evidence: '+str(path))
    audit=make_figures(root); refs=evidence['reference_records']; ucheck=evidence['uniform_analytic_checks']; authors=evidence['author_feap_check']
    residual=max(max(x['metadata']['diagnostics'][key] for key in ('thermal_relative_residual','mechanical_relative_residual')) for x in refs)
    balance=max(abs(x['metadata']['diagnostics']['heat_balance']) for x in refs)
    r=LatexReport(root,stem='FEM_REPORT_KO',header='이질 재료 열탄성 문제의 FEniCS 유한요소 검증',
                  status='FEM 독립 검증 보고서 | 저장 기준해 19개 + 균질 해석해 검증 6개')
    r.p('이질 재료 열탄성 문제의<br/>FEniCS 유한요소 정식화 및 검증','title')
    r.p('Finite Element Formulation and Verification of Heterogeneous Thermoelasticity','small')
    r.p('Harandi 등의 혼합 PINN 연구에 대한 독립 기준해 보고서 | AM Project Reproduce','small')
    r.p('작성일: '+datetime.now().strftime('%Y-%m-%d')+' | XeLaTeX 조판 · 계산 자료 기반 재생성','small')
    r.h('국문 초록',2)
    r.p(f'본 보고서는 정상 열전도와 평면변형률 열탄성 문제에 대한 FEniCS 유한요소 기준해의 정식화, 구현 및 검증을 제시한다. 원형 및 사각형 개재물 문제에서 연속 쌍선형 요소를 사용하고, 열전도 해를 열하중으로 전달하여 역학 문제를 순차적으로 해석하였다. 저장된 19개 기준해의 최대 선형 시스템 상대 잔차는 {residual:.3e}, 최대 열수지 불균형은 {balance:.3e}이었다. 별도의 균질 해석해 검증에서는 절점 일치와 요소 내부 보간 오차를 구분하였다. 저자 공개 FEAP 자료와 비교한 온도의 상대 L2 차이는 {pct(authors["temperature_relative_l2"])}%, 변위 벡터의 차이는 {pct(authors["displacement_relative_l2_raw"])}%였다. 변위의 약 0.98배 진폭 관계는 관측 결과로만 기록하며 수치해를 보정하지 않았다. 본 보고서는 신경망 학습과 독립적으로 FEM 기준해의 신뢰성과 한계를 설명한다.')
    r.p('핵심용어: 유한요소법, 열탄성, 평면변형률, 이질 재료, FEniCS, 격자 수렴','small')
    r.p('이 문서의 FEM 검증은 완료되었다. 전체 PINN·전이학습 연구의 완료 여부는 별도의 전체 보고서에 표시한다. 저장된 선형 시스템 잔차는 이산 방정식의 해법 정확도이며, 연속체 문제에 대한 총 오차가 아니다.')
    r.page(); r.h('Abstract',2)
    r.p(f'This report documents a sequential finite element solution of steady heat conduction and plane-strain thermoelasticity in heterogeneous domains using FEniCS. Bilinear quadrilateral elements are used for temperature and displacement. Nineteen saved reference solutions satisfy the discrete equations with a maximum relative algebraic residual of {residual:.3e}. Six additional uniform-material runs distinguish nodal exactness from interpolation error. Comparisons with the authors\' released FEAP data give relative differences of {pct(authors["temperature_relative_l2"])}% in temperature and {pct(authors["displacement_relative_l2_raw"])}% in the displacement vector. The observed amplitude ratio is reported without fitting or rescaling the computed fields.')
    r.h('기호',2)
    r.table(['기호','정의'],[[r'$\Omega,\Gamma$','단위 정사각형 영역과 경계'],[r'$\boldsymbol u,T$','변위 벡터와 온도'],[r'$\boldsymbol\sigma,\boldsymbol\varepsilon,\boldsymbol q$','Cauchy 응력, 소변형률, 열유속'],[r'$E,\nu,\alpha,k,T_0$','탄성계수, Poisson 비, 열팽창계수, 열전도도, 기준온도'],[r'$\lambda,\mu,\beta$','Lamé 상수와 열응력 계수'],[r'$n,h,Q_1,DQ_0$','방향별 요소 수, 요소 크기, 연속 쌍선형 및 불연속 상수 공간']],[.3,.7])
    r.toc()
    r.page(); r.h('1. 연구 범위와 원저자 자료의 역할')
    r.p('원논문은 혼합 물리정보 신경망의 성능을 유한요소해와 비교한다. 본 연구는 이를 검토할 독립 기준해를 FEniCS/DOLFIN으로 구성하였다. 저자 공개 저장소는 SciANN 기반 신경망 코드와 FEAP 수치 자료를 제공하며, 본 프로젝트의 FEniCS 코드를 제공한 출처는 아니다. 따라서 해석 대상과 공개 수치 자료를 대조하되, 해석 프로그램·계수 표현·후처리가 같은 것으로 가정하지 않는다.')
    r.table(['자료','역할'],[['원논문 DOI 10.1002/nme.7388','기하·재료·경계조건 및 비교 문제의 근거'],['phyml4e/PINNs의 Thermo_mechanic.py','출력 변수, 구성식, 일 및 경계조건 표현의 대조'],['node.txt 및 rectan_inclus.csv','사각형 재료 분포와 외부 FEAP 검증 자료'],['harandi/fem.py, harandi/physics.py','독립적인 FEniCS 조립·해법과 공유 물리 모델'],['results/fem_report_verification.json','본 보고서에 사용한 해시·잔차·해석해·외부 자료 비교']],[.45,.55])
    r.p('고정한 원저자 커밋은 5b426dc8eb2255b1bb3b8630619228b36cdb464e이다. 저자 코드의 독립 출력과 유도 출력 구분을 따르되, FEM에서는 온도와 변위를 주 미지수로 풀고 응력과 열유속을 구성식으로 계산한다. 신경망에서 독립 응력을 출력하는 혼합 정식화와 FEM의 주 변수 정식화를 혼동하지 않는다.')
    r.page(); r.h('2. 지배방정식과 평면변형률 구성식')
    r.p(r'체적력과 내부 열원이 없는 $\Omega=(0,1)^2$에서 정상 문제를 고려한다. 열전도는 변위에 의존하지 않으며, 온도 변화가 열팽창을 통해 역학에만 영향을 준다.')
    r.equation(r'\nabla\!\cdot\boldsymbol\sigma=\boldsymbol0,\qquad\nabla\!\cdot\boldsymbol q=0,\qquad\boldsymbol q=-k\nabla T')
    r.equation(r'\boldsymbol\varepsilon=\tfrac12\left[\nabla\boldsymbol u+(\nabla\boldsymbol u)^{\mathsf T}\right],\qquad\boldsymbol\varepsilon^{\mathrm e}=\boldsymbol\varepsilon-\alpha(T-T_0)\boldsymbol I')
    r.equation(r'\mu=\frac{E}{2(1+\nu)},\qquad\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},\qquad\beta=\frac{E\alpha}{1-2\nu}')
    r.equation(r'\boldsymbol\sigma=2\mu\boldsymbol\varepsilon+\lambda\operatorname{tr}(\boldsymbol\varepsilon)\boldsymbol I-\beta(T-T_0)\boldsymbol I')
    r.p(r'평면변형률은 $\varepsilon_{zz}=0$을 뜻한다. 면외 탄성변형률 $\varepsilon^{\mathrm e}_{zz}=-\alpha(T-T_0)$는 일반적으로 0이 아니므로, 면내 열응력 계수를 평면응력 계수로 대체해서는 안 된다. 면외 응력과 면내 전단응력은 다음과 같다.')
    r.equation(r'\sigma_{zz}=\lambda(u_{x,x}+u_{y,y})-\beta(T-T_0),\qquad\sigma_{xy}=\mu(u_{x,y}+u_{y,x})=2\mu\varepsilon_{xy}')
    r.p('저자 공개 파일의 strain_xy는 텐서 전단변형률이지만 sigma_xy의 계수는 위 표준 구성식의 절반에 해당한다. 본 FEM과 PINN은 위 식을 공통으로 적용한다. 이 선택은 결과 차이의 가능한 원인이지만, 공개 FEAP 자료의 0.98배 변위 관계를 이 차이만으로 설명할 근거는 없다.')
    r.page(); r.h('3. 경계조건과 재료 분포')
    r.equation(r'\begin{aligned}u_x&=0 &&\text{on }x=0,1,& u_y&=0 &&\text{on }y=0,1,\\ T&=1 &&\text{on }x=0,& T&=0 &&\text{on }x=1,\\\boldsymbol t\cdot\boldsymbol\tau&=0 &&\text{on }\partial\Omega,&\boldsymbol q\cdot\boldsymbol n&=0 &&\text{on }y=0,1.\end{aligned}')
    r.p(r'$\boldsymbol t=\boldsymbol\sigma\boldsymbol n$이고 $\boldsymbol\tau$는 경계 접선이다. 변위의 법선 성분만 구속하는 롤러 조건이므로 구속 방향의 반력은 일반적으로 0이 아니다. 네 경계를 완전 고정 경계로 바꾸거나 전체 표면력을 0으로 두면 다른 문제가 된다.')
    r.table(['문제','기지 (E,k)','개재물 (E,k)','위치'],[['원형','(0.3,0.3)','(1,1)','중심 (0.60,0.35), 반지름 0.15'],['사각형','(1,1)','(0.5,0.5)','아래 네 사각형 영역'],['재료비 문제','(1,1)','(1/r,1/r)','사각형; r=1,...,10,15']],[.16,.18,.19,.47])
    r.table(['개재물','x 최소','x 최대','y 최소','y 최대'],[[str(i+1)]+list(bounds) for i,bounds in enumerate(RECTANGLES)],[.2]*5)
    r.p(r'공통으로 $\nu=0.3$, $\alpha=1$, $T_0=0$을 사용한다. $\alpha=1$과 단위 온도차는 논문 벤치마크를 위한 수치 조건이며 실제 재료의 설계 열하중으로 해석하지 않는다. 논문 내 길이 단위 표기가 일치하지 않으므로 본 문서의 해는 수치 단위 영역을 기준으로 비교한다. 별도의 SI 단위 환산을 수행한 결과는 아니다.')
    r.p('원형 중심은 논문 도해에서 재구성한 값이다. 사각형 중 좌상단 개재물의 높이는 저자 재료 자료에서 0.22이며 인쇄된 치수 0.20과 다르다. 본 보고서는 사용한 실제 좌표를 위 표에 고정하여 재현할 수 있게 하였다.')
    r.page(); r.h('4. 약형 정식화')
    r.p(r'온도 시험함수는 온도 Dirichlet 경계에서 0이고, 변위 시험함수는 롤러가 구속한 성분에서 0이다. 재료 계수가 양수이고 $-1<\nu<1/2$인 조건을 사용한다.')
    r.equation(r'\begin{aligned}V_T^0&=\{w\in H^1(\Omega):w=0\text{ on }x=0,1\},\\V_u^0&=\{\boldsymbol v\in[H^1(\Omega)]^2:v_x=0\text{ on }x=0,1,\ v_y=0\text{ on }y=0,1\}.\end{aligned}')
    r.h('4.1 열전도 약형',2)
    r.p('열보존식에 시험함수를 곱해 부분적분한다. Dirichlet 경계에서는 시험함수가 0이고 나머지 경계에서는 법선 열유속이 0이므로 경계 적분이 소거된다.')
    r.equation(r'\int_\Omega k\nabla T\cdot\nabla w\,\mathrm d\Omega=-\int_{\Gamma_{N,T}}(\boldsymbol q\cdot\boldsymbol n)w\,\mathrm d\Gamma=0,\qquad \forall w\in V_T^0')
    r.h('4.2 역학 약형',2)
    r.p('대칭 응력을 이용해 변위 구배를 대칭 구배로 치환한다. 롤러 경계의 허용 변위 방향에서 접선력이 0이므로 외부 일은 소거된다. 열응력 항을 우변으로 옮기면 다음 선형 문제를 얻는다.')
    r.equation(r'\begin{aligned}a_u(\boldsymbol u,\boldsymbol v)&=\int_\Omega\left[2\mu\boldsymbol\varepsilon(\boldsymbol u):\boldsymbol\varepsilon(\boldsymbol v)+\lambda(\nabla\cdot\boldsymbol u)(\nabla\cdot\boldsymbol v)\right]\mathrm d\Omega,\\\ell_u(\boldsymbol v;T)&=\int_\Omega\beta(T-T_0)\nabla\cdot\boldsymbol v\,\mathrm d\Omega,\\a_u(\boldsymbol u,\boldsymbol v)&=\ell_u(\boldsymbol v;T)\qquad\forall\boldsymbol v\in V_u^0.\end{aligned}')
    r.p(r'재료 계면에서는 완전 접합을 가정한다. 변위와 온도는 연속이고, 계면 양쪽의 표면력과 법선 열유속이 균형을 이룬다. 이 조건은 전역 약형에 포함되므로 계면의 $E$나 $k$를 고전적으로 미분할 필요가 없다. 다만 개별 응력 성분이나 온도 구배가 모두 연속이라는 뜻은 아니다.')
    r.page(); r.h('5. 공간 이산화와 FEniCS 구현')
    r.equation(r'T_h=\sum_a N_a T_a,\qquad\boldsymbol u_h=\sum_a N_a\boldsymbol u_a,\qquad N_a|_K\in Q_1(K),\qquad h=1/n')
    r.table(['항목','구현'],[['격자','UnitSquareMesh.create: n×n 정형 사각형 요소'],['온도 / 변위 공간','Q1 스칼라 / 2성분 Q1 벡터'],['주 재료 계수','요소 중심에서 상을 판별하여 DQ0에 저장'],['보간 재료 계수','50×50 재료 격자의 절점 평균·쌍선형 보간; 별도 FEM100'],['적분 / 선형 해법','quadrature_degree=4 / MUMPS 직접해법'],['실행 환경','Ubuntu WSL, DOLFIN 2019.2.0.64.dev0, 단일 MPI rank'],['평가 격자','101×101 공통 좌표; 요소 내 Q1 보간']],[.28,.72])
    r.equation(r'\boldsymbol K_T\boldsymbol T=\boldsymbol b_T,\qquad\boldsymbol K_u\boldsymbol U=\boldsymbol f_{\mathrm{th}}(\boldsymbol T)')
    r.p('첫째, 기하와 재료 함수를 생성한다. 둘째, 온도 Dirichlet 조건을 적용한 열전도 시스템을 조립하고 푼다. 셋째, 구한 온도를 열하중으로 대입하여 롤러 조건의 역학 시스템을 푼다. 넷째, 구성식 응력과 Fourier 열유속을 요소 평균으로 투영한다. 마지막으로 원시 값, 복원 값 및 검증 잔차를 저장한다. 단방향 연성의 선형 문제이므로 FEM에서는 열·역학 사이의 반복 학습이 필요하지 않다.')
    r.table(['n','요소 수','온도 자유도','변위 자유도'],[[n,n*n,(n+1)**2,2*(n+1)**2] for n in (50,100,200)],[.16,.24,.30,.30])
    r.p('자유도 수는 경계조건 제거 전의 값이다. 원형 계면은 요소 중심의 상 판별로 근사하므로 격자 세분화 시 기하 근사도 바뀐다. 반면 제시한 사각형 좌표는 50·100·200 격자의 요소 경계에 정렬된다. 두 기하에서 수렴 양상이 다른 이유를 해석할 때 이 차이를 고려한다.')
    r.page(); r.h('6. 응력·열유속 복원과 색상 기준')
    r.equation(r'\overline{\boldsymbol g}_K=\frac1{|K|}\int_K\boldsymbol g_h\,\mathrm d\Omega,\qquad\boldsymbol g_i^{\mathrm{rec}}=\frac{\sum_{K\ni i}|K|\overline{\boldsymbol g}_K}{\sum_{K\ni i}|K|},\qquad\boldsymbol g=(\sigma_{xx},\sigma_{yy},\sigma_{xy},q_x,q_y)')
    r.p('원시 요소 평균은 raw_cell_fields.xdmf와 HDF5에 불연속 자료로 저장한다. 그림과 공통 평가 좌표의 연속 장은 주변 요소의 면적 가중 평균을 절점에 부여한 뒤 Q1으로 보간한다. 따라서 계면의 급격한 변화는 복원 과정에서 완만해질 수 있다. 원시 장을 보존해야 이 효과와 해법 오차를 분리할 수 있다.')
    r.figure('figures/fem_report/recovery.png','원형 FEM100의 원시 요소 평균과 복원 절점 장. 표본 행이 각각 y=0.355와 y=0.350이므로 순수한 복원 효과만을 분리한 오차 곡선은 아니다.',85)
    r.p('아래 전체장 그림은 논문 인쇄 색막대에서 추출한 팔레트와 논문의 FEM 색상 범위를 사용한다. 범위를 벗어난 값은 양끝 색으로 표시하고 색막대 양끝에 확장 표시를 둔다. 이 처리는 시각 비교를 위한 것이며 원시 배열을 자르거나 진폭을 변경하지 않는다. 범위 밖 비율과 실제 최솟값·최댓값은 color_scale_audit.json에 기록한다.')
    r.page(); r.h('7. 원형 개재물의 FEM 기준해')
    r.figure('figures/fem_report/circle_mechanical.png','원형 개재물 FEM200의 변위와 응력. 검은 선은 연속 원형 기하의 위치이며 실제 재료는 요소 중심 판별을 따른다.',150)
    r.figure('figures/fem_report/circle_thermal.png','원형 개재물 FEM200의 온도와 열유속. 성분마다 논문에서 사용한 색상 범위가 다르다.',75)
    r.page(); r.h('8. 사각형 개재물의 FEM 기준해')
    r.figure('figures/fem_report/rectangles_mechanical.png','사각형 개재물 FEM200의 변위와 응력. 모서리 주변의 국소 변화는 최대오차와 복원 방식에 민감하다.',150)
    r.figure('figures/fem_report/rectangles_thermal.png','사각형 개재물 FEM200의 온도와 열유속. 온도 연속성과 열유속 성분의 공간 변화를 구분한다.',75)
    r.page(); r.h('9. 선형 해법 잔차와 열수지 검증')
    r.equation(r'\rho_A=\frac{\|\boldsymbol A\boldsymbol x-\boldsymbol b\|_2}{\max(\|\boldsymbol b\|_2,10^{-30})},\qquad R_T=R_{\mathrm L}+R_{\mathrm R}')
    r.p('열수지는 경계에서 점별 온도 기울기를 미분해 적분한 값이 아니라, Dirichlet 조건으로 수정하기 전의 조립 행렬에 온도 해를 대입하여 얻은 절점 반력을 합산한다. 이 부호 규약에서 왼쪽 반력은 양수, 오른쪽 반력은 음수이다. 외향 열유속 적분은 이 반력의 부호와 반대이다.')
    rows=[]
    for item in refs:
        m=item['metadata']; d=m['diagnostics']; name=item['folder'].split('/')[-1].replace('fem_','')
        rows.append([name,fmt(d['thermal_relative_residual']),fmt(d['mechanical_relative_residual']),fmt(d['heat_balance'])])
    r.table(['기준해','열 상대 잔차','역학 상대 잔차','열 반력 합'],rows,[.37,.21,.21,.21],numeric=True)
    r.p(f'19개 기준해에서 최대 상대 잔차는 {residual:.3e}, 최대 열 반력 합의 절댓값은 {balance:.3e}이다. 모든 저장 평가점에서 Dirichlet 온도와 구속 변위의 최대 오차는 {max(x["dirichlet_max_error"] for x in refs):.3e}이다. 작은 잔차와 경계 오차는 조립된 이산 문제의 해를 확인하며, 재료 계면의 공간 근사 오차를 제거하지는 않는다.')
    r.page(); r.h('10. 균질 해석해를 이용한 독립 검증')
    r.p(r'$E=k=1$인 균질 문제에서는 다음 해석해를 얻는다. 여기서 $c=\beta/[2(\lambda+2\mu)]$이며 $T_0=0$이다.')
    r.equation(r'\begin{aligned}T&=1-x,& u_x&=cx(1-x),&u_y&=0,\\q_x&=1,&q_y&=0,&\sigma_{xy}&=0,\\\sigma_{xx}&=-\tfrac12\beta,&\sigma_{yy}&=\lambda c(1-2x)-\beta(1-x).\end{aligned}')
    r.p('서로 독립적인 세 격자와 두 재료 계수 모드에서 실제 FEniCS 계산을 수행하였다. 온도와 변위의 절점 값은 해석해와 부동소수점 정밀도 수준에서 일치한다. 그러나 수평변위 해석해는 이차 함수이므로 Q1 요소 내부에서는 보간 오차가 남는다. 절점만 검사하고 전체 변위장이 정확하다고 결론내리지 않는다.')
    r.table(['재료 모드','n','절점 u,T 최대오차','내부 포함 ux 최대오차','T 최대오차'],[[x['mode'],x['n'],fmt(x['nodal_uT_max_abs']),fmt(x['full_grid_ux_max_abs']),fmt(x['T_max_abs'])] for x in ucheck],[.18,.09,.25,.28,.20],numeric=True)
    r.equation(r'\|u_x-u_{x,h}\|_\infty=\frac{c\,h^2}{4}\qquad\text{(uniform mesh; exact nodal values)}')
    r.p('표의 내부 평가 격자는 각 요소 중점을 포함한다. 요소 크기를 절반으로 줄이면 수평변위의 최대 보간 오차가 약 4분의 1로 줄어드는 것을 확인한다. 이는 균질 문제에 대한 이차 보간 수렴의 확인이며, 불연속 재료 계면에서 모든 성분에 같은 수렴률을 보장하는 근거는 아니다.')
    r.page(); r.h('11. 이질 재료의 격자 수렴')
    r.equation(r'e_{n\mid200}(f)=\frac{\|\boldsymbol f_n-\boldsymbol f_{200}\|_2}{\|\boldsymbol f_{200}\|_2}')
    r.p('모든 해를 동일한 101×101 점에 평가하고 FEM200을 상대 비교 기준으로 사용한다. 아래 값은 연속체 해석해에 대한 오차 추정량이 아니라 두 이산 해 사이의 차이이다. 특히 응력·열유속은 앞서 정의한 절점 복원을 거친 값이다.')
    for geom in ('circle','rectangles'):
        ref=load(root,f'fem_{geom}_n200')['fields']
        rows=[[str(n)]+[pct(v) for v in err(load(root,f'fem_{geom}_n{n}')['fields'],ref)] for n in (50,100)]
        r.table(['n']+list(FIELDS),rows,[.10]+[.1125]*8,numeric=True,caption=geom+'의 FEM200 대비 상대 차이 (%)')
    r.figure('figures/fem_report/convergence.png','방향별 요소 수에 따른 8성분의 상대 차이. FEM200 자체를 0 오차의 해석적 정답으로 표시하지 않는다.',93)
    r.p('원형에서는 계면 근사와 복원 영향 때문에 횡열유속·전단응력의 상대 차이가 변위·온도보다 크다. 사각형에서도 국소 모서리장과 작은 성분의 정규화가 상대값을 키울 수 있다. 따라서 “수렴했다”는 판단에는 평가 성분과 허용오차를 명시해야 하며, 하나의 격자를 모든 장의 정답으로 선언하지 않는다.')
    r.page(); r.h('12. 재료 표현에 따른 기준해 차이')
    r.p('주 FEM은 요소별 불연속 계수를 사용하지만 PINN은 50×50 재료 격자의 절점 평균을 보간한다. 이 두 문제는 계면 주변에서 서로 다른 계수를 갖는다. 이를 분리해 보기 위해 두 기하의 FEM100에 동일 절점 재료장을 사용하는 별도 기준해를 보존하였다.')
    rows=[]
    for geom in ('circle','rectangles'):
        sharp=load(root,f'fem_{geom}_n100')['fields']; nodal=load(root,f'fem_{geom}_n100_nodal')['fields']
        rows.append([geom]+[pct(v) for v in err(nodal,sharp)])
    r.table(['기하']+list(FIELDS),rows,[.20]+[.10]*8,numeric=True,caption='FEM100 절점 재료장과 불연속 재료장의 상대 차이 (%)')
    r.p('위 차이는 신경망 오차가 아니라 재료 계수 표현 변경에 따른 두 FEM 해의 차이이다. PINN과 불연속 FEM을 비교하면 신경망 근사·최적화·경계 오차에 이 효과도 함께 나타난다. 절점 재료장 FEM과의 비교는 동일 계수 문제의 비교에 더 가깝지만, 양쪽의 공간 미분 및 후처리까지 동일하다는 뜻은 아니다.')
    r.h('논문과 그림이 다른 이유의 구분',2)
    r.table(['원인','현재 확인한 사실','단정할 수 없는 점'],[['구성식','저자 공개 코드 전단 계수와 표준식 사이의 차이','FEAP 변위 진폭 차이의 단일 원인'],['재료 표현','sharp/nodal 기준해 사이의 실제 수치 차이','PINN 총 오차 중 각 원인의 정확한 기여율'],['후처리','요소 평균과 복원 절점 장의 구분','논문의 모든 그림에 적용한 내부 복원 설정'],['그림 범위','논문 팔레트·범위와 범위 밖 비율 기록','색상 일치만으로 수치 정확도 입증']],[.16,.42,.42])
    r.page(); r.h('13. 저자 공개 FEAP 자료와의 외부 비교')
    r.p('저자 자료의 51×51 좌표와 동일한 좌표에서 FEM50을 새로 계산하였다. 비교 파일의 열 순서를 확인하고, 온도와 두 변위 성분을 그대로 사용하였다. 논문 그림을 눈대중으로 맞추는 대신 공개 배열에 대한 차이를 계산하였다.')
    r.table(['비교량','측정값'],[['온도 상대 L2 차이 (%)',pct(authors['temperature_relative_l2'])],['변위 벡터 상대 L2 차이 (%)',pct(authors['displacement_relative_l2_raw'])],['최소제곱 진폭비: 저자/FEniCS',f"{authors['least_squares_author_over_fem_amplitude']:.10f}"],['진폭 보정 적용 여부','적용하지 않음'],['평가점 수',authors['evaluation_points']]],[.66,.34])
    r.equation(r's^*=\frac{\sum_i\boldsymbol u_i^{\mathrm{FEM}}\cdot\boldsymbol u_i^{\mathrm{author}}}{\sum_i\|\boldsymbol u_i^{\mathrm{FEM}}\|^2}')
    r.p('온도는 공개 FEAP 자료와 매우 가깝다. 변위에서는 약 0.98배의 일정한 진폭 관계가 관측된다. 그러나 공개 자료만으로 열팽창계수, 하중, 단위, 후처리 등 어떤 설정이 원인인지 확정할 수 없다. 위 진폭비는 진단 지표이며 FEM/PINN 결과를 보정하는 계수로 사용하지 않는다. 이 차이를 숨긴 채 저자 수치와 완전히 일치한다고 주장하지 않는다.')
    r.page(); r.h('14. 결론 및 재현 절차')
    r.p('본 보고서는 실제 FEniCS로 생성한 19개 기준해와 새로 수행한 균질 해석해 검증, 외부 FEAP 자료 비교를 통해 구현을 점검하였다. 선형 시스템 잔차와 열수지는 수치 정밀도 수준이며, 균질 변위의 요소 내부 보간 오차는 예상되는 이차 수렴을 보였다. 이질 재료의 응력·열유속에는 격자와 복원의 영향이 남으므로 FEM200을 절대적인 정답으로 간주하지 않는다.')
    r.p('본 검증은 지정한 평면변형률·롤러 경계·완전 접합·정상 열전도 문제에 한정된다. 면외 변형, 접촉 열저항, 비선형 재료, 시간 의존 열전도, 열원 및 양방향 연성은 검증 범위에 포함하지 않는다. 신경망 전체 실험의 완료와 성능 평가는 전체 보고서를 따른다.')
    r.h('14.1 독립 실행',2)
    r.p('프로젝트 루트에서 다음 명령을 실행한다. 첫 번째 명령은 하나의 기준해를 새 출력 폴더에 생성하는 예이고, 두 번째는 보존된 기준해를 검사하고 균질·외부 자료 검증을 다시 수행한다. 세 번째는 문서용 Python에서 보고서를 생성한다.')
    r.code('python3 -m harandi.fem --geometry circle --n 100 --output results/fem_circle_recheck\npython3 scripts/verify_fem_report.py\npython scripts/build_fem_report.py')
    r.p('첫 두 명령은 FEniCS가 설치된 Ubuntu에서 실행한다. Windows의 WSL 명령과 XeLaTeX 설치 절차는 docs/LATEX_REPORTS_KO.md를 따른다. 보고서 생성기는 저장된 NPZ의 해시가 검증 기록과 다르면 중단한다. 신경망 학습을 자동으로 시작하지 않는다.')
    r.h('14.2 결과 파일',2)
    r.table(['파일','내용'],[['FEM_REPORT_KO.tex / .md / .pdf','편집 가능한 LaTeX·Markdown과 최종 PDF'],['scripts/verify_fem_report.py','실제 FEM 검증 및 외부 자료 비교'],['scripts/build_fem_report.py','수치 재계산·그림·문서 생성'],['results/fem_report_verification.json','19개 기준해 해시와 추가 검증의 원시 수치'],['figures/fem_report/*','본 보고서의 그림과 색상 범위 감사 자료'],['results/fem_*/raw_cell_fields.xdmf','불연속 요소 평균장; ParaView에서 열기']],[.48,.52])
    r.page(); r.h('참고문헌 및 출처')
    r.p('Harandi, A., Moeineddin, A., Kaliske, M., Reese, S., Rezaei, S. (2024). Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering, 125(4), e7388. <link href="https://doi.org/10.1002/nme.7388">DOI 10.1002/nme.7388</link>. 온라인 선게재 2023.','small')
    r.p('phyml4e/PINNs. <link href="https://github.com/phyml4e/PINNs/blob/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic/Thermo_mechanic.py">Thermo_mechanic.py</link> 및 같은 폴더의 FEAP·재료 자료. 확인일 2026-11-04. 저자 코드 표기와 수치 자료를 참고하였으며 FEniCS 구현은 본 프로젝트에서 작성하였다.','small')
    r.p('국문 편집 형식은 한국전산구조공학회의 <link href="https://www.jcoseik.or.kr/journal-information/instructions-for-authors/">논문 작성 지침</link> 및 docs/EDITORIAL_REFERENCES_KO.md에 기록한 국내 논문의 서술 방식을 참고하였다. 참고한 국문 논문의 계산 결과를 본 FEM 검증 자료로 사용하지 않았다.','small')
    r.h('부록: 색상 범위 밖 비율',2)
    r.table(['기하','성분','표시 최소','표시 최대','하한 미만 (%)','상한 초과 (%)'],[[v['geometry'],v['field'],fmt(v['vmin']),fmt(v['vmax']),fmt(v['below_percent']),fmt(v['above_percent'])] for v in audit],[.19,.11,.17,.17,.18,.18],numeric=True)
    path=r.build(); print(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--root',type=Path,default=ROOT)
    build(ap.parse_args().root.resolve())
