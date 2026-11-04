"""Generate the current Korean full-paper reproduction notebook."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''# 열탄성 혼합 PINN: 논문 전체 실험의 실행과 검증

Harandi et al., *Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains*, [DOI: 10.1002/nme.7388](https://doi.org/10.1002/nme.7388).

혼합 PINN·표준 PINN·DEM, 동시·순차 학습, 경계점 보강, 에너지 제외, 강제 경계조건·최적화기, 구조·시간 연구, 재료비 전이학습을 실제 계산 자료와 연결하는 노트북이다. 원문에서 불명확한 설정은 재구성 선택으로 구분한다.

**기본값 `RUN_FULL_STUDY=False`는 완료를 검증한 결과만 읽는다.** 누락·진행 중 실행은 대기 표에 남긴다. 오차는 예측·FEM NPZ 배열에서 다시 계산하며 보고서의 정적 숫자를 가져오지 않는다. 전체를 실행하려면 `RUN_FULL_STUDY=True`로 바꾸고 **Run All Cells**를 실행한다. 체크포인트부터 이어서 계산하고 완료된 작업은 재사용한다. 실행 시간은 장비에 따라 길어질 수 있다.

최신 재현 노트북은 이 파일이다. [전체 보고서](../reports/FULL_PROJECT_REPORT_KO.pdf)와 [FEM 보고서](../reports/FEM_REPORT_KO.pdf)를 함께 참고한다. 자세한 실행법은 [전체 실행 안내](../docs/README_FULL_STUDY_KO.md)에 있다.''')
md('''## 1. 환경과 실행 설정

프로젝트 전체를 유지하고 이 폴더에서 노트북을 연다. Windows에서는 `start_notebook.ps1`로 JupyterLab을 시작하여 **Harandi CUDA Python 3.11** 커널을 고른다. FEniCS는 WSL Ubuntu의 Python, 신경망은 현재 커널 Python을 사용한다. Linux에서는 `FENICS_PYTHON`으로 DOLFIN 환경을 지정한다.

신경망 작업은 1개 또는 2개씩 실행한다. 시간 측정은 신경망 작업이 모두 끝난 뒤 **항상 단일 작업**으로 실행한다. 다른 GPU 작업과 겹친 시간 측정은 통제된 비용 비교가 아니다.''')
code(r'''from pathlib import Path
from io import BytesIO
import os, sys, json, csv, subprocess, contextlib, html
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML, Image
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'harandi/full_study.py').exists()), None)
if ROOT is None: raise RuntimeError('harandi/가 있는 프로젝트 폴더에서 실행하세요.')
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
RUN_FULL_STUDY = False
RUN_SMOKE = False                   # 작은 실제 FEM + Adam/L-BFGS 검증
NEURAL_WORKERS = 1                  # 1 또는 2; 시간 연구는 별도로 1
BUILD_FIGURE_PACK = False
RESUME_JOB_ID = None                # 예: 'transfer_w0.1'; 단일 작업 재시작
REFERENCE_SUFFIX = 'n200'           # n50, n100, n200, n100_nodal
OUTPUT_KIND = 'constitutive_outputs' # 또는 direct_mixed_outputs
FIELD = 'T'                        # ux,uy,sxx,syy,sxy,T,qx,qy
STUDY = ROOT/'results/full_study'
WORK = ROOT/'.notebook_work/full_study'
WORK.mkdir(parents=True, exist_ok=True)
assert NEURAL_WORKERS in (1,2)
assert OUTPUT_KIND in ('direct_mixed_outputs','constitutive_outputs')
assert REFERENCE_SUFFIX in ('n50','n100','n200','n100_nodal')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.facecolor':'white'})
COLORS = {'mixed':'#2563a6','standard':'#b8860b','dem':'#a34672',
          '0':'#60666d','0.1':'#2563a6','0.5':'#b8860b','1':'#a34672'}
def show_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    display(Image(data=buffer.getvalue(), format='png'))
    plt.close(fig)
def table(rows, columns=None):
    rows = list(rows)
    if not rows:
        display(Markdown('이 조건으로 검증된 데이터가 없습니다. 대기 상태를 확인하세요.'))
        return
    columns = list(rows[0]) if columns is None else columns
    def value(v):
        if isinstance(v,(float,np.floating)): return f'{v:.5g}' if np.isfinite(v) else '—'
        return html.escape(str(v))
    body = '<tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in columns)+'</tr>'
    body += ''.join('<tr>'+''.join(f'<td>{value(r.get(k,"—"))}</td>' for k in columns)+'</tr>' for r in rows)
    display(HTML('<div style="overflow-x:auto"><table style="font-size:13px;text-align:left">'+body+'</table></div>'))
def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).exists() else default
    except json.JSONDecodeError:
        # A live worker may be replacing a status file. It is not complete evidence.
        return default
from harandi.full_study import catalogue, job_complete, fem_references
from harandi.physics import Problem, FIELDS, grid, material
JOBS = catalogue()
JOB_MAP = {j['id']:j for j in JOBS}
print('Project:',ROOT,'\nPython:',sys.executable)
print('Torch:',torch.__version__,'| GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Accuracy jobs:',sum(j['kind']!='timing' for j in JOBS),'| isolated timing jobs:',sum(j['kind']=='timing' for j in JOBS))''')
md(r'''## 2. 지배방정식·재현 가정·출처 충돌

$$\nabla\cdot\sigma=0,\qquad\nabla\cdot q=0,\qquad q=-k\nabla T,$$
$$\sigma=2\mu\varepsilon+\lambda\operatorname{tr}(\varepsilon)I-\frac{E\alpha(T-T_0)}{1-2\nu}I.$$

평면변형률, $\nu=0.3,\alpha=1,T_0=0$이다. 좌우 $u_x=0$, 상하 $u_y=0$, 전 경계 접선력 0, 좌측 $T=1$, 우측 $T=0$, 상하 $q_y=0$이다. 열전도에는 변위의 역영향이 없는 일방향 연성이다. 직접 혼합 출력과 미분으로 구한 응력·열유속을 분리해서 평가한다.

- 원문의 μm/mm 및 일부 좌표는 불명확하다. 수치 단위 정사각형과 명시된 재료값을 사용하며 자동 SI 환산을 가정하지 않는다. 원형 중심은 도해 추정, 사각형은 저자 재료 자료를 따른다.
- 구성식·전단·Fourier 부호·롤러 경계조건의 수정과 `energy='authors'` 해석을 공개한다. 원문의 모든 인쇄식을 오류까지 그대로 구현한 것은 아니다.
- Table 1은 동시 100,000회, 순차 각 장 100,000회/총 200,000회, 20,000회 간격 전환이다. Remark 7의 100만 회 문구와 충돌하므로 목록의 해당 추가 실행을 별도로 표시한다. 일반 주 실행과 합쳐서 같은 실험이라고 부르지 않는다.
- 깊이 1,2,4,5,6과 너비 5,10,20,40,60의 정확도 연구는 L-BFGS 10,000회이며 겹치는 5×40은 재사용한다. Table 4의 25쌍은 각 100회 **시간 측정**이다.
- 그림 22의 재료비는 matrix/inclusion이다. 전역 log10 비율 입력, 데이터 정규화, 경계조건, Adam 상태 전이 방식은 실제 설정과 함께 보고한다. 가중치 0,0.1,0.5,1 중 **0.5는 원문에 명시되지 않은 보완 선택**이다.
- 순차 학습·큰 망·물리 항이 항상 우세하다고 가정하지 않는다. 원문에도 성분별 예외와 비단조성이 있으므로 평균·최대·8성분 오차를 함께 비교한다.

**원형의 학습점 분포는 재구성 선택이다.** 51×51 직교 격자와 시드가 고정된 내부 균등 난수 2,399점을 합쳐 5,000점을 사용한다. 원문 그림 5의 계면·경계 보강 좌표는 공개되지 않아 그 분포를 그대로 재현하지 않았다. 같은 점 수가 같은 배치나 정확도를 뜻하지 않으며, 동일 점 가중치도 면적에 정확한 적분 공식과 구분한다.''')
code(r'''fig,axes=plt.subplots(1,2,figsize=(8,3.4),constrained_layout=True)
for ax,geometry in zip(axes,('circle','rectangles')):
    xy=grid(101); E,_=material(xy,Problem(geometry),'sharp')
    im=ax.imshow(E.reshape(101,101),origin='lower',extent=(0,1,0,1),cmap='viridis')
    ax.set(title=geometry,xlabel='x / L',ylabel='y / L')
    fig.colorbar(im,ax=ax,label='E [benchmark units]')
show_figure(fig)
inventory = read_json(ROOT/'docs/PAPER_EXPERIMENT_INVENTORY.json')
if inventory:
    table([{'기록':s.get('id',''),'내용':s.get('title',s.get('name','')),
            '출처':'; '.join(source.get('locator','') for source in s.get('sources',[]))} for s in inventory.get('studies',[])])
    table([{'ID':s['id'],'주제':s['topic'],'재현 처리':s['reproduction_implication']}
           for s in inventory.get('source_issues',[])])
else:
    display(Markdown('**실험 감사 목록 대기:** docs/PAPER_EXPERIMENT_INVENTORY.json이 아직 없습니다. 실제 실행 목록은 아래 catalogue를 사용합니다.'))
print('Protocol:', 'available' if (ROOT/'docs/FULL_STUDY_PROTOCOL.json').exists() else 'not yet supplied')''')
md('''## 3. 완료·대기 상태와 실제 예산

`job_complete`는 설정·실제 업데이트·최종 상태·예측 파일을 확인한다. 체크포인트가 있어도 완료 기준을 충족하지 않으면 진행/재시작 가능으로 표시한다. 이전의 짧은 학습을 전체 실험 완료로 세지 않는다.''')
code(r'''def requested_updates(job):
    c=job['config']
    if job['kind']=='transfer': return len(c['ratios'])*c['steps']
    factor=1 if c['mode']=='coupled' else 2
    return factor*(c['steps'] if job['kind']=='baseline' else c['adam_steps']+c['lbfgs_steps'])
def status_rows():
    rows=[]
    for job in JOBS:
        folder=STUDY/job['id']; done=job_complete(job,STUDY)
        status=read_json(folder/'run_status.json',{})
        info=read_json(folder/'experiment.json',{})
        progress=read_json(folder/'progress.json',{})
        count=status.get('completed')
        actual=sum(count.values()) if isinstance(count,dict) else info.get('completed_updates',info.get('actual_optimizer_updates',progress.get('actual_optimizer_updates')))
        rows.append({'실행':job['id'],'분류':job['kind'],'상태':'검증 완료' if done else '진행/재시작 가능' if (folder/'checkpoint.pt').exists() else '대기',
                     '실제 업데이트':actual,'요구 업데이트':requested_updates(job),'완료':done})
    return rows
STATUS=status_rows()
completed_ids={r['실행'] for r in STATUS if r['완료']}
table([{'분류':k,'완료':sum(r['완료'] for r in STATUS if r['분류']==k),'전체':sum(r['분류']==k for r in STATUS)} for k in ('mixed','baseline','transfer','timing')])
table(STATUS,['실행','분류','상태','실제 업데이트','요구 업데이트'])
queue=read_json(STUDY/'queue_status.json',{})
print('Queue:',queue.get('status'),'| running:',queue.get('running',[]),'| failed:',queue.get('failed',{}))''')
md('''## 4. 전체 실행·단일 작업 재시작

기본값에서는 장기 작업을 시작하지 않는다. 전체 실행을 켜면 누락된 FEM을 생성하고, 신경망 전체가 끝난 다음 단일 작업으로 시간을 측정한다. 재시작은 같은 설정의 `checkpoint.pt`에서 이어진다. 한 출력 폴더를 두 프로세스가 동시에 쓰지 않는다. 조건을 바꾼 추가 연구는 새 출력 폴더를 사용한다.''')
code(r'''def fem_command(arguments):
    command=(['wsl.exe','-d','Ubuntu','--cd',str(ROOT),'--','env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','python3'] if os.name=='nt'
             else [os.environ.get('FENICS_PYTHON','/usr/bin/python3')])+arguments
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    if result.returncode: raise RuntimeError(result.stdout[-4000:]+result.stderr[-4000:])
    return result.stdout
if RUN_FULL_STUDY:
    if not torch.cuda.is_available() and any(j['config'].get('fast_adam') for j in JOBS):
        raise RuntimeError('전체 목록은 CUDA 그래프 항목을 포함합니다. CUDA 커널 또는 별도 CPU 구성을 사용하세요.')
    for geometry in ('circle','rectangles'):
        for n,suffix,coefficient in [(50,'n50','sharp'),(100,'n100','sharp'),(200,'n200','sharp'),(100,'n100_nodal','nodal')]:
            folder=ROOT/'results'/f'fem_{geometry}_{suffix}'
            if not (folder/'solution.npz').exists():
                fem_command(['-m','harandi.fem','--geometry',geometry,'--n',str(n),'--coefficient-mode',coefficient,'--output',folder.relative_to(ROOT).as_posix()])
    fem_references(STUDY)
    subprocess.run([sys.executable,'-m','harandi.full_study','--out',str(STUDY),'--groups','mixed','baseline','transfer','--workers',str(NEURAL_WORKERS)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','harandi.full_study','--out',str(STUDY),'--groups','timing','--workers','1'],cwd=ROOT,check=True)
elif RESUME_JOB_ID:
    if RESUME_JOB_ID not in JOB_MAP: raise ValueError('목록에 없는 실행 ID입니다.')
    if not job_complete(JOB_MAP[RESUME_JOB_ID],STUDY):
        subprocess.run([sys.executable,'-m','harandi.full_study','--job',RESUME_JOB_ID,'--out',str(STUDY)],cwd=ROOT,check=True)
    else: print('Already verified complete:',RESUME_JOB_ID)
else: print('Preview: no full training or timing job was started.')
STATUS=status_rows()
completed_ids={r['실행'] for r in STATUS if r['완료']}
print('Verified complete:',len(completed_ids),'/',len(JOBS))''')
md(r'''## 5. 독립적인 소규모 FEM·학습 검증

`RUN_SMOKE=True`는 별도 `.notebook_work/full_study/`에서 실제 FEM과 짧은 Adam/L-BFGS 학습을 실행한다. $T=1-x$, $u_x=\frac{13}{14}x(1-x)$, 열수지·선형 잔차·하드 경계를 검사한다. 기본값은 이전에 저장된 검증의 출처를 표시한다. 작은 시험 통과는 장기 실험의 완료나 정확도를 증명하지 않는다.''')
code(r'''if RUN_SMOKE:
    from harandi.train import TrainConfig,run
    fem_command(['-m','harandi.fem','--geometry','homogeneous','--n','12','--output',(WORK/'fem_exact').relative_to(ROOT).as_posix()])
    with (WORK/'smoke.log').open('w') as log,contextlib.redirect_stdout(log):
        run(TrainConfig(geometry='homogeneous',width=6,layers=1,n=7,adam_steps=4,lbfgs_steps=2,device='cpu',log_every=2),WORK/'pinn_smoke')
checks=[]
fem_smoke=next((p for p in (WORK/'fem_exact',ROOT/'.notebook_work/fem_exact') if (p/'solution.npz').exists()),None)
pinn_smoke=next((p for p in (WORK/'pinn_smoke',ROOT/'.notebook_work/pinn_smoke') if (p/'model.pt').exists()),None)
if fem_smoke:
    with np.load(fem_smoke/'solution.npz') as d:
        x=d['xy'][:,0]; exact=13/14*x*(1-x)
        u_error=float(np.linalg.norm(d['fields'][:,0]-exact)/np.linalg.norm(exact))
        t_error=float(abs(d['fields'][:,5]-(1-x)).max())
    diagnostics=read_json(fem_smoke/'metadata.json')['diagnostics']
    assert u_error<.01 and t_error<1e-10
    assert max(diagnostics['thermal_relative_residual'],diagnostics['mechanical_relative_residual'])<1e-8
    assert abs(diagnostics['heat_balance'])<1e-8
    checks.append({'검증':'FEM analytic/residual/heat balance','상태':'재검산 통과','ux 상대 L2':u_error,'출처':str(fem_smoke.relative_to(ROOT))})
else: checks.append({'검증':'FEM smoke','상태':'대기: RUN_SMOKE=True'})
if pinn_smoke:
    ck=torch.load(pinn_smoke/'model.pt',map_location='cpu',weights_only=False)
    d=read_json(pinn_smoke/'metrics.json')['diagnostics_independent_grid']
    assert d['max_temperature_boundary_error']==d['max_normal_displacement_error']==0
    checks.append({'검증':'Actual Adam/L-BFGS + hard boundaries','상태':str(ck['completed']),'출처':str(pinn_smoke.relative_to(ROOT))})
else: checks.append({'검증':'PINN smoke','상태':'대기: RUN_SMOKE=True'})
table(checks,['검증','상태','ux 상대 L2','출처'])''')
md(r'''## 6. 8성분 오차의 독립 재계산

기본 정량 기준은 200×200 FEM이며 `REFERENCE_SUFFIX`로 50·100·동일 절점 재료장 기준을 선택할 수 있다. 전이학습은 비율 15의 별도 FEM만 쓴다. 기준해의 요소 수·재료장 모드도 표시한다.

$$e_{L2}=\frac{\|\hat f-f\|_2}{\|f\|_2},\quad e_{avg}=\frac{\operatorname{mean}|\hat f-f|}{\max|f|},\quad e_{max}=\frac{\max|\hat f-f|}{\max|f|}.$$

세 지표는 서로 다르다. 작은 전단·횡열유속의 상대오차는 분모가 작아 커질 수 있어 절대오차도 남긴다. 아래는 직접 출력과 미분 출력의 오차를 **모두** NPZ에서 재계산한다.''')
code(r'''def errors_from_arrays(pred,ref):
    if pred.shape!=ref.shape or pred.shape[1]!=8: raise ValueError('8성분 배열의 크기가 다릅니다.')
    if not (np.isfinite(pred).all() and np.isfinite(ref).all()): raise ValueError('비유한 값이 있습니다.')
    result={}
    for j,field in enumerate(FIELDS):
        error=abs(pred[:,j]-ref[:,j]); peak=max(float(abs(ref[:,j]).max()),1e-15)
        result[field]={'relative_L2':float(np.linalg.norm(error)/max(np.linalg.norm(ref[:,j]),1e-15)),
            'MAE':float(error.mean()),'max_absolute_error':float(error.max()),
            'mean_error_over_peak':float(error.mean()/peak),'max_error_over_peak':float(error.max()/peak)}
    return result

def load_evaluation(job):
    from scipy.spatial import cKDTree
    geometry='rectangles' if job['kind']=='transfer' else job['config']['geometry']
    ratio=15 if job['kind']=='transfer' else None
    reference=STUDY/'parametric_fem/ratio_15/solution.npz' if ratio else ROOT/'results'/f'fem_{geometry}_{REFERENCE_SUFFIX}/solution.npz'
    with np.load(STUDY/job['id']/'prediction.npz') as p,np.load(reference) as r:
        xy=p['xy'].copy(); direct=p['fields'].copy(); derived=direct.copy()
        derived[:,2:5]=p['constitutive_stress']; derived[:,6:8]=p['fourier_flux']
        distance,index=cKDTree(r['xy']).query(xy)
        if distance.max()>1e-9: raise ValueError('평가 좌표가 일치하지 않습니다.')
        fields=r['fields'][index].copy()
    metadata=read_json(reference.parent/'metadata.json',{})
    if metadata.get('problem',{}).get('geometry')!=geometry or metadata.get('problem',{}).get('ratio')!=ratio:
        raise ValueError('FEM 메타데이터의 기하/재료비가 다릅니다.')
    return {'xy':xy,'direct_mixed_outputs':direct,'constitutive_outputs':derived,'reference':fields,
            'reference_path':reference,'metadata':metadata,
            'errors':{kind:errors_from_arrays(v,fields) for kind,v in [('direct_mixed_outputs',direct),('constitutive_outputs',derived)]}}
EVALUATIONS={}; evaluation_pending=[]
for job in JOBS:
    if job['kind']=='timing' or job['id'] not in completed_ids: continue
    try: EVALUATIONS[job['id']]=load_evaluation(job)
    except (FileNotFoundError,ValueError,KeyError) as e:
        evaluation_pending.append({'실행':job['id'],'평가':'대기/불일치','이유':str(e)})
table([{'실행':name,'점수':len(v['xy']),'FEM 요소/변':v['metadata'].get('n'),
        '재료장':v['metadata'].get('coefficient_mode'),'기준':str(v['reference_path'].relative_to(ROOT))} for name,v in EVALUATIONS.items()])
if evaluation_pending: table(evaluation_pending)
(WORK/'recomputed_errors.json').write_text(json.dumps({k:v['errors'] for k,v in EVALUATIONS.items()},indent=2),encoding='utf-8')
print('Recomputed accuracy runs:',len(EVALUATIONS),'| displayed:',OUTPUT_KIND)''')
md('''### 공통 표와 단면 함수

주요 단면은 네 성분을 보지만 오차표는 8개 모두 유지한다. `OUTPUT_KIND`를 바꾸면 직접/미분 출력을 선택할 수 있다. 단면은 실제 저장 좌표에서 선택한다.''')
code(r'''def metric_table(ids,metric='relative_L2'):
    rows=[]
    for name in ids:
        if name not in EVALUATIONS: continue
        errors=EVALUATIONS[name]['errors'][OUTPUT_KIND]
        rows.append({'실행':name,**{f:(100 if metric!='MAE' else 1)*errors[f][metric] for f in FIELDS}})
    table(rows)
    print('Metric:',metric,'| percent' if metric!='MAE' else '| benchmark field units')
def section(item,kind,x):
    mask=np.isclose(item['xy'][:,0],x,atol=1e-10); order=np.argsort(item['xy'][mask,1])
    if not mask.any(): raise ValueError(f'x={x} 단면이 없습니다.')
    return item['xy'][mask,1][order],item[kind][mask][order]
def plot_sections(ids,x=.5,title='Cross-section'):
    available=[name for name in ids if name in EVALUATIONS]
    if not available:
        print('Pending figure:',title); return
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,field in zip(axes.flat,('ux','sxx','T','qx')):
        j=FIELDS.index(field); base=EVALUATIONS[available[0]]
        y,ref=section(base,'reference',x)
        ax.plot(y,ref[:,j],color='#25282b',lw=2,label='FEM')
        for name in available:
            y,pred=section(EVALUATIONS[name],OUTPUT_KIND,x)
            if name.startswith('transfer_'):
                weight=f"{JOB_MAP[name]['config']['weight']:g}"
                color=COLORS[weight]; style=['-','--','-.',':'][['0','0.1','0.5','1'].index(weight)]; label='w='+weight
            else:
                method='standard' if 'standard' in name else 'dem' if 'dem' in name else 'mixed'
                color=COLORS[method]; style='--' if name.endswith('sequential') else '-'
                label=name.replace('baseline_circle_','').replace('mixed_circle_','mixed_')
            ax.plot(y,pred[:,j],color=color,ls=style,lw=1.4,label=label)
        ax.set(xlabel='y / L',ylabel=field+' [benchmark units]',title=f'{field}, x / L = {x:g}')
        ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8); fig.suptitle(title); show_figure(fig)''')
md('''## 7. 혼합·표준 PINN·DEM, 동시·순차 학습

원형의 세 방법 × 두 학습 방식과 사각형 혼합 PINN 두 방식을 비교한다. 표준 PINN의 2차 미분·절점 재료장 정규화, DEM의 변분 에너지·자연 경계 처리도 결과에 영향을 준다. 선택은 실행별 `experiment.json`에 공개한다. 큰 오차나 실패를 숨기지 않는다.

**동시 DEM의 역학 에너지는 T를 처방값으로 취급하여 온도망으로 가는 기울기를 차단한다.** 동시 표준·혼합 PINN에서는 역학 손실의 기울기가 온도망으로 전달된다. DEM의 이 선택은 단방향 열탄성 물리를 위한 변분 재구성이며, 원문 식 14의 u,T 공동 최소화를 문자 그대로 구현한 것은 아니다. 방법 비교에는 이 최적화 경로 차이가 포함되므로 원문의 방법 이름만으로 완전히 동일한 비교라고 단정하지 않는다. 선택은 `docs/FULL_STUDY_PROTOCOL.json`에도 기록된다.''')
code(r'''circle_ids=['mixed_circle_coupled','mixed_circle_sequential','baseline_circle_standard_coupled',
            'baseline_circle_standard_sequential','baseline_circle_dem_coupled','baseline_circle_dem_sequential']
rectangle_ids=['mixed_rectangles_coupled','mixed_rectangles_sequential']
metric_table(circle_ids); plot_sections(circle_ids,.5,'Geometry 1: methods and training modes')
metric_table(rectangle_ids); plot_sections(rectangle_ids,.5,'Geometry 2: coupled and sequential')
additional_baselines=[j['id'] for j in JOBS if j['kind']=='baseline' and j['id'] not in circle_ids]
if additional_baselines:
    display(Markdown('**추가 공정성/민감도 연구:** 아래 실행은 원문 기본 비교와 분리한 추가 조건이다. '
                     '유한한 소프트 경계 페널티가 DEM 최소화 해를 바꿀 수 있어 실제 경계 선택을 함께 확인한다.'))
    table([{'실행':name,'hard 경계':JOB_MAP[name]['config'].get('hard'),
            '목적':JOB_MAP[name].get('purpose','추가 연구')} for name in additional_baselines])
    metric_table(additional_baselines)
print('Comparison pending:',[name for name in circle_ids+rectangle_ids if name not in EVALUATIONS])''')
md('''### 편집 가능한 전체장과 직접·미분 출력 비교

`FIELD`와 `selected_run`을 바꾼다. 예측·FEM에는 같은 색 범위, 절대오차에는 0에서 시작하는 별도 범위를 쓴다. 원문 고정 색 범위와 범위 초과 감사는 뒤의 그림집에서 제공한다.''')
code(r'''selected_run='mixed_circle_coupled'
if selected_run in EVALUATIONS:
    item=EVALUATIONS[selected_run]; j=FIELDS.index(FIELD)
    pred,ref=item[OUTPUT_KIND][:,j],item['reference'][:,j]
    limits=(min(pred.min(),ref.min()),max(pred.max(),ref.max()))
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
    for ax,values,label in zip(axes,(pred,ref,abs(pred-ref)),('Prediction','FEM','Absolute error')):
        lo,hi=(0,max(float(values.max()),1e-15)) if label=='Absolute error' else limits
        im=ax.tripcolor(item['xy'][:,0],item['xy'][:,1],values,shading='gouraud',cmap='viridis',vmin=lo,vmax=hi)
        ax.set(xlabel='x / L',ylabel='y / L',title=f'{FIELD}: {label}',aspect='equal'); fig.colorbar(im,ax=ax,shrink=.85)
    show_figure(fig)
    table([{'성분':f,'직접 상대 L2 (%)':100*item['errors']['direct_mixed_outputs'][f]['relative_L2'],
            '미분 상대 L2 (%)':100*item['errors']['constitutive_outputs'][f]['relative_L2']} for f in FIELDS])
else: print('Pending full-field example:',selected_run)''')
md('''## 8. 경계점 보강·에너지 제외·하드 경계

예측 배열의 실제 경계점에서 경계 오차를 다시 계산한다. 하드 Dirichlet의 0 오차를 로그 그림에서 작은 양수로 바꾸지 않는다. 약 1,000개 왼쪽 보강점의 효과는 전체 오차와 경계 만족도를 함께 읽는다. 에너지 항 제외 결과도 그대로 남긴다.''')
code(r'''boundary_ids=['mixed_rectangles_coupled','mixed_rectangles_sequential','refine_rectangles_left1000','hard_rectangles_adam10k','arch_l5_w40']
boundaries=[]
for name in boundary_ids:
    if name not in EVALUATIONS: continue
    v=EVALUATIONS[name]; x,y=v['xy'].T; f=v['direct_mixed_outputs']
    boundaries.append({'실행':name,'max |ux| 좌우':float(abs(f[(x==0)|(x==1),0]).max()),
        'max |uy| 상하':float(abs(f[(y==0)|(y==1),1]).max()),
        'max |T-1| 왼쪽':float(abs(f[x==0,5]-1).max()),'max |T| 오른쪽':float(abs(f[x==1,5]).max())})
table(boundaries)
metric_table(['mixed_circle_coupled','ablation_circle_noenergy'],'mean_error_over_peak')
if boundaries:
    fig,ax=plt.subplots(figsize=(9,3.4),constrained_layout=True)
    values=[r['max |T-1| 왼쪽'] for r in boundaries]
    bars=ax.barh([r['실행'].replace('rectangles_','') for r in boundaries],values,color='#2563a6')
    ax.bar_label(bars,labels=[f'{v:.3g}' for v in values],padding=4)
    ax.set(xlabel='Maximum left temperature boundary error [benchmark K]',title='Boundary checks from prediction arrays')
    ax.set_xlim(0,max(max(values)*1.2,1e-12)); show_figure(fig)''')
md('''## 9. 하드 경계 Adam·L-BFGS와 실제 학습 이력

두 하드 실행은 각 10,000회이다. 소프트 Adam 100,000회와는 예산도 달라 경계조건 하나만 바꾼 효과로 해석할 수 없다. L-BFGS 한 업데이트는 여러 closure 평가를 포함할 수 있다. 순차 학습의 비활성 장 미관측 손실은 비워 두며, 방법마다 다른 총 손실 크기를 정확도 순위로 사용하지 않는다.''')
code(r'''hard_ids=['hard_rectangles_adam10k','arch_l5_w40']
metric_table(hard_ids)
fig,ax=plt.subplots(figsize=(8,3.8),constrained_layout=True); plotted=0
for name in hard_ids:
    path=STUDY/name/'history.csv'
    if name not in completed_ids or not path.exists(): continue
    with path.open(newline='') as stream: rows=list(csv.DictReader(stream))
    steps=np.array([int(r['step']) for r in rows]); losses=np.array([float(r['total']) for r in rows])
    mask=np.isfinite(losses)&(losses>0)
    ax.semilogy(steps[mask],losses[mask],label=name,lw=1.6); plotted+=1
if plotted:
    ax.set(xlabel='Actual optimizer updates',ylabel='Recorded objective',title='Hard constraints: Adam and L-BFGS')
    ax.legend(fontsize=9); ax.grid(alpha=.18); show_figure(fig)
else: plt.close(fig); print('Pending hard-optimizer histories.')
remark_ids=[j['id'] for j in JOBS if 'remark' in j['id']]
if remark_ids:
    display(Markdown('**Remark 7의 별도 100만 회 주장:** 아래 실행은 Table 1의 주 실행과 분리해 확인한다. 학습을 이어 확장했다면 실제 체크포인트·설정이 그 계보를 기록한다.'))
    metric_table(remark_ids)''')
md('''## 10. 깊이·너비 정확도와 25개 독립 시간 측정

평균/최대 오차를 함께 보고 성분별 예외와 비단조성을 남긴다. 시간은 별도 실행의 경과시간/실제 업데이트 수이며 기록 오버헤드를 포함할 수 있다. 로컬 측정과 원문 Apple M2 Pro 결과의 장비·소프트웨어 조건은 다르다.''')
code(r'''def architecture_plot(axis_name):
    specs=[(l,40) for l in (1,2,4,5,6)] if axis_name=='depth' else [(5,w) for w in (5,10,20,40,60)]
    available=[(l,w,f'arch_l{l}_w{w}') for l,w in specs if f'arch_l{l}_w{w}' in EVALUATIONS]
    if not available: print('Pending architecture:',axis_name); return
    fig,axes=plt.subplots(2,2,figsize=(9,6.4),constrained_layout=True)
    for ax,f in zip(axes.flat,('ux','sxx','T','qx')):
        x=[l if axis_name=='depth' else w for l,w,_ in available]
        for metric,label,color,marker in [('mean_error_over_peak','Mean / peak','#2563a6','o'),('max_error_over_peak','Maximum / peak','#b8860b','s')]:
            y=[100*EVALUATIONS[name]['errors'][OUTPUT_KIND][f][metric] for _,_,name in available]
            ax.plot(x,y,color=color,marker=marker,label=label)
        ax.set(title=f,xlabel='Hidden layers' if axis_name=='depth' else 'Neurons per layer',ylabel='Error (%)')
        ax.set_xticks([l if axis_name=='depth' else w for l,w in specs]); ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8); show_figure(fig)
architecture_plot('depth'); architecture_plot('width')
timing=[]
for layers in (1,2,4,5,6):
    row={'층':layers}
    for width in (5,10,20,40,60):
        name=f'timing_l{layers}_w{width}'
        if name not in completed_ids: row[f'{width} 뉴런 s/update']='대기'; continue
        ck=torch.load(STUDY/name/'model.pt',map_location='cpu',weights_only=False)
        row[f'{width} 뉴런 s/update']=ck['elapsed_seconds']/sum(ck['completed'].values())
    timing.append(row)
table(timing)
print('Timing requires verified completion; missing cells are not zero.')''')
md(r'''## 11. 비율 1…10 전이학습과 미학습 비율 15

각 비율 10,000회 학습 후 최종 가중치로 15를 평가한다. 15는 조기 종료·최적 체크포인트·가중치 선택에 쓰지 않는다. 네 고정 가중치의 테스트 결과를 공개하는 것은 비교 평가이며 독립 검증을 통한 최적 가중치 확정이 아니다.

단면은 그림 25의 패널 제목·본문을 따라 **x=0.35**이다. 캡션의 x=0.5와 충돌한다. 원문에도 $q_x$ 최대오차가 약간 커지는 예외가 있어 평균·최대 오차를 모두 공개한다. 아래 실제 설정 표에서 hard/soft 및 정규화 선택을 확인한다.''')
code(r'''transfer_ids=[f'transfer_w{w:g}' for w in (0,.1,.5,1)]
audit=[]
for name in transfer_ids:
    config=JOB_MAP[name]['config']; info=read_json(STUDY/name/'experiment.json',{})
    ratios=info.get('training_ratios',config['ratios']); assert 15 not in ratios
    audit.append({'실행':name,'hard 경계':config.get('hard'),'정규화':config.get('normalization'),
        '학습 비율':str(ratios),'실제 업데이트':info.get('actual_optimizer_updates'),
        '요구 업데이트':requested_updates(JOB_MAP[name]),'상태':info.get('status','대기'),
        '학습초':info.get('training_elapsed_seconds')})
table(audit)
metric_table(transfer_ids,'mean_error_over_peak'); metric_table(transfer_ids,'max_error_over_peak')
plot_sections(transfer_ids,.35,'Unseen ratio 15: four fixed physics weights')
if all(n in EVALUATIONS for n in ('transfer_w0','transfer_w0.1')):
    differences=[]
    for f in FIELDS:
        a=EVALUATIONS['transfer_w0']['errors'][OUTPUT_KIND][f]; b=EVALUATIONS['transfer_w0.1']['errors'][OUTPUT_KIND][f]
        differences.append({'성분':f,'평균 변화 pp':100*(b['mean_error_over_peak']-a['mean_error_over_peak']),
                            '최대 변화 pp':100*(b['max_error_over_peak']-a['max_error_over_peak'])})
    table(differences)
    display(Markdown('변화량은 **w=0.1 − w=0**이다. 음수는 오차 감소, 양수는 증가이다. 성분별 결과를 일반적인 우세로 확대하지 않는다.'))''')
md('''## 12. FEM 50·100·200 격자 수렴

200×200은 비교 기준이며 해석적 정답은 아니다. 원문의 “100에서 거의 수렴”에는 명시된 허용오차가 없으므로 아래 성분별 상대 L2 변화량을 보고 판단한다.''')
code(r'''convergence=[]
for geometry in ('circle','rectangles'):
    fine_path=ROOT/'results'/f'fem_{geometry}_n200/solution.npz'
    if not fine_path.exists(): continue
    with np.load(fine_path) as d: xy,fine=d['xy'].copy(),d['fields'].copy()
    for n in (50,100):
        path=ROOT/'results'/f'fem_{geometry}_n{n}/solution.npz'
        if not path.exists(): continue
        with np.load(path) as d:
            assert np.allclose(xy,d['xy'],atol=1e-12,rtol=0)
            errors=errors_from_arrays(d['fields'],fine)
        convergence.append({'기하 / 요소':f'{geometry} / {n}',**{f:100*errors[f]['relative_L2'] for f in FIELDS}})
table(convergence)
print('Relative L2 (%) against 200x200, recomputed from FEM arrays.')''')
md('''## 13. 전체 그림집과 최종 완료 점검

그림집은 검증 완료 실행만 사용한다. 미완료 작업이 남으면 `--partial`로 생성하고 누락 작업·기준해를 명시한다. 원문 고정 색 범위를 넘는 값은 감사표에 남긴다. 편집 가능한 위 그림·오차 계산은 그림집의 유무와 독립적이다.''')
code(r'''if (RUN_FULL_STUDY or BUILD_FIGURE_PACK) and (ROOT/'harandi/full_figures.py').exists():
    command=[sys.executable,'-m','harandi.full_figures','--root',str(ROOT)]
    if len(completed_ids)!=len(JOBS): command.append('--partial')
    subprocess.run(command,cwd=ROOT,check=True)
report=read_json(ROOT/'validation/FULL_STUDY_RESULTS.json')
if report:
    print('Figure status:',report.get('status'),'| pending jobs:',report.get('pending_jobs',[]))
    print('Pending references:',report.get('pending_references',[]))
    table([{'ID':f.get('id'),'내용':f.get('title'),'원문':f.get('source_figure'),'PNG':f.get('png')} for f in report.get('figures',[])])
    display(Markdown('[그림 PDF](../reports/FULL_STUDY_FIGURES.pdf) · [HTML 갤러리](../reports/FULL_STUDY_GALLERY_KO.html)'))
else:
    display(Markdown('**그림집 대기.** BUILD_FIGURE_PACK=True로 완료된 결과를 묶을 수 있다. 위 계산은 NPZ를 직접 읽는다.'))
STATUS=status_rows(); completed=sum(r['완료'] for r in STATUS)
if completed==len(JOBS):
    display(Markdown(f'현재 목록의 **{completed}/{len(JOBS)}개 작업**이 완료 기준을 충족한다. 기준해가 확인된 실행만 정확도 평가에 포함하며 재구성 설정의 한계는 남는다.'))
else:
    display(Markdown(f'현재 **{completed}/{len(JOBS)}개 작업만 검증 완료**이다. 아래 작업이 남아 있어 전체 연구 완료로 해석하지 않는다.'))
    table([r for r in STATUS if not r['완료']],['실행','상태','실제 업데이트','요구 업데이트'])''')
md('''## 14. 재실행과 추가 연구

- **상태·오차 갱신:** 기본 설정으로 Run All Cells. 전체 학습 없이 NPZ로 재계산한다.
- **전체 이어서:** RUN_FULL_STUDY=True. 완료 작업 재사용, 같은 설정의 체크포인트 재시작.
- **단일 작업:** RUN_FULL_STUDY=False, RESUME_JOB_ID 지정. 실행별 console.log에서 실패 원인을 확인한다.
- **변경 실험:** 별도 출력 폴더와 설정을 사용한다. 원문 재현 결과와 추가 연구를 섞지 않는다.
- **셀 원본:** scripts/build_full_notebook.py. 다시 생성하면 저장된 실행 출력이 초기화된다. 일상적인 갱신은 현재 노트북 실행으로 수행한다.
- **구현 시험:** python -m pytest -q. 단위시험 통과는 장기 실험 완료 또는 논문 수치 일치의 증거를 대신하지 않는다.

수치가 원문과 다르면 해당 차이와 구현 선택을 남긴다. 원문의 결론이나 인쇄 범위에 맞추기 위해 예측 배열을 보정하지 않는다.''')

notebook=nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({'kernelspec':{'display_name':'Harandi CUDA Python 3.11','language':'python','name':'harandi'},
    'language_info':{'name':'python','version':'3.11'},
    'codex':{'scope':'full manuscript execution with independently recomputed metrics',
             'default_mode':'completed-only preview; no long training','generator':'scripts/build_full_notebook.py'}})
nbf.validate(notebook)
for index,cell in enumerate(notebook.cells):
    if cell.cell_type=='code': compile(cell.source,f'Full_Paper_Reproduction_KO:cell_{index}','exec')
target=ROOT/'notebooks/Full_Paper_Reproduction_KO.ipynb'
target.parent.mkdir(parents=True,exist_ok=True)
nbf.write(notebook,target)
print(f'{target}: {len(cells)} cells; all code cells compile; notebook schema valid')
