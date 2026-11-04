"""Build the Korean reproducibility notebook; execute with nbclient/nbconvert."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))

md(r'''# 열탄성 혼합 PINN과 FEniCS 재현 실험

Harandi et al., *Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains*, DOI [10.1002/nme.7388](https://doi.org/10.1002/nme.7388).

이 노트북은 원형 및 사각형 개재물 문제를 실제 FEniCS와 PyTorch로 재현하는 실행 안내서이다. **기본 모드**는 제공된 연구 결과를 읽고, 별도의 작은 FEM·PINN 검증 계산을 실제 실행한 뒤 그림과 오차를 다시 계산한다. **처음부터 재현**하려면 아래의 세 재계산 설정을 `True`로 바꾼다. 기존 결과를 사용한 것을 새 학습으로 간주하지 않는다.

저장된 주 실험에서 온도 상대 L2 오차는 0.24–0.43%, 수평 변위 오차는 1.28–1.63%이다. 전단응력·횡방향 열유속 오차는 더 크다. 정확한 수치는 아래에서 파일로부터 다시 계산한다. 결과는 100k/200k Adam 실험 전체와 동일하지 않다.

한국어 상세 설명은 `PROJECT_REPORT_KO.pdf`, 그림 비교는 `PAPER_FIGURES_KO.html`, 구현은 `harandi/`에 있다. 노트북과 이 폴더 전체를 함께 보관한다.''')
md(r'''## 1 환경과 실행 설정

이 컴퓨터에서는 `start_notebook.ps1`로 열고 **Harandi CUDA Python 3.11** 커널을 선택한다. 새 Windows 컴퓨터에서는 Python 3.11 이상과 WSL Ubuntu를 준비한 뒤 `setup.ps1`를 실행한다. FEniCS는 WSL의 Python, PINN은 이 노트북 커널의 Python에서 실행한다. Linux에서는 `FENICS_PYTHON`을 DOLFIN이 설치된 실행 파일로 지정할 수 있다.

주 실험을 다시 학습하면 이 컴퓨터에서 약 35–40분, 중간 상태 실험은 약 9분이 추가로 필요했다. CPU는 더 오래 걸릴 수 있다.''')
code(r'''from pathlib import Path
import os, sys, subprocess, json, csv, contextlib, hashlib, io
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image

def show_figure(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close(figure)

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'harandi/physics.py').exists()), None)
assert ROOT is not None, '프로젝트 폴더에서 노트북을 여세요.'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
RECOMPUTE_FEM = False
RETRAIN_PINN = False
RETRAIN_PROCESS = False
REGENERATE_PAPER_FIGURES = True
WORK = ROOT / '.notebook_work'
WORK.mkdir(exist_ok=True)
torch.set_num_threads(4)
print('Project:', ROOT)
print('Python:', sys.version.split()[0], '| Torch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('Mode:', dict(FEM=RECOMPUTE_FEM, PINN=RETRAIN_PINN, process=RETRAIN_PROCESS))''')
md(r'''## 2 지배방정식과 가정

단위 정사각형에서 정상 열전도와 선형 평면변형률 열탄성을 계산한다.

$$\nabla\cdot\sigma=0,\quad\nabla\cdot q=0,\quad q=-k\nabla T,$$
$$\sigma=2\mu\varepsilon+\lambda\,\mathrm{tr}(\varepsilon)I-\frac{E\alpha(T-T_0)}{1-2\nu}I.$$

$\nu=0.3,\alpha=1,T_0=0$이며 왼쪽 $T=1$, 오른쪽 $T=0$이다. 좌우에서 $u_x=0$, 상하에서 $u_y=0$, 모든 변에서 전단력이 0이다. 상하 열유속도 0이다. 열전도 문제는 변위의 영향을 받지 않는 일방향 연성이다.

원형 중심 (0.60, 0.35)는 논문 그림에서 추정하였다. 사각형 좌표는 저자의 재료 데이터와 일치한다. 본문 그림의 mm, GPa, K, W/mK 표기는 원문을 따르는 수치 벤치마크 표기이며 자동 SI 환산을 의미하지 않는다.''')
code(r'''from harandi.physics import Problem, FIELDS, grid, material, RECTANGLES
from harandi.paper_figures import paper_cmap, load_fields

cmap = paper_cmap(ROOT)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, geometry in zip(axs, ('circle', 'rectangles')):
    xy = grid(101)
    E, k = material(xy, Problem(geometry), 'sharp')
    im = ax.imshow(E.reshape(101, 101), origin='lower', extent=(0,1,0,1), cmap=cmap)
    ax.set(title=geometry, xlabel='x / L', ylabel='y / L')
    fig.colorbar(im, ax=ax, label='E (benchmark value)')
fig.tight_layout()
show_figure(fig)
print('Rectangles (xmin, xmax, ymin, ymax):', RECTANGLES)''')
md(r'''## 3 실제 FEniCS 검증 계산

다음 셀은 저장된 표를 읽는 대신 균질 재료의 경계값 문제를 새로 푼다. Q1 사각형 유한요소로 열전도 해를 구한 뒤, 온도를 열하중으로 넣어 변위를 계산한다. 균질 문제의 해 $T=1-x$, $u_x=\frac{13}{14}x(1-x)$와 비교한다. 이 작은 검증 계산은 주 연구 결과를 덮어쓰지 않는다.''')
code(r'''def fem_command(arguments):
    if sys.platform == 'win32':
        command = ['wsl.exe','-d','Ubuntu','--cd',str(ROOT),'--','env',
                   'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','python3',*arguments]
    else:
        command = [os.environ.get('FENICS_PYTHON','/usr/bin/python3'),*arguments]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout

fem_command(['-m','harandi.fem','--geometry','homogeneous','--n','12',
             '--output','.notebook_work/fem_exact'])
exact = np.load(WORK/'fem_exact/solution.npz')
x = exact['xy'][:,0]
assert np.max(abs(exact['fields'][:,5]-(1-x))) < 1e-10
u_exact = 13/14*x*(1-x)
u_error = np.linalg.norm(exact['fields'][:,0]-u_exact)/np.linalg.norm(u_exact)
assert u_error < 0.01
diagnostics = json.loads((WORK/'fem_exact/metadata.json').read_text())['diagnostics']
assert diagnostics['thermal_relative_residual'] < 1e-8
assert diagnostics['mechanical_relative_residual'] < 1e-8
assert abs(diagnostics['heat_balance']) < 1e-8
print('Actual FEniCS solve passed. Displacement L2 interpolation error:', f'{100*u_error:.3f}%')
print('Heat imbalance:', diagnostics['heat_balance'])''')
md(r'''## 4 주 FEM 기준해 생성

기하별 50×50, 100×100, 200×200 요소 해와 PINN의 보간 재료장에 맞춘 100×100 기준해를 준비한다. `RECOMPUTE_FEM=True`이면 전부 다시 풀며, 기본 모드에서도 파일이 없으면 해당 해석을 실행한다.''')
code(r'''for geometry in ('circle','rectangles'):
    for n, suffix, coefficient in [(50,'n50','sharp'),(100,'n100','sharp'),
                                  (200,'n200','sharp'),(100,'n100_nodal','nodal')]:
        out = Path('results') / f'fem_{geometry}_{suffix}'
        if RECOMPUTE_FEM or not (out/'metadata.json').exists():
            fem_command(['-m','harandi.fem','--geometry',geometry,'--n',str(n),
                         '--coefficient-mode',coefficient,'--output',out.as_posix()])
            print('Solved:', out)
        else:
            print('Loaded:', out)
print('Eight FEM references available.')''')
md(r'''## 5 혼합 PINN과 실제 학습 동작 확인

8개 독립 신경망이 $u_x,u_y,\sigma_x^o,\sigma_y^o,\sigma_{xy}^o,T,q_x^o,q_y^o$를 출력한다. 응력·열유속 직접 출력과 주 변수 미분으로 얻는 응력·열유속은 연결 손실로 일치시킨다. 아래 작은 학습은 자동미분과 최적화가 이 커널에서 실제 작동하는지 확인하는 실행 검증이다. 정확도 연구의 최종 모델은 다음 절의 5×40 네트워크를 사용한다.''')
code(r'''from harandi.train import TrainConfig, run, relative_errors
from harandi.pinn import MixedPINN, make_batch, mixed_loss

smoke_config = TrainConfig(geometry='homogeneous', width=8, layers=2, n=9,
                          adam_steps=8, lbfgs_steps=4, log_every=4, device='cpu')
with (WORK/'pinn_smoke.log').open('w') as log, contextlib.redirect_stdout(log):
    smoke_model, smoke_metrics = run(smoke_config, WORK/'pinn_smoke')
assert smoke_metrics['diagnostics_independent_grid']['max_temperature_boundary_error'] == 0
assert smoke_metrics['diagnostics_independent_grid']['max_normal_displacement_error'] == 0
print('Actual Adam / L-BFGS updates passed; hard boundary errors are exactly zero.')''')
md(r'''## 6 주 PINN 실험 재현

각 장당 Adam 2,000회와 L-BFGS 3,000회를 사용한다. 동시 학습은 총 5,000회, 순차 학습은 총 10,000회 업데이트한다. 순차 학습은 한 장을 최적화할 때 다른 장의 파라미터를 고정한다. 학습 손실에는 FEM 정답을 넣지 않는다.

`RETRAIN_PINN=True`이면 기존 결과를 같은 폴더에 새로 쓴다. 원문 100k/200k Adam 스케줄은 `run.ps1 -Task paper`에 별도로 준비되어 있다. 본 노트북의 기본 결과는 원문의 모든 스케줄을 실행한 결과가 아니다.''')
code(r'''for geometry in ('circle','rectangles'):
    for mode in ('coupled','sequential'):
        out = ROOT/'results'/f'pinn_{geometry}_{mode}'
        if RETRAIN_PINN or not (out/'model.pt').exists():
            config = TrainConfig(geometry=geometry, mode=mode,
                                 n=51 if geometry=='circle' else 71,
                                 adam_steps=2000, lbfgs_steps=3000, switch=2000)
            with (WORK/f'train_{geometry}_{mode}.log').open('w') as log, contextlib.redirect_stdout(log):
                run(config, out)
            print('Trained:', geometry, mode)
        else:
            checkpoint = torch.load(out/'model.pt', map_location='cpu', weights_only=False)
            print('Loaded:', geometry, mode, '| completed updates:', checkpoint['completed'])''')
md(r'''## 7 오차의 독립 재계산

전체 101×101 균일 평가점에서 상대 L2 오차를 다시 계산한다. 아래 표는 FEniCS 200×200 기준의 **미분으로 계산한 응력과 열유속**을 사용한다. 직접 혼합 출력 오차는 각 실행 폴더의 `metrics.json`에 별도로 저장되어 있다. 작은 전단·횡방향 성분은 상대오차가 클 수 있으므로 함께 확인한다.''')
code(r'''records=[]
for geometry in ('circle','rectangles'):
    xy, reference = load_fields(ROOT/f'results/fem_{geometry}_n200/solution.npz')
    for mode in ('coupled','sequential'):
        xy_pred, prediction = load_fields(ROOT/f'results/pinn_{geometry}_{mode}/prediction.npz')
        assert np.array_equal(xy,xy_pred)
        errors = relative_errors(prediction, reference)
        saved = json.loads((ROOT/f'results/pinn_{geometry}_{mode}/metrics.json').read_text())
        for field in FIELDS:
            assert np.isclose(errors[field]['relative_L2'], saved['comparisons']['FEM_n200']['constitutive_outputs'][field]['relative_L2'])
        records.append((geometry,mode,errors))
header='| 기하 / 학습 | ux | uy | sxx | syy | sxy | T | qx | qy |\n|---|'+'---:|'*8+'\n'
table=header+'\n'.join('| '+g+' / '+m+' | '+' | '.join(f"{100*e[f]['relative_L2']:.3f}%" for f in FIELDS)+' |' for g,m,e in records)
display(Markdown(table))''')
md(r'''## 8 논문 형식의 편집 가능한 시각화

원문 그림 8의 색상 막대를 추출한 팔레트와 그림 8·9·13·14의 인쇄 범위를 사용한다. 이 절의 비교 기준은 원문과 같은 50×50 FEM이며, 이전 절의 정량 기준은 200×200 FEM이다. **색상 범위 일치와 수치해 일치는 서로 다른 조건**이다. 범위를 벗어난 값은 색상 막대의 삼각형과 감사표로 표시한다.''')
code(r'''from harandi.paper_figures import scale_audit
geometry='circle'
field='T'                    # 'ux', 'sxx', 'qx' 등으로 수정 가능
j=FIELDS.index(field)
limits=json.loads((ROOT/'data/paper_color_limits.json').read_text())[geometry][field]
_,prediction=load_fields(ROOT/f'results/pinn_{geometry}_coupled/prediction.npz')
_,reference=load_fields(ROOT/f'results/fem_{geometry}_n50/solution.npz')
values=(prediction[:,j],reference[:,j],abs(prediction[:,j]-reference[:,j]))
fig,axs=plt.subplots(1,3,figsize=(10,3.3))
for ax,role,title,value in zip(axs,('pinn','fem','error'),('PINNs','FEniCS 50x50','Absolute error'),values):
    lo,hi=limits[role]
    info=scale_audit(value,(lo,hi))
    im=ax.imshow(value.reshape(101,101),origin='lower',extent=(0,1,0,1),cmap=cmap,vmin=lo,vmax=hi)
    ax.set_title(field+' '+title)
    ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(im,ax=ax,ticks=(lo,hi),extend=info['extend'])
fig.tight_layout()
show_figure(fig)''')
md(r'''## 9 실제 학습 중간 상태와 과정 그림

별도의 동일 시드 원형 동시 학습에서 0, 100, 500, 2,000, 3,000, 5,000번째 업데이트 직후 상태를 저장한다. 중간 이미지는 보간으로 만든 장면이 아니다. `RETRAIN_PROCESS=True`이면 이 기록 실험도 다시 실행한다.''')
code(r'''process_dir=ROOT/'results/process_circle_coupled'
if RETRAIN_PROCESS or not (process_dir/'process_metadata.json').exists():
    from harandi.process import experiment
    with (WORK/'process_training.log').open('w') as log, contextlib.redirect_stdout(log):
        experiment(process_dir)
    print('Process experiment trained.')
else:
    print('Loaded real snapshot experiment:',json.loads((process_dir/'process_metadata.json').read_text())['snapshot_updates'])

from harandi.paper_figures import generate
if REGENERATE_PAPER_FIGURES:
    with (WORK/'paper_figures.log').open('w') as log, contextlib.redirect_stdout(log):
        generate(ROOT)
    print('Paper-style figures regenerated from numerical files.')
display(Image(filename=str(ROOT/'figures/paper_style/process_training_evolution.png'),width=1000))''')
md(r'''### 항별 손실 곡선

원문처럼 열·역학 손실을 위아래로 분리한다. x축은 실제 최적화 업데이트 수이며 원문의 100,000회로 늘려 표시하지 않는다. 정확히 0인 강제 경계 손실은 로그 축에 표시할 수 없다.''')
code(r'''from harandi.paper_figures import history_rows
rows=history_rows(process_dir/'history.csv')
fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True)
for ax,prefix in zip(axs,('T','M')):
    for term in ('equilibrium','connection','neumann','energy'):
        y=np.array([r.get(prefix+'_'+term,np.nan) for r in rows]);y[y<=0]=np.nan
        ax.semilogy([r['update'] for r in rows],y,label=term,lw=.8)
    ax.axvline(2000,color='black',ls='--',lw=.8)
    ax.set_ylabel(prefix+' loss');ax.legend(ncol=2,fontsize=8)
axs[-1].set_xlabel('Actual optimizer updates')
fig.tight_layout()
show_figure(fig)''')
md(r'''## 10 산출물과 재현성 검사

색상 포화는 시각화의 범위 제한이며 원시 값 수정이 아니다. 전단응력의 원문 PINN/FEM 색상 하한은 서로 다르므로 대응 범위를 별도로 보존하였다. 원문의 그림과 수치가 다른 원인에는 원형 위치 추정, 재료 보간, 평면변형률 식의 해석, 학습 예산·에너지 목적함수 차이가 포함된다. 저자 FEAP 변위에 발견된 약 0.98 계수도 임의로 보정하지 않는다.''')
code(r'''manifest=json.loads((ROOT/'results/paper_figure_manifest.json').read_text(encoding='utf-8'))
assert len(manifest['items']) >= 20
assert all((ROOT/item['png']).exists() and (ROOT/item['svg']).exists() for item in manifest['items'])
snapshot_files=sorted((process_dir/'snapshots').glob('step_*.npz'))
assert len(snapshot_files)==6
assert int(np.load(snapshot_files[-1])['step'])==5000
print('Figure count:',len(manifest['items']))
print('Scale audit panels:',len(manifest['scale_audit']))
print('Snapshots:',len(snapshot_files))
print('Notebook completed successfully. Full experiment recomputation settings:',
      RECOMPUTE_FEM, RETRAIN_PINN, RETRAIN_PROCESS)
display(Markdown('[논문 형식 그림 모음](PAPER_FIGURES_KO.html) · [한국어 프로젝트 보고서](PROJECT_REPORT_KO.pdf) · [실행 안내](README_KO.md)'))''')

nb=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Harandi CUDA Python 3.11','language':'python','name':'harandi'},
    'language_info':{'name':'python','version':'3.11.9'},
    'authors':[{'name':'Reproduction project'}]})
nbf.validate(nb)
nbf.write(nb,ROOT/'Harandi_Reproduction_KO.ipynb')
print('Wrote notebook:',len(cells),'cells')
