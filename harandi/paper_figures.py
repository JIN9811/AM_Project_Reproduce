"""Paper-style scientific figures from saved numerical data, with scale audits.

No values are fitted to the article. Printed ranges may saturate colors;
every such sample is counted, and colorbar triangles disclose saturation.
"""
import argparse,csv,hashlib,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Rectangle,Circle,FancyBboxPatch,FancyArrowPatch,Polygon
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from .physics import FIELDS,Problem,grid,material,RECTANGLES

ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=(r'$u_x$',r'$u_y$',r'$\sigma_x$',r'$\sigma_y$',r'$\sigma_{xy}$',r'$T$',r'$q_x$',r'$q_y$')
UNITS=('mm','mm','GPa','GPa','GPa','K','W/mK','W/mK')
TERMS=('equilibrium','energy','connection','dirichlet','neumann')
COLORS=('#0072BD','#D95319','#008000','#FF0000','#8B42D2')


def paper_cmap(root=ROOT):
    return LinearSegmentedColormap.from_list('published_pdf',json.loads((Path(root)/'data/paper_palette.json').read_text())['rgb'],N=256)


def scale_audit(values,limits):
    a=np.asarray(values);lo,hi=limits
    below=int(np.count_nonzero(a<lo));above=int(np.count_nonzero(a>hi))
    extend='both' if below and above else 'min' if below else 'max' if above else 'neither'
    return {'scale_min':lo,'scale_max':hi,'data_min':float(a.min()),'data_max':float(a.max()),
            'below_count':below,'above_count':above,'sample_count':int(a.size),
            'outside_percent':100*(below+above)/a.size,'extend':extend}


def history_rows(path):
    with Path(path).open() as f:raw=list(csv.DictReader(f))
    offset=max([int(r['step']) for r in raw if r['optimizer']=='adam'] or [0])
    output=[]
    for row in raw:
        r={k:(v if k in ('optimizer','active') else float(v) if v else np.nan) for k,v in row.items()}
        r['update']=int(row['step'])+(offset if row['optimizer']=='lbfgs' else 0)
        output.append(r)
    return output


def load_fields(path):
    with np.load(path) as a:
        fields=a['fields'].copy()
        if 'constitutive_stress' in a:fields[:,2:5]=a['constitutive_stress'];fields[:,6:8]=a['fourier_flux']
        return a['xy'].copy(),fields


class FigureWriter:
    def __init__(self,root):
        self.root=Path(root);self.out=self.root/'figures/paper_style';self.out.mkdir(parents=True,exist_ok=True)
        self.items=[];self.audit=[];self.pdf=PdfPages(self.root/'PAPER_STYLE_FIGURES.pdf')

    def save(self,fig,name,title,note,source=None):
        fig.savefig(self.out/(name+'.png'),dpi=220,facecolor='white')
        fig.savefig(self.out/(name+'.svg'),facecolor='white')
        self.pdf.savefig(fig,facecolor='white')
        self.items.append({'id':name,'title':title,'note':note,'source_figure':source,
                           'png':f'figures/paper_style/{name}.png','svg':f'figures/paper_style/{name}.svg'})
        plt.close(fig)

    def close(self):self.pdf.close()


def field_figure(writer,geometry,mode='coupled',kind='mechanical',released=False):
    root=writer.root
    run='authors_released_checkpoint' if released else f'pinn_{geometry}_{mode}'
    _,pred=load_fields(root/f'results/{run}/prediction.npz')
    _,ref=load_fields(root/f'results/fem_{geometry}_n50/solution.npz')
    # 51x51 displayed samples match the paper's panel density. Quantitative
    # main-report comparisons retain their full 101x101 grid and FEM200.
    pred=pred.reshape(101,101,8)[::2,::2];ref=ref.reshape(101,101,8)[::2,::2]
    indices=list(range(5)) if kind=='mechanical' else [5,6,7]
    limits=json.loads((root/'data/paper_color_limits.json').read_text())[geometry]
    height=10.4 if len(indices)==5 else 6.65
    fig,axes=plt.subplots(len(indices),3,figsize=(7.35,height))
    fig.subplots_adjust(left=.075,right=.925,top=.962,bottom=.065,wspace=.38,hspace=.23)
    source=limits[FIELDS[indices[0]]]['source_figure']
    name=f'fig{source:02d}_{geometry}_{"released" if released else mode}_{kind}'
    cmap=paper_cmap(root)
    for i,j in enumerate(indices):
        for col,(role,values) in enumerate(zip(('pinn','fem','error'),(pred[:,:,j],ref[:,:,j],abs(pred[:,:,j]-ref[:,:,j])))):
            ax=axes[i,col];bound=limits[FIELDS[j]][role];audit=scale_audit(values,bound)
            im=ax.imshow(values,origin='lower',extent=(0,1,0,1),interpolation='nearest',cmap=cmap,
                         vmin=bound[0],vmax=bound[1],aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
            title=(('Released' if released else 'PINNs'),'FE','error')[col]
            ax.set_title(f'{SYMBOLS[j]}  {title} [{UNITS[j]}]',fontsize=9,pad=4,fontstyle='italic')
            bar=fig.colorbar(im,ax=ax,fraction=.048,pad=.055,ticks=bound,extend=audit['extend'],extendfrac=.035)
            bar.ax.set_yticklabels([f'{v:.5g}' for v in bound]);bar.ax.tick_params(labelsize=8,length=0,pad=3)
            bar.outline.set_linewidth(.45)
            writer.audit.append({'figure':name,'geometry':geometry,'run':run,'field':FIELDS[j],'panel':role,**audit})
    fig.text(.075,.028,f'Reproduction of Fig. {source} | {geometry}, {"released checkpoint" if released else mode} | FEniCS 50 x 50\n'
             'Printed color limits; triangles indicate saturation. Benchmark unit labels follow the article.',fontsize=7.5)
    writer.save(fig,name,f'그림 {source} 형식  {geometry} {mode} {kind}',
                '원문의 패널별 색상 범위를 고정하였다. 삼각형은 범위 밖 계산값을 뜻한다. 오차는 본 재현 PINN과 FEniCS 50×50의 절대 차이다.',source)


def point_mesh_figures(writer,geometry):
    p=Problem(geometry);root=writer.root;cmap=paper_cmap(root)
    config=json.loads((root/f'results/pinn_{geometry}_coupled/config.json').read_text())
    xy=grid(config['n'])
    if geometry=='circle' and len(xy)<5000:xy=np.r_[xy,np.random.default_rng(config['seed']).random((5000-len(xy),2))]
    test=grid(51);E,_=material(xy,p,'nodal');Et,_=material(test,p,'nodal')
    nodes=grid(51).reshape(51,51,2);centers=grid(51) # actual mesh edges shown below
    mid=(np.arange(50)+.5)/50;mx,my=np.meshgrid(mid,mid)
    Ec,_=material(np.c_[mx.ravel(),my.ravel()],p,'sharp')
    fig,axes=plt.subplots(1,3,figsize=(8.8,3.5));fig.subplots_adjust(left=.055,right=.99,bottom=.18,top=.89,wspace=.18)
    for ax,a,e,title in zip(axes[:2],(xy,test),(E,Et),('Train PINNs','Test PINNs')):
        ax.scatter(a[:,0],a[:,1],c=e,cmap=cmap,vmin=min(p.phases),vmax=max(p.phases),s=.7 if len(a)>3000 else 1.6,linewidths=0)
        ax.set_title(title,fontstyle='italic',fontsize=10)
    axes[2].pcolormesh(nodes[:,:,0],nodes[:,:,1],Ec.reshape(50,50),cmap=cmap,
                       vmin=min(p.phases),vmax=max(p.phases),edgecolors='black',linewidth=.15,rasterized=True)
    axes[2].set_title('FE Mesh',fontsize=10,fontstyle='italic')
    for ax in axes:ax.set(xlim=(0,1),ylim=(0,1),aspect='equal');ax.tick_params(labelsize=8,length=2)
    source=5 if geometry=='circle' else 12
    fig.text(.055,.045,f'Fig. {source} style | Training: {len(xy):,} actual points. Display: 51 x 51 subset of the 101 x 101 evaluation grid.\n'
             'FEM: 50 x 50 quadrilateral elements. Color identifies the prescribed material.',fontsize=8)
    writer.save(fig,f'fig{source:02d}_{geometry}_points_mesh',f'그림 {source} 형식  학습점과 평가점 및 유한요소 격자',
                f'실제 학습점 {len(xy):,}개를 표시하였다. 평가점 그림은 가독성을 위한 51×51 부분집합이며 전체 오차 평가는 101×101에서 수행한다.',source)
    _,f=load_fields(root/f'results/pinn_{geometry}_coupled/prediction.npz')
    _,r=load_fields(root/f'results/fem_{geometry}_n50/solution.npz')
    f=f.reshape(101,101,8)[::2,::2];r=r.reshape(101,101,8)[::2,::2]
    deformed_p=nodes+f[:,:,:2];deformed_f=nodes+r[:,:,:2]
    fig,axes=plt.subplots(1,3,figsize=(8.8,3.5));fig.subplots_adjust(left=.055,right=.99,bottom=.18,top=.89,wspace=.18)
    for ax,a,title in zip(axes[:2],(nodes,deformed_p),('Undeformed','Deformed configuration PINNs')):
        ax.scatter(a[:,:,0].ravel(),a[:,:,1].ravel(),c=Et,cmap=cmap,vmin=min(p.phases),vmax=max(p.phases),s=1.8,linewidths=0)
        ax.set_title(title,fontsize=9,fontstyle='italic')
    polygons=np.stack((deformed_f[:-1,:-1],deformed_f[:-1,1:],deformed_f[1:,1:],deformed_f[1:,:-1]),axis=2).reshape(-1,4,2)
    pc=PolyCollection(polygons,cmap=cmap,norm=Normalize(min(p.phases),max(p.phases)),edgecolors='black',linewidths=.18)
    pc.set_array(Ec);axes[2].add_collection(pc);axes[2].set_title('Deformed configuration FE',fontsize=9,fontstyle='italic')
    for ax in axes:ax.set(xlim=(0,1),ylim=(0,1),aspect='equal');ax.tick_params(labelsize=8,length=2)
    fig.text(.055,.055,'Fig. 7 style | Actual displacement added to coordinates; deformation factor = 1.0.\n'
             'Material color follows the undeformed coordinates. The benchmark uses an artificial alpha = 1.',fontsize=8)
    writer.save(fig,f'fig07_{geometry}_deformation','그림 7 형식  변형 전후 형상',
                '변형 배율은 1이다. 좌표에 실제 예측 변위를 더했으며 재료 색상은 변형 전 좌표에서 정의하였다.',7)


def _blocks(rows):
    blocks=[];start=0;last=None
    for r in rows:
        key=(r['optimizer'],r['active'])
        if key!=last:
            if last is not None:blocks.append((start,previous,last))
            start=0 if last is None else previous;last=key
        previous=r['update']
    if last is not None:blocks.append((start,previous,last))
    return blocks


def loss_figure(writer,geometry,mode):
    run=writer.root/f'results/pinn_{geometry}_{mode}'
    dense=writer.root/f'results/process_{geometry}_{mode}'
    if (dense/'process_metadata.json').exists():
        meta=json.loads((dense/'process_metadata.json').read_text())
        if meta.get('status','complete')=='complete':run=dense
    rows=history_rows(run/'history.csv');config=json.loads((run/'config.json').read_text())
    blocks=_blocks(rows);fig,axes=plt.subplots(2,1,figsize=(7.6,7.2))
    fig.subplots_adjust(left=.12,right=.97,top=.95,bottom=.20,hspace=.34)
    suffix=('SF','EF','CNC','DBC','NBC')
    for ax,field,prefix in zip(axes,('thermal','mechanical'),('T','M')):
        for lo,hi,(optimizer,active) in blocks:
            if mode=='sequential' and active!=field:ax.axvspan(lo,hi,color='#D9D9D9',zorder=0)
        for term,color,mark in zip(TERMS,COLORS,suffix):
            y=np.array([r.get(prefix+'_'+term,np.nan) for r in rows]);x=np.array([r['update'] for r in rows])
            # Exact zeros are not plotted at a made-up logarithmic floor.
            y[y<=0]=np.nan
            for lo,hi,key in blocks:
                if mode=='sequential' and key[1]!=field:continue
                mask=(x>lo)&(x<=hi)
                ax.plot(x[mask],y[mask],color=color,lw=.75,
                        label=rf'$\mathcal{{L}}_{{{prefix}}}^{{{mark}}}$' if lo==next(b[0] for b in blocks if mode!='sequential' or b[2][1]==field) else None)
        switch=(1 if mode=='coupled' else 2)*config['adam_steps']
        ax.axvline(switch,color='.25',lw=.8,ls='--')
        ax.set(yscale='log',xlabel='Optimizer updates',ylabel='Loss',xlim=(0,rows[-1]['update']))
        ax.set_title(field.capitalize()+' field',fontsize=10)
        ax.legend(loc='upper right',fontsize=9,framealpha=.9)
        ax.text(.02,.04,'Hard Dirichlet loss = 0 (not visible on log scale)',transform=ax.transAxes,fontsize=8)
    source=6 if mode=='coupled' else 10
    fig.text(.12,.035,f'Fig. {source} style | {geometry}, {mode}. Dashed vertical line: Adam to L-BFGS.\n'
             'Actual logged updates; no rescaled epochs. Gray: other field active, this loss unobserved.\n'
             'EF denotes the implemented balance objective; it differs from the printed manuscript energy.',fontsize=8)
    writer.save(fig,f'fig{source:02d}_{geometry}_{mode}_losses',f'그림 {source} 형식  {geometry} {mode} 학습 손실',
                '실제 기록된 항별 손실을 표시하였다. 회색 구간은 다른 장을 학습한 구간이며 미관측 손실을 연결하거나 보간하지 않았다. 영인 경계 손실은 로그 축에 그리지 않는다.',source)


def section_figure(writer,geometry,convergence=False):
    fig,axes=plt.subplots(2,2,figsize=(7.7,6.8));fig.subplots_adjust(left=.10,right=.97,top=.95,bottom=.19,wspace=.25,hspace=.38)
    y=np.linspace(0,1,101)
    if convergence:runs=[(f'fem_{geometry}_n{n}',f'FE {n} by {n}','black',ls) for n,ls in ((50,'-'),(100,'--'),(200,':'))]
    else:runs=[(f'pinn_{geometry}_coupled','Mix. PINNs. Coupled','blue','--'),
               (f'pinn_{geometry}_sequential','Mix. PINNs. Sequential','blue',':'),(f'fem_{geometry}_n50','FE','black','-')]
    for run,label,color,ls in runs:
        file=writer.root/'results'/run/('solution.npz' if run.startswith('fem') else 'prediction.npz')
        _,f=load_fields(file);section=f.reshape(101,101,8)[:,50,:]
        for ax,j in zip(axes.ravel(),(0,2,5,6)):
            ax.plot(y,section[:,j],color=color,ls=ls,lw=.9,label=label)
            ax.set(title=SYMBOLS[j]+r' at $x=0.5$',xlabel=r'$y$ [mm]',ylabel=SYMBOLS[j]+f' [{UNITS[j]}]')
            ax.legend(fontsize=7.5,framealpha=.9)
    source='A1' if convergence else 11 if geometry=='circle' else 16
    note='FEM mesh refinement: 50, 100 and 200 elements per edge.' if convergence else 'Actual local mixed-PINN results. Unexecuted standard PINN / DEM curves are not included.'
    fig.text(.10,.035,f'Fig. {source} style | {geometry}. {note}\nBenchmark unit labels follow the article; no fitted amplitude factors.',fontsize=8)
    writer.save(fig,f'fig{source}_{geometry}_sections',f'그림 {source} 형식  단면 비교',note,source)


def _box(ax,x,y,w,h,text,color='#FFFFFF',fontsize=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.012',
                 facecolor=color,edgecolor='#666666',linewidth=.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)


def _arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=.8,color='#333333'))


def workflow_figures(writer):
    fig,axes=plt.subplots(1,3,figsize=(10,5.2));fig.subplots_adjust(left=.02,right=.98,top=.92,bottom=.10,wspace=.08)
    for ax,title in zip(axes,('Sequential training','Coupled training','Mixed PINN')):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,fontsize=13)
        ax.add_patch(FancyBboxPatch((.025,.025),.95,.95,boxstyle='round,pad=.01,rounding_size=.06',
                                  facecolor='#C4C4C4' if ax!=axes[2] else '#F7B383',edgecolor='none'))
    ax=axes[0];_box(ax,.17,.80,.66,.11,'Coordinates and material')
    _box(ax,.16,.59,.46,.12,r'Minimize $\mathcal{L}_T$','#F7B383');_box(ax,.68,.59,.24,.12,'Mechanics\nfrozen',fontsize=8)
    _box(ax,.16,.38,.46,.12,r'Minimize $\mathcal{L}_M$','#F7B383');_box(ax,.68,.38,.24,.12,'Thermal\nfrozen',fontsize=8)
    _box(ax,.17,.18,.66,.11,'Repeat blocks to budget','#DDEBF7');_box(ax,.26,.055,.48,.075,'Solution','#DDEBF7')
    for a,b in [((.39,.80),(.39,.71)),((.39,.59),(.39,.50)),((.39,.38),(.39,.29)),((.50,.18),(.50,.13))]:_arrow(ax,a,b)
    ax=axes[1];_box(ax,.17,.80,.66,.11,'Coordinates and material');_box(ax,.10,.53,.80,.16,r'Minimize $\mathcal{L}_M+\mathcal{L}_T$','#F7B383')
    _box(ax,.17,.28,.66,.13,'Update every network\nuntil budget reached','#DDEBF7');_box(ax,.26,.08,.48,.10,'Solution','#DDEBF7')
    for a,b in [((.5,.80),(.5,.69)),((.5,.53),(.5,.41)),((.5,.28),(.5,.18))]:_arrow(ax,a,b)
    ax=axes[2];_box(ax,.10,.77,.80,.14,'Inputs\n$x,y$ and prescribed material')
    _box(ax,.10,.48,.80,.16,'Eight independent networks\n5 hidden layers x 40 tanh units')
    _box(ax,.10,.20,.80,.15,'Primary and mixed outputs\n'+r'$u_x,u_y,T,\sigma_x,\sigma_y,\sigma_{xy},q_x,q_y$')
    _arrow(ax,(.5,.77),(.5,.64));_arrow(ax,(.5,.48),(.5,.35))
    fig.text(.03,.025,'Fig. 1 style | Diagram reflects the implemented algorithm. Global work-balance terms and hard Dirichlet conditions are used.',fontsize=8)
    writer.save(fig,'fig01_training_workflow','그림 1 형식  순차 학습과 동시 학습 절차','논문의 색조와 세 패널 구성을 사용하여 실제 구현의 학습 절차를 다시 그렸다.',1)
    fig,ax=plt.subplots(figsize=(10.8,5.6));fig.subplots_adjust(left=.01,right=.99,bottom=.10,top=.95);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    _box(ax,.02,.42,.12,.16,'$x,y$\ninputs','#FFF2CC')
    labels=(r'$u_x$',r'$u_y$',r'$\sigma_x^o$',r'$\sigma_y^o$',r'$\sigma_{xy}^o$',r'$T$',r'$q_x^o$',r'$q_y^o$')
    for i,label in enumerate(labels):
        y=.89-i*.105;_box(ax,.23,y-.035,.16,.065,'5 x 40  →  '+label,'#DDEBF7',9)
        _arrow(ax,(.14,.50),(.23,y));_arrow(ax,(.39,y),(.49,.53))
    _box(ax,.49,.26,.22,.52,'First derivatives\n\nKinematics\n'+r'$\varepsilon=\mathrm{sym}(\nabla u)$'+'\n\nPlane-strain law\n'+r'$\sigma=C:(\varepsilon-\varepsilon_T)$'+'\n\nFourier law\n'+r'$q=-k\nabla T$','#DDE9D2',10)
    _box(ax,.78,.55,.20,.34,'Mechanical losses\nEquilibrium\nConnection\nBoundary\nGlobal balance','#F7C9A6',10)
    _box(ax,.78,.12,.20,.34,'Thermal losses\nEquilibrium\nConnection\nBoundary\nGlobal balance','#F7C9A6',10)
    _arrow(ax,(.71,.63),(.78,.72));_arrow(ax,(.71,.40),(.78,.29))
    fig.text(.03,.025,'Fig. 4 style | Eight independent scalar networks; derivative evaluation uses automatic differentiation.\nHard constraints: ux = x(1-x)Nx, uy = y(1-y)Ny, T = 1-x+x(1-x)NT.',fontsize=9)
    writer.save(fig,'fig04_network_losses','그림 4 형식  혼합 PINN 구조와 손실 항','직접 출력한 응력과 열유속을 변위 및 온도 미분으로 계산한 값과 연결하는 구조를 나타낸다.',4)


def evolution_figure(writer):
    run=writer.root/'results/process_circle_coupled';folder=run/'snapshots'
    if not (run/'process_metadata.json').exists():return
    meta=json.loads((run/'process_metadata.json').read_text())
    if meta.get('status','complete')!='complete':return
    config=json.loads((run/'config.json').read_text())
    steps=meta['snapshot_updates'];files=[folder/f'step_{step:06d}.npz' for step in steps]
    if any(not file.exists() for file in files):raise FileNotFoundError('Missing snapshot listed in completed process metadata')
    if not files:return
    values=[]
    for file,step in zip(files,steps):
        xy,f=load_fields(file)
        if int(np.load(file)['step'])!=step:raise ValueError('Snapshot step does not match metadata')
        values.append(f)
    fig,axes=plt.subplots(3,len(files),figsize=(2.15*len(files),6.9));fig.subplots_adjust(left=.045,right=.935,top=.91,bottom=.13,wspace=.18,hspace=.25)
    cmap=paper_cmap(writer.root);limits=((0,1),(0,.23),(-.65,-.25))
    row_extensions=[scale_audit(np.concatenate([f[:,j] for f in values]),limits[row])['extend']
                    for row,j in enumerate((5,0,2))]
    for col,(f,step) in enumerate(zip(values,steps)):
        for row,j in enumerate((5,0,2)):
            ax=axes[row,col];lo,hi=limits[row]
            im=ax.imshow(f[:,j].reshape(51,51),extent=(0,1,0,1),origin='lower',cmap=cmap,vmin=lo,vmax=hi)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'Update {step:,}',fontsize=10)
            if col==0:ax.set_ylabel(SYMBOLS[j],fontsize=13)
            audit=scale_audit(f[:,j],(lo,hi))
            writer.audit.append({'figure':'process_training_evolution','geometry':'circle',
                'run':'process_circle_coupled','field':FIELDS[j],'panel':f'update_{step}',**audit})
            if col==len(files)-1:fig.colorbar(im,ax=ax,fraction=.045,pad=.03,ticks=(lo,hi),extend=row_extensions[row])
    factor=1 if config['mode']=='coupled' else 2;adam_end=factor*config['adam_steps']
    total=meta['total_optimizer_updates']
    schedule=(f'Adam: 1–{adam_end:,}; ' if adam_end else 'No Adam updates; ')
    schedule+=(f'L-BFGS: {adam_end+1:,}–{total:,}.' if total>adam_end else 'No L-BFGS updates.')
    fig.text(.045,.045,f'Real post-update snapshots, circle {config["mode"]} run; no interpolated frames. '+schedule+'\n'
             'Each row uses a fixed scale. Early fields may exceed it; full values remain in the snapshot NPZ files.',fontsize=9)
    writer.save(fig,'process_training_evolution','추가 과정 그림  실제 학습 단계별 장의 변화',f'총 {total:,}회 업데이트 중 현재 완료된 실험의 메타데이터에 명시된 단계만 표시하였다. 각 단계의 색상 포화 통계를 감사표에 기록한다.')


def generate(root=ROOT):
    root=Path(root);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.6,
        'axes.spines.top':True,'axes.spines.right':True,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    writer=FigureWriter(root)
    try:
        workflow_figures(writer)
        for geom in ('circle','rectangles'):
            point_mesh_figures(writer,geom)
            for mode in ('coupled','sequential'):
                loss_figure(writer,geom,mode)
                for kind in ('mechanical','thermal'):field_figure(writer,geom,mode,kind)
            section_figure(writer,geom)
        section_figure(writer,'rectangles',convergence=True)
        for kind in ('mechanical','thermal'):field_figure(writer,'rectangles','released',kind,released=True)
        evolution_figure(writer)
    finally:writer.close()
    audit=root/'results/paper_scale_audit.csv'
    with audit.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(writer.audit[0]));w.writeheader();w.writerows(writer.audit)
    manifest={'palette':'RGB sampled from supplied PDF Fig.8; data/paper_palette.json',
        'panel_limits':'Transcribed from Figs.8,9,13,14; data/paper_color_limits.json',
        'reference':'paper-style panels use FEniCS 50x50; original accuracy report retains FEniCS200',
        'field_representation':'stress/flux derived from u/T gradients, as in authors plotting code',
        'color_saturation':'disclosed by triangles and complete audit CSV; no data modification',
        'items':writer.items,'scale_audit':writer.audit}
    (root/'results/paper_figure_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    gallery=''
    for item in writer.items:
        gallery+=f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["note"])}</p><img src="{item["png"]}" alt="{html.escape(item["title"])}"><p><a href="{item["svg"]}">SVG 벡터 파일</a></p></section>'
    comparisons=''
    for number,geom,kind in ((8,'circle','mechanical'),(9,'circle','thermal'),(13,'rectangles','mechanical'),(14,'rectangles','thermal')):
        comparisons+=f'<h2>원문 그림 {number}와 재현 그림</h2><div class="pair"><div><h3>원문 PDF 발췌</h3><img src="figures/paper_reference/original_fig{number:02d}.png"></div><div><h3>실제 재현 계산</h3><img src="figures/paper_style/fig{number:02d}_{geom}_coupled_{kind}.png"></div></div>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Harandi 논문 형식 재현 그림</title><style>body{max-width:1120px;margin:40px auto;padding:0 22px;font:16px/1.7 "Malgun Gothic",sans-serif;color:#222}h1,h2{line-height:1.4}h2{margin-top:40px}img{display:block;max-width:100%;height:auto;margin:auto}section{border-top:1px solid #ddd;margin-top:32px;padding-top:15px}a{color:#245b86}.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px}.pair img{max-height:1150px}section>img{max-width:900px}@media(max-width:760px){.pair{display:block}}</style><h1>Harandi 열탄성 혼합 PINN 논문 형식 재현 그림</h1><p>원문 PDF의 팔레트와 패널별 색상 범위를 사용하였다. 계산값 및 학습 횟수는 실제 재현 결과를 유지한다. 색상 막대의 삼각형은 원문 범위를 벗어난 값을 나타내며, <a href="results/paper_scale_audit.csv">전체 색상 범위 감사표</a>에 수량과 최솟값·최댓값을 기록하였다.</p><p>원문 출처는 Harandi et al., <a href="https://doi.org/10.1002/nme.7388">10.1002/nme.7388</a>이며 원문 이미지는 사용자가 제공한 PDF의 비교용 발췌이다. 재현 오차는 원문 이미지의 픽셀 차이가 아니라 본 PINN과 본 FEniCS의 차이다. 원문에 수록된 모든 연구를 실행한 것은 아니다.</p><p><a href="PROJECT_REPORT_KO.pdf">한국어 학술 프로젝트 보고서</a> · <a href="PAPER_STYLE_FIGURES.pdf">전체 그림 PDF</a> · <a href="Harandi_Reproduction_KO.ipynb">재현 노트북</a> · <a href="REPORT.html">전체 범위의 기존 정확도 보고서</a></p>'''+comparisons+gallery+'</html>'
    (root/'PAPER_FIGURES_KO.html').write_text(page,encoding='utf-8')
    print(f'Generated {len(writer.items)} paper-style figures; {len(writer.audit)} scale audits.')
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT)
    generate(ap.parse_args().root)
