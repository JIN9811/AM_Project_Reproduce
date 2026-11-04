"""Regenerate static, publication-style figures and an offline HTML report."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .physics import FIELDS,Problem,grid,material,RECTANGLES
from .train import relative_errors

LABELS=(r'$u_x$',r'$u_y$',r'$\sigma_{xx}$',r'$\sigma_{yy}$',r'$\sigma_{xy}$',
        r'$T$',r'$q_x$',r'$q_y$')
COLORS={'coupled':'#156082','sequential':'#a34920'}


def _save(fig,path):
    fig.savefig(path,dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def fields_figure(pred,ref,indices,title,path,representation='constitutive'):
    fig,axs=plt.subplots(len(indices),3,figsize=(11.5,3.05*len(indices)),layout='constrained')
    for row,j in enumerate(indices):
        v=ref[:,j];p=pred[:,j];err=np.abs(p-v)
        low=min(v.min(),p.min());high=max(v.max(),p.max())
        if abs(high-low)<1e-12:high=low+1e-12
        cmap='RdBu_r' if low<0<high else 'viridis'
        if low<0<high: high=max(abs(low),abs(high));low=-high
        arrays=(p,v,err)
        for k,a in enumerate(arrays):
            ax=axs[row,k]
            im=ax.imshow(a.reshape(101,101),extent=(0,1,0,1),origin='lower',
                         interpolation='nearest',cmap='magma' if k==2 else cmap,
                         vmin=0 if k==2 else low,vmax=max(err.max(),1e-12) if k==2 else high)
            ax.set_xlabel('x / L');ax.set_ylabel('y / L')
            left='PINN-derived' if representation=='constitutive' else 'mixed output'
            ax.set_title(f'{LABELS[j]} — '+(left,'FEniCS Q1, 200×200','absolute difference')[k],fontsize=10)
            fig.colorbar(im,ax=ax,fraction=.047,pad=.035).ax.tick_params(labelsize=8)
    convention='Stress/flux from derivatives of u and T (authors’ plotting convention)' if representation=='constitutive' else 'Independent mixed stress/flux outputs'
    fig.suptitle(title+'\n'+convention+'; benchmark numerical units',fontsize=11)
    _save(fig,path)


def generate(root):
    root=Path(root);results=root/'results';figdir=root/'figures';figdir.mkdir(exist_ok=True)
    required=[results/f'fem_{geom}_n{n}'/'solution.npz' for geom in ('circle','rectangles') for n in (50,100,200)]
    if any(not f.exists() for f in required):
        raise FileNotFoundError('Generate all FEM references first: run.ps1 -Task fem')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,geom in zip(axs,('circle','rectangles')):
        xy=grid(301);e,_=material(xy,Problem(geom),'sharp')
        im=ax.imshow(e.reshape(301,301),extent=(0,1,0,1),origin='lower',cmap='cividis',interpolation='nearest')
        ax.set(title=geom,xlabel='x / L',ylabel='y / L');fig.colorbar(im,ax=ax,label='E / GPa; k has same numerical values')
    _save(fig,figdir/'geometry.png')
    fig,axs=plt.subplots(2,4,figsize=(13,6.8),layout='constrained')
    for row,geom in enumerate(('circle','rectangles')):
        pf=results/f'pinn_{geom}_coupled'/'prediction.npz'
        if not pf.exists():
            for ax in axs[row]:ax.set_axis_off()
            axs[row,0].text(.1,.5,geom+': training incomplete')
            continue
        pred=np.load(pf)['fields'];ref=np.load(results/f'fem_{geom}_n200'/'solution.npz')['fields']
        for pair,j in enumerate((5,0)):
            high=max(ref[:,j].max(),pred[:,j].max());low=min(ref[:,j].min(),pred[:,j].min())
            for k,a in enumerate((ref[:,j],pred[:,j])):
                ax=axs[row,pair*2+k]
                im=ax.imshow(a.reshape(101,101),extent=(0,1,0,1),origin='lower',cmap='viridis',vmin=low,vmax=high,interpolation='nearest')
                title=('FEniCS' if k==0 else 'mixed PINN')+' '+LABELS[j]
                if k:
                    error=100*np.linalg.norm(pred[:,j]-ref[:,j])/np.linalg.norm(ref[:,j])
                    title+=f' | L2 error {error:.2f}%'
                ax.set_title(title,fontsize=9);ax.set_xlabel('x / L')
                ax.set_ylabel(geom+'\ny / L' if pair==k==0 else 'y / L')
                fig.colorbar(im,ax=ax,fraction=.045,pad=.025).ax.tick_params(labelsize=7)
    fig.suptitle('FEniCS and mixed PINNs — primary benchmark fields\nCoupled training; 200×200 sharp-interface FEM reference; numerical benchmark units',fontsize=12)
    _save(fig,figdir/'overview.png')
    metrics={};figures=[]
    for geom in ('circle','rectangles'):
        ref_file=results/f'fem_{geom}_n200'/'solution.npz'
        if not ref_file.exists():continue
        ref=np.load(ref_file)['fields']
        fig,axs=plt.subplots(2,4,figsize=(14,6),layout='constrained')
        for j,ax in enumerate(axs.ravel()):
            y=np.linspace(0,1,101)
            ax.plot(y,ref.reshape(101,101,8)[:,50,j],color='#222',lw=1.5,label='FEM 200×200')
            ax.set(xlabel='y / L at x/L=0.5',title=LABELS[j])
        any_prediction=False
        for mode in ('coupled','sequential'):
            run=results/f'pinn_{geom}_{mode}';pf=run/'prediction.npz'
            if not pf.exists():continue
            any_prediction=True;data=np.load(pf);direct=data['fields'];pred=direct.copy()
            pred[:,2:5]=data['constitutive_stress'];pred[:,6:8]=data['fourier_flux']
            met=json.loads((run/'metrics.json').read_text());metrics[f'{geom}_{mode}']=met
            for indices,name in ((range(5),'mechanical'),(range(5,8),'thermal')):
                filename=f'{geom}_{mode}_{name}.png'
                fields_figure(pred,ref,indices,f'{geom.capitalize()} inclusion(s), {mode} training',figdir/filename)
                figures.append((f'{geom}, {mode}: {name} fields',filename))
            filename=f'{geom}_{mode}_direct_secondary.png'
            fields_figure(direct,ref,(2,3,4,6,7),f'{geom.capitalize()}, {mode}: direct-output audit',figdir/filename,'direct')
            figures.append((f'{geom}, {mode}: direct mixed stress and flux audit',filename))
            for j,ax in enumerate(axs.ravel()):
                ax.plot(y,pred.reshape(101,101,8)[:,50,j],color=COLORS[mode],lw=1.3,ls='--' if mode=='sequential' else '-',label=f'PINN {mode}')
        for ax in axs.ravel():ax.legend(fontsize=7)
        _save(fig,figdir/f'{geom}_sections.png')
        if any_prediction:figures.append((geom+' — independent centerline comparison',f'{geom}_sections.png'))
    # Convergence is evaluated on identical coordinates against the 200 mesh.
    conv=[]
    for geom in ('circle','rectangles'):
        refpath=results/f'fem_{geom}_n200'/'solution.npz'
        if not refpath.exists():continue
        ref=np.load(refpath)['fields']
        for n in (50,100):
            a=np.load(results/f'fem_{geom}_n{n}'/'solution.npz')['fields']
            for name,v in relative_errors(a,ref).items():conv.append({'geometry':geom,'n':n,'field':name,**v})
    if conv:
        with (results/'fem_convergence.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(conv[0]));w.writeheader();w.writerows(conv)
        fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,geom in zip(axs,('circle','rectangles')):
            for j in (0,2,5,6):
                a=[r for r in conv if r['geometry']==geom and r['field']==FIELDS[j]]
                ax.plot([r['n'] for r in a],[100*r['relative_L2'] for r in a],marker='o',label=LABELS[j])
            ax.set(xlabel='elements per edge',ylabel='relative L2 difference from 200×200 (%)',title=geom,yscale='log')
            ax.set_xticks((50,100));ax.legend()
        _save(fig,figdir/'fem_convergence.png')
    if metrics:
        fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained',sharey='row')
        for row,geom in enumerate(('circle','rectangles')):
            for col,group in enumerate(('direct_mixed_outputs','constitutive_outputs')):
                ax=axs[row,col]
                for offset,mode in ((-.18,'coupled'),(.18,'sequential')):
                    met=metrics.get(f'{geom}_{mode}')
                    if met:
                        err=met['comparisons']['FEM_n200'][group]
                        ax.bar(np.arange(8)+offset,[100*err[n]['relative_L2'] for n in FIELDS],width=.36,color=COLORS[mode],label=mode)
                ax.set_xticks(range(8),LABELS);ax.set(title=geom+' — '+('direct outputs' if col==0 else 'constitutive-derived'),ylabel='relative L2 error vs FEM 200×200 (%)')
                if ax.get_legend_handles_labels()[0]:ax.legend()
        _save(fig,figdir/'pinn_errors.png')
        fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,geom in zip(axs,('circle','rectangles')):
            for mode in ('coupled','sequential'):
                f=results/f'pinn_{geom}_{mode}'/'history.csv'
                if f.exists():
                    with f.open() as s: rows=list(csv.DictReader(s))
                    for field,ls in (('mechanical','-'),('thermal','--')):
                        good=[r for r in rows if float(r[field])>0]
                        ax.semilogy([float(r['elapsed_seconds']) for r in good],[float(r[field]) for r in good],color=COLORS[mode],ls=ls,label=mode+' '+field)
            ax.set(title=geom,xlabel='wall-clock seconds',ylabel='logged active objective')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        _save(fig,figdir/'loss_history.png')
    # Compare independent author FEAP data without fitting material constants.
    authors={}
    af=root/'data/authors_geometry2_fe.csv';f50=results/'fem_rectangles_n50/solution.npz'
    if af.exists() and f50.exists():
        a=np.genfromtxt(af,delimiter=',',names=True);names=a.dtype.names
        ref=np.column_stack([a[names[i]] for i in (0,1,3,4,6,2,15,16)])
        pred=np.load(f50)['fields'].reshape(101,101,8)[::2,::2].reshape(-1,8)
        authors=relative_errors(pred,ref)
        authors['displacement_fit_diagnostic_only']=float(np.sum(pred[:,:2]*ref[:,:2])/np.sum(pred[:,:2]**2))
        (results/'authors_feap_comparison.json').write_text(json.dumps(authors,indent=2))
    regularization={}
    for geom in ('circle','rectangles'):
        sharp=results/f'fem_{geom}_n100/solution.npz';smooth=results/f'fem_{geom}_n100_nodal/solution.npz'
        if sharp.exists() and smooth.exists():
            regularization[geom]=relative_errors(np.load(smooth)['fields'],np.load(sharp)['fields'])
    released={};released_html=''
    released_path=results/'authors_released_checkpoint/metrics.json'
    if released_path.exists():
        released=json.loads(released_path.read_text())
        released_rows=[]
        for field in FIELDS:
            errs=[100*released['comparisons']['FEM_n200'][group][field]['relative_L2']
                  for group in ('direct_mixed_outputs','constitutive_outputs')]
            released_rows.append('<tr><td>'+field+'</td>'+''.join(f'<td>{e:.3f}%</td>' for e in errs)+'</tr>')
        released_html='<h2>Authors’ released checkpoint: independent evaluation</h2><p>The public SciANN weights were ported without retraining. A raw-coordinate NumPy evaluation agrees with the converted PyTorch model within 7×10⁻¹⁶ on 101 test points. The original training history is unavailable. This model is separate from the four locally trained models above. Errors below use the same rectangular FEM200 reference and no amplitude fitting.</p><table><thead><tr><th>Field</th><th>Direct L2</th><th>Derived L2</th></tr></thead><tbody>'+''.join(released_rows)+'</tbody></table><p>See <a href="results/authors_released_checkpoint/metrics.json">complete released-model metrics</a> and <a href="results/authors_released_checkpoint/conversion_validation.json">conversion validation</a>.</p>'
    summary={'PINN':metrics,'FEM_convergence':conv,'material_regularization':regularization,
             'authors_FEAP':authors,'authors_released_checkpoint':released}
    (results/'summary.json').write_text(json.dumps(summary,indent=2))
    flattened=[]
    for name,met in metrics.items():
        for reference,groups in met['comparisons'].items():
            for representation,fields in groups.items():
                for field,values in fields.items():
                    flattened.append({'run':name,'reference':reference,'representation':representation,'field':field,**values})
    if flattened:
        with (results/'error_metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flattened[0]));w.writeheader();w.writerows(flattened)
    rows=[];md=['# Executed results','','Global relative L2 errors against the 200×200 sharp-interface FEniCS reference. Both stress/flux representations are shown.','',
               '| Geometry / training / representation | ux L2 | T L2 | sxx L2 | qx L2 |','|---|---:|---:|---:|---:|']
    for name,met in metrics.items():
        for group in ('direct_mixed_outputs','constitutive_outputs'):
            e=met['comparisons']['FEM_n200'][group]
            vals=[100*e[n]['relative_L2'] for n in ('ux','T','sxx','qx')]
            label=name.replace('_',' / ')+' / '+('direct' if group.startswith('direct') else 'derived')
            rows.append('<tr><td>'+label+'</td>'+''.join(f'<td>{v:.3f}%</td>' for v in vals)+'</tr>')
            md.append('| '+label+' | '+' | '.join(f'{v:.3f}%' for v in vals)+' |')
    if released:
        md+=['','The authors’ released SciANN checkpoint was also ported and evaluated without retraining. Its derived errors vs rectangular FEM200 are: '+
             ', '.join(f"{field} {100*released['comparisons']['FEM_n200']['constitutive_outputs'][field]['relative_L2']:.3f}%" for field in ('ux','T','sxx','qx'))+'.',
             'All eight released-model fields, including less accurate shear/transverse components, are reported separately in REPORT.html and results/authors_released_checkpoint/metrics.json.']
    md+=['','Full eight-field errors and independent-grid residuals: `results/summary.json`.','',
         'Main runs use hard Dirichlet constraints and a consistent global balance loss. They are an accelerated reproduction, not a rerun of all 100k/200k Adam studies.',
         'Manuscript energy expressions and original Adam schedules, with corrected constitutive/roller-boundary conventions, are available in `run.ps1 -Task paper`.',
         'Circle center (0.60, 0.35) is inferred from Figure 5. Rectangles match the authors’ material data exactly.',
         'PINN uses the authors-style nodal material interpolation; sharp FEM errors include material regularization. Separate `FEM_n100_nodal` metrics compare the same coefficient field.',
         'Author FEAP displacement is approximately 0.98 times the independently assembled solution. No amplitude correction is applied.']
    (root/'RESULTS.md').write_text('\n'.join(md),encoding='utf-8')
    gallery=''
    for title,name in figures:
        block=f'<img loading="lazy" src="figures/{name}" alt="{html.escape(title)}">'
        if 'direct_secondary' in name:block=f'<details><summary>{html.escape(title)}</summary>{block}</details>'
        else:block=f'<h2>{html.escape(title)}</h2>{block}'
        gallery+='<section>'+block+'</section>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harandi thermoelasticity reproduction</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 20px;color:#202b32;background:#fff}h1{line-height:1.15}h2{font-size:1.25rem;margin-top:2rem}a{color:#156082}img{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #ccd4da;padding:10px;text-align:right}td:first-child,th:first-child{text-align:left}.note{border-left:4px solid #a34920;background:#fbf7ef;padding:16px}code{background:#edf1f4;padding:2px 5px}section{break-inside:avoid}summary{cursor:pointer}@media(max-width:650px){body{margin:20px auto;padding:0 12px;font-size:14px}table{font-size:11px}td,th{padding:5px}}@media print{body{max-width:none}section{page-break-before:auto}}</style>
<h1>Mixed PINNs and FEniCS for heterogeneous thermoelasticity</h1>
<p>Executed local reproduction · 3 November 2026 · Harandi et al., <a href="https://doi.org/10.1002/nme.7388">DOI 10.1002/nme.7388</a></p>
<p>Eight independent 5×40 tanh networks, first-order mixed residuals, exact Dirichlet transforms, coupled and sequential training. Thermal and displacement FEM solutions use FEniCS/DOLFIN Q1 quadrilaterals; original cell fluxes and stresses are retained alongside recovered nodal fields.</p>
<div class="note"><b>Scope and scientific limits.</b> Main trained models use an accelerated Adam + L-BFGS schedule and a consistent global balance objective. They do not recreate every original 100k/200k learning curve or architecture sweep. Manuscript/author energy variants retain explicitly corrected constitutive and roller-boundary conventions. PINN coefficients interpolate the authors-style 50×50 material grid; the sharp FEM comparison therefore includes material regularization error. Separate matched-coefficient FEM comparisons are saved in the numerical files. Verified rectangular training uses 71×71 points. The circular center is inferred from the paper figure. The supplied FEAP data has a 0.98 displacement amplitude factor; results here are not fitted to remove it. See README for conventions and source provenance.</div>
<h2>Measured errors against the refined FEM reference</h2><div style="overflow-x:auto"><table><thead><tr><th>Geometry / training</th><th>ux L2</th><th>T L2</th><th>sxx L2</th><th>qx L2</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<p>These are global relative L2 errors, not pointwise relative errors. Small transverse fields can have larger relative errors. Main maps follow the authors’ plotting convention: stress and flux derived from displacement/temperature gradients. The direct mixed-output maps are available in expandable audits below. Complete eight-field errors, matched-coefficient comparisons and independent-grid residuals are in <a href="results/summary.json">summary.json</a> and <a href="results/error_metrics.csv">error_metrics.csv</a>.</p>
<img src="figures/overview.png" alt="FEniCS and coupled mixed-PINN temperature and displacement fields, with measured relative errors">
<h2>Benchmarks</h2><img src="figures/geometry.png" alt="Material moduli in a circular inclusion and four rectangular inclusions">
<h2>FEM refinement</h2><img src="figures/fem_convergence.png" alt="FEM differences for meshes with 50 and 100 elements per edge against the 200-element reference">
<h2>Training and error comparison</h2><img src="figures/pinn_errors.png" alt="Eight-field PINN errors compared with refined FEM"><img src="figures/loss_history.png" alt="Mechanical and thermal losses over elapsed training time">'''+released_html+gallery+'''
<h2>Reproduce and inspect</h2><p>Run <code>.\run.ps1 -Task test</code>, <code>.\run.ps1 -Task verified</code>, or <code>.\run.ps1 -Task paper</code>. Checkpoints, raw predictions, CSV histories and FEniCS XDMF/HDF5 files are in the results directory. No FEM labels are used to train the main physics-only models.</p>
<p>Sources: <a href="https://github.com/phyml4e/PINNs/tree/main/Thermomechanic">authors’ code and data</a>; <a href="https://fenicsproject.org/download/archive/">FEniCS installation</a>. Reproduction code and limitations: <a href="README.md">README.md</a>.</p></html>'''
    (root/'REPORT.html').write_text(page,encoding='utf-8')
    print(f'Wrote REPORT.html, RESULTS.md and {len(list(figdir.glob("*.png")))} figures; {len(metrics)} main trained models found.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).parents[1])
    generate(parser.parse_args().root)
