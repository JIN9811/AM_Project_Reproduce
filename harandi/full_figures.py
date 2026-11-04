"""Evidence-only scientific figures for the complete Harandi study catalogue.

Default generation requires every catalogue job and reference. ``--partial`` is
an explicit progress mode: incomplete jobs never contribute prediction curves.
Plots read NPZ/CSV/JSON files; catalogue completion delegates to its checkpoint
validator (CPU schema checking). This module never trains or requests GPU work.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Circle

from .full_study import catalogue, job_complete
from .physics import FIELDS, Problem, grid, material, RECTANGLES
from .paper_figures import paper_cmap, scale_audit, SYMBOLS, UNITS

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (1, 2, 4, 5, 6)
WIDTHS = (5, 10, 20, 40, 60)
TERMS = ('equilibrium', 'energy', 'connection', 'dirichlet', 'neumann')
COLORS = ('#0072BD', '#D95319', '#008000', '#FF0000', '#8B42D2')
LIMITS_PARAM = {
    'ux': {'pinn': [-.0002, .4709], 'fem': [-.0002, .4709], 'error': [0, .0387]},
    'sxx': {'pinn': [-1.257, -.102], 'fem': [-1.257, -.102], 'error': [0, .1831]},
    'T': {'pinn': [.0010, .9988], 'fem': [.0010, .9988], 'error': [0, .0268]},
    'qx': {'pinn': [.049, 1.299], 'fem': [.049, 1.299], 'error': [0, .2787]},
}
LIMITS_PARAM_DATA_ERROR = {'ux': .0969, 'sxx': .4214, 'T': .058, 'qx': .2573}


def _json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prediction(path):
    """Keep the independent mixed heads and constitutive-derived fields apart."""
    with np.load(path, allow_pickle=False) as data:
        xy, direct = data['xy'].copy(), data['fields'].copy()
        if direct.shape != (len(xy), 8) or xy.shape != (len(xy), 2):
            raise ValueError(f'Expected N×2 coordinates and N×8 fields: {path}')
        if not np.isfinite(xy).all() or not np.isfinite(direct).all():
            raise ValueError(f'Nonfinite prediction data: {path}')
        derived = direct.copy()
        has_stress, has_flux = 'constitutive_stress' in data, 'fourier_flux' in data
        if has_stress != has_flux:
            raise ValueError(f'Incomplete constitutive export: {path}')
        if has_stress:
            derived[:, 2:5] = data['constitutive_stress']
            derived[:, 6:8] = data['fourier_flux']
        if not np.isfinite(derived).all():
            raise ValueError(f'Nonfinite constitutive fields: {path}')
    return {'xy': xy, 'direct': direct, 'derived': derived,
            'derived_available': has_stress, 'source': str(path), 'source_sha256': _hash(path)}


def align_reference(xy, reference_xy, reference_fields):
    """Reorder identical coordinate sets; never silently interpolate FEM data."""
    xy, reference_xy = np.asarray(xy), np.asarray(reference_xy)
    if xy.shape != reference_xy.shape:
        raise ValueError('Prediction/reference coordinates have different shapes')
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    ref_order = np.lexsort((reference_xy[:, 0], reference_xy[:, 1]))
    if not np.allclose(xy[order], reference_xy[ref_order], atol=1e-12, rtol=0):
        raise ValueError('Prediction/reference coordinates differ; interpolation not authorized')
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('Duplicate prediction coordinates')
    aligned = np.empty_like(reference_fields)
    aligned[order] = np.asarray(reference_fields)[ref_order]
    return aligned


def field_metrics(prediction, reference):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.shape[1] != 8:
        raise ValueError('Metrics need matching N×8 arrays')
    result = {}
    for j, name in enumerate(FIELDS):
        error = prediction[:, j] - reference[:, j]
        peak = float(np.max(np.abs(reference[:, j])))
        norm = float(np.linalg.norm(reference[:, j]))
        mean, maximum = float(np.mean(np.abs(error))), float(np.max(np.abs(error)))
        result[name] = {'mean_absolute_error': mean, 'max_absolute_error': maximum,
                        'root_mean_squared_error': float(np.sqrt(np.mean(error**2))),
                        'relative_l2': float(np.linalg.norm(error)/norm) if norm else None,
                        'mean_absolute_error_over_reference_peak_percent': 100*mean/peak if peak else None,
                        'max_absolute_error_over_reference_peak_percent': 100*maximum/peak if peak else None,
                        'normalization_reference_peak': peak, 'sample_count': len(error),
                        'prediction_min': float(prediction[:, j].min()),
                        'prediction_max': float(prediction[:, j].max()), 'unit': UNITS[j]}
    return result


def read_history(path):
    """Preserve raw decimal text and blank cells alongside parsed observations."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8') as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for entry in raw:
        row = {}
        for key, value in entry.items():
            if value is None or value == '':
                row[key] = None
                continue
            try:
                number = float(value)
                row[key] = number if np.isfinite(number) else None
            except ValueError:
                row[key] = value
        rows.append(row)
    return {'source': str(path), 'source_sha256': _hash(path), 'raw_rows': raw, 'rows': rows,
            'policy': 'Original CSV untouched; raw strings retained; no rounding, interpolation, smoothing or missing-loss filling.'}


def timing_measurement(history, environment):
    first = next((r for r in history['rows'] if r.get('optimizer') == 'lbfgs' and r.get('step') == 100), None)
    elapsed = first.get('elapsed_seconds') if first else None
    return {'iterations': 100, 'elapsed_seconds_first_100': elapsed,
            'seconds_per_iteration_first_100': elapsed/100 if elapsed is not None else None,
            'closure_evaluations_first_100': first.get('closure_evaluations') if first else None,
            'device': environment.get('device'), 'GPU': environment.get('GPU'),
            'platform': environment.get('platform'),
            'measurement_kind': 'Cumulative wall time through update100 /100; includes setup, closures, logging and checkpoint overhead; not isolated GPU kernel time.',
            'available': elapsed is not None,
            'status': 'measured' if elapsed is not None else 'missing exact update100 timing; not interpolated'}


def _regular_grid(xy, fields):
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if len(x)*len(y) != len(xy):
        raise ValueError('Field visualization requires a complete tensor grid')
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    return x, y, fields[order].reshape(len(y), len(x), fields.shape[-1])


def extract_section(xy, fields, x_value):
    x, y, values = _regular_grid(xy, fields)
    matches = np.flatnonzero(np.isclose(x, x_value, atol=1e-12, rtol=0))
    if len(matches):
        section = values[:, matches[0], :]
        method = 'exact grid line'
    else:
        if not x[0] <= x_value <= x[-1]:
            raise ValueError('Section outside coordinate domain')
        section = np.array([[np.interp(x_value, x, row[:, j]) for j in range(8)] for row in values])
        method = 'linear interpolation in x'
    return y, section, {'x': x_value, 'method': method}


class Writer:
    def __init__(self, root, partial):
        self.root, self.partial = Path(root), partial
        self.out = self.root/'figures/full_study'
        self.out.mkdir(parents=True, exist_ok=True)
        self.items, self.audit = [], []
        self.pdf = PdfPages(self.root/'FULL_STUDY_FIGURES.pdf')
        self.cmap = paper_cmap(self.root)

    def save(self, fig, name, title, note, source=None):
        # Keep an explicit footer below axes; no tight-bbox cropping of captions.
        mode = 'PARTIAL PROGRESS: completed jobs only' if self.partial else 'FULL CATALOGUE: verified completed jobs'
        fig.text(.055, .018, mode+' | '+note, fontsize=7, va='bottom', wrap=True)
        fig.savefig(self.out/(name+'.png'), dpi=170, facecolor='white')
        fig.savefig(self.out/(name+'.svg'), facecolor='white')
        self.pdf.savefig(fig, facecolor='white')
        self.items.append({'id': name, 'title': title, 'note': note, 'source_figure': source,
                           'png': f'figures/full_study/{name}.png', 'svg': f'figures/full_study/{name}.svg',
                           'status': 'completed-data-only', 'reused': False})
        plt.close(fig)

    def close(self):
        self.pdf.close()

    def image(self, fig, ax, xy, values, bounds, name, job, field, panel):
        x, y, z = _regular_grid(xy, values[:, None])
        audit = scale_audit(values, bounds)
        im = ax.imshow(z[:, :, 0], origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
                       interpolation='nearest', cmap=self.cmap, vmin=bounds[0], vmax=bounds[1], aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        bar = fig.colorbar(im, ax=ax, fraction=.048, pad=.04, extend=audit['extend'], extendfrac=.035)
        bar.ax.tick_params(labelsize=7, length=0)
        bar.outline.set_linewidth(.45)
        self.audit.append({'figure': name, 'job': job, 'field': field, 'panel': panel, **audit})


def _finite_bounds(values, zero=False):
    lo, hi = (0., float(np.max(values))) if zero else (float(np.min(values)), float(np.max(values)))
    if lo == hi:
        delta = max(abs(lo)*.01, 1e-12)
        lo, hi = (0., delta) if zero else (lo-delta, hi+delta)
    return [lo, hi]


def _field_panels(writer, name, job_id, dataset, reference, indices, source, limits=None, title=None):
    height=2.18*len(indices)+.65
    fig, axes = plt.subplots(len(indices), 3, figsize=(8.0, height), squeeze=False)
    fig.subplots_adjust(left=.055, right=.96, top=1-.65/height, bottom=.095, wspace=.33, hspace=.28)
    pred = dataset['derived']
    for row, j in enumerate(indices):
        values = (pred[:, j], reference[:, j], np.abs(pred[:, j]-reference[:, j]))
        for col, (role, val) in enumerate(zip(('pinn', 'fem', 'error'), values)):
            bound = limits[FIELDS[j]][role] if limits and FIELDS[j] in limits else _finite_bounds(val, role=='error')
            writer.image(fig, axes[row,col], dataset['xy'], val, bound, name, job_id, FIELDS[j], role)
            axes[row,col].set_title(f'{SYMBOLS[j]}  {("Prediction", "FEM50", "Absolute error")[col]} [{UNITS[j]}]', fontsize=9)
    fig.suptitle(title or job_id, fontsize=11, y=1-.10/height)
    writer.save(fig, name, title or job_id,
                'Derived stress/flux; exact saved values. Printed limits where available; triangles disclose saturation.', source)


def _sections(writer, name, entries, x_value, source, note):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    fig.subplots_adjust(left=.09, right=.98, bottom=.14, top=.94, hspace=.34, wspace=.29)
    methods = []
    for dataset, label, color, style in entries:
        y, section, info = extract_section(dataset['xy'], dataset['derived'], x_value)
        methods.append({'label': label, **info})
        for ax, j in zip(axes.ravel(), (0,2,5,6)):
            ax.plot(y, section[:,j], color=color, ls=style, lw=1, label=label)
            ax.set(title=SYMBOLS[j]+f' at x={x_value:g}', xlabel='y [mm]', ylabel=SYMBOLS[j]+f' [{UNITS[j]}]')
            ax.legend(fontsize=6.8, framealpha=.9)
    writer.save(fig, name, f'Figure {source}: section x={x_value:g}', note, source)
    return methods


def _history_updates(rows):
    offset = max([r.get('step', 0) for r in rows if r.get('optimizer')=='adam'] or [0])
    return np.array([r.get('completed_updates', r.get('step', 0)) +
                     (offset if r.get('optimizer')=='lbfgs' else 0) for r in rows])


def _losses(writer, name, history, config, source):
    rows = history['rows']
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0))
    fig.subplots_adjust(left=.10, right=.97, top=.94, bottom=.14, hspace=.32)
    x = _history_updates(rows)
    for ax, prefix, active in zip(axes, ('T','M'), ('thermal','mechanical')):
        for term, color in zip(TERMS, COLORS):
            val = np.array([r.get(prefix+'_'+term) if r.get(prefix+'_'+term) is not None and
                            r.get('active','coupled') in ('coupled',active) else np.nan for r in rows], dtype=float)
            val[val <= 0] = np.nan
            if np.isfinite(val).any():
                # Separate active blocks prevents connecting across unobserved frozen-field intervals.
                cuts = [0]+[i for i in range(1,len(rows)) if rows[i].get('active')!=rows[i-1].get('active')]+[len(rows)]
                labeled = False
                for a,b in zip(cuts[:-1],cuts[1:]):
                    if np.isfinite(val[a:b]).any():
                        ax.plot(x[a:b], val[a:b], lw=.9, color=color, label=term if not labeled else None)
                        labeled=True
        ax.set(yscale='log', xlabel='Actual optimizer updates', ylabel='Loss', title=active.capitalize())
        if ax.lines: ax.legend(fontsize=8)
        if config.get('mode')=='sequential':
            switch=config.get('switch',20000)
            for boundary in range(switch,int(x[-1])+1,switch): ax.axvline(boundary,color='.8',lw=.5)
    writer.save(fig,name,f'Figure {source}: recorded objective terms',
                'Original logged samples only. Zero terms omitted on log axes; inactive terms are not filled or bridged.',source)


def _error_grid(writer, name, entries, source, indices=(0,2,5,6), note=''):
    fig, axes = plt.subplots(len(indices), len(entries), figsize=(2.5*len(entries)+.6, 2.25*len(indices)+.6), squeeze=False)
    fig.subplots_adjust(left=.055,right=.96,top=.94,bottom=.09,wspace=.30,hspace=.28)
    for row,j in enumerate(indices):
        maximum=max(float(np.abs(d['derived'][:,j]-ref[:,j]).max()) for _,d,ref in entries)
        bounds=_finite_bounds(np.array([0.,maximum]),zero=True)
        for col,(label,d,ref) in enumerate(entries):
            val=np.abs(d['derived'][:,j]-ref[:,j])
            writer.image(fig,axes[row,col],d['xy'],val,bounds,name,label,FIELDS[j],'absolute_error')
            axes[row,col].set_title(f'{label}\n{SYMBOLS[j]} error [{UNITS[j]}]',fontsize=8)
    writer.save(fig,name,f'Figure {source}: error-field comparison',note+' Shared per-field limits across cases; derived fields.',source)


def _boundary_optimizer_panels(writer, entries):
    name='fig17_boundary_optimizer'
    fig,axes=plt.subplots(3,6,figsize=(15,8.4))
    fig.subplots_adjust(left=.035,right=.975,top=.92,bottom=.10,wspace=.34,hspace=.34)
    labels=['Soft Adam100k','Hard Adam10k','Hard L-BFGS10k']
    for row,j in enumerate((0,1,5)):
        field_bounds=_finite_bounds(np.concatenate([d['derived'][:,j] for _,d,_ in entries]))
        error_bounds=_finite_bounds(np.concatenate([np.abs(d['derived'][:,j]-ref[:,j]) for _,d,ref in entries]),True)
        for col,(job,dataset,reference) in enumerate(entries):
            for side,values,bounds,role in [(0,dataset['derived'][:,j],field_bounds,'prediction'),
                                           (1,np.abs(dataset['derived'][:,j]-reference[:,j]),error_bounds,'absolute_error')]:
                ax=axes[row,2*col+side]
                writer.image(fig,ax,dataset['xy'],values,bounds,name,job,FIELDS[j],role)
                ax.set_title(labels[col]+'\n'+SYMBOLS[j]+(' error' if side else '')+f' [{UNITS[j]}]',fontsize=8)
    writer.save(fig,name,'Figure17: boundary enforcement and optimizer comparison',
                'Field/error pairs. Soft Adam100k versus hard Adam10k and hard L-BFGS10k; unequal budgets disclosed.',17)


def _collocation(config):
    """Deterministic reconstruction of make_batch's saved configuration."""
    xy=grid(config.get('n',51));seed=config.get('seed',42)
    if config['geometry']=='circle' and config.get('n',51)>=51 and len(xy)<5000:
        xy=np.r_[xy,np.random.default_rng(seed).random((5000-len(xy),2))]
    count=config.get('left_refine',0)
    if count:
        xy=np.r_[xy,np.c_[np.zeros(count),np.random.default_rng(seed+91).uniform(0,1,count)]]
    return xy


def _points_and_deformation(writer,geometry,config,dataset,reference):
    p=Problem(geometry);xy=_collocation(config);values,_=material(xy,p,'nodal')
    evaluation=dataset['xy'];evalvalues,_=material(evaluation,p,'nodal')
    source=5 if geometry=='circle' else 12
    fig,axes=plt.subplots(1,3,figsize=(10,4.1));fig.subplots_adjust(left=.05,right=.98,top=.88,bottom=.18,wspace=.22)
    for ax,coords,col,title in [(axes[0],xy,values,f'Train: {len(xy):,} points'),
                               (axes[1],evaluation,evalvalues,f'Evaluate: {len(evaluation):,} points')]:
        ax.scatter(coords[:,0],coords[:,1],c=col,cmap=writer.cmap,s=.8,linewidths=0)
        ax.set(title=title,xlim=(0,1),ylim=(0,1),aspect='equal')
    nodes=np.linspace(0,1,51);mid=(np.arange(50)+.5)/50;x,y=np.meshgrid(mid,mid)
    phase,_=material(np.c_[x.ravel(),y.ravel()],p,'sharp')
    axes[2].pcolormesh(nodes,nodes,phase.reshape(50,50),cmap=writer.cmap,edgecolors='k',linewidth=.12,rasterized=True)
    axes[2].set(title='FEM50 material mesh',aspect='equal')
    writer.save(fig,f'fig{source:02d}_{geometry}_points_mesh',f'Figure{source}: training/evaluation/FEM points',
                'Train points reconstructed deterministically from saved config and sampler; evaluation coordinates read from prediction NPZ.',source)
    if geometry=='circle':
        fig,axes=plt.subplots(1,3,figsize=(10,4.1));fig.subplots_adjust(left=.05,right=.98,top=.88,bottom=.18,wspace=.22)
        for ax,coords,title in [(axes[0],evaluation,'Undeformed'),(axes[1],evaluation+dataset['derived'][:,:2],'PINN deformation'),
                               (axes[2],evaluation+reference[:,:2],'FEM50 deformation')]:
            ax.scatter(coords[:,0],coords[:,1],c=evalvalues,cmap=writer.cmap,s=.8,linewidths=0)
            ax.set(title=title,aspect='equal')
        writer.save(fig,'fig07_circle_deformation','Figure7: measured deformation',
                    'Displacement factor1; no fitted amplitude or coordinate clipping. Artificial benchmark alpha=1.',7)


def _refinement_panels(writer, entries, config):
    fig,axes=plt.subplots(2,3,figsize=(10.5,7.2));fig.subplots_adjust(left=.05,right=.96,top=.92,bottom=.12,wspace=.32,hspace=.32)
    for row,(name,dataset,reference) in enumerate(entries):
        local=dict(config,left_refine=0 if row==0 else config.get('left_refine',1000))
        xy=_collocation(local);value,_=material(xy,Problem('rectangles'),'nodal')
        axes[row,0].scatter(xy[:,0],xy[:,1],c=value,cmap=writer.cmap,s=1,linewidths=0)
        axes[row,0].set(title=f'{"Baseline" if row==0 else "Left refined"}: {len(xy)} points',xlim=(-.02,1),ylim=(0,1),aspect='equal')
        for col,j in enumerate((0,5),1):
            maximum=max(float(np.abs(d['derived'][:,j]-ref[:,j]).max()) for _,d,ref in entries)
            error=np.abs(dataset['derived'][:,j]-reference[:,j])
            writer.image(fig,axes[row,col],dataset['xy'],error,_finite_bounds(np.array([0,maximum]),True),
                         'fig15_left_refinement',name,FIELDS[j],'absolute_error')
            axes[row,col].set_title(SYMBOLS[j]+f' error [{UNITS[j]}]')
    writer.save(fig,'fig15_left_refinement','Figure15: left-boundary refinement',
                'Same deterministic sampler;1000left-edge points added. Shared error scales; actual outcomes may improve or deteriorate.',15)


def _remark_comparison(writer, report, datasets, reference):
    ids=['mixed_rectangles_coupled','remark_rectangles_adam1m','arch_l5_w40']
    rows=[];timings={}
    for name in ids:
        job=report['jobs'][name];history=job['history']['rows']
        elapsed=history[-1].get('elapsed_seconds') if history else None
        config=job['config'];factor=2 if config.get('mode')=='sequential' else 1
        updates=factor*(config.get('adam_steps',0)+config.get('lbfgs_steps',0))
        timings[name]={'actual_scheduled_updates':updates,'cumulative_training_wall_seconds':elapsed,
            'device':job['environment'].get('device'),'GPU':job['environment'].get('GPU'),
            'source':'last original history row; continuation keeps cumulative history',
            'not_comparable_as_identical_hardware_to':'Paper Apple M2 Pro timings'}
        rows.append([name,f'{updates:,}',f'{elapsed:.4g}' if elapsed is not None else 'unavailable',
                     job['environment'].get('GPU') or job['environment'].get('device') or 'unknown'])
    fig,ax=plt.subplots(figsize=(12,4.5));ax.axis('off');fig.subplots_adjust(left=.04,right=.96,top=.82,bottom=.25)
    table=ax.table(cellText=rows,colLabels=['Case','Updates','Wall seconds','Device'],cellLoc='left',loc='center',colWidths=[.38,.12,.14,.36])
    table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2)
    ax.set_title('Remark7: separately measured100k /1M Adam and10k hard L-BFGS',pad=22)
    writer.save(fig,'remark07_runtime','Remark7: observed cumulative runtime',
                '1M continues the100k checkpoint; original100k remains unchanged. Different hardware from paper; wall-time overhead included.','Remark7')
    _error_grid(writer,'remark07_error_fields',[(name,datasets[name],reference(name,'rectangles_n50')) for name in ids],
                'Remark7',note='Separate1M continuation versus100k baseline and10k hard L-BFGS; main baseline figures still100k.')
    report['remark7_timing']=timings


def _sweep(writer, name, ids, xvalues, jobs, label, source, representation):
    fig, axes = plt.subplots(4,2,figsize=(10.0,11.0))
    fig.subplots_adjust(left=.09,right=.98,top=.95,bottom=.085,hspace=.48,wspace=.30)
    for ax,field,symbol,unit in zip(axes.ravel(),FIELDS,SYMBOLS,UNITS):
        for key,lab,marker in [('mean_absolute_error','Mean absolute','o'),('max_absolute_error','Maximum absolute','s')]:
            y=[jobs[j]['comparisons']['FEM_n50'][representation][field][key] for j in ids]
            ax.plot(xvalues,y,marker=marker,lw=1,label=lab)
        ax.set(title=symbol+f' [{unit}]',xlabel=label,ylabel='Absolute error',xticks=xvalues)
        ax.legend(fontsize=7)
        if all(v>0 for line in ax.lines for v in line.get_ydata()): ax.set_yscale('log')
    fig.suptitle(f'Figure {source}: {representation} outputs, FEM50 reference',fontsize=12)
    writer.save(fig,name,f'Figure {source}: {label}, {representation}',
                'All eight fields. Maximum and mean measured errors; no assumption of monotonic improvement.',source)


def _timing_figure(writer, jobs):
    values=[];devices=set()
    for layer in DEPTHS:
        row=[]
        for width in WIDTHS:
            item=jobs.get(f'timing_l{layer}_w{width}',{}).get('timing',{})
            val=item.get('seconds_per_iteration_first_100')
            row.append('pending' if val is None else f'{val:.6g}')
            if item.get('device'): devices.add(f"{item['device']} / {item.get('GPU') or item.get('platform') or 'unspecified hardware'}")
        values.append(row)
    fig,ax=plt.subplots(figsize=(10,4.5));ax.axis('off')
    fig.subplots_adjust(left=.1,right=.95,top=.85,bottom=.24)
    tab=ax.table(cellText=values,rowLabels=[str(v) for v in DEPTHS],colLabels=[str(v) for v in WIDTHS],loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.9)
    ax.set_title('Table 4 reproduction: seconds / iteration through first100 L-BFGS updates\nRows: hidden layers; columns: neurons per layer',pad=18)
    writer.save(fig,'table04_architecture_timing','Table4: measured architecture timing',
                'Wall time, not isolated GPU time. '+('; '.join(sorted(devices)) or 'No completed timing jobs yet')+'.',4)


def _setup_figures(writer):
    # Recompute geometry from the shared Problem implementation, rather than
    # reuse old result figures whose captions describe different energy/BCs.
    fig,axes=plt.subplots(1,2,figsize=(8,4.5))
    fig.subplots_adjust(left=.07,right=.94,bottom=.20,top=.90,wspace=.28)
    xy=grid(101)
    for ax,geom in zip(axes,('circle','rectangles')):
        values,_=material(xy,Problem(geom),'nodal')
        writer.image(fig,ax,xy,values,_finite_bounds(values),'fig02_geometry','setup',geom,'Young modulus')
        ax.set_title(f'{geom}: E [GPa]');ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
    writer.save(fig,'fig02_geometry','Figure2: implemented material maps',
                'Recomputed from shared geometry code; alpha=1,T0=0 and origins are reconstruction choices. Paper unit-label conflict disclosed.',2)
    fig,ax=plt.subplots(figsize=(9.3,4.8));ax.axis('off')
    ax.text(.05,.83,'Mechanical: ux=0 at x=0,1; uy=0 at y=0,1',fontsize=12)
    ax.text(.05,.68,'Thermal: T=1 left, T=0 right; qy=0 top and bottom',fontsize=12)
    ax.text(.05,.50,'Soft baseline: boundary penalties in the loss',fontsize=12)
    ax.text(.05,.35,r'Hard cases: $\hat u_x=x(x-1)u_x$, $\hat u_y=y(y-1)u_y$',fontsize=12)
    ax.text(.05,.20,r'$\hat T=x(x-1)T+1-x$',fontsize=12)
    writer.save(fig,'fig03_boundary_conditions','Figure3: boundary conditions',
                'Baseline comparator penalty choices and printed mechanical-Neumann inconsistencies are documented in the protocol.',3)
    fig,axes=plt.subplots(1,3,figsize=(11,4.7));fig.subplots_adjust(left=.03,right=.98,top=.87,bottom=.18,wspace=.23)
    panels=[('Sequential','Thermal20k\nMechanical frozen\n↓\nMechanical20k\nThermal frozen\n↓\nFive cycles;200k total'),
            ('Coupled','Thermal + mechanical loss\n↓\nAll networks update\n↓\n100k Adam baseline'),
            ('Mixed architecture','Eight separate scalar networks\nDefault5 hidden layers×40tanh\n↓\nux,uy,sxx,syy,sxy,T,qx,qy\n↓\nFirst-derivative physical losses')]
    for ax,(title,text) in zip(axes,panels):
        ax.axis('off');ax.set_title(title);ax.text(.5,.5,text,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=1',fc='#FFF0DD',ec='.6'))
    writer.save(fig,'fig01_04_training_architecture','Figures1/4: implemented training and architecture',
                'New schematic. Baseline soft BCs and authors-energy variant; hard/L-BFGS and transfer cases have separate configurations.','1,4')
    fig,ax=plt.subplots(figsize=(10,4.5));ax.axis('off')
    ax.text(.05,.75,'Paired material ratios (1,1) → … → (10,10)',fontsize=14)
    ax.text(.05,.53,'10,000 Adam updates per ratio; transfer network weights',fontsize=12)
    ax.text(.05,.32,'Evaluate held-out (15,15) without further training',fontsize=12)
    ax.text(.05,.13,'w = 0, 0.1, 0.5, 1; 0.5 is an explicit reconstruction choice',fontsize=12)
    writer.save(fig,'fig22_transfer_protocol','Figure22: transfer-learning protocol',
                'Global log10 ratios, BC enforcement and normalization are implementation choices; Fig22 defines matrix/inclusion ratios.',22)


def _boundary_metrics(dataset):
    xy,values=dataset['xy'],dataset['direct']
    left=np.isclose(xy[:,0],0);right=np.isclose(xy[:,0],1)
    top=np.isclose(xy[:,1],1);bottom=np.isclose(xy[:,1],0)
    return {'left_max_abs_ux':float(np.abs(values[left,0]).max()),
            'left_max_abs_T_minus1':float(np.abs(values[left,5]-1).max()),
            'max_normal_displacement_error':float(max(np.abs(values[left|right,0]).max(),np.abs(values[top|bottom,1]).max())),
            'max_temperature_dirichlet_error':float(max(np.abs(values[left,5]-1).max(),np.abs(values[right,5]).max()))}


def _comparison_summary(jobs, left, right, key='FEM_n50'):
    """Observed deltas, including unfavorable ones; never encode expected wins."""
    if left not in jobs or right not in jobs:
        return {'status':'pending','required_jobs':[left,right]}
    if key not in jobs[left]['comparisons'] or key not in jobs[right]['comparisons']:
        return {'status':'reference pending','reference':key}
    differences={}
    for representation in ('direct','derived'):
        differences[representation]={}
        for field in FIELDS:
            a=jobs[left]['comparisons'][key][representation][field]
            b=jobs[right]['comparisons'][key][representation][field]
            differences[representation][field]={}
            for metric in ('mean_absolute_error','max_absolute_error'):
                differences[representation][field][metric]={'left':a[metric],'right':b[metric],
                    'left_minus_right':a[metric]-b[metric],
                    'left_over_right':a[metric]/b[metric] if b[metric] else None,
                    'lower_error':'left' if a[metric]<b[metric] else 'right' if a[metric]>b[metric] else 'equal'}
    return {'status':'measured','left':left,'right':right,'reference':key,'differences':differences}


def _write_gallery(root, report):
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
      '<title>Harandi 전체 연구 재현</title><style>body{max-width:1160px;margin:40px auto;padding:0 22px;font:16px/1.7 "Malgun Gothic",sans-serif;color:#20252b}h1,h2{line-height:1.4}img{display:block;max-width:100%;height:auto;margin:auto}section{margin-top:32px;border-top:1px solid #ccc;padding-top:15px}.status{padding:16px;background:#fff1d5}code{overflow-wrap:anywhere}td,th{padding:6px 12px;border-bottom:1px solid #ddd;text-align:left}a{color:#165d99}details{margin:18px 0}</style>',
      '<h1>Harandi 전체 실험군 재현 결과</h1>',
      f'<p class="status"><strong>{"부분 진행 보고서" if report["status"]=="partial" else "등록된 전체 작업 완료"}</strong> · 완료 {len(report["completed_jobs"])} / {report["catalogue_jobs"]} 작업. 미완료 작업의 예측은 그림에 포함하지 않았습니다.</p>',
      '<p>그림은 저장된 수치 결과만 사용합니다. 응력·열유속 그림은 구성식으로 계산한 값을 사용하며, JSON에는 직접 출력과 구성식 출력의 모든 8개 장 오차를 각각 보존합니다. 색 막대 삼각형은 원문 색 범위를 벗어난 값을 나타냅니다.</p>',
      '<p>비교 기준은 본 프로젝트의 FEM50 및 FEM200입니다. 매개변수 학습은 비율15의 별도 FEM50과만 비교합니다. 논문 수치와의 일치 또는 혼합 PINN의 우수성을 가정하지 않습니다.</p>',
      '<p><a href="FULL_STUDY_FIGURES.pdf">전체 그림 PDF</a> · <a href="FULL_STUDY_RESULTS.json">전체 수치·출처 JSON</a></p>',
      '<p>설정 해석: α=1, T₀=0, 원점 좌표, 일부 경계 손실, 전이학습 입력 및 w=0.5는 재구성 선택입니다. 그림25는 패널 제목과 본문에 따른 x=0.35를 사용합니다(원문 캡션 x=0.5와 충돌).</p>',
      '<details open><summary>미완료 작업 및 참조</summary><ul>']
    for name in report['pending_jobs']+report['pending_references']:
        parts.append('<li><code>'+html.escape(name)+'</code></li>')
    parts.append('</ul></details><details><summary>실제 완료 작업의 설정과 이력</summary><table><tr><th>작업</th><th>분류</th><th>장치</th><th>이력</th></tr>')
    for name,job in report['jobs'].items():
        history_path=f'results/full_study/{name}/history.csv'
        parts.append(f'<tr><td>{html.escape(name)}</td><td>{job["kind"]}</td><td>{html.escape(str(job["environment"].get("GPU") or job["environment"].get("device")))}</td><td><a href="{history_path}">원본 CSV</a></td></tr>')
    parts.append('</table></details>')
    for item in report['figures']:
        parts.append(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["note"])}</p><img src="{item["png"]}" alt="{html.escape(item["title"])}"><p><a href="{item["svg"]}">SVG 벡터 그림</a></p></section>')
    parts.append('<details><summary>자료 부족으로 생략한 그림</summary><ul>')
    for item in report['skipped_figures']:
        parts.append('<li>'+html.escape(item['id']+': '+', '.join(item['missing']))+'</li>')
    parts.append('</ul></details></html>')
    (root/'FULL_STUDY_GALLERY_KO.html').write_text(''.join(parts),encoding='utf-8')


def generate(root=ROOT, partial=False):
    root=Path(root); out=root/'results/full_study'; specs=catalogue()
    completed=[j for j in specs if job_complete(j,out)]
    pending=[j['id'] for j in specs if j not in completed]
    reference_paths={f'{geom}_n{n}':root/f'results/fem_{geom}_n{n}/solution.npz'
                     for geom in ('circle','rectangles') for n in ((50,200) if geom=='circle' else (50,100,200))}
    reference_paths['ratio15_n50']=out/'parametric_fem/ratio_15/solution.npz'
    missing_refs=[str(p.relative_to(root)) for p in reference_paths.values() if not p.exists()]
    if not partial and (pending or missing_refs):
        raise RuntimeError('Incomplete full study: '+str(len(pending))+' jobs and '+str(len(missing_refs))+
                           ' references pending. Use --partial only for a labeled progress report. '+', '.join(pending+missing_refs))
    # Require histories for completed jobs: silent omission would hide training provenance.
    for job in completed:
        if not (out/job['id']/'history.csv').exists():
            raise FileNotFoundError('Completed job lacks original history.csv: '+job['id'])
    refs={key:load_prediction(path) for key,path in reference_paths.items() if path.exists()}
    report={'status':'partial' if pending or missing_refs else 'complete','partial_requested':partial,
            'catalogue_jobs':len(specs),'catalogue_neural_jobs':sum(j['kind']!='timing' for j in specs),
            'catalogue_timing_jobs':sum(j['kind']=='timing' for j in specs),
            'completed_jobs':[j['id'] for j in completed],'pending_jobs':pending,'pending_references':missing_refs,
            'jobs':{},'figures':[],'skipped_figures':[],'scale_audit':[],
            'metric_definition':'Absolute mean/max/RMSE, relative L2, and absolute error/reference peak; zero denominator yields null. No claim of matching manuscript relative-error convention.',
            'prediction_representation':'Figures use constitutive-derived stress and Fourier-derived flux. JSON retains independent direct heads and derived fields separately.',
            'source_limits':'data/paper_color_limits.json (Figs8/9/13/14); parametric limits transcribed from Figs23/24. Saturation audited.',
            'source_caveats':['Fig25 captionx=.5 conflicts with titles/prosex=.35; use.35.',
                'w=.5 is reconstructed; complete four-value paper set is absent.',
                '5layer stress and physics-data qx maximum-error exceptions are hypotheses to measure, not guaranteed results.',
                'Setup figures are newly computed/schematic, not reused accelerated-run result images.',
                'Remark7 one-million Adam claim is a separate continuation comparison when present; primary soft baseline stays100k.'],
            'references':{k:{'source':str(reference_paths[k].relative_to(root)),'sha256':v['source_sha256']} for k,v in refs.items()}}
    datasets={}
    for job in completed:
        name=job['id'];folder=out/name;dataset=load_prediction(folder/'prediction.npz')
        if not dataset['derived_available']:
            raise ValueError('Completed neural export lacks separate derived fields: '+name)
        datasets[name]=dataset
        history=read_history(folder/'history.csv');env=_json(folder/'environment.json',{})
        status_metadata=_json(folder/'experiment.json',_json(folder/'run_status.json',{}))
        actual_config=_json(folder/'config.json',status_metadata.get('config',job['config']))
        record={'kind':job['kind'],'config':actual_config,'catalogue_config':job['config'],'environment':env,
                'prediction_sha256':dataset['source_sha256'],'history':history,'comparisons':{},
                'boundary_errors':_boundary_metrics(dataset),
                'status_metadata':status_metadata}
        names=['ratio15_n50'] if job['kind']=='transfer' else [job['config']['geometry']+'_n'+str(n) for n in (50,200)]
        for refname in names:
            if refname not in refs: continue
            reference=align_reference(dataset['xy'],refs[refname]['xy'],refs[refname]['direct'])
            key='FEM_ratio15_n50' if job['kind']=='transfer' else 'FEM_'+refname.split('_')[-1]
            record['comparisons'][key]={mode:field_metrics(dataset[mode],reference) for mode in ('direct','derived')}
        if job['kind']=='timing':record['timing']=timing_measurement(history,env)
        report['jobs'][name]=record
    def ready(name, required, reference=None):
        missing=[j for j in required if j not in datasets]
        if reference and reference not in refs:missing.append('reference:'+reference)
        if missing:
            report['skipped_figures'].append({'id':name,'missing':missing});return False
        return True
    def reference(name,key):
        return align_reference(datasets[name]['xy'],refs[key]['xy'],refs[key]['direct'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':.6,
                         'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    writer=Writer(root,report['status']=='partial')
    try:
        _setup_figures(writer)
        limits=_json(root/'data/paper_color_limits.json')
        for geom in ('circle','rectangles'):
            for mode in ('coupled','sequential'):
                name=f'mixed_{geom}_{mode}'
                if ready(name+'_fullfields',[name],geom+'_n50'):
                    if mode=='coupled':
                        _points_and_deformation(writer,geom,report['jobs'][name]['config'],datasets[name],reference(name,geom+'_n50'))
                    for kind,indices in [('mechanical',range(5)),('thermal',(5,6,7))]:
                        source=(8 if kind=='mechanical' else 9) if geom=='circle' else (13 if kind=='mechanical' else 14)
                        _field_panels(writer,f'fig{source:02d}_{geom}_{mode}_{kind}',name,datasets[name],reference(name,geom+'_n50'),list(indices),source,limits[geom])
                if ready(name+'_losses',[name]):
                    source=6 if mode=='coupled' else 10
                    _losses(writer,f'fig{source:02d}_{geom}_{mode}_losses',report['jobs'][name]['history'],report['jobs'][name]['config'],source)
        methods=[('baseline_circle_standard_coupled','Standard coupled','red','--'),
                 ('baseline_circle_standard_sequential','Standard sequential','red',':'),
                 ('baseline_circle_dem_coupled','DEM coupled','green','--'),
                 ('baseline_circle_dem_sequential','DEM sequential','green',':'),
                 ('mixed_circle_coupled','Mixed coupled','blue','--'),('mixed_circle_sequential','Mixed sequential','blue',':')]
        if ready('fig11_circle_methods',[v[0] for v in methods],'circle_n50'):
            entries=[(datasets[n],lab,col,sty) for n,lab,col,sty in methods]+[(refs['circle_n50'],'FEM50','black','-')]
            report['fig11_sections']=_sections(writer,'fig11_circle_methods',entries,.5,11,
                'All six completed full-budget cases, x=.5. Negative outcomes shown without ranking assumptions.')
        sensitivities=[('baseline_circle_dem_coupled','DEM soft coupled','red','--'),
                       ('baseline_circle_dem_sequential','DEM soft sequential','red',':'),
                       ('sensitivity_circle_dem_hard_coupled','DEM hard coupled','blue','--'),
                       ('sensitivity_circle_dem_hard_sequential','DEM hard sequential','blue',':')]
        if any(j['id']=='sensitivity_circle_dem_hard_coupled' for j in specs):
            if ready('dem_boundary_sensitivity',[v[0] for v in sensitivities],'circle_n50'):
                _sections(writer,'dem_boundary_sensitivity',[(datasets[n],lab,c,s) for n,lab,c,s in sensitivities]+
                          [(refs['circle_n50'],'FEM50','black','-')],.5,'supplement',
                          'Additional DEM soft/hard boundary sensitivity; not a separately reported manuscript experiment.')
        pair=['mixed_rectangles_coupled','mixed_rectangles_sequential']
        if ready('fig16_rectangles_sections',pair,'rectangles_n50'):
            _sections(writer,'fig16_rectangles_sections',[(datasets[pair[0]],'Mixed coupled','blue','--'),
                (datasets[pair[1]],'Mixed sequential','blue',':'),(refs['rectangles_n50'],'FEM50','black','-')],.5,16,
                'Saved full-budget coupled and sequential predictions; no fitted scaling.')
        refine=['mixed_rectangles_coupled','refine_rectangles_left1000']
        if ready('fig15_left_refinement',refine,'rectangles_n50'):
            _refinement_panels(writer,[(n,datasets[n],reference(n,'rectangles_n50')) for n in refine],
                               report['jobs'][refine[1]]['config'])
        hard=['mixed_rectangles_coupled','hard_rectangles_adam10k','arch_l5_w40']
        if ready('fig17_boundary_optimizer',hard,'rectangles_n50'):
            _boundary_optimizer_panels(writer,[(n,datasets[n],reference(n,'rectangles_n50')) for n in hard])
        if ready('fig18_hard_objective',hard[1:]):
            fig,ax=plt.subplots(figsize=(8,4.9));fig.subplots_adjust(left=.1,right=.97,top=.94,bottom=.18)
            for name in hard[1:]:
                rows=report['jobs'][name]['history']['rows'];x=_history_updates(rows)
                y=np.array([r.get('total',np.nan) for r in rows],float);y[y<=0]=np.nan
                ax.plot(x,y,label=name,lw=1)
            ax.set(yscale='log',xlabel='Actual optimizer updates',ylabel='Recorded total objective');ax.legend(fontsize=8)
            writer.save(fig,'fig18_hard_objective','Figure18: hard-boundary objectives',
                        'Original logged objectives; Adam and L-BFGS logging evaluation moments may differ. No resampling.',18)
        selected=['arch_l2_w40','arch_l5_w20','arch_l5_w40','arch_l6_w40']
        if ready('fig19_architectures',selected,'rectangles_n50'):
            _error_grid(writer,'fig19_architectures',[(n,datasets[n],reference(n,'rectangles_n50')) for n in selected],19,
                        note='Selected2x40,5x20,5x40,6x40; each10k L-BFGS.')
        for source,values,label,ids,stem in [(20,DEPTHS,'Hidden layers',[f'arch_l{v}_w40' for v in DEPTHS],'depth'),
                                         (21,WIDTHS,'Neurons per hidden layer',[f'arch_l5_w{v}' for v in WIDTHS],'width')]:
            if ready(f'fig{source}_{stem}',ids,'rectangles_n50'):
                for representation in ('derived','direct'):
                    _sweep(writer,f'fig{source}_{stem}_{representation}',ids,values,report['jobs'],label,source,representation)
        _timing_figure(writer,report['jobs'])
        remark_ids=['mixed_rectangles_coupled','remark_rectangles_adam1m','arch_l5_w40']
        if any(j['id']=='remark_rectangles_adam1m' for j in specs):
            if ready('remark07_comparison',remark_ids,'rectangles_n50'):
                _remark_comparison(writer,report,datasets,reference)
        ablation=['mixed_circle_coupled','ablation_circle_noenergy']
        if ready('energy_omission_all_fields',ablation,'circle_n50'):
            _error_grid(writer,'energy_omission_all_fields',[(n,datasets[n],reference(n,'circle_n50')) for n in ablation],
                        'qualitative p13',range(8),'Energy omission is a reconstruction of the unquantified paper claim.')
        for name,source in [('transfer_w0.1',23),('transfer_w0',24)]:
            if ready(f'fig{source}_transfer',[name],'ratio15_n50'):
                local={k:{role:list(bound) for role,bound in v.items()} for k,v in LIMITS_PARAM.items()}
                if source==24:
                    for key,maximum in LIMITS_PARAM_DATA_ERROR.items():local[key]['error']=[0,maximum]
                _field_panels(writer,f'fig{source}_transfer',name,datasets[name],reference(name,'ratio15_n50'),[0,2,5,6],source,local,
                              f'Held-out ratio15: {name}')
        if ready('fig25_transfer_sections',['transfer_w0','transfer_w0.1'],'ratio15_n50'):
            report['fig25_sections']=_sections(writer,'fig25_transfer_sections',[
                (datasets['transfer_w0.1'],'Physics+data w=.1','blue','--'),
                (datasets['transfer_w0'],'Data w=0','green',':'),(refs['ratio15_n50'],'FEM ratio15','black','-')],.35,25,
                'x=.35 follows panel titles and body; paper caption says.5. Held-out ratio15, no retraining.')
        transfer_ids=['transfer_w0','transfer_w0.1','transfer_w0.5','transfer_w1']
        if ready('transfer_four_weight_errors',transfer_ids,'ratio15_n50'):
            fig,axes=plt.subplots(4,2,figsize=(10,11));fig.subplots_adjust(left=.09,right=.98,top=.95,bottom=.09,hspace=.48,wspace=.3)
            for ax,field,symbol,unit in zip(axes.ravel(),FIELDS,SYMBOLS,UNITS):
                for metric,marker in [('mean_absolute_error','o'),('max_absolute_error','s')]:
                    val=[report['jobs'][n]['comparisons']['FEM_ratio15_n50']['derived'][field][metric] for n in transfer_ids]
                    ax.plot([0,.1,.5,1],val,marker=marker,label=metric.replace('_',' '))
                ax.set(title=symbol+f' [{unit}]',xlabel='Physics weight w',ylabel='Absolute error',xticks=[0,.1,.5,1]);ax.legend(fontsize=7)
            writer.save(fig,'transfer_four_weight_errors','Four-weight held-out errors: all eight fields',
                        'w=.5 is reconstructed. Unseen ratio15; no uniform benefit of physics assumed.','23–25')
        arch='arch_l5_w40'
        if ready('appendixB_hard_fullfields',[arch],'rectangles_n50'):
            for suffix,indices in [('B1',range(5)),('B2',(5,6,7))]:
                _field_panels(writer,f'fig{suffix}_hard_fullfields',arch,datasets[arch],reference(arch,'rectangles_n50'),list(indices),suffix)
        if all('rectangles_n'+str(n) in refs for n in (50,100,200)):
            _sections(writer,'figA1_mesh_convergence',[(refs[f'rectangles_n{n}'],f'FEM {n}x{n}','black',style)
                for n,style in [(50,'-'),(100,'--'),(200,':')]],.5,'A1','Independent FE mesh convergence; quantitative norms in JSON.')
            fine=refs['rectangles_n200']
            report['fem_mesh_convergence']={}
            for n in (50,100):
                coarse=refs[f'rectangles_n{n}'];aligned=align_reference(coarse['xy'],fine['xy'],fine['direct'])
                report['fem_mesh_convergence'][f'FEM{n}_vs_FEM200']=field_metrics(coarse['direct'],aligned)
        else:report['skipped_figures'].append({'id':'figA1_mesh_convergence','missing':[f'rectangles_n{n}' for n in (50,100,200) if f'rectangles_n{n}' not in refs]})
    finally:
        writer.close()
    report['figures']=writer.items;report['scale_audit']=writer.audit
    report['observed_comparisons']={
        'sequential_vs_coupled_circle':_comparison_summary(report['jobs'],'mixed_circle_sequential','mixed_circle_coupled'),
        'sequential_vs_coupled_rectangles':_comparison_summary(report['jobs'],'mixed_rectangles_sequential','mixed_rectangles_coupled'),
        'refined_vs_baseline':_comparison_summary(report['jobs'],'refine_rectangles_left1000','mixed_rectangles_coupled'),
        'noenergy_vs_energy':_comparison_summary(report['jobs'],'ablation_circle_noenergy','mixed_circle_coupled'),
        'physics_data_vs_data':_comparison_summary(report['jobs'],'transfer_w0.1','transfer_w0','FEM_ratio15_n50'),
        'depth5_vs_depth4':_comparison_summary(report['jobs'],'arch_l5_w40','arch_l4_w40')}
    if any(j['id']=='remark_rectangles_adam1m' for j in specs):
        report['observed_comparisons']['remark_1m_vs_100k']=_comparison_summary(report['jobs'],'remark_rectangles_adam1m','mixed_rectangles_coupled')
    for mode in ('coupled','sequential'):
        if any(j['id']==f'sensitivity_circle_dem_hard_{mode}' for j in specs):
            report['observed_comparisons'][f'dem_hard_vs_soft_{mode}']=_comparison_summary(report['jobs'],
                f'sensitivity_circle_dem_hard_{mode}',f'baseline_circle_dem_{mode}')
    target=root/'FULL_STUDY_RESULTS.json';temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(target)
    _write_gallery(root,report)
    print(f"Full-study figures: {report['status']}; {len(completed)}/{len(specs)} jobs; {len(writer.items)} figures")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--partial',action='store_true',help='Labeled progress report using only verified completed jobs')
    args=parser.parse_args();generate(args.root,args.partial)


if __name__=='__main__':main()
