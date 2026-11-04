import csv
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


def api():
    spec = importlib.util.find_spec('harandi.full_figures')
    assert spec is not None, 'full-study figure module must exist'
    return importlib.import_module('harandi.full_figures')


def synthetic_completion_gate(job,out):
    """Test-only metadata gate; production delegates to the real model validator."""
    folder=Path(out)/job['id']
    if not (folder/'model.pt').exists() or not (folder/'prediction.npz').exists():return False
    try:
        config=json.loads((folder/'config.json').read_text())
        if config!=job['config']:return False
        if job['kind']=='transfer':
            meta=json.loads((folder/'experiment.json').read_text())
            return meta['status']=='complete' and meta['actual_optimizer_updates']==100000
        if job['kind']=='baseline':
            meta=json.loads((folder/'experiment.json').read_text())
            return meta['complete'] and meta['completed_updates']==config['steps']*(2 if config['mode']=='sequential' else 1)
        meta=json.loads((folder/'run_status.json').read_text())
        expected={k:config[k+'_steps']*(2 if config['mode']=='sequential' else 1) for k in ('adam','lbfgs')}
        return meta['status']=='complete' and meta['completed']==expected
    except (FileNotFoundError,KeyError,ValueError):return False


def prepare(root):
    from harandi.physics import grid
    from harandi.full_study import catalogue
    project = Path(__file__).parents[1]
    (root/'data').mkdir()
    for name in ('paper_palette.json', 'paper_color_limits.json'):
        shutil.copy(project/'data'/name, root/'data'/name)
    xy = grid(5)
    reference = np.tile(np.arange(1., 9.), (len(xy), 1))
    for n in (50, 200):
        folder = root/f'results/fem_rectangles_n{n}'
        folder.mkdir(parents=True)
        np.savez(folder/'solution.npz', xy=xy, fields=reference)
    job = next(j for j in catalogue() if j['id']=='mixed_rectangles_coupled')
    folder = root/'results/full_study'/job['id']
    folder.mkdir(parents=True)
    fields = reference + 1
    np.savez(folder/'prediction.npz', xy=xy, fields=fields,
             constitutive_stress=reference[:,2:5]+2, fourier_flux=reference[:,6:]+3)
    (folder/'model.pt').write_bytes(b'not loaded by plotting')
    (folder/'config.json').write_text(json.dumps(job['config']))
    (folder/'run_status.json').write_text(json.dumps({'status':'complete',
        'completed':{'adam':100000,'lbfgs':0}}))
    (folder/'environment.json').write_text(json.dumps({'device':'cpu','GPU':None,'training_points':2601}))
    (folder/'history.csv').write_text('optimizer,step,active,elapsed_seconds,total,M_energy,T_energy\n'
        'adam,1,coupled,0.12345678901234567,2.1234567890123457,1,1\n'
        'adam,100000,coupled,12.345678901234567,0.5,0.25,0.25\n')
    return folder, xy, reference


def test_default_refuses_incomplete_catalogue_without_writing_outputs(tmp_path):
    module = api()
    with pytest.raises(RuntimeError, match='Incomplete'):
        module.generate(tmp_path)
    assert not (tmp_path/'validation/FULL_STUDY_RESULTS.json').exists()
    assert not (tmp_path/'reports/FULL_STUDY_FIGURES.pdf').exists()


def test_metrics_keep_direct_derived_and_zero_reference_explicit(tmp_path):
    module = api()
    folder, xy, reference = prepare(tmp_path)
    dataset = module.load_prediction(folder/'prediction.npz')
    metrics = module.field_metrics(dataset['direct'], reference)
    derived = module.field_metrics(dataset['derived'], reference)
    assert set(metrics)=={'ux','uy','sxx','syy','sxy','T','qx','qy'}
    assert metrics['sxx']['mean_absolute_error']==1
    assert derived['sxx']['mean_absolute_error']==2
    assert derived['qx']['max_absolute_error']==3
    zero = module.field_metrics(np.ones_like(reference), np.zeros_like(reference))
    assert zero['qx']['relative_l2'] is None
    assert zero['qx']['normalization_reference_peak']==0
    np.testing.assert_array_equal(dataset['direct'][:,2], reference[:,2]+1)


def test_coordinate_alignment_reorders_but_rejects_different_points():
    module = api()
    xy = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    values = np.arange(32.).reshape(4,8)
    np.testing.assert_array_equal(module.align_reference(xy, xy[::-1], values[::-1]), values)
    bad = xy.copy(); bad[0,0]=.1
    with pytest.raises(ValueError, match='coordinates'):
        module.align_reference(xy, bad, values)


def test_history_preserves_source_and_does_not_fill_inactive_losses(tmp_path):
    module = api()
    path = tmp_path/'history.csv'
    path.write_text('optimizer,step,active,elapsed_seconds,total,M_energy,T_energy\n'
        'adam,1,thermal,.12345678901234567,1.1234567890123457,,.5\n'
        'adam,20001,mechanical,25.0,2,.9,\n')
    original = path.read_bytes()
    history = module.read_history(path)
    assert history['rows'][0]['total']==float('1.1234567890123457')
    assert history['rows'][0]['M_energy'] is None
    assert history['rows'][1]['T_energy'] is None
    assert history['source_sha256']==hashlib.sha256(original).hexdigest()
    assert history['raw_rows'][0]['elapsed_seconds']=='.12345678901234567'
    assert path.read_bytes()==original


def test_partial_build_only_admits_complete_jobs_and_marks_pending(tmp_path,monkeypatch):
    module = api()
    monkeypatch.setattr(module,'job_complete',synthetic_completion_gate)
    folder, _, _ = prepare(tmp_path)
    before = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir()}
    incomplete = tmp_path/'results/full_study/mixed_circle_coupled'
    incomplete.mkdir()
    shutil.copy(folder/'prediction.npz', incomplete/'prediction.npz')
    report = module.generate(tmp_path, partial=True)
    assert report['status']=='partial'
    assert report['completed_jobs']==['mixed_rectangles_coupled']
    from harandi.full_study import catalogue
    assert len(report['pending_jobs'])==len(catalogue())-1
    assert 'mixed_circle_coupled' in report['pending_jobs']
    assert report['jobs']['mixed_rectangles_coupled']['comparisons']['FEM_n200']['derived']['qx']['max_absolute_error']==3
    assert report['skipped_figures']
    assert (tmp_path/'reports/FULL_STUDY_FIGURES.pdf').stat().st_size>1000
    assert (tmp_path/'reports/FULL_STUDY_GALLERY_KO.html').exists()
    assert any((tmp_path/'figures/full_study').glob('*.svg'))
    assert json.loads((tmp_path/'validation/FULL_STUDY_RESULTS.json').read_text(encoding='utf-8'))['pending_jobs']==report['pending_jobs']
    assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir()}


def test_timing_uses_first_hundred_updates_and_labels_device(tmp_path):
    module = api()
    path = tmp_path/'history.csv'
    path.write_text('optimizer,step,active,elapsed_seconds,total\nlbfgs,1,coupled,2,1\n'
                    'lbfgs,100,coupled,42,.1\nlbfgs,200,coupled,70,.05\n')
    timing = module.timing_measurement(module.read_history(path), {'device':'cuda','GPU':'test GPU'})
    assert timing['seconds_per_iteration_first_100']==.42
    assert timing['elapsed_seconds_first_100']==42
    assert timing['device']=='cuda'
    assert timing['GPU']=='test GPU'
    assert 'wall' in timing['measurement_kind'].lower()


def test_section_coordinate_is_point35_and_linear_interpolation_is_disclosed():
    module = api()
    from harandi.physics import grid
    xy=grid(5); fields=np.repeat(xy[:,0,None],8,axis=1)
    y, section, info=module.extract_section(xy,fields,.35)
    np.testing.assert_allclose(section,.35)
    assert info['x']==.35 and info['method']=='linear interpolation in x'


def test_parametric_temperature_lower_color_limit_is_positive():
    # Enlarged supplied PDF page28/29 colorbar: 0.0010, not -0.0010.
    module=api()
    assert module.LIMITS_PARAM['T']['pinn']==[.0010,.9988]
    assert module.LIMITS_PARAM['T']['fem']==[.0010,.9988]


def test_complete_synthetic_catalogue_exercises_all_plot_families(tmp_path,monkeypatch):
    """Synthetic data live only in pytest temp directories, never study outputs."""
    module=api()
    monkeypatch.setattr(module,'job_complete',synthetic_completion_gate)
    from harandi.full_study import catalogue
    first,xy,reference=prepare(tmp_path)
    for geometry in ('circle','rectangles'):
        for n in (50,100,200):
            folder=tmp_path/f'results/fem_{geometry}_n{n}'
            folder.mkdir(parents=True,exist_ok=True)
            np.savez(folder/'solution.npz',xy=xy,fields=reference)
    folder=tmp_path/'results/full_study/parametric_fem/ratio_15';folder.mkdir(parents=True)
    np.savez(folder/'solution.npz',xy=xy,fields=reference)
    jobs=catalogue()
    for index,job in enumerate(jobs):
        folder=tmp_path/'results/full_study'/job['id'];folder.mkdir(exist_ok=True)
        fields=reference+(index+1)*.01
        np.savez(folder/'prediction.npz',xy=xy,fields=fields,
                 constitutive_stress=fields[:,2:5],fourier_flux=fields[:,6:])
        (folder/'model.pt').write_bytes(b'synthetic fixture; never load')
        (folder/'config.json').write_text(json.dumps(job['config']))
        (folder/'environment.json').write_text(json.dumps({'device':'cpu','platform':'synthetic-test'}))
        if job['kind']=='transfer':
            (folder/'experiment.json').write_text(json.dumps({'status':'complete','actual_optimizer_updates':100000,
                'training_ratios':list(range(1,11)),'physics_weight':job['config']['weight'],'config':job['config']}))
            history='ratio,step,completed_updates,data,physics,loss,seconds\n1,1,1,1,1,2,1\n10,10000,100000,.1,.1,.2,10\n'
        elif job['kind']=='baseline':
            updates=job['config']['steps']*(2 if job['config']['mode']=='sequential' else 1)
            (folder/'experiment.json').write_text(json.dumps({'complete':True,'completed_updates':updates}))
            history=f'optimizer,step,active,elapsed_seconds,total,M_energy,T_energy\nadam,1,coupled,1,2,1,1\nadam,{updates},coupled,10,.2,.1,.1\n'
        else:
            factor=2 if job['config']['mode']=='sequential' else 1
            completed={key:job['config'][key+'_steps']*factor for key in ('adam','lbfgs')}
            (folder/'run_status.json').write_text(json.dumps({'status':'complete','completed':completed}))
            optimizer='lbfgs' if completed['lbfgs'] else 'adam';updates=completed[optimizer]
            history=f'optimizer,step,active,elapsed_seconds,total,M_energy,T_energy\n{optimizer},1,coupled,1,2,1,1\n{optimizer},{updates},coupled,10,.2,.1,.1\n'
        (folder/'history.csv').write_text(history)
    report=module.generate(tmp_path)
    names={f['id'] for f in report['figures']}
    expected={'fig11_circle_methods','fig15_left_refinement','fig17_boundary_optimizer','fig18_hard_objective',
              'fig19_architectures','fig20_depth_derived','fig20_depth_direct','fig21_width_derived',
              'fig21_width_direct','fig23_transfer','fig24_transfer','fig25_transfer_sections',
              'transfer_four_weight_errors','figB1_hard_fullfields','figB2_hard_fullfields','figA1_mesh_convergence',
              'energy_omission_all_fields'}
    assert expected<=names
    assert report['status']=='complete' and not report['pending_jobs']
    assert not report['skipped_figures']
    assert len(report['jobs'])==len(jobs)
    assert report['jobs']['transfer_w0']['comparisons'].keys()=={'FEM_ratio15_n50'}
    assert report['jobs']['timing_l5_w40']['timing']['seconds_per_iteration_first_100']==.1
    assert report['observed_comparisons']['physics_data_vs_data']['status']=='measured'
    if any(j['id']=='remark_rectangles_adam1m' for j in jobs):
        assert {'remark07_runtime','remark07_error_fields'}<=names
    if any(j['id']=='sensitivity_circle_dem_hard_coupled' for j in jobs):
        assert 'dem_boundary_sensitivity' in names
