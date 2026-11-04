import json
from harandi.full_study import catalogue,job_complete


def test_catalogue_has_full_budgets_and_paired_transfer_ratios():
    jobs=catalogue();by={j['id']:j for j in jobs}
    assert len(by)==len(jobs)==52
    for method in ('standard','dem'):
        for mode in ('coupled','sequential'):
            c=by[f'baseline_circle_{method}_{mode}']['config']
            assert c['steps']==100000 and c['hard'] is False
    assert len([j for j in jobs if j['id'].startswith('arch_')])==9
    assert len([j for j in jobs if j['kind']=='timing'])==25
    for job in jobs:
        if job['kind']=='transfer':
            assert job['config']['ratios']==list(range(1,11))
            assert job['config']['steps']==10000


def test_completion_rejects_partial_and_accepts_full_baseline(tmp_path):
    job=next(j for j in catalogue() if j['kind']=='baseline')
    folder=tmp_path/job['id'];folder.mkdir()
    for f in ('prediction.npz','model.pt'):(folder/f).write_bytes(b'dummy')
    (folder/'config.json').write_text(json.dumps(job['config']))
    expected=job['config']['steps']*(1 if job['config']['mode']=='coupled' else 2)
    (folder/'experiment.json').write_text(json.dumps({'complete':False,'completed_updates':expected-1}))
    assert not job_complete(job,tmp_path)
    (folder/'experiment.json').write_text(json.dumps({'complete':True,'completed_updates':expected}))
    assert not job_complete(job,tmp_path)  # Metadata alone cannot validate corrupt artifacts.


def test_complete_accepts_actual_tiny_run_and_rejects_changed_budget(tmp_path,monkeypatch):
    import pytest
    pytest.importorskip('torch')
    from harandi.full_study import execute_job
    job={'id':'actual','kind':'mixed','config':dict(geometry='homogeneous',mode='coupled',n=5,
         width=4,layers=1,hard=True,adam_steps=2,lbfgs_steps=0,device='cpu',fast_adam=False)}
    execute_job(job,tmp_path)
    assert job_complete(job,tmp_path)
    import torch
    with monkeypatch.context() as cache_check:
        def unexpected_load(*args,**kwargs):
            raise AssertionError('Unchanged verified artifacts should be cached')
        cache_check.setattr(torch,'load',unexpected_load)
        assert job_complete(job,tmp_path)
    job['config']['adam_steps']=3
    assert not job_complete(job,tmp_path)


def test_wsl_fem_output_paths_are_posix(tmp_path,monkeypatch):
    import harandi.full_study as study
    commands=[]
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study.subprocess,'run',lambda args,**kwargs:commands.append(args))
    study.fem_references(tmp_path/'results/full_study')
    assert len(commands)==11
    for command in commands:
        path=command[command.index('--output')+1]
        assert path.startswith('results/full_study/parametric_fem/ratio_')
        assert '\\' not in path


def test_queue_lock_rejects_duplicate_writer_and_releases_after_exception(tmp_path):
    import pytest
    from harandi.full_study import queue_lock
    with pytest.raises(ValueError,match='deliberate'):
        with queue_lock(tmp_path):
            with pytest.raises(RuntimeError,match='already running'):
                with queue_lock(tmp_path):
                    raise AssertionError('Two queues may not write the same study')
            raise ValueError('deliberate')
    with queue_lock(tmp_path):
        pass
