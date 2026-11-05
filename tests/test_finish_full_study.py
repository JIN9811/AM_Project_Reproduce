"""Finalization must reject partial evidence and keep the compact archive honest."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def document_folders(tmp_path):
    for name in ('reports', 'notebooks', 'docs', 'validation'):
        (tmp_path/name).mkdir(exist_ok=True)


def test_neural_gate_never_trusts_queue_completion_without_verified_jobs():
    from scripts.finish_full_study import neural_gate, PipelineError
    queue = {'status': 'complete', 'running': [], 'pending': [], 'failed': {}}
    with pytest.raises(PipelineError, match='missing'):
        neural_gate(queue, {'a'}, {'a', 'b'}, heartbeat_age=0)
    assert neural_gate(queue, {'a', 'b'}, {'a', 'b'}, heartbeat_age=0)
    queue['status'] = 'running'
    assert not neural_gate(queue, {'a', 'b'}, {'a', 'b'}, heartbeat_age=0)
    queue['failed'] = {'b': 1}
    with pytest.raises(PipelineError, match='failed'):
        neural_gate(queue, {'a', 'b'}, {'a', 'b'}, heartbeat_age=0)


def test_neural_gate_detects_stale_or_missing_worker_heartbeat():
    from scripts.finish_full_study import neural_gate, PipelineError
    with pytest.raises(PipelineError, match='stale'):
        neural_gate({'status': 'running'}, set(), {'a'}, heartbeat_age=901)
    assert not neural_gate(None, set(), {'a'}, heartbeat_age=10)


def test_archive_excludes_optimizer_states_but_keeps_models_results_and_code(tmp_path):
    from scripts.finish_full_study import archive_members
    names = ['harandi/train.py', 'results/full_study/job/model.pt',
             'results/full_study/job/prediction.npz', 'results/full_study/job/console.log',
             'results/full_study/job/checkpoint.pt', 'results/full_study/job/checkpoints/ratio_1.pt',
             'results/full_study/job/progress.pt', '.notebook_work/scratch.py',
             '.venv/python.exe', 'harandi/__pycache__/train.pyc', 'any.tmp', 'old.zip',
             'reports/FULL_STUDY_PROGRESS.html', 'validation/full_pipeline_status.json',
             'results/full_study/.queue.lock', 'reports/FULL_STUDY_TRAINING_PREVIEW.png',
             'validation/FULL_STUDY_TRAINING_PREVIEW.json']
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'test')
    actual = {path.relative_to(tmp_path).as_posix() for path in archive_members(tmp_path)}
    assert actual == set(names[:4])


def test_notebook_defaults_prevent_retraining_and_completed_cells_have_no_errors(tmp_path):
    from scripts.finish_full_study import validate_notebook_defaults, validate_notebook, PipelineError
    path = tmp_path/'notebook.ipynb'
    cell = {'cell_type': 'code', 'source': ['RUN_FULL_STUDY = False\nRUN_SMOKE = False\n',
            'BUILD_FIGURE_PACK = False\nRESUME_JOB_ID = None'], 'execution_count': 1, 'outputs': []}
    path.write_text(json.dumps({'cells': [cell]}))
    validate_notebook_defaults(path)
    assert validate_notebook(path)['code_cells'] == 1
    cell['execution_count'] = None
    path.write_text(json.dumps({'cells': [cell]}))
    with pytest.raises(PipelineError, match='unexecuted'):
        validate_notebook(path)
    cell['execution_count'] = 1
    cell['outputs'] = [{'output_type': 'error', 'ename': 'ValueError', 'evalue': 'failed'}]
    path.write_text(json.dumps({'cells': [cell]}))
    with pytest.raises(PipelineError, match='error'):
        validate_notebook(path)
    cell['source'] = 'RUN_FULL_STUDY = True'
    path.write_text(json.dumps({'cells': [cell]}))
    with pytest.raises(PipelineError, match='RUN_FULL_STUDY'):
        validate_notebook_defaults(path)


def test_source_hash_mismatch_fails_instead_of_refreshing_provenance(tmp_path):
    from scripts.finish_full_study import verify_sources, PipelineError
    paper = tmp_path/'paper.pdf'
    paper.write_bytes(b'original paper')
    data = tmp_path/'data'; data.mkdir()
    (data/'source.csv').write_bytes(b'original data')
    provenance = {'paper': {'path': str(paper), 'sha256': hashlib.sha256(paper.read_bytes()).hexdigest()},
                  'included_data': {'source.csv': hashlib.sha256(b'original data').hexdigest()}}
    path = tmp_path/'docs/SOURCE_PROVENANCE.json'
    path.write_text(json.dumps(provenance))
    assert len(verify_sources(tmp_path)) == 2
    (data/'source.csv').write_bytes(b'modified data')
    with pytest.raises(PipelineError, match='hash'):
        verify_sources(tmp_path)
    assert json.loads(path.read_text()) == provenance


def test_reference_validation_rejects_nonfinite_fields_and_large_residual(tmp_path):
    from scripts.finish_full_study import verify_reference, PipelineError
    from harandi.physics import Problem, grid, homogeneous_exact
    problem = Problem('homogeneous')
    xy = grid(101)
    fields = homogeneous_exact(xy, problem)
    meta = {'problem': problem.to_dict(), 'n': 50, 'coefficient_mode': 'sharp',
            'diagnostics': {'thermal_relative_residual': 1e-12, 'mechanical_relative_residual': 1e-12,
                            'heat_balance': 1e-13, 'heat_left_reaction': 1, 'heat_right_reaction': -1}}
    np.savez(tmp_path/'solution.npz', xy=xy, fields=fields)
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    assert verify_reference(tmp_path, problem.to_dict(), 50, 'sharp')['points'] == 10201
    meta['diagnostics']['mechanical_relative_residual'] = 1e-2
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(PipelineError, match='residual'):
        verify_reference(tmp_path, problem.to_dict(), 50, 'sharp')
    meta['diagnostics']['mechanical_relative_residual'] = 1e-12
    (tmp_path/'metadata.json').write_text(json.dumps(meta))
    fields[0, 0] = np.nan
    np.savez(tmp_path/'solution.npz', xy=xy, fields=fields)
    with pytest.raises(PipelineError, match='finite'):
        verify_reference(tmp_path, problem.to_dict(), 50, 'sharp')


def test_public_source_audit_reports_unbundled_paper_and_checks_included_data(tmp_path):
    from scripts.finish_full_study import verify_sources, PipelineError
    data = tmp_path/'data'; data.mkdir()
    (data/'source.csv').write_bytes(b'licensed data')
    provenance = {'paper': {'path': 'external/paper.pdf', 'sha256': hashlib.sha256(b'paper').hexdigest()},
                  'distribution': {'mode': 'public', 'external_paper_and_crops_included': False},
                  'included_data': {'source.csv': hashlib.sha256(b'licensed data').hexdigest()},
                  'paper_visual_style': {'original_comparison_crops': {'original.png': '0'*64}}}
    (tmp_path/'docs/SOURCE_PROVENANCE.json').write_text(json.dumps(provenance))
    records = verify_sources(tmp_path)
    assert sum(r.get('status') == 'not_distributed_not_verified' for r in records) == 2
    assert any(r.get('status') == 'verified' and r['path'].endswith('source.csv') for r in records)
    (data/'source.csv').write_bytes(b'corrupted')
    with pytest.raises(PipelineError, match='hash'):
        verify_sources(tmp_path)


@pytest.mark.parametrize('damage', ['missing', 'modified'])
def test_public_source_audit_checks_bundled_figure_even_when_pdf_is_unbundled(tmp_path, damage):
    from scripts.finish_full_study import verify_sources, PipelineError
    image = tmp_path/'figures/paper_reference/paper_fig5.png'
    image.parent.mkdir(parents=True)
    image.write_bytes(b'licensed original figure')
    provenance = {
        'paper': {'path': 'external/paper.pdf', 'sha256': hashlib.sha256(b'paper').hexdigest()},
        'distribution': {'mode': 'public', 'external_paper_and_crops_included': False},
        'bundled_artifacts': {'figures/paper_reference/paper_fig5.png': hashlib.sha256(image.read_bytes()).hexdigest()}}
    (tmp_path/'docs/SOURCE_PROVENANCE.json').write_text(json.dumps(provenance))
    records = verify_sources(tmp_path)
    assert any(r['path'] == str(image) and r['status'] == 'verified' for r in records)
    if damage == 'missing':
        image.unlink()
    else:
        image.write_bytes(b'changed image')
    with pytest.raises(PipelineError, match='Source hash verification failed'):
        verify_sources(tmp_path)


def test_public_source_audit_rejects_wrong_external_paper_when_present(tmp_path):
    from scripts.finish_full_study import verify_sources, PipelineError
    external=tmp_path/'external'; external.mkdir()
    (external/'paper.pdf').write_bytes(b'wrong paper')
    provenance={'paper': {'path':'external/paper.pdf', 'sha256': hashlib.sha256(b'correct').hexdigest()},
                'distribution': {'mode':'public', 'external_paper_and_crops_included':False}}
    (tmp_path/'docs/SOURCE_PROVENANCE.json').write_text(json.dumps(provenance))
    with pytest.raises(PipelineError, match='hash'):
        verify_sources(tmp_path)


def test_bundle_contains_manifest_and_file_hashes_preserves_previous_archive(tmp_path):
    from scripts.finish_full_study import write_bundle
    root = tmp_path/'project'; root.mkdir()
    (root/'model.pt').write_bytes(b'compact model')
    (root/'checkpoint.pt').write_bytes(b'large optimizer')
    previous = tmp_path/'harandi_reproduction.zip'; previous.write_bytes(b'preserve me')
    target = tmp_path/'harandi_full_study.zip'
    result = write_bundle(root, target, {'verified_jobs': 52})
    manifest = json.loads((root/'validation/FULL_STUDY_MANIFEST.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['files']['model.pt']['sha256'] == hashlib.sha256(b'compact model').hexdigest()
    assert result['sha256'] == hashlib.sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {'harandi_reproduction/model.pt',
                                          'harandi_reproduction/validation/FULL_STUDY_MANIFEST.json'}
        assert json.loads(archive.read('harandi_reproduction/validation/FULL_STUDY_MANIFEST.json')) == manifest
    assert previous.read_bytes() == b'preserve me'


def test_failed_queue_marks_pipeline_failed_without_starting_other_stages(tmp_path):
    from scripts.finish_full_study import Finisher
    study = tmp_path/'results/full_study'; study.mkdir(parents=True)
    (study/'queue_status.json').write_text(json.dumps({'status': 'failed',
        'running': [], 'pending': [], 'failed': {'mixed_circle_coupled': 1}}))
    finisher = Finisher(tmp_path)
    assert finisher.run() == 1
    state = json.loads((tmp_path/'validation/full_pipeline_status.json').read_text())
    assert state['status'] == 'failed'
    assert 'Neural queue failed' in state['detail']
    assert not (study/'pipeline_logs').exists()
    assert not (tmp_path/'validation/FULL_STUDY_MANIFEST.json').exists()
    assert not (tmp_path/'.full_pipeline.lock').exists()


def test_archive_publish_failure_never_creates_complete_manifest(tmp_path):
    from scripts.finish_full_study import write_bundle
    root = tmp_path/'project'; root.mkdir()
    (root/'model.pt').write_bytes(b'compact model')
    destination = tmp_path/'blocked.zip'; destination.mkdir()
    with pytest.raises(OSError):
        write_bundle(root, destination, {'verified_jobs': 52})
    assert not (root/'validation/FULL_STUDY_MANIFEST.json').exists()
    assert not (tmp_path/'blocked.zip.tmp').exists()


def test_progress_counts_actual_history_without_loading_numeric_artifacts(tmp_path):
    from scripts.finish_full_study import progress_rows
    job = {'id': 'example', 'kind': 'baseline', 'config': {'method': 'standard',
           'mode': 'sequential', 'steps': 10, 'width': 4, 'layers': 1}}
    folder = tmp_path/'results/full_study/example'; folder.mkdir(parents=True)
    (folder/'history.csv').write_text('step,active,total\n7,mechanical,0.25\n')
    (folder/'model.pt').write_bytes(b'intentionally not a model')
    row = progress_rows(tmp_path, [job], {'running': ['example']}, set())[0]
    assert row['status'] == 'running'
    assert row['actual_logged_or_checkpoint_updates'] == 7
    assert row['requested_updates'] == 20
    assert row['last_loss'] == .25


def test_archive_heartbeats_update_progress_without_archiving_live_snapshot(tmp_path, monkeypatch):
    import itertools
    from scripts import finish_full_study as finish
    root = tmp_path/'project'; root.mkdir()
    (root/'model.pt').write_bytes(b'a'*(3*1024*1024))
    clock = itertools.count(0, 31)
    monkeypatch.setattr(finish.time, 'monotonic', lambda: next(clock))
    def heartbeat():
        path = root/'validation/full_pipeline_status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(path.read_text()) if path.exists() else {'heartbeats': 0}
        path.write_text(json.dumps({'heartbeats': previous['heartbeats']+1, 'status': 'running'}))
    finish.write_bundle(root, tmp_path/'full.zip', {'verified_jobs': 52}, heartbeat=heartbeat)
    assert json.loads((root/'validation/full_pipeline_status.json').read_text())['heartbeats'] >= 2
    with zipfile.ZipFile(tmp_path/'full.zip') as archive:
        assert not any(name.endswith('validation/full_pipeline_status.json') for name in archive.namelist())


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows execution-state contract')
def test_keep_awake_is_scoped_to_process_work_and_does_not_request_display(monkeypatch):
    from scripts import finish_full_study as finish
    requests = []
    def native(flags):
        requests.append(flags)
        return 0x80000000
    monkeypatch.setattr(finish, '_set_execution_state', native)
    guard = finish.SleepGuard()
    guard.enable()
    assert guard.active
    assert requests[-1] == 0x80000001
    guard.disable()
    assert not guard.active
    assert requests == [0x80000001, 0x80000000]


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows execution-state contract')
def test_keep_awake_failure_is_recorded_without_claiming_protection(monkeypatch):
    from scripts import finish_full_study as finish
    monkeypatch.setattr(finish, '_set_execution_state', lambda flags: 0)
    guard = finish.SleepGuard()
    guard.enable()
    assert not guard.active
    assert guard.warning and 'failed' in guard.warning


def test_pdf_qa_must_cover_every_page_and_match_the_current_report(tmp_path):
    from scripts.finish_full_study import verify_report_qa, PipelineError
    report = tmp_path/'reports/FULL_PROJECT_REPORT_KO.pdf'; report.write_bytes(b'new PDF')
    record = {'status': 'pass', 'automatic_checks_passed': True, 'pdf_sha256': 'stale',
              'page_count': 2, 'rendered_page_count': 2,
              'manual_visual_review': {'performed': False}}
    qa = tmp_path/'validation/FULL_REPORT_LAYOUT_QA.json'
    qa.write_text(json.dumps(record))
    with pytest.raises(PipelineError, match='hash'):
        verify_report_qa(tmp_path)
    record['pdf_sha256'] = hashlib.sha256(report.read_bytes()).hexdigest()
    record['rendered_page_count'] = 1
    qa.write_text(json.dumps(record))
    with pytest.raises(PipelineError, match='page'):
        verify_report_qa(tmp_path)
    record['rendered_page_count'] = 2
    qa.write_text(json.dumps(record))
    result = verify_report_qa(tmp_path)
    assert result['automatic_checks_passed'] is True
    assert result['manual_visual_review']['performed'] is False


def ready_pipeline(tmp_path, monkeypatch, failing_phase=None):
    """Only external command execution and expensive artifact checks are replaced."""
    import nbformat
    from scripts import finish_full_study as finish
    study = tmp_path/'results/full_study'; study.mkdir(parents=True)
    (study/'queue_status.json').write_text(json.dumps({'status': 'complete',
        'running': [], 'pending': [], 'failed': {}}))
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'RUN_FULL_STUDY=False\nRUN_SMOKE=False\nBUILD_FIGURE_PACK=False\nRESUME_JOB_ID=None',
        execution_count=1)])
    nbformat.write(notebook, tmp_path/'notebooks/Full_Paper_Reproduction_KO.ipynb')
    finalizer = finish.Finisher(tmp_path, archive=tmp_path/'full.zip')
    events = []
    def verify_jobs(jobs):
        if len(jobs) == 27:
            events.append('neural_gate')
        if len(jobs) == 25:
            return set()  # timing still has actual work to do
        return {job['id'] for job in jobs}
    def command(phase, arguments):
        events.append(phase)
        if phase == 'timing':
            assert arguments[arguments.index('--groups')+1] == 'timing'
            assert arguments[arguments.index('--workers')+1] == '1'
        if phase == failing_phase:
            raise finish.PipelineError(phase+' simulated command failure')
    def validate():
        events.append('validation')
        assert not (tmp_path/'full.zip').exists()
        return {'verified_jobs': 52}
    monkeypatch.setattr(finalizer, 'verify_jobs', verify_jobs)
    monkeypatch.setattr(finalizer, 'command', command)
    monkeypatch.setattr(finalizer, 'verify_deliverables', validate)
    return finalizer, events


def test_pipeline_runs_strict_stages_in_order_before_real_archive(tmp_path, monkeypatch):
    finalizer, events = ready_pipeline(tmp_path, monkeypatch)
    assert finalizer.run() == 0
    assert events == ['neural_gate', 'timing', 'figures', 'notebook', 'report',
                      'report_layout_qa', 'validation']
    assert (tmp_path/'full.zip').exists()
    assert (tmp_path/'notebooks/Full_Paper_Reproduction_KO.html').exists()
    assert json.loads((tmp_path/'validation/full_pipeline_status.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('phase', ['timing', 'figures', 'notebook', 'report', 'report_layout_qa'])
def test_failed_stage_prevents_final_validation_and_archive(tmp_path, monkeypatch, phase):
    finalizer, events = ready_pipeline(tmp_path, monkeypatch, phase)
    assert finalizer.run() == 1
    assert 'validation' not in events
    assert not (tmp_path/'full.zip').exists()
    assert not (tmp_path/'validation/FULL_STUDY_MANIFEST.json').exists()
    state = json.loads((tmp_path/'validation/full_pipeline_status.json').read_text())
    assert state['status'] == 'failed'
    assert phase+' simulated command failure' in state['detail']


def test_final_validation_failure_prevents_archive(tmp_path, monkeypatch):
    from scripts.finish_full_study import PipelineError
    finalizer, _ = ready_pipeline(tmp_path, monkeypatch)
    def reject():
        raise PipelineError('final evidence mismatch')
    monkeypatch.setattr(finalizer, 'verify_deliverables', reject)
    assert finalizer.run() == 1
    assert not (tmp_path/'full.zip').exists()
    assert not (tmp_path/'validation/FULL_STUDY_MANIFEST.json').exists()


def test_restart_rejects_live_stage_even_if_old_finalizer_removed_its_lock(tmp_path, monkeypatch):
    from scripts import finish_full_study as finish
    finalizer = finish.Finisher(tmp_path)
    (tmp_path/'validation/full_pipeline_status.json').write_text(json.dumps({'child_pid': 4321}))
    monkeypatch.setattr(finish, 'pid_alive', lambda pid: pid == 4321)
    with pytest.raises(finish.PipelineError, match='still running'):
        finalizer.acquire()
    assert not (tmp_path/'.full_pipeline.lock').exists()
