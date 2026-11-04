"""Wait for the neural queue, then strictly finish and package the full study.

This script never starts neural training. It starts only missing timing jobs,
after the neural queue has stopped and every neural artifact is verified.
Run with the project Python; set --document-python or HARANDI_DOCUMENT_PYTHON
to a Python containing requirements-documents.txt (including PDF layout QA).
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {'.git', '.venv', 'venv', 'env', 'pinn-env', '__pycache__',
                  '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints',
                  '.notebook_work', 'node_modules', 'checkpoints', 'work', 'tmp'}
EXCLUDED_NAMES = {'checkpoint.pt', 'progress.pt', 'FULL_STUDY_MANIFEST.json',
                  'FULL_STUDY_PROGRESS.html', 'full_pipeline_status.json', '.queue.lock',
                  'FULL_STUDY_TRAINING_PREVIEW.png', 'FULL_STUDY_TRAINING_PREVIEW.json'}


class PipelineError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path, heartbeat=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
            if heartbeat:
                heartbeat()
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def neural_gate(queue, complete_ids, required_ids, heartbeat_age, stale_after=900):
    """Queue metadata can delay/fail the pipeline, but cannot prove completion."""
    if queue and (queue.get('status') == 'failed' or queue.get('failed')):
        raise PipelineError('Neural queue failed: '+str(queue.get('failed', {})))
    if queue and queue.get('status') == 'complete':
        if queue.get('running') or queue.get('pending'):
            raise PipelineError('Queue claims complete while jobs remain running or pending')
        missing = set(required_ids)-set(complete_ids)
        if missing:
            raise PipelineError('Queue stopped with missing/unverified neural jobs: '+', '.join(sorted(missing)))
        return True
    if heartbeat_age > stale_after:
        raise PipelineError(f'Neural queue heartbeat is stale ({heartbeat_age:.0f}s); no timing or publication started')
    return False


def requested_updates(job):
    config = job['config']
    if job['kind'] == 'transfer':
        return len(config['ratios'])*config['steps']
    factor = 1 if config['mode'] == 'coupled' else 2
    return factor*(config['steps'] if job['kind'] == 'baseline'
                   else config['adam_steps']+config['lbfgs_steps'])


def last_history_row(path):
    """Read only a small tail; a worker may currently be appending a row."""
    try:
        with Path(path).open('rb') as stream:
            header = stream.readline().decode('utf-8-sig').strip()
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size-32768))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        fields = next(csv.reader([header]))
        for line in reversed(lines):
            values = next(csv.reader([line]))
            if len(values) == len(fields) and line != header:
                return dict(zip(fields, values))
    except (OSError, ValueError, StopIteration, csv.Error):
        pass
    return {}


def progress_rows(root, jobs, queue, verified):
    """Lightweight metadata only: never load model.pt/NPZ in active training."""
    rows = []
    queue = queue or {}
    for job in jobs:
        name = job['id']
        folder = Path(root)/'results/full_study'/name
        config = read_json(folder/'config.json', job['config'])
        state = read_json(folder/'run_status.json', {})
        info = read_json(folder/'experiment.json', {})
        latest = last_history_row(folder/'history.csv')
        actual = info.get('completed_updates', info.get('actual_optimizer_updates', 0))
        counters = state.get('completed', {})
        if isinstance(counters, dict):
            actual = max(actual, sum(counters.values()))
        try:
            logged = int(latest.get('completed_updates', latest.get('step', 0)))
            if job['kind'] not in ('baseline', 'transfer') and latest.get('optimizer') == 'lbfgs':
                logged += counters.get('adam', 0)
            actual = max(actual, logged)
        except (TypeError, ValueError):
            pass
        if name in verified:
            status, detail = 'complete', 'Artifact verified'
        elif name in queue.get('failed', {}) or state.get('status') == 'failed':
            status, detail = 'failed', str(queue.get('failed', {}).get(name, 'worker failed'))
        elif name in queue.get('running', []):
            status, detail = 'running', latest.get('active', latest.get('ratio', 'worker active'))
        else:
            status = 'pending'
            detail = ('Worker reports complete; artifact verification pending'
                      if name in queue.get('completed', []) else 'Waiting')
        loss = latest.get('total', latest.get('loss'))
        try:
            loss = float(loss)
            if not math.isfinite(loss):
                loss = None
        except (TypeError, ValueError):
            loss = None
        rows.append({'id': name, 'kind': job['kind'], 'status': status, 'detail': detail,
                     'actual_logged_or_checkpoint_updates': int(actual),
                     'requested_updates': requested_updates(job), 'last_loss': loss,
                     'optimizer': config.get('optimizer', 'Adam/L-BFGS'),
                     'width': config.get('width'), 'layers': config.get('layers'),
                     'backend': config.get('backend', config.get('jacobian_backend', 'autograd'))})
    return rows


def validate_notebook_defaults(path):
    book = read_json(path)
    if not isinstance(book, dict):
        raise PipelineError('Notebook is missing or invalid: '+str(path))
    required = {'RUN_FULL_STUDY': False, 'RUN_SMOKE': False,
                'BUILD_FIGURE_PACK': False, 'RESUME_JOB_ID': None}
    seen = set()
    for cell in book.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = ''.join(source) if isinstance(source, list) else source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in required:
                        seen.add(target.id)
                        if not isinstance(node.value, ast.Constant) or node.value.value is not required[target.id]:
                            raise PipelineError('Notebook must retain safe default '+target.id)
    if set(required)-seen:
        raise PipelineError('Notebook lacks safe default switches: '+', '.join(sorted(set(required)-seen)))


def validate_notebook(path):
    book = read_json(path)
    if not isinstance(book, dict):
        raise PipelineError('Notebook is missing or invalid')
    counts = []
    for index, cell in enumerate(book.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        if not ''.join(source).strip():
            continue
        if not isinstance(cell.get('execution_count'), int):
            raise PipelineError(f'Notebook has an unexecuted code cell: {index}')
        if any(value.get('output_type') == 'error' for value in cell.get('outputs', [])):
            raise PipelineError(f'Notebook contains an error output: cell {index}')
        counts.append(cell['execution_count'])
    if not counts:
        raise PipelineError('Notebook contains no executed code cells')
    return {'code_cells': len(counts), 'execution_counts': counts, 'error_outputs': 0,
            'sha256': sha256(path)}


def export_notebook_html(path, target):
    import nbformat
    from nbconvert import HTMLExporter
    book = nbformat.read(path, as_version=4)
    body, _ = HTMLExporter(template_name='lab', theme='light', embed_images=True).from_notebook_node(book)
    css = '''<style id="portable-light-code">
body,.jp-Notebook{background:#fff!important;color:#172b44!important}
.jp-CodeCell .jp-InputArea-editor,.jp-CodeCell .highlight,.highlight pre{background:#f6f8fa!important;color:#1f2328!important}
.highlight span{color:#1f2328!important}.highlight .k,.highlight .kn,.highlight .kd,.highlight .ow{color:#8250df!important}
.highlight .s,.highlight .s1,.highlight .s2{color:#0a3069!important}.highlight .c,.highlight .c1{color:#57606a!important}
.jp-InputPrompt,.jp-OutputPrompt{color:#57606a!important}</style>'''
    body = body.replace('</head>', css+'\n</head>').replace('<html lang="en">', '<html lang="ko">')
    atomic_text(target, body)


def verify_reference(folder, expected_problem, n, mode):
    import numpy as np
    from harandi.physics import grid
    folder = Path(folder)
    metadata = read_json(folder/'metadata.json')
    if not metadata or metadata.get('problem') != expected_problem:
        raise PipelineError('FEM problem metadata mismatch: '+str(folder))
    if metadata.get('n') != n or metadata.get('coefficient_mode') != mode:
        raise PipelineError('FEM mesh/coefficient metadata mismatch: '+str(folder))
    diagnostics = metadata.get('diagnostics', {})
    for name in ('thermal_relative_residual', 'mechanical_relative_residual'):
        value = diagnostics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1e-8:
            raise PipelineError('FEM residual check failed: '+str(folder)+' '+name)
    balance = diagnostics.get('heat_balance', float('nan'))
    scale = max(1., abs(diagnostics.get('heat_left_reaction', 0)), abs(diagnostics.get('heat_right_reaction', 0)))
    if not math.isfinite(balance) or abs(balance) > 1e-8*scale:
        raise PipelineError('FEM heat-balance check failed: '+str(folder))
    with np.load(folder/'solution.npz', allow_pickle=False) as data:
        if data['fields'].shape != (10201, 8) or not np.array_equal(data['xy'], grid(101)):
            raise PipelineError('FEM evaluation grid/field shape mismatch: '+str(folder))
        if any(not np.isfinite(data[name]).all() for name in data.files if np.issubdtype(data[name].dtype, np.number)):
            raise PipelineError('FEM fields are not finite: '+str(folder))
    return {'folder': str(folder), 'points': 10201, 'n': n, 'coefficient_mode': mode,
            'diagnostics': diagnostics, 'solution_sha256': sha256(folder/'solution.npz'),
            'metadata_sha256': sha256(folder/'metadata.json')}


def verify_references(root):
    from harandi.physics import Problem
    records = []
    for geometry in ('circle', 'rectangles'):
        for suffix, n, mode in (('n50', 50, 'sharp'), ('n100', 100, 'sharp'),
                                ('n200', 200, 'sharp'), ('n100_nodal', 100, 'nodal')):
            records.append(verify_reference(Path(root)/f'results/fem_{geometry}_{suffix}',
                                             Problem(geometry).to_dict(), n, mode))
    for ratio in list(range(1, 11))+[15]:
        records.append(verify_reference(Path(root)/f'results/full_study/parametric_fem/ratio_{ratio}',
                                         Problem('rectangles', ratio=ratio).to_dict(), 50, 'sharp'))
    return records


def verify_sources(root):
    root = Path(root)
    provenance = read_json(root/'SOURCE_PROVENANCE.json')
    if not provenance:
        raise PipelineError('Missing SOURCE_PROVENANCE.json')
    public = (provenance.get('distribution', {}).get('mode') == 'public'
              and provenance.get('distribution', {}).get('external_paper_and_crops_included') is False)
    paper = Path(provenance['paper']['path'])
    if not paper.is_absolute():
        paper = root/paper
    expected = [(paper, provenance['paper']['sha256'], public)]
    expected += [(root/'data'/name, digest, False) for name, digest in provenance.get('included_data', {}).items()]
    style = provenance.get('paper_visual_style', {})
    for key in ('palette', 'ranges'):
        if key in style:
            expected.append((root/style[key]['path'], style[key]['sha256'], False))
    expected += [(root/'figures/paper_reference'/name, digest, public)
                 for name, digest in style.get('original_comparison_crops', {}).items()]
    records = []
    for path, digest, external in expected:
        if external and not path.is_file():
            records.append({'path': str(path), 'expected_sha256': digest,
                            'status': 'not_distributed_not_verified'})
            continue
        if not path.is_file() or sha256(path) != digest:
            raise PipelineError('Source hash verification failed: '+str(path))
        records.append({'path': str(path), 'sha256': digest, 'status': 'verified'})
    return records


def verify_report_qa(root):
    root = Path(root)
    path = root/'FULL_REPORT_LAYOUT_QA.json'
    record = read_json(path, {})
    if record.get('status') != 'pass' or record.get('automatic_checks_passed') is not True:
        raise PipelineError('Final PDF automatic layout QA did not pass')
    if record.get('pdf_sha256') != sha256(root/'FULL_PROJECT_REPORT_KO.pdf'):
        raise PipelineError('Final PDF layout QA hash is stale')
    pages = record.get('page_count')
    if not isinstance(pages, int) or pages < 1 or record.get('rendered_page_count') != pages:
        raise PipelineError('Final PDF layout QA did not render every page')
    return {'status': 'pass', 'automatic_checks_passed': True, 'pdf_sha256': record['pdf_sha256'],
            'page_count': pages, 'rendered_page_count': pages, 'qa_json_sha256': sha256(path),
            'manual_visual_review': record.get('manual_visual_review', {'performed': False})}


def archive_members(root):
    root = Path(root)
    result = []
    for directory, folders, names in os.walk(root):
        folders[:] = [name for name in folders if name.lower() not in EXCLUDED_PARTS
                      and not (Path(directory)/name).is_symlink()]
        for name in names:
            path = Path(directory)/name
            if not path.is_file() or path.is_symlink():
                continue
            if (name in EXCLUDED_NAMES or name.startswith('.full_pipeline')
                    or name.lower().startswith('checkpoint') or path.suffix.lower() in ('.zip', '.tmp', '.pyc', '.pyo', '.part')):
                continue
            result.append(path)
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def write_bundle(root, target, verification, heartbeat=None):
    """The manifest becomes complete only after the temporary ZIP verifies."""
    root, target = Path(root), Path(target)
    if target.name.lower() == 'harandi_reproduction.zip':
        raise PipelineError('The previous harandi_reproduction.zip must be preserved')
    last_beat = time.monotonic()
    def pulse():
        nonlocal last_beat
        current = time.monotonic()
        if heartbeat and current-last_beat >= 30:
            heartbeat()
            last_beat = current
    members = archive_members(root)
    files = {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha256(path, pulse)}
             for path in members}
    manifest = {'status': 'complete', 'created_utc': now(), 'verification': verification,
                'archive_name': target.name, 'archive_root': 'harandi_reproduction',
                'archive_policy': 'Compact model.pt retained; optimizer checkpoints, environments, caches, temporary files and live progress snapshots excluded. Scripts can rerun the study.',
                'files': files}
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for path in members:
                info = zipfile.ZipInfo.from_file(path, 'harandi_reproduction/'+path.relative_to(root).as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as destination:
                    for chunk in iter(lambda: source.read(1024*1024), b''):
                        destination.write(chunk)
                        pulse()
            archive.writestr('harandi_reproduction/FULL_STUDY_MANIFEST.json', payload)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise PipelineError('ZIP CRC validation failed')
            for name, record in files.items():
                digest = hashlib.sha256()
                with archive.open('harandi_reproduction/'+name) as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(chunk)
                        pulse()
                if digest.hexdigest() != record['sha256']:
                    raise PipelineError('File changed during packaging: '+name)
        temporary.replace(target)
        atomic_text(root/'FULL_STUDY_MANIFEST.json', payload.decode('utf-8'))
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha256(target, pulse),
            'included_files': len(files)+1}


def pid_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: conservatively live
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _set_execution_state(flags):
    import ctypes
    from ctypes import wintypes
    function = ctypes.WinDLL('kernel32', use_last_error=True).SetThreadExecutionState
    function.argtypes = [wintypes.DWORD]
    function.restype = wintypes.DWORD
    return function(flags)


class SleepGuard:
    """Thread-scoped system-awake request; never changes a global power plan."""
    def __init__(self):
        self.active = False
        self.warning = None

    def enable(self):
        if os.name != 'nt':
            return
        try:
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; deliberately no display flag.
            self.active = bool(_set_execution_state(0x80000001))
            if not self.active:
                raise OSError('SetThreadExecutionState returned zero')
        except OSError as error:
            self.warning = 'Windows keep-awake request failed: '+str(error)
            print(self.warning, file=sys.stderr, flush=True)

    def disable(self):
        if self.active:
            try:
                if not _set_execution_state(0x80000000):
                    raise OSError('SetThreadExecutionState reset returned zero')
            except OSError as error:
                self.warning = 'Windows keep-awake reset failed: '+str(error)
                print(self.warning, file=sys.stderr, flush=True)
            finally:
                self.active = False  # process exit also releases the native request


class Finisher:
    def __init__(self, root, document_python=None, poll_seconds=45, stale_after=900,
                 notebook_timeout=3600, archive=None, qa_workdir=None):
        self.root = Path(root).resolve()
        sys.path.insert(0, str(self.root))
        from harandi import full_study
        self.study_module = full_study
        self.jobs = full_study.catalogue()
        if len(self.jobs) != 52 or sum(j['kind'] == 'timing' for j in self.jobs) != 25:
            raise PipelineError('Expected the audited 52-job catalogue (27 neural +25 timing)')
        self.study = self.root/'results/full_study'
        self.document_python = str(document_python or os.environ.get('HARANDI_DOCUMENT_PYTHON') or sys.executable)
        self.poll, self.stale = poll_seconds, stale_after
        self.notebook_timeout = notebook_timeout
        self.qa_workdir = Path(qa_workdir) if qa_workdir else self.root.parents[1]/'work/full_study/report_layout_qa'
        self.archive = Path(archive) if archive else self.root.parent/'harandi_full_study.zip'
        self.started, self.started_clock = now(), time.time()
        self.phase, self.state, self.detail = 'waiting_neural', 'pending', 'Waiting for the existing neural queue'
        self.verified = set()
        self.child = None
        self.archive_info = None
        self.lock = self.root/'.full_pipeline.lock'
        self.awake = SleepGuard()

    def acquire(self):
        state = read_json(self.root/'full_pipeline_status.json', {})
        if pid_alive(state.get('child_pid')):
            raise PipelineError('Another finalizer stage process is still running')
        if self.lock.exists():
            previous = read_json(self.lock, {})
            if pid_alive(previous.get('pid')):
                raise PipelineError('Another finalizer or its stage process is still running')
            self.lock.unlink()
        with self.lock.open('x', encoding='utf-8') as stream:
            json.dump({'pid': os.getpid(), 'started_utc': self.started}, stream)

    def progress(self):
        queue = read_json(self.study/'queue_status.json')
        rows = progress_rows(self.root, self.jobs, queue, self.verified)
        status = {'status': self.state, 'phase': self.phase, 'detail': self.detail,
                  'started_utc': self.started, 'updated_utc': now(), 'pid': os.getpid(),
                  'child_pid': None if self.child is None else self.child.pid,
                  'windows_keep_awake': {'active': self.awake.active, 'warning': self.awake.warning},
                  'verified_complete_jobs': len(self.verified), 'registered_jobs': len(self.jobs),
                  'queue_reported_complete_jobs': len((queue or {}).get('completed', [])),
                  'archive': self.archive_info, 'jobs': rows}
        atomic_json(self.root/'full_pipeline_status.json', status)
        cells = []
        for row in rows:
            loss = '—' if row['last_loss'] is None else f'{row["last_loss"]:.6g}'
            cells.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in (
                row['id'], row['status'], row['actual_logged_or_checkpoint_updates'],
                row['requested_updates'], loss, row['detail']))+'</tr>')
        preview = read_json(self.root/'FULL_STUDY_TRAINING_PREVIEW.json', {})
        preview_html = ''
        if preview.get('status') == 'intermediate_only' and (self.root/'FULL_STUDY_TRAINING_PREVIEW.png').exists():
            count = sum(preview.get('completed', {}).values())
            caption = (f'Static intermediate preview: {preview.get("job", "unknown")}, '
                       f'{count:,} checkpoint updates, generated {preview.get("generated_utc", "unknown")}. '
                       'This image is not live and is not a completed experiment.')
            preview_html = ('<figure><img style="max-width:1100px;width:100%" src="FULL_STUDY_TRAINING_PREVIEW.png" '
                            'alt="Static intermediate training preview"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
        document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                    f'<meta http-equiv="refresh" content="{self.poll}"><title>Harandi full-study progress</title>'
                    '<style>body{font:15px system-ui;margin:32px;color:#172b44;background:#fff}table{border-collapse:collapse;width:100%}'
                    'th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}th{background:#eef3f7}.note{color:#58616d}</style>'
                    '<h1>Harandi full-study progress</h1>'
                    f'<p><b>{html.escape(self.state)} · {html.escape(self.phase)}</b> — {html.escape(self.detail)}</p>'
                    f'<p>Artifact-verified jobs: {len(self.verified)}/{len(self.jobs)}. Updated {html.escape(status["updated_utc"])}.</p>'
                    '<p class="note">Counts below come from the worker’s latest history/status. A worker completion label is verified only after the neural queue stops. '
                    'The final report and ZIP are complete only when this pipeline says complete.</p>'
                    '<table><thead><tr><th>Job</th><th>Status</th><th>Observed updates</th><th>Requested</th><th>Last loss</th><th>Evidence/status</th></tr></thead>'
                    '<tbody>'+''.join(cells)+'</tbody></table>'+preview_html+'</html>')
        atomic_text(self.root/'FULL_STUDY_PROGRESS.html', document)

    def verify_jobs(self, jobs):
        self.study_module._COMPLETE_CACHE.clear()
        return {job['id'] for job in jobs if self.study_module.job_complete(job, self.study)}

    def wait_neural(self):
        neural = [job for job in self.jobs if job['kind'] != 'timing']
        required = {job['id'] for job in neural}
        path = self.study/'queue_status.json'
        while True:
            queue = read_json(path)
            age = time.time()-(path.stat().st_mtime if path.exists() else self.started_clock)
            if queue and queue.get('status') == 'complete' and not queue.get('failed'):
                self.detail = 'Queue stopped; independently verifying all neural artifacts'
                self.progress()
                self.verified = self.verify_jobs(neural)
            if neural_gate(queue, self.verified, required, age, self.stale):
                self.progress()
                return
            self.state = 'running'
            self.progress()
            time.sleep(self.poll)

    def command(self, phase, arguments):
        self.phase, self.state = phase, 'running'
        self.detail = 'Executing '+phase+'; failures are retained in the stage log'
        self.progress()
        logs = self.study/'pipeline_logs'
        logs.mkdir(parents=True, exist_ok=True)
        log = logs/(phase+'.log')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', HARANDI_PYTHON=sys.executable)
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        print(phase, flush=True)
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\n'+now()+' '+json.dumps(list(map(str, arguments)))+'\n')
            stream.flush()
            self.child = subprocess.Popen(list(map(str, arguments)), cwd=self.root, env=env,
                stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            self.progress()
            while True:
                try:
                    code = self.child.wait(timeout=self.poll)
                    break
                except subprocess.TimeoutExpired:
                    self.progress()
            self.child = None
        if code:
            raise PipelineError(f'{phase} failed with exit {code}; inspect {log.relative_to(self.root)}')
        self.progress()

    def verify_deliverables(self):
        self.verified = self.verify_jobs(self.jobs)
        missing = {job['id'] for job in self.jobs}-self.verified
        if missing:
            raise PipelineError('Final artifact verification failed: '+', '.join(sorted(missing)))
        report = read_json(self.root/'FULL_STUDY_RESULTS.json', {})
        if (report.get('status') != 'complete' or report.get('partial_requested')
                or set(report.get('completed_jobs', [])) != self.verified
                or report.get('pending_jobs') or report.get('pending_references') or report.get('skipped_figures')):
            raise PipelineError('Full-study figures/results are incomplete or stale')
        for name, row in report.get('jobs', {}).items():
            if row.get('prediction_sha256') != sha256(self.study/name/'prediction.npz'):
                raise PipelineError('Figure source prediction changed: '+name)
        for figure in report.get('figures', []):
            if not (self.root/figure['png']).is_file():
                raise PipelineError('Missing final figure: '+figure['png'])
        for name in ('FULL_PROJECT_REPORT_KO.pdf', 'FULL_PROJECT_REPORT_KO.md',
                     'Full_Paper_Reproduction_KO.html', 'README_FULL_STUDY_KO.md',
                     'PAPER_EXPERIMENT_INVENTORY.json', 'FULL_STUDY_PROTOCOL.json'):
            path = self.root/name
            if not path.is_file() or path.stat().st_size == 0:
                raise PipelineError('Missing final deliverable: '+name)
        pdf = self.root/'FULL_PROJECT_REPORT_KO.pdf'
        with pdf.open('rb') as stream:
            header = stream.read(5)
            stream.seek(max(0, pdf.stat().st_size-1024))
            trailer = stream.read()
        if header != b'%PDF-' or b'%%EOF' not in trailer:
            raise PipelineError('Final report is not a complete PDF')
        return {'verified_jobs': len(self.verified), 'catalogue': self.jobs,
                'fem_references': verify_references(self.root), 'source_hashes': verify_sources(self.root),
                'notebook': validate_notebook(self.root/'Full_Paper_Reproduction_KO.ipynb'),
                'figure_count': len(report.get('figures', [])), 'strict_figure_status': report['status'],
                'report_layout_qa': verify_report_qa(self.root),
                'report_pdf_check': 'ReportLab and automatic all-page layout/render checks succeeded. No new manual visual-review claim is implied.'}

    def run(self):
        self.acquire()
        try:
            self.awake.enable()
            self.progress()
            self.wait_neural()
            timing = [job for job in self.jobs if job['kind'] == 'timing']
            if self.verify_jobs(timing) != {job['id'] for job in timing}:
                # wait_neural has just required queue complete/no live workers.
                self.command('timing', [sys.executable, '-m', 'harandi.full_study',
                             '--groups', 'timing', '--workers', '1', '--out', self.study])
            self.verified = self.verify_jobs(self.jobs)
            if len(self.verified) != len(self.jobs):
                raise PipelineError('Timing finished without all 52 jobs verified')
            self.command('figures', [sys.executable, '-m', 'harandi.full_figures', '--root', self.root])
            notebook = self.root/'Full_Paper_Reproduction_KO.ipynb'
            validate_notebook_defaults(notebook)
            self.command('notebook', [sys.executable, '-m', 'nbconvert', '--execute', '--to', 'notebook',
                '--inplace', '--ExecutePreprocessor.kernel_name=harandi',
                f'--ExecutePreprocessor.timeout={self.notebook_timeout}', notebook])
            validate_notebook(notebook)
            self.phase = 'notebook_html'
            self.detail = 'Exporting executed notebook with a portable light theme'
            self.progress()
            export_notebook_html(notebook, self.root/'Full_Paper_Reproduction_KO.html')
            self.command('report', [self.document_python, self.root/'scripts/build_full_korean_report.py',
                         '--root', self.root, '--pinn-python', sys.executable])
            self.command('report_layout_qa', [self.document_python, self.root/'scripts/check_full_report_layout.py',
                         '--root', self.root, '--workdir', self.qa_workdir])
            self.phase, self.detail = 'validation', 'Verifying all jobs, 19 FEM references, notebook and source hashes'
            self.progress()
            verification = self.verify_deliverables()
            self.phase, self.detail = 'archive', 'Building and checksum-verifying the new compact archive'
            self.progress()
            self.archive_info = write_bundle(self.root, self.archive, verification, heartbeat=self.progress)
            self.phase, self.state, self.detail = 'finished', 'complete', 'All registered jobs and final deliverables verified; new archive verified'
            self.progress()
            print('Complete:', self.archive, flush=True)
            return 0
        except BaseException as error:
            self.state, self.detail = 'failed', str(error) or type(error).__name__
            self.progress()
            print('FAILED:', self.detail, file=sys.stderr, flush=True)
            return 1
        finally:
            self.awake.disable()
            try:
                self.progress()
            finally:
                self.lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--document-python', help='Python with requirements-documents.txt; also HARANDI_DOCUMENT_PYTHON')
    parser.add_argument('--poll-seconds', type=int, default=45)
    parser.add_argument('--stale-after', type=int, default=900, help='Fail when a running queue stops updating its heartbeat')
    parser.add_argument('--notebook-timeout', type=int, default=3600)
    parser.add_argument('--archive', type=Path, help='Default: sibling harandi_full_study.zip; earlier archive is protected')
    parser.add_argument('--qa-workdir', type=Path, help='PDF page render scratch; default outside the project/archive')
    args = parser.parse_args()
    if not 30 <= args.poll_seconds <= 60 or args.stale_after < 2*args.poll_seconds:
        parser.error('poll-seconds must be 30..60 and stale-after at least two polls')
    return Finisher(**vars(args)).run()


if __name__ == '__main__':
    raise SystemExit(main())
