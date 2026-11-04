"""Complete identifiable manuscript experiment families, with restartable jobs.

Manuscript equations/settings contain ambiguities. See docs/PAPER_EXPERIMENT_INVENTORY.json
and docs/FULL_STUDY_PROTOCOL.json; full budgets do not imply bitwise author equivalence.
"""
import argparse,json,os,subprocess,sys,time,hashlib,shutil,pickle
from contextlib import contextmanager
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/full_study'
_COMPLETE_CACHE={}


def catalogue():
    jobs=[]
    def mixed(name,**kw):
        config=dict(geometry='rectangles',mode='coupled',n=51,width=40,layers=5,hard=False,
                    energy='authors',adam_steps=100000,lbfgs_steps=0,switch=20000,
                    log_every=1000,checkpoint_every=1000,fast_adam=True,dtype='float64',left_refine=0)
        config.update(kw)
        jobs.append(dict(id=name,kind='mixed',config=config))
    for geometry in ('rectangles','circle'):
        for mode in ('coupled','sequential'):mixed(f'mixed_{geometry}_{mode}',geometry=geometry,mode=mode,reset_mechanical_adam=mode=='sequential')
    for method in ('standard','dem'):
        for mode in ('coupled','sequential'):
            jobs.append(dict(id=f'baseline_circle_{method}_{mode}',kind='baseline',config=dict(
                method=method,geometry='circle',mode=mode,steps=100000,switch=20000,hard=False,
                width=40,layers=5,n=51,optimizer='adam',lr=.001,dtype='float64',device='auto',
                seed=42,log_every=1000,checkpoint_every=1000,fast_adam=True,
                backend='jets' if method=='standard' else 'autograd')))
    for mode in ('coupled','sequential'):
        job=dict(next(j for j in jobs if j['id']==f'baseline_circle_dem_{mode}'))
        job['id']=f'sensitivity_circle_dem_hard_{mode}'
        job['config']=dict(job['config'],hard=True)
        job['purpose']='Additional fairness check: finite unit soft penalty biases the DEM energy minimizer.'
        jobs.append(job)
    mixed('refine_rectangles_left1000',left_refine=1000)
    mixed('ablation_circle_noenergy',geometry='circle',energy='none')
    mixed('hard_rectangles_adam10k',hard=True,adam_steps=10000,log_every=100)
    architectures=sorted({(l,40) for l in (1,2,4,5,6)}|{(5,w) for w in (5,10,20,40,60)})
    for layers,width in architectures:
        mixed(f'arch_l{layers}_w{width}',hard=True,layers=layers,width=width,
              adam_steps=0,lbfgs_steps=10000,fast_adam=False,log_every=100)
    for weight in (0.,.1,.5,1.):
        jobs.append(dict(id=f'transfer_w{weight:g}',kind='transfer',config=dict(
            ratios=list(range(1,11)),steps=10000,weight=weight,n=51,width=40,layers=5,
            device='auto',energy='authors',dtype='float64',hard=False,normalization='none',
            fast_adam=True,log_every=1000,checkpoint_every=1000)))
    # Table4 is a timing measurement over100updates, not25convergence studies.
    for layers in (1,2,4,5,6):
        for width in (5,10,20,40,60):
            mixed(f'timing_l{layers}_w{width}',hard=True,layers=layers,width=width,
                  adam_steps=0,lbfgs_steps=100,fast_adam=False,log_every=100)
            jobs[-1]['kind']='timing'
    mixed('remark_rectangles_adam1m',adam_steps=1000000)
    jobs[-1]['continue_from']='mixed_rectangles_coupled'
    jobs[-1]['purpose']='Remark7 text-only1M Adam claim; explicit soft-BC reconstruction choice, same trajectory extended from100k.'
    return jobs


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def job_complete(job,out=OUT):
    folder=Path(out)/job['id']
    if not (folder/'prediction.npz').exists() or not (folder/'model.pt').exists():return False
    try:
        signature=tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(folder.glob('*'))
                        if p.name in ('prediction.npz','model.pt','config.json','experiment.json','run_status.json','held_out_ratio15_errors.json'))
        if job['kind']=='transfer':
            signature+=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size)
                for p in sorted((Path(out)/'parametric_fem').rglob('*')) if p.name in ('solution.npz','metadata.json'))
        key=(str(folder.resolve()),json.dumps(job,sort_keys=True))
        if _COMPLETE_CACHE.get(key)==signature:return True
        if job['kind']=='transfer':
            meta=json.loads((folder/'experiment.json').read_text())
            config=json.loads((folder/'config.json').read_text())
            ignored=('device','log_every','checkpoint_every')
            if any(config.get(k)!=v for k,v in job['config'].items() if k not in ignored):return False
            if not (meta['status']=='complete' and meta['actual_optimizer_updates']==100000
                    and meta['training_ratios']==list(range(1,11)) and meta['physics_weight']==job['config']['weight']):return False
            evaluation=meta.get('held_out_evaluation') or {}
            if evaluation.get('used_for_training_or_selection') is not False or evaluation.get('points')!=10201:return False
            if not (folder/'held_out_ratio15_errors.json').exists():return False
            fingerprints=list(meta['training_references'])+[evaluation['reference']]
            for ref in fingerprints:
                reference=Path(out)/'parametric_fem'/ref['file']
                if hashlib.sha256(reference.read_bytes()).hexdigest()!=ref['sha256']:return False
                metadata=reference.parent/'metadata.json'
                if ref.get('metadata_verified') and hashlib.sha256(metadata.read_bytes()).hexdigest()!=ref['metadata_sha256']:return False
        elif job['kind']=='baseline':
            config=json.loads((folder/'config.json').read_text())
            meta=json.loads((folder/'experiment.json').read_text())
            expected=job['config']['steps']*(1 if job['config']['mode']=='coupled' else 2)
            if not (all(config.get(k)==v for k,v in job['config'].items())
                    and meta['complete'] and meta['completed_updates']==expected):return False
        else:
            config=json.loads((folder/'config.json').read_text());meta=json.loads((folder/'run_status.json').read_text())
            config.setdefault('reset_mechanical_adam',False)
            if any(config.get(k)!=v for k,v in job['config'].items()):return False
            factor=1 if config['mode']=='coupled' else 2
            expected={name:factor*config[name+'_steps'] for name in ('adam','lbfgs')}
            if not (meta['status']=='complete' and meta['completed']==expected):return False
        import numpy as np
        import torch
        from .physics import Problem,grid
        from .pinn import MixedPINN
        with np.load(folder/'prediction.npz') as data:
            if data['fields'].shape!=(10201,8) or not np.array_equal(data['xy'],grid(101)):return False
            if data['constitutive_stress'].shape!=(10201,3) or data['fourier_flux'].shape!=(10201,2):return False
            if any(not np.isfinite(data[k]).all() for k in ('fields','constitutive_stress','fourier_flux')):return False
        checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        saved_config=checkpoint['transfer_config'] if job['kind']=='transfer' else checkpoint['config']
        for setting in ('geometry','width','layers','hard','dtype','energy','mode','n','normalization'):
            if setting in config and saved_config.get(setting)!=config[setting]:return False
        if not all(torch.isfinite(t).all() for t in checkpoint['model'].values()):return False
        if job['kind']=='baseline':
            from .baselines import PrimaryPINN
            model=PrimaryPINN(Problem(config['geometry']),config['width'],config['layers'],config['hard'])
            if checkpoint['completed']!=expected:return False
        else:
            model=MixedPINN(config['width'],config['layers'],config['hard'],parametric=job['kind']=='transfer')
            if job['kind']=='transfer':
                if checkpoint['completed_updates']!=100000:return False
            elif checkpoint['completed']!=expected:return False
        model.load_state_dict(checkpoint['model'],strict=True)
        _COMPLETE_CACHE[key]=signature
        return True
    except (OSError,KeyError,ValueError,RuntimeError,EOFError,pickle.UnpicklingError):return False


def execute_job(job,out=OUT):
    folder=Path(out)/job['id'];folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'study_job.json',job)
    resume=folder/'checkpoint.pt'
    if job.get('continue_from') and not resume.exists():
        source_job=next(j for j in catalogue() if j['id']==job['continue_from'])
        if not job_complete(source_job,out):raise RuntimeError('Required100k precursor is incomplete')
        source=Path(out)/job['continue_from']
        shutil.copy2(source/'checkpoint.pt',resume)
        shutil.copy2(source/'history.csv',folder/'history.csv')
        atomic_json(folder/'continuation.json',dict(source=job['continue_from'],
            source_checkpoint_sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),
            reused_updates=100000,additional_requested_updates=900000))
    if job['kind']=='transfer':
        from .transfer import run
        run(Path(out)/'parametric_fem',folder,resume=resume if resume.exists() else None,**job['config'])
    elif job['kind']=='baseline':
        from .baselines import BaselineConfig,run_config
        run_config(BaselineConfig(**job['config']),folder,resume=resume if resume.exists() else None)
    else:
        from .train import TrainConfig,run
        run(TrainConfig(**job['config']),folder,resume=resume if resume.exists() else None)
    if not job_complete(job,out):raise RuntimeError('Job returned without verified completion: '+job['id'])


def fem_references(out=OUT):
    for ratio in list(range(1,11))+[15]:
        folder=Path(out)/'parametric_fem'/f'ratio_{ratio}'
        if (folder/'metadata.json').exists() and (folder/'solution.npz').exists():continue
        args=['-m','harandi.fem','--geometry','rectangles','--ratio',str(ratio),
              '--n','50','--output',folder.relative_to(ROOT).as_posix()]
        if os.name=='nt':cmd=['wsl.exe','-d','Ubuntu','--cd',str(ROOT),'--','env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','python3']+args
        else:cmd=[os.environ.get('FENICS_PYTHON','/usr/bin/python3')]+args
        subprocess.run(cmd,cwd=ROOT,check=True)


@contextmanager
def queue_lock(out):
    """OS-released writer lock: a second notebook must not corrupt active runs."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    stream=(out/'.queue.lock').open('a+b')
    locked=False
    try:
        if stream.seek(0,os.SEEK_END)==0:
            stream.write(b' ');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as error:
            raise RuntimeError('A study queue is already running in '+str(out)) from error
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def run_queue(out=OUT,groups=None,workers=1,exclude=()):
    with queue_lock(out):
        status=Path(out)/'queue_status.json'
        # Protect a still-running worker started before the OS lock was added.
        if status.exists() and time.time()-status.stat().st_mtime<10:
            previous=json.loads(status.read_text(encoding='utf-8'))
            if previous.get('status')=='running':
                raise RuntimeError('A study queue is already running (recent heartbeat).')
        return _run_queue_unlocked(out,groups,workers,exclude)


def _run_queue_unlocked(out=OUT,groups=None,workers=1,exclude=()):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    jobs=[j for j in catalogue() if (groups is None or j['kind'] in groups) and j['id'] not in exclude]
    if workers<1:raise ValueError('workers must be positive')
    if any(j['kind']=='timing' for j in jobs) and workers!=1:raise ValueError('Timing measurements require one worker and no other GPU jobs')
    atomic_json(out/'catalogue.json',catalogue())
    running={};failed={};started=time.time()
    pending=[j for j in jobs if not job_complete(j,out)]
    while pending or running:
        while pending and len(running)<workers:
            job=pending.pop(0);folder=out/job['id'];folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'console.log').open('a',encoding='utf-8')
            process=subprocess.Popen([sys.executable,'-m','harandi.full_study','--job',job['id'],'--out',str(out)],
                                     cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            running[job['id']]=(job,process,log)
            print('START',job['id'],'pid',process.pid,flush=True)
        for name,(job,process,log) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();del running[name]
            if code or not job_complete(job,out):failed[name]=code if code else 'incomplete'
            print('FINISH',name,'exit',code,flush=True)
        atomic_json(out/'queue_status.json',dict(status='running',started_unix=started,
            completed=[j['id'] for j in jobs if job_complete(j,out)],running=list(running),
            pending=[j['id'] for j in pending],failed=failed))
        if running:time.sleep(2)
    atomic_json(out/'queue_status.json',dict(status='failed' if failed else 'complete',started_unix=started,
        completed=[j['id'] for j in jobs if job_complete(j,out)],running=[],pending=[],failed=failed))
    if failed:raise RuntimeError(f'Failed jobs, inspect logs and resume: {failed}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--job')
    ap.add_argument('--groups',nargs='+',choices=['mixed','baseline','transfer','timing'])
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--exclude',nargs='*',default=[])
    ap.add_argument('--fem-only',action='store_true');ap.add_argument('--list',action='store_true')
    a=ap.parse_args()
    if a.list:print(json.dumps(catalogue(),indent=2));return
    if a.fem_only:fem_references(a.out);return
    if a.job:
        matches=[j for j in catalogue() if j['id']==a.job]
        if not matches:raise ValueError('Unknown study job')
        execute_job(matches[0],a.out);return
    run_queue(a.out,a.groups,a.workers,a.exclude)


if __name__=='__main__':main()
