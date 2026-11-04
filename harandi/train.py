"""Train, resume and export mixed PINNs. FEM is used only for final evaluation.

Example: python -m harandi.train --geometry rectangles --mode coupled
         --adam-steps 2000 --lbfgs-steps 3000 --output results/pinn_rectangles_coupled
"""
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import time
import platform
from itertools import islice
import numpy as np
import torch
from .physics import Problem, grid, material, FIELDS
from .pinn import MixedPINN, make_batch, mixed_loss, physical_state


@dataclass
class TrainConfig:
    geometry:str='rectangles'
    mode:str='coupled'
    width:int=40
    layers:int=5
    hard:bool=True
    energy:str='balance'
    energy_weight:float=1.
    adam_steps:int=2000
    lbfgs_steps:int=3000
    switch:int=2000
    lr:float=.001
    n:int=51
    seed:int=42
    dtype:str='float64'
    device:str='auto'
    log_every:int=100
    ratio:float|None=None
    checkpoint_every:int=1000
    left_refine:int=0
    fast_adam:bool=False
    reset_mechanical_adam:bool=False
    jacobian_backend:str='autograd'


def training_schedule(mode,per_field_steps,switch):
    if mode=='coupled':
        yield from ('coupled' for _ in range(per_field_steps))
    else:
        for start in range(0,per_field_steps,switch):
            count=min(switch,per_field_steps-start)
            for active in ('thermal','mechanical'):
                yield from (active for _ in range(count))


def validate_resume(c,ck):
    previous=ck['config']
    for key in ('geometry','width','layers','hard','energy','energy_weight','mode',
                'dtype','n','ratio','switch','seed'):
        if previous[key]!=getattr(c,key):raise ValueError(f'resume config differs: {key}')
    for key in ('left_refine','fast_adam','reset_mechanical_adam'):
        if previous.get(key,getattr(TrainConfig(),key))!=getattr(c,key):raise ValueError(f'resume config differs: {key}')
    stages=1 if c.mode=='coupled' else 2
    for optimizer in ('adam','lbfgs'):
        done=ck['completed'][optimizer]
        steps=getattr(c,optimizer+'_steps')
        if steps*stages<done:raise ValueError('requested budget is below completed updates')
        old_steps=previous[optimizer+'_steps']
        if c.mode=='sequential' and old_steps!=steps:
            old=islice(training_schedule(c.mode,old_steps,c.switch),done)
            new=islice(training_schedule(c.mode,steps,c.switch),done)
            if any(a!=b for a,b in zip(old,new)):
                raise ValueError('new sequential schedule reorders already completed blocks')
    if ck['completed']['lbfgs'] and c.adam_steps!=previous['adam_steps']:
        raise ValueError('cannot add Adam updates after L-BFGS has started')


def _lbfgs(params):
    return torch.optim.LBFGS(params,lr=1.,max_iter=1,max_eval=25,
             history_size=80,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14)


def relative_errors(pred,ref):
    result={}
    for j,name in enumerate(FIELDS):
        d=pred[:,j]-ref[:,j]
        scale=max(float(np.max(np.abs(ref[:,j]))),1e-15)
        result[name]={'relative_L2':float(np.linalg.norm(d)/max(np.linalg.norm(ref[:,j]),1e-15)),
                     'MAE':float(np.mean(np.abs(d))),
                     'mean_error_over_peak':float(np.mean(np.abs(d))/scale),
                     'max_error_over_peak':float(np.max(np.abs(d))/scale)}
    return result


def export(model,p,device,dtype,out,config=None):
    xy=grid(101);x=torch.as_tensor(xy,device=device,dtype=dtype)
    E,k=material(xy,p,'nodal')
    params=None
    if model.parametric:
        if p.ratio is None:raise ValueError('parametric inference requires ratio')
        params=torch.full((len(x),2),np.log(p.ratio)/np.log(10.),device=device,dtype=dtype)
    # Chunk derivative exports so the dense evaluation grid fits smaller GPUs.
    fs=[]; gs=[]; cs=[]; qs=[]
    for start in range(0,len(x),2048):
        sl=slice(start,start+2048)
        with torch.enable_grad():
            f,g=model.state(x[sl],create_graph=False,parameters=None if params is None else params[sl])
            st=physical_state(f,g,torch.as_tensor(E[sl],device=device,dtype=dtype),
                              torch.as_tensor(k[sl],device=device,dtype=dtype),p)
        fs.append(f.detach().cpu().numpy());gs.append(g.detach().cpu().numpy())
        cs.append(st['stress'].detach().cpu().numpy());qs.append(st['flux'].detach().cpu().numpy())
    fields=np.concatenate(fs); grad=np.concatenate(gs)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'prediction.npz',xy=xy,fields=fields,gradients=grad,
                        constitutive_stress=np.concatenate(cs),fourier_flux=np.concatenate(qs))
    np.savetxt(out/'prediction.csv',np.c_[xy,fields],delimiter=',',
               header=','.join(('x','y')+FIELDS),comments='')
    off=make_batch(p,n=67,device=device,dtype=dtype,seed=2026,circle_points=4489)
    # Independent 67x67 points; not the collocation grid or FEM labels.
    with torch.enable_grad():
        losses=mixed_loss(model,off,p,energy='none')
    diag={k:float(v.detach().cpu()) for k,v in losses.items()}
    diag['max_normal_displacement_error']=float(max(np.max(np.abs(fields[xy[:,0]==0,0])),
             np.max(np.abs(fields[xy[:,0]==1,0])),np.max(np.abs(fields[xy[:,1]==0,1])),
             np.max(np.abs(fields[xy[:,1]==1,1]))))
    diag['max_temperature_boundary_error']=float(max(np.max(np.abs(fields[xy[:,0]==0,5]-1)),
                                                    np.max(np.abs(fields[xy[:,0]==1,5]))))
    # Original manuscript plots constitutive-derived stress and flux. Preserve
    # both and explicitly compare both against the same nodal FEM reference.
    derived=fields.copy();derived[:,2:5]=np.concatenate(cs);derived[:,6:]=np.concatenate(qs)
    metrics={'problem':p.to_dict(),'diagnostics_independent_grid':diag,'comparisons':{}}
    for suffix in ('n50','n100','n200','n100_nodal'):
        if p.ratio is not None:continue # fixed-ratio main references would be the wrong BVP
        candidates=[parent/f'fem_{p.geometry}_{suffix}'/'solution.npz' for parent in list(out.parents)[:2]]
        ref=next((path for path in candidates if path.exists()),None)
        if ref is not None:
            with np.load(ref) as data:r={key:data[key].copy() for key in ('xy','fields')}
            if r['fields'].shape==fields.shape and np.allclose(r['xy'],xy):
                metrics['comparisons'][f'FEM_{suffix}']={
                    'direct_mixed_outputs':relative_errors(fields,r['fields']),
                    'constitutive_outputs':relative_errors(derived,r['fields'])}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics


def run(c,output,resume=None,observer=None,max_updates=None):
    if c.mode not in ('coupled','sequential'):raise ValueError('unknown training mode')
    if min(c.adam_steps,c.lbfgs_steps)<0 or min(c.switch,c.log_every,c.checkpoint_every)<1:
        raise ValueError('steps must be nonnegative; switch, log and checkpoint intervals positive')
    if c.adam_steps+c.lbfgs_steps<1:raise ValueError('at least one optimizer step is required')
    torch.set_num_threads(4)
    torch.manual_seed(c.seed);np.random.seed(c.seed)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if c.device=='auto' else c.device
    dtype=getattr(torch,c.dtype)
    torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends.cudnn,'allow_tf32'):torch.backends.cudnn.allow_tf32=False
    p=Problem(c.geometry,ratio=c.ratio)
    b=make_batch(p,c.n,device,dtype,c.seed,left_refine=c.left_refine)
    if c.fast_adam:
        if not device.startswith('cuda'):raise ValueError('fast_adam requires CUDA')
        from .fast_adam import CapturedAdam,index_batch
        b=index_batch(b)
    model=MixedPINN(c.width,c.layers,c.hard,jacobian_backend=c.jacobian_backend).to(device=device,dtype=dtype)
    ck=None
    if resume:
        ck=torch.load(resume,map_location=device,weights_only=False)
        validate_resume(c,ck)
        if 'optimizers' not in ck:raise ValueError('Resume requires checkpoint.pt; model.pt is for inference.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(c),indent=2))
    env={'python':platform.python_version(),'torch':torch.__version__,
         'numpy':np.__version__,'platform':platform.platform(),'device':device,
         'GPU':torch.cuda.get_device_name(0) if device.startswith('cuda') else None,
         'training_points':len(b['xy']),'parameter_count':sum(a.numel() for a in model.parameters())}
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    stages=('coupled',) if c.mode=='coupled' else ('thermal','mechanical')
    opts={}
    for stage in stages:
        params=list(model.parameters()) if stage=='coupled' else list(getattr(model,stage).parameters())
        opts[('adam',stage)]=torch.optim.Adam(params,lr=c.lr,capturable=c.fast_adam,foreach=False if c.fast_adam else None)
        opts[('lbfgs',stage)]=_lbfgs(params)
    completed={'adam':0,'lbfgs':0};evaluation_count=0;offset_seconds=0.
    if resume:
        model.load_state_dict(ck['model']);completed=ck['completed']
        if ck['config'].get('jacobian_backend','autograd')!=c.jacobian_backend:
            (out/'derivative_backend_transition.json').write_text(json.dumps({
                'previous':ck['config'].get('jacobian_backend','autograd'),'current':c.jacobian_backend,
                'at_completed_updates':completed,'note':'Mathematically exact chain-rule backend; floating-point operation ordering differs. See derivative parity tests.'},indent=2))
        for key,value in ck['optimizers'].items():opts[key].load_state_dict(value)
        for stage in stages:
            for group in opts[('adam',stage)].param_groups:group['lr']=c.lr
        evaluation_count=ck.get('closure_evaluations',0);offset_seconds=ck.get('elapsed_seconds',0.)
    start=time.perf_counter();last_stage=None;history=[];fast_engine=None;invocation_updates=0
    status_path=out/'run_status.json'
    def status(name):
        status_path.write_text(json.dumps({'status':name,'completed':completed.copy(),
            'expected':{'adam':c.adam_steps*len(stages),'lbfgs':c.lbfgs_steps*len(stages)}},indent=2))
    status('running')
    optimizer_active={name:next(islice(training_schedule(c.mode,getattr(c,name+'_steps'),c.switch),
                                     max(completed[name]-1,0),completed[name]),None)
                      for name in ('adam','lbfgs')}

    def checkpoint(filename='checkpoint.pt',include_optimizers=True):
        path=out/filename;temp=path.with_suffix('.tmp')
        payload={'model':model.state_dict(),'config':asdict(c),'problem':p.to_dict(),
                    'completed':completed.copy(),'closure_evaluations':evaluation_count,
                    'elapsed_seconds':offset_seconds+time.perf_counter()-start}
        if include_optimizers:payload['optimizers']={key:op.state_dict() for key,op in opts.items()}
        torch.save(payload,temp)
        temp.replace(path)

    if not resume:checkpoint()
    if observer is not None:observer(model,p,completed.copy())
    logpath=out/'history.csv';append=bool(resume and logpath.exists())
    if append:
        with logpath.open(newline='') as f:
            reader=csv.DictReader(f);columns=reader.fieldnames
            retained=[r for r in reader if int(r['step'])<=completed[r['optimizer']]]
        temporary=logpath.with_suffix('.tmp')
        with temporary.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(retained)
        temporary.replace(logpath)
    keys=['optimizer','step','active','closure_evaluations','elapsed_seconds','total','mechanical','thermal',
          'M_connection','M_equilibrium','M_dirichlet','M_neumann','M_energy',
          'T_connection','T_equilibrium','T_dirichlet','T_neumann','T_energy']
    with logpath.open('a' if append else 'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        if not append:writer.writeheader()
        for optimizer_name,per_field_steps in (('adam',c.adam_steps),('lbfgs',c.lbfgs_steps)):
            total_steps=per_field_steps*len(stages)
            for step,active in enumerate(training_schedule(c.mode,per_field_steps,c.switch)):
                if step<completed[optimizer_name]:continue
                if active!=last_stage:
                    fast_engine=None
                    for name in ('mechanical','thermal'):
                        for v in getattr(model,name).parameters():
                            v.requires_grad_(active=='coupled' or active==name)
                    last_stage=active
                if optimizer_name=='lbfgs' and active!='coupled' and optimizer_active[optimizer_name]!=active:
                    # A new temperature block changes the mechanical objective.
                    # Old quasi-Newton curvature pairs must not cross that change.
                    opts[(optimizer_name,active)]=_lbfgs(list(getattr(model,active).parameters()))
                if (optimizer_name=='adam' and active=='mechanical' and c.reset_mechanical_adam
                        and optimizer_active[optimizer_name]!=active):
                    opts[(optimizer_name,active)]=torch.optim.Adam(model.mechanical.parameters(),lr=c.lr,
                        capturable=c.fast_adam,foreach=False if c.fast_adam else None)
                optimizer_active[optimizer_name]=active
                optimizer=opts[(optimizer_name,active)]
                latest={}
                def closure():
                    nonlocal evaluation_count,latest
                    optimizer.zero_grad(set_to_none=True)
                    losses=mixed_loss(model,b,p,c.energy,c.energy_weight,active=active)
                    loss=losses['total']
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective; checkpoint preserved.')
                    loss.backward();evaluation_count+=1
                    latest=losses
                    return loss
                if optimizer_name=='adam' and c.fast_adam:
                    if fast_engine is None:
                        fast_engine=CapturedAdam(optimizer,lambda active=active:mixed_loss(model,b,p,c.energy,c.energy_weight,active=active))
                    latest=fast_engine.step();evaluation_count+=1
                elif optimizer_name=='adam':closure();optimizer.step()
                else:optimizer.step(closure)
                completed[optimizer_name]=step+1
                invocation_updates+=1
                if observer is not None:observer(model,p,completed.copy())
                if step==0 or (step+1)%c.log_every==0 or step+1==total_steps:
                    row={k:float(v.detach().cpu()) for k,v in latest.items()}
                    if not np.isfinite(row['total']):
                        status('failed');raise FloatingPointError('Nonfinite captured objective; last valid checkpoint preserved')
                    row.update(optimizer=optimizer_name,step=step+1,active=active,
                               closure_evaluations=evaluation_count,
                               elapsed_seconds=offset_seconds+time.perf_counter()-start)
                    writer.writerow(row);stream.flush();history.append(row)
                    print(f'{p.geometry} {c.mode} {optimizer_name} {step+1}/{total_steps} '
                          f'{active}: loss={row["total"]:.6g}, seconds={row["elapsed_seconds"]:.1f}',flush=True)
                if (step+1)%c.checkpoint_every==0:checkpoint();status('running')
                if max_updates is not None and invocation_updates>=max_updates:
                    checkpoint();status('partial');return model,{'status':'partial','completed':completed.copy()}
            checkpoint()
    for v in model.parameters():v.requires_grad_(True)
    checkpoint('model.pt',include_optimizers=False)
    metrics=export(model,p,device,dtype,out,c)
    status('complete')
    return model,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','mode','energy','dtype','device','jacobian_backend'):
        parser.add_argument('--'+name,default=getattr(TrainConfig(),name))
    for name in ('width','layers','adam_steps','lbfgs_steps','switch','n','seed','log_every','checkpoint_every','left_refine'):
        parser.add_argument('--'+name.replace('_','-'),type=int,default=getattr(TrainConfig(),name))
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--energy-weight',type=float,default=1.)
    parser.add_argument('--ratio',type=float)
    parser.add_argument('--soft',action='store_true',help='soft Dirichlet BCs as in Sections 4.1-4.2')
    parser.add_argument('--fast-adam',action='store_true',help='CUDA graph replay, with unchanged float precision')
    parser.add_argument('--reset-mechanical-adam',action='store_true')
    parser.add_argument('--max-updates',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    args=vars(parser.parse_args());out=args.pop('output');resume=args.pop('resume');maximum=args.pop('max_updates')
    args['hard']=not args.pop('soft')
    run(TrainConfig(**args),out,resume,max_updates=maximum)


if __name__=='__main__':main()
