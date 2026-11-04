"""Record actual intermediate fields during a reproducible training experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .physics import grid,material
from .pinn import physical_state
from .train import TrainConfig,run


class SnapshotRecorder:
    def __init__(self,output,steps):
        self.output=Path(output)/'snapshots';self.output.mkdir(parents=True,exist_ok=True)
        self.steps=set(steps)

    def __call__(self,model,problem,completed):
        step=sum(completed.values())
        if step not in self.steps:return
        parameter=next(model.parameters());device=parameter.device;dtype=parameter.dtype
        xy=grid(51);E,k=material(xy,problem,'nodal')
        with torch.enable_grad():
            f,g=model.state(torch.as_tensor(xy,device=device,dtype=dtype),create_graph=False)
            s=physical_state(f,g,torch.as_tensor(E,device=device,dtype=dtype),
                             torch.as_tensor(k,device=device,dtype=dtype),problem)
        np.savez_compressed(self.output/f'step_{step:06d}.npz',xy=xy,
            fields=f.detach().cpu().numpy(),constitutive_stress=s['stress'].detach().cpu().numpy(),
            fourier_flux=s['flux'].detach().cpu().numpy(),step=step,
            adam_updates=completed['adam'],lbfgs_updates=completed['lbfgs'])


def experiment(output,geometry='circle',mode='coupled',adam=2000,lbfgs=3000):
    if min(adam,lbfgs)<0 or adam+lbfgs<1:raise ValueError('positive total training budget required')
    c=TrainConfig(geometry=geometry,mode=mode,n=51 if geometry=='circle' else 71,
                  adam_steps=adam,lbfgs_steps=lbfgs,log_every=10)
    factor=1 if mode=='coupled' else 2;total=factor*(adam+lbfgs)
    steps=sorted(set([0,min(100,total),min(500,total),factor*adam,
                      min(factor*adam+1000,total),total]))
    out=Path(output);observer=SnapshotRecorder(out,steps)
    metadata={'status':'running','snapshot_updates':steps,'adam_steps':adam,'lbfgs_steps':lbfgs,
              'mode':mode,'total_optimizer_updates':total}
    meta_path=out/'process_metadata.json'
    meta_path.write_text(json.dumps(metadata,indent=2))
    model,metrics=run(c,out,observer=observer)
    metadata.update({
        'status':'complete',
        'source':'new same-seed experiment with real intermediate field snapshots',
        'snapshot_updates':steps,'log_interval':10,'total_optimizer_updates':total,
        'history_semantics':'reported objective is the most recent closure evaluation; snapshots are post-update',
        'comparison_to_paper':'accelerated Adam plus L-BFGS, not the paper 100k Adam schedule'})
    meta_path.write_text(json.dumps(metadata,indent=2))
    return model,metrics


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('results/process_circle_coupled'))
    ap.add_argument('--geometry',default='circle',choices=('circle','rectangles'))
    ap.add_argument('--mode',default='coupled',choices=('coupled','sequential'))
    ap.add_argument('--adam-steps',type=int,default=2000);ap.add_argument('--lbfgs-steps',type=int,default=3000)
    a=ap.parse_args();experiment(a.output,a.geometry,a.mode,a.adam_steps,a.lbfgs_steps)
