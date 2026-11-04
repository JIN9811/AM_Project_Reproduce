"""CPU-only illustration of an actual intermediate mixed checkpoint, never a final result."""
import argparse, csv, hashlib, io, json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from harandi.pinn import MixedPINN
from harandi.physics import grid
from harandi.paper_figures import paper_cmap


def preview(job='mixed_rectangles_coupled'):
    torch.set_num_threads(2)
    folder=ROOT/'results/full_study'/job
    raw=(folder/'checkpoint.pt').read_bytes()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    c=ck['config'];done=ck['completed']
    if c.get('ratio') is not None:raise ValueError('This preview expects a non-parametric mixed job')
    model=MixedPINN(c['width'],c['layers'],c['hard']).double()
    model.load_state_dict(ck['model'],strict=True)
    xy=grid(101)
    with torch.no_grad():fields=model.predict(torch.as_tensor(xy,dtype=torch.float64)).numpy()
    reference=ROOT/f"results/fem_{c['geometry']}_n50/solution.npz"
    with np.load(reference) as data:
        if not np.array_equal(data['xy'],xy):raise ValueError('Reference grid mismatch')
        fem=data['fields'].copy()
    limits=json.loads((ROOT/'data/paper_color_limits.json').read_text())[c['geometry']]
    with (folder/'history.csv').open(newline='') as stream:history=list(csv.DictReader(stream))
    history=[r for r in history if r['optimizer']=='adam' and int(r['step'])<=done['adam']]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig=plt.figure(figsize=(11,9.6),layout='constrained')
    gs=fig.add_gridspec(3,3,height_ratios=[1,1,.65])
    audit=[]
    for row,(name,j,symbol) in enumerate([('ux',0,r'$u_x$'),('T',5,r'$T$')]):
        for col,(kind,values,title) in enumerate([('pinn',fields[:,j],f'PINN {done["adam"]:,} Adam updates'),
                ('fem',fem[:,j],'FEniCS 50 x 50'),('error',abs(fields[:,j]-fem[:,j]),'Absolute difference')]):
            lo,hi=limits[name][kind];ax=fig.add_subplot(gs[row,col])
            im=ax.imshow(values.reshape(101,101),origin='lower',extent=[0,1,0,1],
                         vmin=lo,vmax=hi,cmap=paper_cmap(ROOT),interpolation='none')
            ax.set(xlabel='x',ylabel='y',title=symbol+' | '+title)
            fig.colorbar(im,ax=ax,fraction=.047,pad=.035,extend='both')
            audit.append(dict(field=name,panel=kind,range=[lo,hi],minimum=float(values.min()),
                         maximum=float(values.max()),outside_fraction=float(np.mean((values<lo)|(values>hi)))))
    ax=fig.add_subplot(gs[2,:])
    for term in ('total','mechanical','thermal'):
        observed=[r for r in history if r.get(term) not in ('',None)]
        if observed:ax.semilogy([int(r['step']) for r in observed],[float(r[term]) for r in observed],label=term)
    ax.set(xlabel='Actual optimizer update',ylabel='Loss',title='Recorded training history (no smoothing)')
    ax.legend(ncol=3);ax.grid(alpha=.2)
    fig.suptitle('INTERMEDIATE CHECKPOINT - NOT A COMPLETED STUDY\n'+job,fontsize=15)
    fig.supxlabel('Paper color ranges; triangle ends indicate values outside the displayed range. Benchmark coordinates.',fontsize=9)
    output=ROOT/'reports/FULL_STUDY_TRAINING_PREVIEW.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150);plt.close(fig)
    metadata=dict(status='intermediate_only',job=job,completed=done,
        generated_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        reference=str(reference.relative_to(ROOT)),color_scale_audit=audit)
    metadata_path=ROOT/'validation/FULL_STUDY_TRAINING_PREVIEW.json'
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'figure':str(output),'completed':done}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',default='mixed_rectangles_coupled')
    preview(parser.parse_args().job)
