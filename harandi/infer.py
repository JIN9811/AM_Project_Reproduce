"""Evaluate a trained checkpoint, without retraining."""
import argparse
from pathlib import Path
import torch
from .physics import Problem
from .pinn import MixedPINN
from .train import export


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--ratio',type=float)
    args=parser.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    c=ck['config'];dtype=getattr(torch,c['dtype'])
    if ck.get('method'):
        from .baselines import PrimaryPINN
        m=PrimaryPINN(Problem(**ck['problem']),c['width'],c['layers'],c['hard']).to(device,dtype)
    else:m=MixedPINN(c['width'],c['layers'],c['hard'],parametric=ck.get('parametric',False)).to(device,dtype)
    m.load_state_dict(ck['model']);m.eval()
    p=Problem(**ck['problem'])
    if args.ratio is not None:
        if not m.parametric:raise ValueError('a fixed-material checkpoint cannot predict a new material ratio')
        pd=p.to_dict();pd['ratio']=args.ratio;p=Problem(**pd)
    metrics=export(m,p,device,dtype,args.output,c)
    print('Saved prediction and independent-grid residuals to',args.output)


if __name__=='__main__':main()
