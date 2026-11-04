"""Port and evaluate the authors' released SciANN checkpoint, without retraining.

Extract under WSL (h5py); evaluate under Windows (PyTorch). The extracted NPZ
is supplied so evaluation does not need TensorFlow, SciANN, or h5py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

NAMES=('Tn','gT_x','gT_y','u_xn','u_yn','sigma_x','sigma_y','sigma_xy')


def extract(source,output):
    import h5py
    arrays={}
    with h5py.File(source,'r') as f:
        for head,name in enumerate(NAMES):
            for layer in range(6):
                index=head*5+layer+1
                group=name if layer==5 else ('D40b' if index==1 else f'D40b_{index}')
                found={}
                def collect(path,obj):
                    if isinstance(obj,h5py.Dataset):found[path.split('/')[-1]]=obj[:]
                f[group].visititems(collect)
                if set(found)!={'kernel:0','bias:0'}:raise ValueError(f'Unexpected weights in {group}: {found.keys()}')
                expected=(2 if layer==0 else 40,1 if layer==5 else 40)
                if found['kernel:0'].shape!=expected:raise ValueError(f'Unexpected shape in {group}')
                arrays[f'{name}.W{layer}']=found['kernel:0']
                arrays[f'{name}.b{layer}']=found['bias:0']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    provenance={'source_filename':Path(source).name,
        'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'extracted_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'authors_commit':'5b426dc8eb2255b1bb3b8630619228b36cdb464e',
        'source_url':'https://github.com/phyml4e/PINNs/blob/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic/model_v_new.hdf5',
        'license':'MIT; see data/AUTHORS_LICENSE',
        'architecture':'eight independent raw-(x,y) 5x40 tanh networks, linear outputs',
        'status':'released pretrained model; original training history is not supplied'}
    output.with_suffix('.json').write_text(json.dumps(provenance,indent=2))
    print('Extracted',len(arrays),'weight arrays to',output)


def from_arrays(arrays):
    import torch
    from .pinn import MixedPINN
    model=MixedPINN(40,5,hard=True).double()
    groups={'mechanical':('u_xn','u_yn','sigma_x','sigma_y','sigma_xy'),
            'thermal':('Tn','gT_x','gT_y')}
    with torch.no_grad():
        for group,names in groups.items():
            net=getattr(model,group)
            for head,name in enumerate(names):
                for i in range(6):
                    w=np.asarray(arrays[f'{name}.W{i}'],dtype=float).copy()
                    b=np.asarray(arrays[f'{name}.b{i}'],dtype=float).copy()
                    # Original x=(z+1)/2, with our normalized input z.
                    if i==0:b+=.5*w.sum(axis=0);w*=.5
                    # Authors use x*(x-1); our positive hard factor is x*(1-x).
                    if i==5 and name in ('u_xn','u_yn','Tn'):w=-w;b=-b
                    net.weights[i][head].copy_(torch.as_tensor(w))
                    net.biases[i][head,0].copy_(torch.as_tensor(b))
    return model


def original_numpy(arrays,xy):
    """Independent evaluation with the authors' raw coordinates and signs."""
    outputs=[]
    for name in NAMES:
        v=xy.copy()
        for i in range(6):
            v=v@arrays[f'{name}.W{i}']+arrays[f'{name}.b{i}']
            if i<5:v=np.tanh(v)
        outputs.append(v[:,0])
    T,qx,qy,ux,uy,sxx,syy,sxy=outputs;x,y=xy.T
    return np.stack((x*(x-1)*ux,y*(y-1)*uy,sxx,syy,sxy,1-x+x*(x-1)*T,qx,qy),1)


def evaluate(weights,output):
    import torch
    from .physics import Problem
    from .train import export
    torch.set_num_threads(4)
    arrays=np.load(weights);model=from_arrays(arrays).eval()
    xy=np.random.default_rng(2026).random((101,2))
    predicted=model.predict(torch.as_tensor(xy)).detach().numpy()
    original=original_numpy(arrays,xy)
    np.testing.assert_allclose(predicted,original,atol=1e-11,rtol=1e-10)
    p=Problem('rectangles');out=Path(output);out.mkdir(parents=True,exist_ok=True)
    config={'width':40,'layers':5,'hard':True,'dtype':'float64','geometry':'rectangles',
            'mode':'authors_released','energy':'authors'}
    torch.save({'model':model.state_dict(),'config':config,'problem':p.to_dict(),
                'origin':'authors released checkpoint; no local training'},out/'authors_model.pt')
    metrics=export(model,p,'cpu',torch.float64,out,config)
    report={'max_conversion_absolute_difference':float(abs(predicted-original).max()),
            'points_checked':len(xy),'source_weights_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
            'training_status':'released pretrained model; no local training',
            'interpretation':'same corrected plane-strain law and nodal material convention; no amplitude fitting'}
    (out/'conversion_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('Derived L2 errors vs FEM200:',{k:round(100*v['relative_L2'],3)
          for k,v in metrics['comparisons'].get('FEM_n200',{}).get('constitutive_outputs',{}).items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('extract','evaluate'))
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (extract if args.action=='extract' else evaluate)(args.input,args.output)
