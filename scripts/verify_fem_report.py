"""Independent FEM report evidence. Run with the Ubuntu FEniCS Python.

Uses saved reference arrays and small fresh uniform/author-data integrations;
never trains a neural network or modifies existing reference solutions.
"""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harandi.fem import solve
from harandi.physics import Problem, grid, homogeneous_exact, FIELDS


def main():
    folders=[ROOT/'results'/f'fem_{g}_n{n}' for g in ('circle','rectangles') for n in (50,100,200)]
    folders += [ROOT/'results'/f'fem_{g}_n100_nodal' for g in ('circle','rectangles')]
    folders += [ROOT/'results/full_study/parametric_fem'/f'ratio_{i}' for i in [*range(1,11),15]]
    refs=[]
    for folder in folders:
        meta=json.loads((folder/'metadata.json').read_text())
        with np.load(folder/'solution.npz') as d:
            assert d['fields'].shape==(10201,8) and np.isfinite(d['fields']).all()
            xy,fields=d['xy'],d['fields']; x,y=xy.T
            boundary=max(np.abs(fields[(x==0)|(x==1),0]).max(),np.abs(fields[(y==0)|(y==1),1]).max(),
                         np.abs(fields[x==0,5]-1).max(),np.abs(fields[x==1,5]).max())
        refs.append({'folder':folder.relative_to(ROOT).as_posix(),'metadata':meta,
                     'dirichlet_max_error':float(boundary),
                     'solution_sha256':hashlib.sha256((folder/'solution.npz').read_bytes()).hexdigest()})
    uniform=[]
    # Compare Q1 FE functions at nodes AND off nodes; nodal exactness is not field exactness.
    for mode in ('sharp','nodal'):
        for n in (8,16,32):
            p=Problem('homogeneous',material_grid=4); xy=grid(2*n+1)
            r=solve(p,n,xy,coefficient_mode=mode); exact=homogeneous_exact(xy,p)
            nodal=np.isclose(xy[:,0]*n,np.round(xy[:,0]*n)) & np.isclose(xy[:,1]*n,np.round(xy[:,1]*n))
            err=np.abs(r['fields']-exact)
            uniform.append({'mode':mode,'n':n,'nodal_uT_max_abs':float(err[nodal][:,[0,1,5]].max()),
                'full_grid_ux_max_abs':float(err[:,0].max()),'T_max_abs':float(err[:,5].max()),
                'sxx_max_abs':float(err[:,2].max()),'qx_max_abs':float(err[:,6].max()),'diagnostics':r['diagnostics']})
            assert uniform[-1]['nodal_uT_max_abs']<2e-10
            assert uniform[-1]['T_max_abs']<2e-10
            assert uniform[-1]['qx_max_abs']<2e-10
    a=np.genfromtxt(ROOT/'data/authors_geometry2_fe.csv',delimiter=',',names=True)
    names=a.dtype.names; calculated=solve(Problem('rectangles'),50,grid(51))['fields']
    reference=np.column_stack((a[names[0]],a[names[1]],a[names[2]]))
    uv=calculated[:,:2]; refuv=reference[:,:2]
    authors={'temperature_relative_l2':float(np.linalg.norm(calculated[:,5]-reference[:,2])/np.linalg.norm(reference[:,2])),
        'displacement_relative_l2_raw':float(np.linalg.norm(uv-refuv)/np.linalg.norm(refuv)),
        'least_squares_author_over_fem_amplitude':float(np.sum(uv*refuv)/np.sum(uv**2)),
        'amplitude_applied_to_results':False,'evaluation_points':2601}
    assert authors['temperature_relative_l2']<1e-5
    assert authors['displacement_relative_l2_raw']<.025
    result={'status':'pass','reference_count':len(refs),'reference_records':refs,
            'uniform_analytic_checks':uniform,'author_feap_check':authors,
            'notes':['Uniform Q1 nodal exactness does not imply exact interpolation between nodes.',
                     'All stored numerical fields and author FEAP data remain unscaled.']}
    target=ROOT/'results/fem_report_verification.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'pass','references':len(refs),'uniform_runs':len(uniform),'authors':authors},indent=2))


if __name__=='__main__': main()
