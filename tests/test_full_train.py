import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from harandi.physics import Problem
from harandi.pinn import make_batch
from harandi.train import TrainConfig,run


def test_boundary_refinement_adds_exact_requested_points_and_preserves_base():
    p=Problem('rectangles');base=make_batch(p,11)
    refined=make_batch(p,11,left_refine=17)
    assert len(refined['xy'])==len(base['xy'])+17
    torch.testing.assert_close(refined['xy'][:-17],base['xy'],atol=0,rtol=0)
    assert refined['left'][-17:].all() and not refined['body'][-17:].any()
    assert torch.unique(refined['xy'][-17:,1]).numel()==17
    with pytest.raises(ValueError):make_batch(p,11,left_refine=-1)


def test_partial_training_resumes_without_stale_history(tmp_path):
    c=TrainConfig(geometry='homogeneous',n=5,width=4,layers=1,adam_steps=4,lbfgs_steps=0,
                  device='cpu',log_every=1,checkpoint_every=2,left_refine=3)
    run(c,tmp_path/'full')
    run(c,tmp_path/'split',max_updates=2)
    assert json.loads((tmp_path/'split/run_status.json').read_text())['status']=='partial'
    hist=tmp_path/'split/history.csv'
    with hist.open('a') as f:f.write(hist.read_text().splitlines()[-1].replace('adam,2,','adam,3,')+'\n')
    run(c,tmp_path/'split',resume=tmp_path/'split/checkpoint.pt')
    a=torch.load(tmp_path/'full/model.pt',weights_only=False)['model']
    b=torch.load(tmp_path/'split/model.pt',weights_only=False)['model']
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    import csv
    with hist.open() as f:rows=list(csv.DictReader(f))
    assert [int(r['step']) for r in rows]==[1,2,3,4]
    assert json.loads((tmp_path/'split/run_status.json').read_text())['status']=='complete'


def test_nested_study_export_finds_main_fem_references(tmp_path):
    from harandi.physics import grid,homogeneous_exact
    p=Problem('homogeneous');xy=grid(101)
    reference=tmp_path/'results/fem_homogeneous_n50';reference.mkdir(parents=True)
    np.savez(reference/'solution.npz',xy=xy,fields=homogeneous_exact(xy,p))
    c=TrainConfig(geometry='homogeneous',n=5,width=4,layers=1,adam_steps=1,
                  lbfgs_steps=0,device='cpu')
    run(c,tmp_path/'results/full_study/test')
    metrics=json.loads((tmp_path/'results/full_study/test/metrics.json').read_text())
    assert set(metrics['comparisons']['FEM_n50']['direct_mixed_outputs'])=={'ux','uy','sxx','syy','sxy','T','qx','qy'}
