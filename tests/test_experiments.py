import numpy as np
import pytest
torch=pytest.importorskip('torch')
from harandi.physics import Problem,material


def test_differentiable_material_matches_nodal_coefficient_and_gradient():
    from harandi.baselines import differentiable_material
    p=Problem('rectangles');xy=np.array([[.095,.2],[.2,.2],[.7,.7]])
    t=torch.tensor(xy,dtype=torch.float64,requires_grad=True)
    e,k=differentiable_material(t,p)
    ref,_=material(xy,p,'nodal')
    np.testing.assert_allclose(e.detach().numpy(),ref,atol=1e-13)
    g=torch.autograd.grad(e.sum(),t)[0]
    # At x=.095, a .5 phase jump is interpolated over a .04-wide layer;
    # nodal interpolation on the left half gives dE/dx=-12.5.
    np.testing.assert_allclose(g.detach().numpy()[:,0],[-12.5,0,0],atol=1e-12)


def test_standard_body_points_avoid_material_interpolation_knots():
    from harandi.baselines import make_standard_batch
    p=Problem('rectangles')
    b=make_standard_batch(p,51,'cpu',torch.float64)
    coords=b['xy'][b['body']].numpy()*p.material_grid
    assert np.min(abs(coords-np.rint(coords)))>1e-8


def test_standard_and_dem_complete_real_training_steps(tmp_path):
    from harandi.baselines import run
    for method in ('standard','dem'):
        result=run(method,Problem('homogeneous'),steps=2,n=7,width=4,layers=1,
                   device='cpu',output=tmp_path/method)
        assert np.isfinite(result).all()


def test_parametric_transfer_saves_checkpoint_without_seen_test_ratio(tmp_path):
    from harandi.transfer import run
    from harandi.physics import grid,homogeneous_exact
    xy=grid(7);ref=tmp_path/'references';ref.mkdir()
    # A real numerical fixture independent of the network's predictions.
    folder=ref/'ratio_1';folder.mkdir()
    np.savez_compressed(folder/'solution.npz',xy=xy,fields=homogeneous_exact(xy,Problem('homogeneous')))
    run(ref,tmp_path/'transfer',ratios=[1],steps=1,weight=.1,n=7,width=4,layers=1,device='cpu')
    ck=torch.load(tmp_path/'transfer/model.pt',weights_only=False)
    assert ck['training_ratios']==[1]
    assert ck['problem']['ratio']==15


def test_coupled_dem_does_not_apply_mechanical_energy_force_to_temperature():
    from harandi.baselines import PrimaryPINN,loss
    from harandi.pinn import make_batch
    p=Problem('homogeneous');b=make_batch(p,n=7,dtype=torch.float64)
    m=PrimaryPINN(p,4,1).double()
    mechanical=loss(m,b,'dem','coupled')-loss(m,b,'dem','thermal')
    mechanical.backward()
    assert all(v.grad is None or v.grad.abs().max()<1e-12 for v in m.thermal.parameters())


def test_variational_mixed_energy_freezes_temperature_for_mechanical_variation():
    from harandi.pinn import MixedPINN,make_batch,mixed_loss
    p=Problem('homogeneous');b=make_batch(p,n=7,dtype=torch.float64)
    m=MixedPINN(4,1).double()
    mixed_loss(m,b,p,energy='variational')['M_energy'].backward()
    assert all(v.grad is None or v.grad.abs().max()<1e-12 for v in m.thermal.parameters())
