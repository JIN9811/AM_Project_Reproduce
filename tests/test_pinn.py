"""Independent residual/derivative checks; no expensive training in unit tests."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from harandi.physics import Problem
from harandi.pinn import MixedPINN, make_batch, mixed_loss, physical_state


def test_hard_dirichlet_conditions_are_exact_before_training():
    m=MixedPINN(width=8,layers=2,hard=True).double()
    t=torch.linspace(0,1,21,dtype=torch.float64)
    xy=torch.cat((torch.stack((0*t,t),1),torch.stack((0*t+1,t),1),
                  torch.stack((t,0*t),1),torch.stack((t,0*t+1),1)))
    f=m.predict(xy)
    torch.testing.assert_close(f[:21,0],0*t,atol=0,rtol=0)
    torch.testing.assert_close(f[21:42,0],0*t,atol=0,rtol=0)
    torch.testing.assert_close(f[42:,1],0*xy[42:,0],atol=0,rtol=0)
    torch.testing.assert_close(f[:21,5],1+0*t,atol=0,rtol=0)
    torch.testing.assert_close(f[21:42,5],0*t,atol=0,rtol=0)


class Exact(torch.nn.Module):
    """Hand-derived analytic fields, with head-independent coordinates."""
    def state(self,xy,create_graph=True,parameters=None):
        z=xy[None].expand(8,-1,-1).clone().requires_grad_(True)
        x=z[:,:,0]; y=z[:,:,1]
        f=torch.stack((13/14*x[0]*(1-x[0]),0*y[1],-1.25+0*x[2],
             (15/26)*(13/14)*(1-2*x[3])-2.5*(1-x[3]),0*x[4],
             1-x[5],1+0*x[6],0*y[7]))
        g=torch.autograd.grad(f.sum(),z,create_graph=create_graph)[0]
        return f.T,g.permute(1,0,2)


def test_constitutive_equations_are_enforced_at_boundaries_too():
    class WrongBoundary(Exact):
        def state(self,xy,create_graph=True,parameters=None):
            f,g=super().state(xy,create_graph,parameters)
            mask=(xy[:,0]==0).to(xy.dtype)
            f=f.clone();f[:,2]=f[:,2]+mask;f[:,6]=f[:,6]+mask
            return f,g
    p=Problem('homogeneous');b=make_batch(p,n=7,dtype=torch.float64)
    losses=mixed_loss(WrongBoundary(),b,p,energy='none')
    assert float(losses['M_connection'].detach())>.1
    assert float(losses['T_connection'].detach())>.1


def test_exact_fields_make_mixed_pde_and_connection_losses_zero():
    b=make_batch(Problem('homogeneous'),n=11,device='cpu',dtype=torch.float64)
    f,g=Exact().state(b['xy'])
    st=physical_state(f,g,b['E'],b['k'],Problem('homogeneous'))
    torch.testing.assert_close(st['stress'],f[:,2:5],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(st['flux'],f[:,6:8],rtol=1e-12,atol=1e-12)
    loss=mixed_loss(Exact(),b,Problem('homogeneous'),energy='none')
    assert float(loss['total'].detach()) < 1e-25


@pytest.mark.parametrize('energy,mechanical,thermal',[
    ('none',0.,0.),('balance',0.,0.),('authors',0.,.5),
    ('paper',335/504,.5),('variational',1025/1008,.5),
])
def test_energy_conventions_against_hand_integrated_exact_grid(energy,mechanical,thermal):
    # For the five interior x coordinates, mean((x-.5)^2)=1/18. Substituting
    # the independent exact solution into Eq52 gives 5/8 + (5/7)/18.
    # Including zz in the elastic energy gives 15/16 + (10/7)/18.
    p=Problem('homogeneous');b=make_batch(p,n=7,dtype=torch.float64)
    terms=mixed_loss(Exact(),b,p,energy=energy)
    assert float(terms['M_energy'].detach())==pytest.approx(mechanical,abs=1e-13)
    assert float(terms['T_energy'].detach())==pytest.approx(thermal,abs=1e-13)


def test_sequential_schedule_trains_both_fields_equally_in_short_blocks():
    from harandi.train import training_schedule
    s=list(training_schedule('sequential',3,20))
    assert s==['thermal']*3+['mechanical']*3
    s=list(training_schedule('sequential',5,2))
    assert s.count('thermal')==s.count('mechanical')==5


def test_checkpoint_contains_both_completed_sequential_updates(tmp_path):
    from harandi.train import TrainConfig,run
    config=TrainConfig(geometry='homogeneous',width=4,layers=1,n=5,device='cpu',
                       adam_steps=1,lbfgs_steps=0,mode='sequential',log_every=1)
    run(config,tmp_path)
    ck=torch.load(tmp_path/'model.pt',weights_only=False)
    assert ck['completed']['adam']==2
    assert (tmp_path/'prediction.csv').exists()
    before=np.load(tmp_path/'prediction.npz')['fields']
    assert np.isfinite(before).all()
    run(config,tmp_path,resume=tmp_path/'checkpoint.pt')
    np.testing.assert_array_equal(before,np.load(tmp_path/'prediction.npz')['fields'])


def test_resume_rejects_reordered_sequential_blocks():
    from dataclasses import asdict
    from harandi.train import TrainConfig,validate_resume
    old=TrainConfig(mode='sequential',adam_steps=0,lbfgs_steps=1,switch=20)
    new=TrainConfig(mode='sequential',adam_steps=0,lbfgs_steps=2,switch=20)
    ck={'config':asdict(old),'completed':{'adam':0,'lbfgs':2}}
    with pytest.raises(ValueError,match='schedule'):validate_resume(new,ck)


def test_resume_allows_coupled_lbfgs_extension_and_rejects_changed_physics():
    from dataclasses import asdict
    from harandi.train import TrainConfig,validate_resume
    old=TrainConfig(adam_steps=2,lbfgs_steps=3)
    ck={'config':asdict(old),'completed':{'adam':2,'lbfgs':3}}
    validate_resume(TrainConfig(adam_steps=2,lbfgs_steps=5),ck)
    with pytest.raises(ValueError,match='energy_weight'):
        validate_resume(TrainConfig(adam_steps=2,lbfgs_steps=5,energy_weight=.1),ck)


def test_rejected_resume_preserves_original_metadata(tmp_path):
    from harandi.train import TrainConfig,run
    c=TrainConfig(geometry='homogeneous',width=4,layers=1,n=5,device='cpu',adam_steps=1,lbfgs_steps=0)
    run(c,tmp_path);before=(tmp_path/'config.json').read_bytes()
    c.energy_weight=.5
    with pytest.raises(ValueError):run(c,tmp_path,resume=tmp_path/'model.pt')
    assert (tmp_path/'config.json').read_bytes()==before


def test_zero_checkpoint_interval_is_rejected_before_training(tmp_path):
    from harandi.train import TrainConfig,run
    with pytest.raises(ValueError):run(TrainConfig(checkpoint_every=0,device='cpu'),tmp_path)


def test_independent_network_derivatives_match_finite_differences():
    torch.manual_seed(12)
    m=MixedPINN(width=6,layers=2,hard=True).double()
    xy=torch.tensor([[.25,.33],[.73,.61]],dtype=torch.float64)
    f,g=m.state(xy)
    h=1e-6
    for j in range(2):
        d=torch.zeros_like(xy);d[:,j]=h
        fd=(m.predict(xy+d)-m.predict(xy-d))/(2*h)
        torch.testing.assert_close(g[:,:,j],fd,rtol=2e-7,atol=2e-8)


def test_mechanical_loss_updates_temperature_only_in_coupled_mode():
    p=Problem('homogeneous'); b=make_batch(p,n=5,device='cpu',dtype=torch.float64)
    m=MixedPINN(width=6,layers=1).double()
    loss=mixed_loss(m,b,p,energy='none',detach_thermal=True)['mechanical']
    loss.backward()
    assert all(v.grad is None or torch.all(v.grad==0) for v in m.thermal.parameters())
    assert any(v.grad is not None and torch.any(v.grad!=0) for v in m.mechanical.parameters())
