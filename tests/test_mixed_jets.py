import copy
import pytest
torch=pytest.importorskip('torch')
from harandi.pinn import MixedPINN,make_batch,mixed_loss
from harandi.physics import Problem


@pytest.mark.parametrize('hard',[False,True])
@pytest.mark.parametrize('active',['coupled','thermal','mechanical'])
def test_mixed_chain_rule_matches_autograd_in_values_and_weight_gradients(hard,active):
    torch.manual_seed(18)
    a=MixedPINN(6,3,hard).double();b=MixedPINN(6,3,hard,jacobian_backend='jets').double()
    b.load_state_dict(a.state_dict())
    p=Problem('rectangles');batch=make_batch(p,9)
    av,ag=a.state(batch['xy']);bv,bg=b.state(batch['xy'])
    torch.testing.assert_close(av,bv,atol=1e-13,rtol=1e-12)
    torch.testing.assert_close(ag,bg,atol=1e-13,rtol=1e-12)
    aa=mixed_loss(a,batch,p,'authors',active=active);bb=mixed_loss(b,batch,p,'authors',active=active)
    for key in aa:torch.testing.assert_close(aa[key],bb[key],atol=1e-13,rtol=1e-12)
    aa['total'].backward();bb['total'].backward()
    for x,y in zip(a.parameters(),b.parameters()):
        if x.grad is None:assert y.grad is None
        else:torch.testing.assert_close(x.grad,y.grad,atol=1e-12,rtol=1e-11)
