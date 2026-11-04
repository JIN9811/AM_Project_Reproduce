import copy
import pytest
torch=pytest.importorskip('torch')
from harandi.fast_adam import CapturedAdam,index_batch
from harandi.physics import Problem
from harandi.pinn import MixedPINN,make_batch,mixed_loss


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph needs CUDA')
def test_captured_adam_preserves_updates_and_loss():
    torch.manual_seed(121)
    p=Problem('homogeneous')
    batch=index_batch(make_batch(p,9,'cuda',torch.float64))
    eager=MixedPINN(8,2,False).cuda().double();graph=copy.deepcopy(eager)
    oe=torch.optim.Adam(eager.parameters(),lr=.001,capturable=True,foreach=False)
    og=torch.optim.Adam(graph.parameters(),lr=.001,capturable=True,foreach=False)
    def closure():return mixed_loss(graph,batch,p,'authors')
    fast=CapturedAdam(og,closure,warmup=3)
    for step in range(12):
        oe.zero_grad(set_to_none=True)
        expected=mixed_loss(eager,batch,p,'authors');expected['total'].backward();oe.step()
        actual=fast.step()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual['total'],expected['total'],atol=1e-12,rtol=1e-11)
        for a,b in zip(eager.parameters(),graph.parameters()):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-11)
    assert fast.updates==12 and fast.captured
    assert all(int(s['step'])==12 for s in og.state.values())


def test_index_batch_preserves_loss_values():
    p=Problem('homogeneous');m=MixedPINN(5,1).double()
    original=make_batch(p,7);indexed=index_batch(original)
    a=mixed_loss(m,original,p);b=mixed_loss(m,indexed,p)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    assert original['body'].dtype==torch.bool
