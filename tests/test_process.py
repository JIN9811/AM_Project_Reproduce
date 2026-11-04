import numpy as np
import pytest
torch=pytest.importorskip('torch')
from harandi.train import TrainConfig,run


def test_observer_does_not_change_optimization_and_records_actual_steps(tmp_path):
    c=TrainConfig(geometry='homogeneous',width=4,layers=1,n=5,adam_steps=2,lbfgs_steps=1,
                  log_every=10,device='cpu')
    plain,_=run(c,tmp_path/'plain')
    observations=[]
    def observer(model,problem,completed):
        observations.append((sum(completed.values()),model.predict(torch.tensor([[.2,.3]],dtype=torch.float64)).detach().clone()))
    observed,_=run(c,tmp_path/'observed',observer=observer)
    assert [x[0] for x in observations]==[0,1,2,3]
    for key,value in plain.state_dict().items():torch.testing.assert_close(value,observed.state_dict()[key],rtol=0,atol=0)
    assert not torch.equal(observations[0][1],observations[-1][1])
