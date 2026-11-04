"""Check the released-weight port against independent raw-coordinate networks."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from harandi.authors import from_arrays


def test_released_weight_conversion_preserves_original_fields_and_derivatives():
    rng=np.random.default_rng(2026)
    arrays={}
    names=('Tn','gT_x','gT_y','u_xn','u_yn','sigma_x','sigma_y','sigma_xy')
    for name in names:
        dims=[2]+[40]*5+[1]
        for layer,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            arrays[f'{name}.W{layer}']=rng.normal(0,.2,(a,b))
            arrays[f'{name}.b{layer}']=rng.normal(0,.1,b)
    def original(xy):
        values=[]
        for name in names:
            y=xy.copy()
            for i in range(6):
                y=y@arrays[f'{name}.W{i}']+arrays[f'{name}.b{i}']
                if i<5:y=np.tanh(y)
            values.append(y[:,0])
        T,qx,qy,ux,uy,sxx,syy,sxy=values
        x,y=xy.T
        return np.stack((x*(x-1)*ux,y*(y-1)*uy,sxx,syy,sxy,1-x+x*(x-1)*T,qx,qy),1)
    xy=rng.random((17,2))
    m=from_arrays(arrays)
    f,g=m.state(torch.as_tensor(xy),create_graph=False)
    np.testing.assert_allclose(f.detach().numpy(),original(xy),atol=2e-14,rtol=2e-13)
    for j in (0,1):
        delta=np.zeros_like(xy);delta[:,j]=1e-6
        fd=(original(xy+delta)-original(xy-delta))/(2e-6)
        np.testing.assert_allclose(g[:,:,j].detach().numpy(),fd,atol=2e-9,rtol=2e-7)
