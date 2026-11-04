"""Exact manual tanh jets must preserve spatial and parameter derivatives."""
import pytest

torch = pytest.importorskip('torch')
from harandi.pinn import ParallelMLP


def autograd_jet(network, inputs, spatial_dim=2, second_order=False):
    raw = network(inputs)
    jac = torch.autograd.grad(raw.sum(), inputs, create_graph=True)[0][..., :spatial_dim]
    if not second_order:
        return raw, jac
    hess = torch.stack([torch.autograd.grad(jac[..., j].sum(), inputs,
        create_graph=True, retain_graph=True)[0][..., :spatial_dim]
        for j in range(spatial_dim)], dim=-2)
    return raw, jac, hess


@pytest.mark.parametrize('spatial_dim,input_dim', [(1, 2), (2, 2), (2, 4)])
@pytest.mark.parametrize('second_order', [False, True])
def test_forward_jet_matches_autograd_and_weight_gradients(spatial_dim, input_dim, second_order):
    from harandi.jets import forward_jet
    torch.manual_seed(123)
    network = ParallelMLP(3, 5, 2, input_dim).double()
    inputs = torch.randn(3, 7, input_dim, dtype=torch.float64, requires_grad=True)
    expected = autograd_jet(network, inputs, spatial_dim, second_order)
    actual = forward_jet(network, inputs, spatial_dim, second_order=second_order)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-13, atol=2e-14)
    # Gradient through a Hessian-dependent loss is the expensive PINN path;
    # detaching jets or differentiating only activations would fail here.
    eager_loss = sum((i+1)*value.square().mean() for i, value in enumerate(expected))
    manual_loss = sum((i+1)*value.square().mean() for i, value in enumerate(actual))
    eager_grad = torch.autograd.grad(eager_loss, tuple(network.parameters()), retain_graph=True)
    manual_grad = torch.autograd.grad(manual_loss, tuple(network.parameters()), retain_graph=True)
    for left, right in zip(manual_grad, eager_grad):
        torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-14)


def test_forward_jet_derivatives_are_wrt_supplied_input_coordinates():
    from harandi.jets import forward_jet
    network = ParallelMLP(1, 3, 0).double()
    with torch.no_grad():
        network.weights[0][0, :, 0] = torch.tensor([3., -2.])
        network.biases[0].fill_(.7)
    physical = torch.tensor([[[.1, .2], [.4, .8]]], dtype=torch.float64)
    value, jac, hess = forward_jet(network, 2*physical-1, second_order=True)
    torch.testing.assert_close(value, torch.tensor([[-.5, -1.1]], dtype=torch.float64))
    torch.testing.assert_close(jac, torch.tensor([[[3., -2.], [3., -2.]]], dtype=torch.float64))
    assert torch.count_nonzero(hess) == 0
    # For x -> z=2x-1, the physical Jacobian is twice this returned Jacobian.
    torch.testing.assert_close(2*jac, torch.tensor([[[6., -4.], [6., -4.]]], dtype=torch.float64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='tiny CUDA derivative check')
def test_forward_jet_matches_cuda_autograd():
    from harandi.jets import forward_jet
    torch.manual_seed(41)
    network = ParallelMLP(2, 4, 2).cuda().double()
    inputs = torch.randn(2, 5, 2, device='cuda', dtype=torch.float64, requires_grad=True)
    expected = autograd_jet(network, inputs, second_order=True)
    actual = forward_jet(network, inputs, second_order=True)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-13)
    eager_grad = torch.autograd.grad(sum(v.square().mean() for v in expected), tuple(network.parameters()))
    manual_grad = torch.autograd.grad(sum(v.square().mean() for v in actual), tuple(network.parameters()))
    for left, right in zip(manual_grad, eager_grad):
        torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-13)
