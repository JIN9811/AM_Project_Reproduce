"""Scientific and resume contracts for the primary-variable baselines."""
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from harandi import baselines
from harandi.physics import Problem, homogeneous_exact
from harandi.pinn import make_batch


def exact_homogeneous_model():
    """u_x=13/14*x*(1-x), u_y=0, T=1-x for nu=.3, alpha=1."""
    model = baselines.PrimaryPINN(Problem('homogeneous'), 4, 1).double()
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        model.mechanical.biases[-1][0, 0, 0] = 13 / 14
    return model


def test_mechanical_standard_residual_preserves_frozen_temperature_spatial_gradient():
    # Detaching temperature from x deletes the thermoelastic body term.
    model = exact_homogeneous_model()
    batch = make_batch(model.p, 5, dtype=torch.float64)
    value = baselines.loss(model, batch, 'standard', 'mechanical')
    assert value.item() < 1e-24
    value.backward()
    assert all(p.grad is None or p.grad.abs().max() < 1e-24
               for p in model.thermal.parameters())


def test_standard_exact_solution_matches_all_eight_fields():
    model = exact_homogeneous_model()
    xy = torch.tensor([[.17, .31], [.61, .83]], dtype=torch.float64)
    fields = model.predict(xy).detach().numpy()
    np.testing.assert_allclose(fields, homogeneous_exact(xy.numpy(), model.p), atol=1e-14)


def test_dem_raw_mechanical_energy_has_no_temperature_parameter_gradient():
    model = baselines.PrimaryPINN(Problem('homogeneous'), 4, 1, hard=False).double()
    batch = make_batch(model.p, 5, dtype=torch.float64)
    terms = baselines.loss_components(model, batch, 'dem', 'coupled')
    terms['mechanical'].backward()
    assert all(p.grad is None or p.grad.abs().max() < 1e-24
               for p in model.thermal.parameters())
    assert terms['M_equilibrium'].item() == 0
    assert terms['T_equilibrium'].item() == 0


@pytest.mark.parametrize('method', ['standard', 'dem'])
@pytest.mark.parametrize('backend', ['autograd', 'jets'])
def test_checkpoint_resume_matches_uninterrupted_updates_and_export(tmp_path, method, backend):
    # Losing Adam moments or resuming the wrong active block changes parameters.
    config = baselines.BaselineConfig(method=method, geometry='homogeneous', mode='sequential',
        steps=3, switch=2, width=4, layers=1, n=5, device='cpu',
        log_every=1, checkpoint_every=1, hard=False, backend=backend)
    full, _ = baselines.run_config(config, tmp_path / 'full')
    assert full.backend == backend
    partial, status = baselines.run_config(config, tmp_path / 'resumed', max_updates=3)
    assert status['complete'] is False
    resumed, metrics = baselines.run_config(config, tmp_path / 'resumed',
                                            resume=tmp_path / 'resumed/checkpoint.pt')
    assert metrics['complete'] is True
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
    ck = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=False)
    assert ck['completed'] == 6
    assert ck['field_updates'] == {'thermal': 3, 'mechanical': 3}
    saved = torch.load(tmp_path / 'resumed/model.pt', weights_only=False)
    for name, expected in full.state_dict().items():
        torch.testing.assert_close(saved['model'][name], expected, rtol=0, atol=0)
    prediction = np.load(tmp_path / 'resumed/prediction.npz')
    points = torch.as_tensor(prediction['xy'][:9], dtype=torch.float64)
    np.testing.assert_allclose(prediction['fields'][:9], resumed.predict(points).detach().numpy(),
                               rtol=1e-13, atol=1e-13)
    with (tmp_path / 'resumed/history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['step']) for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row['active'] for row in rows] == [
        'thermal', 'thermal', 'mechanical', 'mechanical', 'thermal', 'mechanical']
    assert all(row['evaluation_point'] == 'post_update' for row in rows)
    assert all('M_energy' in row and 'T_equilibrium' in row for row in rows)
    assert json.loads((tmp_path / 'resumed/config.json').read_text())['hard'] is False
    assert json.loads((tmp_path / 'resumed/environment.json').read_text())['device'] == 'cpu'


def test_lbfgs_curvature_resets_after_thermal_field_changes(tmp_path):
    config = baselines.BaselineConfig(method='dem', geometry='homogeneous', mode='sequential',
        optimizer='lbfgs', steps=3, switch=1, width=3, layers=1, n=5, device='cpu',
        log_every=1, checkpoint_every=1)
    baselines.run_config(config, tmp_path, max_updates=6)
    ck = torch.load(tmp_path / 'checkpoint.pt', weights_only=False)
    assert ck['optimizer_resets']['mechanical'] == 3
    assert ck['field_updates'] == {'thermal': 3, 'mechanical': 3}


def test_resume_rejects_changed_problem_before_overwriting_output(tmp_path):
    config = baselines.BaselineConfig(method='dem', geometry='homogeneous', steps=2,
        width=3, layers=1, n=5, device='cpu')
    baselines.run_config(config, tmp_path, max_updates=1)
    before = (tmp_path / 'config.json').read_text()
    with pytest.raises(ValueError, match='problem'):
        baselines.run_config(config, tmp_path, resume=tmp_path / 'checkpoint.pt',
                             problem=Problem('homogeneous', alpha=.5))
    assert (tmp_path / 'config.json').read_text() == before


def test_fast_adam_rejects_cpu_before_creating_run_directory(tmp_path):
    config = baselines.BaselineConfig(method='dem', geometry='homogeneous', steps=1,
        width=3, layers=1, n=5, device='cpu', fast_adam=True)
    destination = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='CUDA'):
        baselines.run_config(config, destination)
    assert not destination.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph needs CUDA')
@pytest.mark.parametrize('method', ['standard', 'dem'])
@pytest.mark.parametrize('backend', ['autograd', 'jets'])
def test_fast_baseline_adam_matches_eager_across_field_switches(tmp_path, method, backend):
    from harandi.fast_adam import index_batch
    config = baselines.BaselineConfig(method=method, geometry='homogeneous', mode='sequential',
        steps=6, switch=6, width=3, layers=1, n=5, device='cuda',
        log_every=6, checkpoint_every=6, hard=False, fast_adam=True, backend=backend)
    torch.manual_seed(config.seed)
    problem = Problem('homogeneous')
    eager = baselines.PrimaryPINN(problem, 3, 1, hard=False).cuda().double()
    batch = (baselines.make_standard_batch(problem, 5, 'cuda', torch.float64)
             if method == 'standard' else make_batch(problem, 5, 'cuda', torch.float64))
    batch = index_batch(batch)
    # Capturable Adam has different scalar-counter arithmetic from default
    # Adam; isolate graph replay by matching the optimizer arithmetic exactly.
    for active in ('thermal', 'mechanical'):
        for name in ('thermal', 'mechanical'):
            for value in getattr(eager, name).parameters():
                value.requires_grad_(name == active)
        optimizer = torch.optim.Adam(getattr(eager, active).parameters(), lr=.001,
                                     capturable=True, foreach=False)
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            baselines.loss_components(eager, batch, method, active)['total'].backward()
            optimizer.step()
    fast, _ = baselines.run_config(config, tmp_path / 'fast')
    for name, expected in eager.state_dict().items():
        torch.testing.assert_close(fast.state_dict()[name], expected, rtol=1e-10, atol=1e-12)
    saved = torch.load(tmp_path / 'fast/checkpoint.pt', weights_only=False)
    assert saved['completed'] == 12
    assert saved['field_updates'] == {'thermal': 6, 'mechanical': 6}
    environment = json.loads((tmp_path / 'fast/environment.json').read_text())
    assert environment['execution_variant'] == 'cuda_graph_capturable_adam'


@pytest.mark.parametrize('method', ['standard', 'dem'])
@pytest.mark.parametrize('active', ['coupled', 'thermal', 'mechanical'])
@pytest.mark.parametrize('hard', [False, True])
def test_jet_backend_preserves_loss_and_parameter_gradients(method, active, hard):
    torch.manual_seed(543)
    problem = Problem('rectangles')
    eager = baselines.PrimaryPINN(problem, 5, 2, hard=hard).double()
    manual = baselines.PrimaryPINN(problem, 5, 2, hard=hard, backend='jets').double()
    manual.load_state_dict(eager.state_dict())
    batch = baselines.make_standard_batch(problem, 7, 'cpu', torch.float64)
    expected = baselines.loss_components(eager, batch, method, active)
    actual = baselines.loss_components(manual, batch, method, active)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=5e-12, atol=1e-12)
    reference_grad = torch.autograd.grad(expected['total'], tuple(eager.parameters()), allow_unused=True)
    manual_grad = torch.autograd.grad(actual['total'], tuple(manual.parameters()), allow_unused=True)
    for left, right, parameter in zip(manual_grad, reference_grad, eager.parameters()):
        left = torch.zeros_like(parameter) if left is None else left
        right = torch.zeros_like(parameter) if right is None else right
        torch.testing.assert_close(left, right, rtol=5e-11, atol=5e-12)


@pytest.mark.parametrize('hard', [False, True])
def test_jet_backend_export_derivatives_match_autograd(hard):
    torch.manual_seed(193)
    problem = Problem('rectangles')
    eager = baselines.PrimaryPINN(problem, 4, 2, hard=hard).double()
    manual = baselines.PrimaryPINN(problem, 4, 2, hard=hard, backend='jets').double()
    manual.load_state_dict(eager.state_dict())
    points = torch.tensor([[.095, .2], [.2, .2], [.7, .7], [.5, .53]], dtype=torch.float64)
    reference = eager.state(points, create_graph=False)
    actual = manual.state(points, create_graph=False)
    for left, right in zip(actual, reference):
        torch.testing.assert_close(left, right, rtol=5e-12, atol=2e-12)


def test_explicit_bilinear_material_gradient_matches_autograd_including_clamp():
    problem = Problem('rectangles')
    xy = torch.tensor([[.095, .2], [.2, .2], [.7, .7], [-.1, .2], [1.1, .4]],
                      dtype=torch.float64, requires_grad=True)
    E, k, dE, dk = baselines.differentiable_material(xy, problem, with_gradient=True)
    reference = torch.autograd.grad(E.sum(), xy)[0]
    torch.testing.assert_close(dE, reference, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(dk, reference, rtol=1e-13, atol=1e-13)
