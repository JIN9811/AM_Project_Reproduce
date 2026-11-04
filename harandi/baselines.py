"""Resumable standard PINN and deep-energy thermoelastic baselines.

The standard PINN uses second derivatives and differentiable bilinear nodal
material regularization. DEM uses first-order variational energies with
prescribed temperature during the mechanical variation. Paper-sized defaults
are 100,000 updates per field, Adam, soft boundaries, and 20,000-update blocks.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import islice
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .physics import Problem, nodal_phase, material
from .pinn import ParallelMLP, physical_state, make_batch
from .train import export, training_schedule
from .jets import forward_jet


@dataclass
class BaselineConfig:
    method: str = 'standard'
    geometry: str = 'circle'
    mode: str = 'coupled'
    steps: int = 100000
    switch: int = 20000
    width: int = 40
    layers: int = 5
    hard: bool = False
    n: int = 51
    optimizer: str = 'adam'
    lr: float = .001
    dtype: str = 'float64'
    device: str = 'auto'
    seed: int = 42
    log_every: int = 100
    checkpoint_every: int = 1000
    boundary_weight: float = 1.
    dem_neumann: bool = False
    fast_adam: bool = False
    backend: str = 'autograd'


def differentiable_material(xy, p, phase_nodes=None, with_gradient=False):
    n = p.material_grid
    v = (torch.as_tensor(nodal_phase(p), device=xy.device, dtype=xy.dtype)
         if phase_nodes is None else phase_nodes)
    a = (xy*n).clamp(0, n)
    ij = a.detach().to(torch.long).clamp(max=n-1)
    f = a-ij
    ix, iy = ij[:, 0], ij[:, 1]
    fx, fy = f[:, 0], f[:, 1]
    v00, v10, v01, v11 = v[iy, ix], v[iy, ix+1], v[iy+1, ix], v[iy+1, ix+1]
    phase = ((1-fx)*(1-fy)*v00 + fx*(1-fy)*v10
             + (1-fx)*fy*v01 + fx*fy*v11)
    em, ei = p.phases
    e = em+(ei-em)*phase
    if p.geometry == 'homogeneous':
        e = 1+0*e
    if with_gradient:
        gradient = (ei-em)*n*torch.stack(((1-fy)*(v10-v00)+fy*(v11-v01),
                                        (1-fx)*(v01-v00)+fx*(v11-v10)), 1)
        gradient = gradient*((xy >= 0) & (xy <= 1)).to(xy.dtype)
        if p.geometry == 'homogeneous':
            gradient = gradient*0
        return e, e, gradient, gradient
    return e, e


def make_standard_batch(p, n, device, dtype, seed=42):
    b = make_batch(p, n, device, dtype, seed)
    xy = b['xy'].cpu().numpy().copy()
    body = b['body'].cpu().numpy()
    # Classical material derivatives are undefined at interpolation knots.
    # Keep boundary points/counts but use reproducible off-knot interior points.
    rng = np.random.default_rng(seed+31)
    points = rng.uniform(1e-6, 1-1e-6, (int(body.sum()), 2))
    bad = np.any(abs(points*p.material_grid-np.rint(points*p.material_grid)) <= 1e-8, axis=1)
    while bad.any():
        points[bad] = rng.uniform(1e-6, 1-1e-6, (int(bad.sum()), 2))
        bad = np.any(abs(points*p.material_grid-np.rint(points*p.material_grid)) <= 1e-8, axis=1)
    xy[body] = points
    E, k = material(xy, p, 'nodal')
    for name, a in (('xy', xy), ('E', E), ('k', k)):
        b[name] = torch.as_tensor(a, device=device, dtype=dtype)
    return b


def _grad(a, x, create_graph=True):
    if not a.requires_grad:
        return x*0
    value = torch.autograd.grad(a.sum(), x, create_graph=create_graph,
                               retain_graph=True, allow_unused=True)[0]
    return x*0 if value is None else value


class PrimaryPINN(nn.Module):
    parametric = False

    def __init__(self, p, width=40, layers=5, hard=True, backend='autograd'):
        super().__init__()
        if backend not in ('autograd', 'jets'):
            raise ValueError('backend must be autograd or jets')
        self.p, self.hard, self.width, self.layers = p, hard, width, layers
        self.backend = backend
        self.mechanical = ParallelMLP(2, width, layers)
        self.thermal = ParallelMLP(1, width, layers)
        self.register_buffer('_phase_nodes', torch.as_tensor(nodal_phase(p)), persistent=False)

    def _group(self, xy, kind, freeze_parameters=False):
        count = 2 if kind == 'mechanical' else 1
        z = xy[None].expand(count, -1, -1)
        network = getattr(self, kind)
        if freeze_parameters:
            # Freeze parameters, not coordinates: div(sigma) contains grad(T).
            parameters = {name: value.detach() for name, value in network.named_parameters()}
            raw = torch.func.functional_call(network, parameters, (2*z-1,))
        else:
            raw = network(2*z-1)
        if not self.hard:
            return raw.T
        if kind == 'mechanical':
            return torch.stack((z[0, :, 0]*(1-z[0, :, 0])*raw[0],
                                z[1, :, 1]*(1-z[1, :, 1])*raw[1]), 1)
        return (1-z[0, :, 0]+z[0, :, 0]*(1-z[0, :, 0])*raw[0])[:, None]

    def _group_jet(self, xy, kind, freeze_parameters=False, second_order=False):
        heads = 2 if kind == 'mechanical' else 1
        network = getattr(self, kind)
        if freeze_parameters:
            network = SimpleNamespace(weights=[v.detach() for v in network.weights],
                                      biases=[v.detach() for v in network.biases])
        result = forward_jet(network, (2*xy-1)[None].expand(heads, -1, -1),
                             second_order=second_order)
        raw, jac = result[0], 2*result[1]
        hess = 4*result[2] if second_order else None
        if self.hard:
            direction = torch.eye(2, device=xy.device, dtype=xy.dtype)[:heads]
            coordinate = xy.T if heads == 2 else xy[:, 0][None]
            factor = coordinate*(1-coordinate)
            first = (1-2*coordinate)[..., None]*direction[:, None]
            if second_order:
                second = -2*direction[:, None, :, None]*direction[:, None, None, :]
                hess = (factor[..., None, None]*hess
                        + first[..., :, None]*jac[..., None, :]
                        + jac[..., :, None]*first[..., None, :]
                        + raw[..., None, None]*second)
            jac = factor[..., None]*jac+raw[..., None]*first
            raw = factor*raw
            if kind == 'thermal':
                raw = raw+1-coordinate
                jac = jac-direction[:, None]
        result = (raw.T, jac.permute(1, 0, 2))
        if second_order:
            result += (hess.permute(1, 0, 2, 3),)
        return result

    def _jet_primal(self, xy, detach_thermal=False, active='coupled', second_order=False):
        if active == 'thermal':
            u = xy.new_zeros((len(xy), 2))
            du = xy.new_zeros((len(xy), 2, 2))
            hu = xy.new_zeros((len(xy), 2, 2, 2)) if second_order else None
        else:
            mechanical = self._group_jet(xy, 'mechanical', second_order=second_order)
            u, du = mechanical[:2]
            hu = mechanical[2] if second_order else None
        thermal = self._group_jet(xy, 'thermal', freeze_parameters=detach_thermal,
                                  second_order=second_order)
        t, dt = thermal[:2]
        zero = 0*t
        f = torch.cat((u, zero.expand(-1, 3), t, zero.expand(-1, 2)), 1)
        g = torch.cat((du, du[:, :1]*0, du[:, :1]*0, du[:, :1]*0, dt, du*0), 1)
        coefficients = differentiable_material(xy, self.p, self._phase_nodes,
                                                with_gradient=second_order)
        E, k = coefficients[:2]
        st = physical_state(f, g, E, k, self.p)
        st['_E'], st['_k'] = E, k
        if second_order:
            dE, dk = coefficients[2:]
            nu = self.p.nu
            mu = 1/(2*(1+nu))
            lam = nu/((1+nu)*(1-2*nu))
            beta = self.p.alpha/(1-2*nu)
            ds = torch.stack(((lam+2*mu)*hu[:, 0, 0]+lam*hu[:, 1, 1]-beta*dt[:, 0],
                              lam*hu[:, 0, 0]+(lam+2*mu)*hu[:, 1, 1]-beta*dt[:, 0],
                              mu*(hu[:, 0, 1]+hu[:, 1, 0])), 1)
            st['stress_gradient'] = (st['stress'][..., None]*(dE/E[:, None])[:, None]
                                     + E[:, None, None]*ds)
            st['flux_gradient'] = (-dt[:, 0, :, None]*dk[:, None]
                                   - k[:, None, None]*thermal[2][:, 0])
        return xy, f, g, st

    def primal(self, xy, detach_thermal=False, active='coupled', second_order=False):
        if self.backend == 'jets':
            return self._jet_primal(xy, detach_thermal, active, second_order)
        x = xy if xy.requires_grad else xy.detach().clone().requires_grad_(True)
        if active == 'thermal':
            u, du = x*0, x[:, None, :].expand(-1, 2, -1)*0
        else:
            u = self._group(x, 'mechanical')
            du = torch.stack((_grad(u[:, 0], x), _grad(u[:, 1], x)), 1)
        t = self._group(x, 'thermal', freeze_parameters=detach_thermal)
        dt = _grad(t[:, 0], x)
        z = 0*t
        f = torch.cat((u, z.expand(-1, 3), t, z.expand(-1, 2)), 1)
        g = torch.cat((du, du[:, :1]*0, du[:, :1]*0, du[:, :1]*0, dt[:, None], du*0), 1)
        E, k = differentiable_material(x, self.p, self._phase_nodes)
        st = physical_state(f, g, E, k, self.p)
        st['_E'], st['_k'] = E, k
        return x, f, g, st

    def state(self, xy, create_graph=True, parameters=None):
        x, f, g, st = self.primal(xy, second_order=self.backend == 'jets')
        full = torch.cat((f[:, :2], st['stress'], f[:, 5:6], st['flux']), 1)
        if self.backend == 'jets':
            grad = torch.cat((g[:, :2], st['stress_gradient'], g[:, 5:6], st['flux_gradient']), 1)
            if not create_graph:
                grad = grad.detach()
        else:
            grad = torch.stack([_grad(full[:, j], x, create_graph) for j in range(8)], 1)
        return full, grad

    def predict(self, xy, parameters=None):
        _, f, _, st = self.primal(xy)
        return torch.cat((f[:, :2], st['stress'], f[:, 5:6], st['flux']), 1)


def loss_components(model, b, method, active='coupled', boundary_weight=1., dem_neumann=False):
    """Return raw components and weighted field/total objectives.

    DEM zero-flux/zero-shear natural boundaries are variational by default;
    optional explicit Neumann penalties support sensitivity studies.
    """
    if method not in ('standard', 'dem') or active not in ('thermal', 'mechanical', 'coupled'):
        raise ValueError('unknown baseline method or active field')
    x, f, g, st = model.primal(b['xy'], detach_thermal=active == 'mechanical', active=active,
                              second_order=method == 'standard')
    body, L, R, B, U = (b[key] for key in ('body', 'left', 'right', 'bottom', 'top'))
    mse = lambda value: value.square().mean()
    zero = f.sum()*0
    terms = {name: zero for name in ('M_equilibrium', 'M_dirichlet', 'M_neumann', 'M_energy',
                                    'T_equilibrium', 'T_dirichlet', 'T_neumann', 'T_energy')}
    if active != 'thermal':
        s = st['stress']
        if method == 'standard':
            if model.backend == 'jets':
                sx, sy, ss = (st['stress_gradient'][:, j] for j in range(3))
            else:
                sx, sy, ss = (_grad(s[:, j], x) for j in range(3))
            terms['M_equilibrium'] = (mse((sx[:, 0]+ss[:, 1])[body])
                                      + mse((ss[:, 0]+sy[:, 1])[body]))
        else:
            fixed = torch.cat((f[:, :5], f[:, 5:].detach()), 1)
            E, k = st['_E'], st['_k']
            terms['M_energy'] = physical_state(fixed, g, E, k, model.p)['elastic_energy'][body].mean()
        terms['M_dirichlet'] = sum(mse(f[mask, j]) for mask, j in ((L, 0), (R, 0), (B, 1), (U, 1)))
        if method == 'standard' or dem_neumann:
            terms['M_neumann'] = sum(mse(s[mask, 2]) for mask in (L, R, B, U))
    if active != 'mechanical':
        q = st['flux']
        if method == 'standard':
            if model.backend == 'jets':
                qx, qy = st['flux_gradient'][:, 0], st['flux_gradient'][:, 1]
            else:
                qx, qy = _grad(q[:, 0], x), _grad(q[:, 1], x)
            terms['T_equilibrium'] = mse((qx[:, 0]+qy[:, 1])[body])
        else:
            k = st['_k']
            terms['T_energy'] = .5*(k*g[:, 5].square().sum(1))[body].mean()
        terms['T_dirichlet'] = mse(f[L, 5]-1)+mse(f[R, 5])
        if method == 'standard' or dem_neumann:
            terms['T_neumann'] = mse(q[B, 1])+mse(q[U, 1])
    for prefix, name in (('M', 'mechanical'), ('T', 'thermal')):
        terms[name] = (terms[prefix+'_equilibrium']+terms[prefix+'_energy']
                       + boundary_weight*(terms[prefix+'_dirichlet']+terms[prefix+'_neumann']))
    terms['total'] = terms['mechanical']+terms['thermal']
    return terms


def loss(model, b, method, active):
    """Backward-compatible scalar objective."""
    return loss_components(model, b, method, active)['total']


def _new_optimizer(config, parameters):
    if config.optimizer == 'adam':
        return torch.optim.Adam(parameters, lr=config.lr, capturable=config.fast_adam,
                                foreach=False if config.fast_adam else None)
    return torch.optim.LBFGS(parameters, lr=1., max_iter=1, max_eval=25,
                            history_size=50, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-14)


def _validate(config):
    if config.method not in ('standard', 'dem') or config.mode not in ('coupled', 'sequential'):
        raise ValueError('unknown method or training mode')
    if config.optimizer not in ('adam', 'lbfgs'):
        raise ValueError('optimizer must be adam or lbfgs')
    if config.backend not in ('autograd', 'jets'):
        raise ValueError('backend must be autograd or jets')
    if config.fast_adam and config.optimizer != 'adam':
        raise ValueError('fast_adam requires Adam')
    if config.dtype not in ('float32', 'float64'):
        raise ValueError('dtype must be float32 or float64')
    if min(config.steps, config.switch, config.width, config.layers,
           config.log_every, config.checkpoint_every) < 1 or config.n < 3:
        raise ValueError('steps, architecture and intervals must be positive; n must be at least 3')
    if config.lr <= 0 or config.boundary_weight <= 0:
        raise ValueError('learning rate and boundary weight must be positive')


def _validate_resume(config, problem, checkpoint):
    if 'optimizers' not in checkpoint:
        raise ValueError('resume requires checkpoint.pt, not the inference model.pt')
    if checkpoint['problem'] != problem.to_dict():
        raise ValueError('resume problem differs')
    old = checkpoint['config']
    for key, value in asdict(config).items():
        previous_value = old.get(key, 'autograd' if key == 'backend' else None)
        if key == 'backend' and checkpoint['completed'] == 0:
            continue  # no optimizer update has occurred in the old backend
        if key not in ('steps', 'device', 'log_every', 'checkpoint_every') and previous_value != value:
            raise ValueError(f'resume config differs: {key}')
    completed = checkpoint['completed']
    if config.steps*(2 if config.mode == 'sequential' else 1) < completed:
        raise ValueError('requested budget is below completed updates')
    previous = islice(training_schedule(config.mode, old['steps'], config.switch), completed)
    requested = islice(training_schedule(config.mode, config.steps, config.switch), completed)
    if list(previous) != list(requested):
        raise ValueError('new sequential schedule reorders completed blocks')


def run_config(config, output, resume=None, observer=None, max_updates=None, problem=None):
    """Train a baseline; max_updates bounds this invocation, not its saved budget.

    Partial invocations save checkpoint and post-update exports, with metadata
    identifying them as incomplete.
    """
    _validate(config)
    if max_updates is not None and max_updates < 0:
        raise ValueError('max_updates must be nonnegative')
    p = problem or Problem(config.geometry)
    if p.geometry != config.geometry:
        raise ValueError('config geometry and problem geometry differ')
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if config.device == 'auto' else config.device
    if config.fast_adam and not device.startswith('cuda'):
        raise ValueError('fast_adam requires CUDA')
    dtype = getattr(torch, config.dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = PrimaryPINN(p, config.width, config.layers, config.hard,
                        backend=config.backend).to(device=device, dtype=dtype)
    batch = (make_standard_batch(p, config.n, device, dtype, config.seed) if config.method == 'standard'
             else make_batch(p, config.n, device, dtype, config.seed))
    if config.fast_adam:
        from .fast_adam import CapturedAdam, index_batch
        batch = index_batch(batch)
    ck = None if resume is None else torch.load(resume, map_location=device, weights_only=False)
    if ck is not None:
        _validate_resume(config, p, ck)
    stages = ('coupled',) if config.mode == 'coupled' else ('thermal', 'mechanical')
    parameters = {stage: list(model.parameters()) if stage == 'coupled'
                  else list(getattr(model, stage).parameters()) for stage in stages}
    optimizers = {stage: _new_optimizer(config, parameters[stage]) for stage in stages}
    completed, evaluations, elapsed_offset = 0, 0, 0.
    field_updates = {'thermal': 0, 'mechanical': 0}
    resets = {stage: 0 for stage in stages}
    last_active, history = None, []
    if ck is not None:
        model.load_state_dict(ck['model'])
        for stage, state in ck['optimizers'].items():
            optimizers[stage].load_state_dict(state)
        completed, field_updates = ck['completed'], ck['field_updates'].copy()
        evaluations, elapsed_offset = ck['closure_evaluations'], ck['elapsed_seconds']
        last_active, resets = ck['last_active'], ck['optimizer_resets'].copy()
        history = [row for row in ck.get('history', []) if row['step'] <= completed]
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        np.random.set_state(ck['numpy_rng_state'])
        if device.startswith('cuda') and ck.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng_state']])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(config), indent=2))
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'numpy': np.__version__, 'platform': platform.platform(), 'device': device,
                   'dtype': config.dtype, 'fast_adam': config.fast_adam,
                   'derivative_backend': config.backend,
                   'execution_variant': 'cuda_graph_capturable_adam' if config.fast_adam else 'eager_'+config.optimizer,
                   'GPU': torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                   'training_points': len(batch['xy']),
                   'parameter_count': sum(value.numel() for value in model.parameters())}
    (out/'environment.json').write_text(json.dumps(environment, indent=2))
    total = config.steps*len(stages)
    start = time.perf_counter()

    def checkpoint(filename='checkpoint.pt', with_optimizers=True):
        payload = {'model': model.state_dict(), 'config': asdict(config), 'problem': p.to_dict(),
                   'method': config.method, 'completed': completed, 'field_updates': field_updates.copy(),
                   'closure_evaluations': evaluations, 'last_active': last_active,
                   'optimizer_resets': resets.copy(), 'history': history,
                   'elapsed_seconds': elapsed_offset+time.perf_counter()-start,
                   'complete': completed == total, 'torch_rng_state': torch.get_rng_state(),
                   'numpy_rng_state': np.random.get_state(),
                   'cuda_rng_state': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
        if with_optimizers:
            payload['optimizers'] = {stage: opt.state_dict() for stage, opt in optimizers.items()}
        temporary = out/(filename+'.tmp')
        torch.save(payload, temporary)
        temporary.replace(out/filename)

    if ck is None:
        checkpoint()
    if observer is not None:
        observer(model, p, {'updates': completed, **field_updates})
    keys = ['optimizer', 'step', 'active', 'thermal_updates', 'mechanical_updates',
            'closure_evaluations', 'elapsed_seconds', 'evaluation_point', 'total', 'mechanical', 'thermal',
            'M_equilibrium', 'M_dirichlet', 'M_neumann', 'M_energy',
            'T_equilibrium', 'T_dirichlet', 'T_neumann', 'T_energy']
    initial_completed = completed
    fast_engine, fast_active = None, None
    with (out/'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
        stream.flush()
        for step, active in enumerate(training_schedule(config.mode, config.steps, config.switch), 1):
            if step <= completed:
                continue
            if max_updates is not None and completed-initial_completed >= max_updates:
                break
            for name in ('mechanical', 'thermal'):
                for value in getattr(model, name).parameters():
                    value.requires_grad_(active == 'coupled' or active == name)
            if active != last_active and active != 'coupled':
                # Thermal updates change the mechanical objective. Stale Adam
                # moments and L-BFGS curvature must not bridge that change.
                if active == 'mechanical' or config.optimizer == 'lbfgs':
                    optimizers[active] = _new_optimizer(config, parameters[active])
                    resets[active] += 1
            last_active = active
            opt = optimizers[active]

            def objective():
                return loss_components(model, batch, config.method, active,
                                       config.boundary_weight, config.dem_neumann)

            def closure():
                nonlocal evaluations
                opt.zero_grad(set_to_none=True)
                value = objective()['total']
                if not torch.isfinite(value):
                    raise FloatingPointError('nonfinite baseline objective; last checkpoint preserved')
                value.backward()
                evaluations += 1
                return value

            if config.fast_adam:
                if fast_engine is None or fast_active != active:
                    fast_engine = CapturedAdam(opt, objective)
                    fast_active = active
                fast_engine.step()
                evaluations += 1
            elif config.optimizer == 'adam':
                closure()
                opt.step()
            else:
                opt.step(closure)
            completed = step
            for field in field_updates:
                field_updates[field] += int(active == 'coupled' or active == field)
            if observer is not None:
                observer(model, p, {'updates': completed, **field_updates})
            if step == 1 or step % config.log_every == 0 or step == total:
                terms = loss_components(model, batch, config.method, active,
                                        config.boundary_weight, config.dem_neumann)
                row = {key: float(value.detach().cpu()) for key, value in terms.items()}
                if not np.isfinite(row['total']):
                    raise FloatingPointError('nonfinite baseline objective; last checkpoint preserved')
                row.update(optimizer=config.optimizer, step=step, active=active,
                           thermal_updates=field_updates['thermal'], mechanical_updates=field_updates['mechanical'],
                           closure_evaluations=evaluations, elapsed_seconds=elapsed_offset+time.perf_counter()-start,
                           evaluation_point='post_update')
                writer.writerow(row)
                stream.flush()
                history.append(row)
                print(f'{config.method} {p.geometry} {config.mode} {step}/{total} '
                      f'{active}: loss={row["total"]:.6g}', flush=True)
            if step % config.checkpoint_every == 0:
                checkpoint()
    checkpoint()
    for value in model.parameters():
        value.requires_grad_(True)
    checkpoint('model.pt', with_optimizers=False)
    (out/'history.json').write_text(json.dumps(history, indent=2))
    experiment = {'method': config.method, 'steps_per_field': config.steps,
                  'requested_total_updates': total, 'completed_updates': completed,
                  'field_updates': field_updates, 'complete': completed == total,
                  'optimizer': config.optimizer, 'mode': config.mode,
                  'execution_variant': environment['execution_variant'],
                  'derivative_backend': config.backend,
                  'material': 'nodal bilinear regularization',
                  'collocation': 'off-knot random interior; original boundaries' if config.method == 'standard' else 'mixed-method batch',
                  'mechanical_energy': 'complete plane-strain elastic energy with prescribed temperature',
                  'dem_natural_boundaries': 'penalized' if config.dem_neumann else 'variational',
                  'optimizer_switch_policy': 'reset mechanics after thermal blocks; reset all sequential L-BFGS blocks',
                  'history_evaluation': 'post_update',
                  'source_note': 'Section 4.1.2 specifies equal optimizer and per-field epochs, but does not fully specify comparator boundary penalties or DEM energy; these choices are explicit here.'}
    (out/'experiment.json').write_text(json.dumps(experiment, indent=2))
    metrics = export(model, p, device, dtype, out)
    metrics.update(complete=completed == total, completed_updates=completed, field_updates=field_updates)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    return model, metrics


def run(method, p, steps=10000, n=51, width=40, layers=5, device='auto',
        output='results/baseline', mode='sequential', optimizer='adam', **kwargs):
    """Legacy API (hard boundaries), returning exported fields."""
    resume, observer = kwargs.pop('resume', None), kwargs.pop('observer', None)
    maximum = kwargs.pop('max_updates', None)
    kwargs.setdefault('hard', True)
    config = BaselineConfig(method=method, geometry=p.geometry, steps=steps, n=n, width=width,
                            layers=layers, device=device, mode=mode, optimizer=optimizer, **kwargs)
    run_config(config, output, resume, observer, maximum, p)
    return np.load(Path(output)/'prediction.npz')['fields']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='BaselineConfig JSON; flags override it')
    for key, choices in (('method', ('standard', 'dem')), ('geometry', ('circle', 'rectangles', 'homogeneous')),
                         ('mode', ('coupled', 'sequential')), ('optimizer', ('adam', 'lbfgs')),
                         ('dtype', ('float32', 'float64')), ('backend', ('autograd', 'jets'))):
        parser.add_argument('--'+key, choices=choices, default=argparse.SUPPRESS)
    parser.add_argument('--device', default=argparse.SUPPRESS)
    for key in ('steps', 'switch', 'width', 'layers', 'n', 'seed', 'log_every', 'checkpoint_every'):
        parser.add_argument('--'+key.replace('_', '-'), type=int, default=argparse.SUPPRESS)
    for key in ('lr', 'boundary_weight'):
        parser.add_argument('--'+key.replace('_', '-'), type=float, default=argparse.SUPPRESS)
    boundaries = parser.add_mutually_exclusive_group()
    boundaries.add_argument('--soft', dest='hard', action='store_false', default=argparse.SUPPRESS)
    boundaries.add_argument('--hard', dest='hard', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--dem-neumann', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--fast-adam', action='store_true', default=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-updates', type=int)
    args = vars(parser.parse_args())
    path, out = args.pop('config'), args.pop('output')
    resume, maximum = args.pop('resume'), args.pop('max_updates')
    values = {} if path is None else json.loads(path.read_text())
    values.update(args)
    run_config(BaselineConfig(**values), out, resume=resume, max_updates=maximum)


if __name__ == '__main__':
    main()
