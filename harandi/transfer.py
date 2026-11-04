"""Section 4.4: sequential transfer over ratios 1..10; ratio 15 is test-only.

The two material inputs are global E_matrix/E_inclusion and k_matrix/k_inclusion
ratios (log10 transformed), so they identify the entire BVP even in the matrix.
Defaults allocate 10,000 Adam updates to each of ten ratios. Network weights
transfer between ratios; Adam moments restart at a ratio boundary. Checkpoints
inside a ratio retain Adam moments and permit exact CPU continuation.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .physics import Problem, grid
from .pinn import MixedPINN, make_batch, mixed_loss
from .train import TrainConfig, export, relative_errors


HELD_OUT_RATIO = 15
HISTORY_KEYS = ('ratio', 'step', 'completed_updates', 'data', 'physics', 'loss', 'seconds')


def _labels(folder, xy):
    from scipy.spatial import cKDTree
    with np.load(Path(folder) / 'solution.npz') as data:
        points, fields = data['xy'], data['fields']
        if points.ndim != 2 or points.shape[1] != 2 or fields.shape != (len(points), 8):
            raise ValueError('FEM reference needs xy[N,2] and fields[N,8]')
        if not np.isfinite(points).all() or not np.isfinite(fields).all():
            raise ValueError('FEM reference contains nonfinite values')
        distance, indices = cKDTree(points).query(xy)
        if np.max(distance) > 1e-9:
            raise ValueError('FEM label grid must include every collocation coordinate')
        return fields[indices].copy()


def _reference(folder, ratio):
    """Fingerprint only a requested ratio; training never calls this for 15."""
    folder = Path(folder)
    path = folder / 'solution.npz'
    info = {'ratio': ratio, 'file': f'{folder.name}/solution.npz',
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'metadata_verified': False}
    metadata = folder / 'metadata.json'
    if metadata.exists():
        raw = metadata.read_bytes()
        details = json.loads(raw)
        expected = Problem('rectangles', ratio=ratio).to_dict()
        actual = details.get('problem', {})
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f'FEM reference {folder.name} has incompatible {key}')
        info.update(metadata_verified=True, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                    coefficient_mode=details.get('coefficient_mode'), fem_n=details.get('n'))
    return info


def _json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def _save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def _validate_resume(config, references, checkpoint):
    if checkpoint.get('format') != 'harandi-transfer-v2' or 'optimizer' not in checkpoint:
        raise ValueError('resume requires a transfer checkpoint.pt, not model.pt or legacy progress.pt')
    for key, value in config.items():
        if checkpoint['transfer_config'].get(key) != value:
            raise ValueError(f'resume config differs: {key}')
    if checkpoint['training_references'] != references:
        raise ValueError('training reference files differ from the checkpoint')
    index, step = checkpoint['ratio_index'], checkpoint['ratio_step']
    if not 0 <= index < len(config['ratios']) or not 0 <= step <= config['steps']:
        raise ValueError('checkpoint has invalid ratio progress')
    if checkpoint['completed_updates'] != index * config['steps'] + step:
        raise ValueError('checkpoint update count is inconsistent')


def run(references, output, ratios=None, steps=10000, weight=.1, n=51, width=40,
        layers=5, device='auto', energy='balance', resume=None, checkpoint_every=1000,
        log_every=100, seed=42, dtype='float64', lr=.001, max_updates=None,
        hard=True, normalization='peak', fast_adam=False):
    """Train the prescribed schedule; max_updates bounds one resumable invocation.

    Held-out labels are first read after the entire training schedule. Neither
    early stopping nor best-checkpoint selection uses that test reference.
    """
    ratios = [float(r) for r in (range(1, 11) if ratios is None else ratios)]
    if (not ratios or any(not np.isfinite(r) or not 1 <= r <= 10 for r in ratios)
            or len(set(ratios)) != len(ratios)):
        raise ValueError('training ratios must be unique finite values in [1,10]; ratio 15 is held out')
    if steps < 1 or not np.isfinite(weight) or weight < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('positive steps/lr and finite nonnegative physics weight required')
    if min(checkpoint_every, log_every, width, layers) < 1 or n < 3:
        raise ValueError('positive checkpoint/log intervals and architecture; n >= 3 required')
    if max_updates is not None and max_updates < 1:
        raise ValueError('max_updates must be positive when specified')
    if dtype not in ('float32', 'float64') or normalization not in ('peak', 'none'):
        raise ValueError('dtype must be float32/float64 and normalization peak/none')
    if energy not in ('balance', 'paper', 'authors', 'variational', 'none'):
        raise ValueError('unknown energy formulation')
    config = dict(ratios=ratios, steps=steps, weight=weight, n=n, width=width, layers=layers,
                  energy=energy, seed=seed, dtype=dtype, lr=lr, hard=hard,
                  normalization=normalization, fast_adam=fast_adam)
    references = Path(references)
    # Preflight the entire training schedule before creating/overwriting output.
    training_references = [_reference(references / f'ratio_{r:g}', r) for r in ratios]
    training_labels = [_labels(references / f'ratio_{r:g}', grid(n)) for r in ratios]
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if checkpoint is not None:
        _validate_resume(config, training_references, checkpoint)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if fast_adam and not device.startswith('cuda'):
        raise ValueError('fast_adam requires a CUDA device')
    torch_dtype = getattr(torch, dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    model = MixedPINN(width, layers, hard, parametric=True).to(device=device, dtype=torch_dtype)
    def new_optimizer():
        # Use identical CUDA Adam arithmetic for eager/graph comparisons.
        return torch.optim.Adam(model.parameters(), lr=lr, foreach=False,
                                capturable=device.startswith('cuda'))
    optimizer = new_optimizer()
    ratio_index = ratio_step = completed_updates = 0
    history = []
    elapsed_offset = 0.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ratio_index, ratio_step = checkpoint['ratio_index'], checkpoint['ratio_step']
        completed_updates = checkpoint['completed_updates']
        history = checkpoint['history']
        elapsed_offset = checkpoint['elapsed_seconds']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        if device.startswith('cuda') and checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    _json(out/'config.json',config)
    (out / 'checkpoints').mkdir(exist_ok=True)
    started = time.perf_counter()
    invocation_start_updates = completed_updates
    expected_updates = steps * len(ratios)
    status = 'running'
    evaluation = None
    elapsed = lambda: elapsed_offset + time.perf_counter() - started

    def metadata():
        completed_ratios = ratios[:ratio_index + (ratio_step == steps)]
        return {
            'status': status, 'training_ratios': ratios, 'test_ratio': HELD_OUT_RATIO,
            'steps_per_ratio': steps, 'requested_optimizer_updates': expected_updates,
            'actual_optimizer_updates': completed_updates, 'completed_training_ratios': completed_ratios,
            'active_ratio': ratios[ratio_index], 'active_ratio_updates': ratio_step,
            'physics_weight': weight, 'energy': energy, 'hard_dirichlet': hard,
            'training_elapsed_seconds': elapsed(), 'optimizer': 'Adam', 'learning_rate': lr,
            'optimizer_execution': 'CUDA graph replay' if fast_adam else 'eager',
            'optimizer_state_between_ratios': 'reset; only network weights transfer',
            'checkpoint_selection': 'last scheduled update; no held-out selection',
            'inputs': 'x, y, log10(global E_matrix/E_inclusion), log10(global k_matrix/k_inclusion)',
            'material_convention': 'Figure 22: matrix/inclusion; matrix E=k=1, inclusion E=k=1/ratio',
            'input_deviation': 'Global ratios identify the BVP; page 29 text describes local E(x,y),k(x,y).',
            'data_loss': ('mean of per-field peak-normalized squared errors (scale floor 0.05)'
                          if normalization == 'peak' else 'mean squared error over all eight fields'),
            'reconstruction_choices': ['log10 global ratio encoding', 'Adam reset at ratio transitions',
                'data normalization and hard-Dirichlet choice not specified in Section 4.4',
                'weight 0.5 in the four-weight sweep is a reconstruction choice'],
            'training_references': training_references, 'held_out_evaluation': evaluation,
            'config': config,
        }

    def save_checkpoint(path=None):
        payload = {'format': 'harandi-transfer-v2', 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'transfer_config': config,
                   'training_references': training_references, 'ratio_index': ratio_index,
                   'ratio_step': ratio_step, 'completed_updates': completed_updates,
                   'elapsed_seconds': elapsed(), 'history': history,
                   'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': np.random.get_state(),
                   'cuda_rng_state': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
        _save(out / 'checkpoint.pt' if path is None else path, payload)
        info = metadata()
        _json(out / 'experiment.json', info)
        _json(out / 'progress.json', {key: info[key] for key in ('status', 'active_ratio',
              'active_ratio_updates', 'actual_optimizer_updates', 'requested_optimizer_updates',
              'training_elapsed_seconds', 'completed_training_ratios')})

    _json(out / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
          'numpy': np.__version__, 'platform': platform.platform(), 'device': device, 'dtype': dtype,
          'GPU': torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
          'training_points_per_ratio': n*n, 'parameter_count': sum(p.numel() for p in model.parameters())})
    # Rebuild from checkpoint history, discarding rows after its committed update.
    with (out / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_KEYS)
        writer.writeheader()
        writer.writerows(history)
        stream.flush()
        save_checkpoint()
        for index in range(ratio_index, len(ratios)):
            if index != ratio_index:
                optimizer = new_optimizer()
                ratio_index, ratio_step = index, 0
            if ratio_step == steps:
                continue
            ratio = ratios[index]
            problem = Problem('rectangles', ratio=ratio)
            batch = make_batch(problem, n, device, torch_dtype, seed)
            expected = torch.full((n*n, 2), np.log10(ratio), device=device, dtype=torch_dtype)
            if 'parameters' not in batch or not torch.equal(batch['parameters'], expected):
                # np.log(r)/np.log(10) can differ from log10 by the last bit.
                if 'parameters' not in batch or not torch.allclose(batch['parameters'], expected,
                                                                  rtol=1e-14, atol=1e-14):
                    raise ValueError('transfer requires two spatially constant global material ratios')
            labels = torch.as_tensor(training_labels[index], device=device, dtype=torch_dtype)
            scale = labels.abs().amax(0).clamp_min(.05) if normalization == 'peak' else torch.ones_like(labels[0])
            def objective():
                prediction = model.predict(batch['xy'], batch['parameters'])
                data_loss = ((prediction-labels)/scale).square().mean()
                physics = mixed_loss(model, batch, problem, energy=energy)['total'] if weight else data_loss*0
                return {'data': data_loss, 'physics': physics, 'total': data_loss + weight*physics}
            engine = None
            if fast_adam:
                from .fast_adam import CapturedAdam, index_batch
                batch = index_batch(batch)
                engine = CapturedAdam(optimizer, objective)
            for step in range(ratio_step, steps):
                if engine is None:
                    optimizer.zero_grad(set_to_none=True)
                    losses = objective()
                    if not torch.isfinite(losses['total']):
                        raise FloatingPointError('nonfinite transfer objective; last checkpoint retained')
                    losses['total'].backward()
                    optimizer.step()
                else:
                    losses = engine.step()
                data_loss, physics, loss = (losses[key].detach() for key in ('data', 'physics', 'total'))
                del losses
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite transfer objective; last checkpoint retained')
                ratio_step = step + 1
                completed_updates += 1
                pause = max_updates is not None and completed_updates-invocation_start_updates >= max_updates
                if step == 0 or ratio_step % log_every == 0 or ratio_step == steps or pause:
                    row = {'ratio': ratio, 'step': ratio_step, 'completed_updates': completed_updates,
                           'data': float(data_loss.detach()), 'physics': float(physics.detach()),
                           'loss': float(loss.detach()), 'seconds': elapsed()}
                    history.append(row)
                    writer.writerow(row)
                    stream.flush()
                    print(f'transfer ratio={ratio:g} Adam {ratio_step}/{steps} '
                          f'total={completed_updates}/{expected_updates} loss={row["loss"]:.6g} '
                          f'seconds={row["seconds"]:.1f}', flush=True)
                if ratio_step == steps:
                    save_checkpoint(out / 'checkpoints' / f'ratio_{ratio:g}.pt')
                    _save(out / 'progress.pt', {'model': model.state_dict(),
                          'last_training_ratio': ratio, 'completed_updates': completed_updates})
                if ratio_step % checkpoint_every == 0 or ratio_step == steps or pause:
                    if pause and completed_updates < expected_updates:
                        status = 'paused'
                    save_checkpoint()
                if pause and completed_updates < expected_updates:
                    return model
            # No prior autograd graph may retain AccumulateGrad nodes when the
            # next ratio starts on a new capture stream.
            engine = None

    status = 'trained'
    save_checkpoint()
    training_seconds = elapsed()
    held = Problem('rectangles', ratio=HELD_OUT_RATIO)
    inference_config = asdict(TrainConfig(geometry='rectangles', width=width, layers=layers,
        hard=hard, ratio=HELD_OUT_RATIO, dtype=dtype, energy=energy, adam_steps=steps, lbfgs_steps=0))
    _save(out / 'model.pt', {'model': model.state_dict(), 'config': inference_config,
          'problem': held.to_dict(), 'parametric': True, 'training_ratios': ratios,
          'physics_weight': weight, 'completed_updates': completed_updates,
          'transfer_config': config, 'elapsed_seconds': training_seconds})
    evaluation_start = time.perf_counter()
    metrics = export(model, held, device, torch_dtype, out)
    test = references / 'ratio_15'
    if (test / 'solution.npz').exists():
        test_reference = _reference(test, HELD_OUT_RATIO)
        with np.load(out / 'prediction.npz') as prediction:
            labels = _labels(test, prediction['xy'])
            direct = prediction['fields']
            derived = direct.copy()
            derived[:, 2:5] = prediction['constitutive_stress']
            derived[:, 6:] = prediction['fourier_flux']
        comparison = {'direct_mixed_outputs': relative_errors(direct, labels),
                      'constitutive_outputs': relative_errors(derived, labels)}
        _json(out / 'held_out_ratio15_errors.json', comparison)
        metrics['comparisons']['FEM_held_out_ratio15'] = comparison
        _json(out / 'metrics.json', metrics)
        evaluation = {'reference': test_reference, 'points': len(labels),
                      'used_for_training_or_selection': False,
                      'seconds': time.perf_counter()-evaluation_start}
    else:
        evaluation = {'available': False, 'reason': 'ratio_15/solution.npz absent',
                      'used_for_training_or_selection': False}
    status = 'complete'
    final_info = metadata()
    final_info['training_elapsed_seconds'] = training_seconds
    final_info['evaluation_elapsed_seconds'] = time.perf_counter()-evaluation_start
    _json(out / 'experiment.json', final_info)
    _json(out / 'progress.json', {key: final_info[key] for key in ('status', 'active_ratio',
          'active_ratio_updates', 'actual_optimizer_updates', 'requested_optimizer_updates',
          'training_elapsed_seconds', 'completed_training_ratios')})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratios', type=float, nargs='+')
    for name, default in [('steps', 10000), ('n', 51), ('width', 40), ('layers', 5),
                          ('checkpoint_every', 1000), ('log_every', 100), ('seed', 42)]:
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--weight', type=float, default=.1)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--energy', default='balance')
    parser.add_argument('--normalization', choices=['peak', 'none'], default='peak')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-updates', type=int, help='pause after this many additional updates')
    parser.add_argument('--soft', action='store_true', help='use soft rather than hard Dirichlet constraints')
    parser.add_argument('--fast-adam', action='store_true', help='replay the unchanged Adam objective with CUDA graphs')
    args = vars(parser.parse_args())
    args['hard'] = not args.pop('soft')
    run(**args)


if __name__ == '__main__':
    main()
