"""Small real CPU runs protect the long transfer study from restart/leakage bugs."""
import csv
import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from harandi.physics import Problem, grid, homogeneous_exact
from harandi.transfer import run


def references(tmp_path, ratios=(1, 2), held_out=False):
    root = tmp_path / 'references'
    for ratio in (*ratios, *((15,) if held_out else ())):
        folder = root / f'ratio_{ratio:g}'
        folder.mkdir(parents=True)
        xy = grid(101)
        fields = homogeneous_exact(xy, Problem('homogeneous'))
        # Stable numerical labels suffice for optimizer continuation; these are
        # not claimed to be the heterogeneous benchmark's reference solution.
        fields[:, 2:5] /= ratio
        fields[:, 6:] /= ratio
        np.savez_compressed(folder / 'solution.npz', xy=xy, fields=fields)
    return root


def tiny_run(ref, out, **kwargs):
    return run(ref, out, ratios=[1, 2], steps=3, n=5, width=4, layers=1,
               device='cpu', weight=.1, log_every=1, checkpoint_every=1, **kwargs)


def test_resume_inside_ratio_preserves_adam_and_exact_final_weights(tmp_path):
    ref = references(tmp_path)
    full = tiny_run(ref, tmp_path / 'full')
    tiny_run(ref, tmp_path / 'resumed', max_updates=2)
    checkpoint = tmp_path / 'resumed' / 'checkpoint.pt'
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['completed_updates'] == 2
    assert saved['ratio_index'] == 0 and saved['ratio_step'] == 2
    assert saved['optimizer']['state']
    assert not (tmp_path / 'resumed' / 'model.pt').exists()
    resumed = tiny_run(ref, tmp_path / 'resumed', resume=checkpoint)
    for key, expected in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], expected, rtol=0, atol=0)
    rows = list(csv.DictReader((tmp_path / 'resumed' / 'history.csv').open()))
    assert [int(row['completed_updates']) for row in rows] == [1, 2, 3, 4, 5, 6]
    info = json.loads((tmp_path / 'resumed' / 'experiment.json').read_text())
    assert info['actual_optimizer_updates'] == 6 and info['status'] == 'complete'
    assert info['completed_training_ratios'] == [1, 2]
    for ratio in (1, 2):
        assert (tmp_path / 'resumed' / 'checkpoints' / f'ratio_{ratio}.pt').exists()


def test_held_out_labels_cannot_change_training_and_both_outputs_are_evaluated(tmp_path):
    ref = references(tmp_path, held_out=True)
    a = tiny_run(ref, tmp_path / 'a')
    held = ref / 'ratio_15' / 'solution.npz'
    with np.load(held) as data:
        xy, fields = data['xy'].copy(), data['fields'].copy()
    np.savez_compressed(held, xy=xy, fields=fields + 100.)
    b = tiny_run(ref, tmp_path / 'b')
    for key, expected in a.state_dict().items():
        torch.testing.assert_close(b.state_dict()[key], expected, rtol=0, atol=0)
    metrics = json.loads((tmp_path / 'a' / 'held_out_ratio15_errors.json').read_text())
    assert set(metrics) >= {'direct_mixed_outputs', 'constitutive_outputs'}
    prediction = np.load(tmp_path / 'a' / 'prediction.npz')
    assert prediction['fields'].shape == (10201, 8)
    assert prediction['constitutive_stress'].shape == (10201, 3)
    assert prediction['fourier_flux'].shape == (10201, 2)
    info = json.loads((tmp_path / 'a' / 'experiment.json').read_text())
    assert all(entry['ratio'] != 15 for entry in info['training_references'])
    assert info['checkpoint_selection'] == 'last scheduled update; no held-out selection'


@pytest.mark.parametrize('ratios', [[1, 15], [1, 1], [float('nan')], [float('inf')], [11]])
def test_rejects_test_ratio_and_invalid_training_schedule_before_writing(tmp_path, ratios):
    with pytest.raises(ValueError):
        run(tmp_path / 'missing', tmp_path / 'out', ratios=ratios, steps=1)
    assert not (tmp_path / 'out').exists()


def test_resume_rejects_changed_schedule_without_overwriting_metadata(tmp_path):
    ref = references(tmp_path)
    out = tmp_path / 'run'
    tiny_run(ref, out, max_updates=2)
    before = (out / 'experiment.json').read_bytes()
    with pytest.raises(ValueError, match='steps'):
        run(ref, out, ratios=[1, 2], steps=4, n=5, width=4, layers=1,
            device='cpu', resume=out / 'checkpoint.pt')
    assert (out / 'experiment.json').read_bytes() == before


def test_resume_rejects_changed_training_labels(tmp_path):
    ref = references(tmp_path)
    out = tmp_path / 'run'
    tiny_run(ref, out, max_updates=2)
    path = ref / 'ratio_2' / 'solution.npz'
    with np.load(path) as data:
        xy, fields = data['xy'].copy(), data['fields'].copy()
    np.savez_compressed(path, xy=xy, fields=fields + .1)
    with pytest.raises(ValueError, match='reference'):
        tiny_run(ref, out, resume=out / 'checkpoint.pt')


def test_material_metadata_rejects_wrong_bvp(tmp_path):
    ref = references(tmp_path)
    (ref / 'ratio_2' / 'metadata.json').write_text(json.dumps({
        'problem': Problem('rectangles', ratio=15).to_dict(),
        'coefficient_mode': 'nodal'}))
    with pytest.raises(ValueError, match='ratio'):
        tiny_run(ref, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_two_global_parameters_identify_bvp_even_at_same_matrix_coordinate():
    from harandi.pinn import make_batch
    one = make_batch(Problem('rectangles', ratio=1), n=5)
    ten = make_batch(Problem('rectangles', ratio=10), n=5)
    assert one['parameters'].shape == (25, 2)
    torch.testing.assert_close(one['parameters'], torch.zeros((25, 2), dtype=torch.float64))
    torch.testing.assert_close(ten['parameters'], torch.ones((25, 2), dtype=torch.float64))
    assert one['E'][0] == ten['E'][0] == 1.
    assert not torch.equal(one['parameters'][0], ten['parameters'][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_fast_transfer_preserves_updates_across_capture_resume_and_ratio_transition(tmp_path):
    ref = references(tmp_path)
    options = dict(ratios=[1, 2], steps=6, n=5, width=4, layers=1,
                   device='cuda', weight=.1, log_every=3, checkpoint_every=3)
    eager = run(ref, tmp_path / 'eager', **options)
    run(ref, tmp_path / 'fast', **options, fast_adam=True, max_updates=5)
    fast = run(ref, tmp_path / 'fast', **options, fast_adam=True,
               resume=tmp_path / 'fast' / 'checkpoint.pt')
    for key, expected in eager.state_dict().items():
        torch.testing.assert_close(fast.state_dict()[key], expected, rtol=1e-10, atol=1e-12)
    checkpoint = torch.load(tmp_path / 'fast' / 'checkpoint.pt', weights_only=False)
    assert checkpoint['completed_updates'] == 12
    assert all(int(state['step']) == 6 for state in checkpoint['optimizer']['state'].values())


def test_fast_transfer_rejects_cpu_before_creating_outputs(tmp_path):
    ref = references(tmp_path)
    with pytest.raises(ValueError, match='CUDA'):
        tiny_run(ref, tmp_path / 'out', fast_adam=True)
    assert not (tmp_path / 'out').exists()
