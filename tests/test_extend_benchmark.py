import importlib.util
from pathlib import Path

import numpy as np
import pytest

from maglev_sim import benchmark as b


def extension():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'extend_benchmark.py'
    assert path.exists(), 'independent extension script missing'
    spec = importlib.util.spec_from_file_location('extend_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_batch(directory, seed, content_seed=None):
    directory.mkdir()
    cfg = b.base_config(duration=.002).to_dict()
    samples = b.sample_specs('id', seed)
    t = np.arange(3, dtype=np.float64) * .001
    Y = np.zeros((1000, 3, 3), dtype=np.float32)
    U = np.zeros((1000, 3, 2), dtype=np.float32)
    quality = dict(finite=True, completed=True, termination='completed', gap_min=.007,
                   gap_max=.009, current_min=2., current_max=4., saturation_fraction=0., retries=0)
    content_seed = seed if content_seed is None else content_seed
    import hashlib
    for index, sample in enumerate(samples):
        Y[index] = (content_seed * 1000 + index) * 1e-8
        sample['quality'] = dict(quality)
        sample['content_sha256'] = hashlib.sha256(Y[index].tobytes() + U[index].tobytes()).hexdigest()
        sample['initial_delta'] = dict(z_c=0., z_m=0., v_c=0., v_m=0.)
    path = directory / 'id.npz'
    np.savez_compressed(path, Y=Y, U=U, t=t, dt=np.float64(.001),
                        split=np.asarray([s['split'] for s in samples], dtype='U5'),
                        Y_names=np.asarray(b.Y_NAMES), U_names=np.asarray(b.U_NAMES))
    source = dict(source_hash='fixture-core', commit='fixture')
    contract = dict(stage='id', master_seed=seed, source=source, config=cfg,
                    convergence_signature='fixture-convergence', selected_config_hash=b._hash(cfg))
    b._write_json(directory / 'id.manifest.json', dict(contract=contract, signature=b._hash(contract),
        sha256=b.file_hash(path), config=cfg, source=source, samples=samples,
        convergence_signature='fixture-convergence', shape=dict(Y=list(Y.shape), U=list(U.shape))))
    return path, cfg, source


def test_two_batches_merge_preserves_splits_and_maps_repeated_ids(tmp_path):
    e = extension()
    one, cfg, source = fixture_batch(tmp_path / 'one', 1)
    two, _, _ = fixture_batch(tmp_path / 'two', 2)
    contract = dict(config=cfg, source=source, convergence_signature='fixture-convergence',
                    seeds=[1, 2], batch_size=1000, script_hash='fixture-script')
    out = tmp_path / 'merged'
    path = e.merge_batches([one, two], out, contract)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {'Y', 'U', 't', 'dt', 'split', 'Y_names', 'U_names'}
        assert data['Y'].shape == (2000, 3, 3)
        assert data['Y'].dtype == np.float32 and data['t'].dtype == np.float64
        assert {s: int(np.sum(data['split'] == s)) for s in ['train', 'val', 'test']} == dict(train=1400, val=300, test=300)
    manifest = b._read_json(out / 'id.manifest.json')
    rows = manifest['samples']
    assert len({s['id'] for s in rows}) == 2000
    assert rows[0]['id'] == 'seed1:' + rows[0]['source_id']
    assert rows[1000]['id'] == 'seed2:' + rows[1000]['source_id']
    assert rows[0]['initial_delta'] == dict(z_c=0., z_m=0., v_c=0., v_m=0.)
    assert all(s['sample_id'] == s['id'] for s in rows)
    assert manifest['quality']['retries'] == 0
    assert e.merge_batches([one, two], out, contract) == path


def test_merge_rejects_duplicate_content_even_with_different_seeds(tmp_path):
    e = extension()
    one, cfg, source = fixture_batch(tmp_path / 'one', 1)
    two, _, _ = fixture_batch(tmp_path / 'two', 2, content_seed=1)
    contract = dict(config=cfg, source=source, convergence_signature='fixture-convergence',
                    seeds=[1, 2], batch_size=1000, script_hash='fixture-script')
    with pytest.raises(ValueError, match='duplicate trajectory content'):
        e.merge_batches([one, two], tmp_path / 'merged', contract)
    assert not (tmp_path / 'merged' / 'id.manifest.json').exists()


def test_resume_rejects_changed_contract(tmp_path):
    e = extension()
    assert e.lock_contract(tmp_path, dict(seeds=[1, 2], script_hash='a'))
    assert e.lock_contract(tmp_path, dict(seeds=[1, 2], script_hash='a'))
    with pytest.raises(ValueError, match='incompatible'):
        e.lock_contract(tmp_path, dict(seeds=[1, 3], script_hash='a'))


def test_merge_resumes_verified_memmap_after_interruption(tmp_path):
    e = extension()
    one, cfg, source = fixture_batch(tmp_path / 'one', 1)
    two, _, _ = fixture_batch(tmp_path / 'two', 2)
    contract = dict(config=cfg, source=source, convergence_signature='fixture-convergence',
                    seeds=[1, 2], batch_size=1000, script_hash='fixture-script')
    out = tmp_path / 'merged'
    def interrupt(**status):
        if status.get('merge_completed_batches') == 1:
            raise KeyboardInterrupt('fixture interruption')
    with pytest.raises(KeyboardInterrupt):
        e.merge_batches([one, two], out, contract, interrupt)
    assert b._read_json(out / '.merge' / 'progress.json')['completed_batches'] == 1
    assert not (out / 'id.manifest.json').exists()
    result = e.merge_batches([one, two], out, contract)
    assert b._read_json(result.with_suffix('.manifest.json'))['unique_trajectory_count'] == 2000
