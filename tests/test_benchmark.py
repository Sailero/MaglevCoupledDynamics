from collections import Counter
import importlib.util

import numpy as np
import pytest


def benchmark():
    assert importlib.util.find_spec('maglev_sim.benchmark') is not None, 'benchmark module missing'
    from maglev_sim import benchmark as b
    return b


def test_split_stratification_and_disjoint_namespaces():
    b = benchmark()
    specs = b.sample_specs('id', 20260922)
    assert len(specs) == 1000
    assert Counter(s['split'] for s in specs) == dict(train=700, val=150, test=150)
    for split, counts in [('train', [70, 140, 140, 350]), ('val', [15, 30, 30, 75]), ('test', [15, 30, 30, 75])]:
        assert [sum(s['split'] == split and s['category'] == c for s in specs) for c in b.CATEGORIES] == counts
    pilot = b.sample_specs('pilot', 20260922)
    assert len(pilot) == 32 and {s['split'] for s in pilot} == {'train', 'val', 'test'}
    assert not {s['sample_id'] for s in pilot} & {s['sample_id'] for s in specs}
    for name, count in [('ood_amplitude', 100), ('ood_frequency', 100), ('ood_long', 50)]:
        ss = b.sample_specs(name, 20260922)
        assert len(ss) == count and all(s['split'] == 'test' for s in ss)


def test_waveforms_reproducible_normalized_and_independent():
    b = benchmark()
    spec = b.sample_specs('id', 20260922)[-1]
    t = np.arange(5001) * .001
    a = b.realization(spec, t, 0)
    again = b.realization(spec, t, 0)
    assert np.array_equal(a['U'], again['U'])
    assert a['initial_delta'] == again['initial_delta']
    assert a['U'][0].tolist() == [0., 0.]
    for j, name in enumerate(['r', 'P']):
        wave = a['waveforms'][name]
        assert np.sqrt(np.mean(a['U'][t >= .2, j] ** 2)) == pytest.approx(wave['target_rms'])
        assert wave['crest'] <= 4
    assert a['subseeds']['r'] != a['subseeds']['P'] != a['subseeds']['initial']
    assert not np.array_equal(a['U'], b.realization(spec, t, 1)['U'])
    for key, scale in [('z_c', 1e-4), ('z_m', 1e-4), ('v_c', 1e-3), ('v_m', 1e-3)]:
        assert abs(a['initial_delta'][key]) <= 3 * scale


def test_comparison_gate_and_quality_aggregation():
    b = benchmark()
    ref = np.zeros((20, 4))
    assert b.compare_arrays(ref, ref)['passed']
    assert not b.compare_arrays(ref + 2e-6, ref)['passed']
    result = b.aggregate_quality([dict(gap_min=.005, gap_max=.01, current_min=2., current_max=6., saturation_fraction=.1, retries=2), dict(gap_min=.004, gap_max=.009, current_min=3., current_max=8., saturation_fraction=.3, retries=1)])
    assert result['gap_min'] == .004 and result['current_max'] == 8.
    assert result['saturation_fraction_mean'] == pytest.approx(.2)
    assert result['retries'] == 3


def test_export_schema_input_closure_and_resume(tmp_path, monkeypatch):
    b = benchmark()
    specs = [s for s in b.sample_specs('id', 20260922) if s['category'] == 'both'][:2]
    monkeypatch.setattr(b, 'sample_specs', lambda stage, seed: specs)
    cfg = b.base_config(duration=.25).to_dict()
    gate = dict(passed=True, selected_config=cfg, signature='unit-test-short-integration', selected_config_hash=b._hash(cfg))
    path = b.generate_stage('id', tmp_path, convergence=gate)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {'Y', 'U', 't', 'dt', 'split', 'Y_names', 'U_names'}
        assert data['Y'].shape == (2, 251, 3)
        assert data['U'].shape == (2, 251, 2)
        assert data['Y'].dtype == np.float32 and data['U'].dtype == np.float32
        assert data['t'].dtype == np.float64 and data['split'].dtype.kind == 'U'
        assert data['dt'].shape == ()
        for index, spec in enumerate(specs):
            realization = b.realization(spec, data['t'], 0)
            np.testing.assert_array_equal(data['U'][index], realization['U'].astype(np.float32))
    manifest = b._read_json(tmp_path / 'id.manifest.json')
    assert [s['id'] for s in manifest['samples']] == [s['id'] for s in specs]
    assert len({s['content_sha256'] for s in manifest['samples']}) == 2
    assert b.generate_stage('id', tmp_path, convergence=gate) == path
    with pytest.raises(ValueError, match='incompatible'):
        b.generate_stage('id', tmp_path, seed=1, convergence=gate)


def test_ood_changes_exactly_one_factor_with_both_inputs_active():
    b = benchmark()
    t = np.arange(5001) * .001
    for stage in ('ood_amplitude', 'ood_frequency'):
        for factor in ('r', 'P'):
            spec = next(s for s in b.sample_specs(stage, 1) if s['group'].startswith(factor + '_'))
            data = b.realization(spec, t, 0)
            assert np.std(data['U'], axis=0).min() > 0
            rw, pw = data['waveforms']['r'], data['waveforms']['P']
            if stage == 'ood_amplitude':
                assert (1e-4 <= rw['target_rms'] <= 1.6e-4) if factor == 'r' else (2e-5 <= rw['target_rms'] <= 8e-5)
                assert (250 <= pw['target_rms'] <= 400) if factor == 'P' else (100 <= pw['target_rms'] <= 200)
            else:
                assert rw['band'] == ([12., 20.] if factor == 'r' else [.5, 10.])
                assert pw['band'] == ([35., 60.] if factor == 'P' else [.5, 30.])


def test_common_nine_modes_requires_every_case_validation(tmp_path, monkeypatch):
    b = benchmark()
    # Expensive integration boundary: one case needs nine; another must fail at nine.
    def case(job):
        spec = job[0]
        forced = len(job) > 4 and job[4]
        needs_nine = spec['index'] == 0
        return dict(sample=spec, selected_modes=9 if needs_nine or forced else 7,
                    passed=not (forced and spec['index'] == 1))
    monkeypatch.setattr(b, '_convergence_case', case)
    with pytest.raises(RuntimeError, match='convergence failed'):
        b.ensure_convergence(tmp_path, 1)
    report = b._read_json(tmp_path / 'convergence.json')
    assert not report['passed']
