"""Reproducible, input-closed DUE benchmark with a mandatory convergence gate.

Run ``python -m maglev_sim.benchmark --stage pilot --out-dir data/benchmark``.
Per-trajectory shards are internal checkpoints; exported NPZs have exactly seven keys.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from maglev_sim.due import Y_NAMES, U_NAMES
from maglev_sim.plant.config import Config, demo_config
from maglev_sim.plant.excitation import Excitation
from maglev_sim.plant.model import initial_state
from maglev_sim.plant.simulate import simulate

CATEGORIES = ('free', 'r_only', 'p_only', 'both')
NAMESPACES = dict(pilot=11, id=23, ood_amplitude=37, ood_frequency=41, ood_long=53, convergence=71)
VERSION = 1
COARSE = dict(method='RK45', rtol=1e-8, atol=1e-10, max_step=.001)
TIGHT = dict(method='RK45', rtol=1e-9, atol=1e-11, max_step=.0005)


def _json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _hash(data):
    return hashlib.sha256(_json(data).encode()).hexdigest()


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)).replace('\\', '/'): file_hash(p)
             for p in sorted((root / 'maglev_sim').rglob('*.py'))}
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    return dict(commit=result.stdout.strip(), source_hash=_hash(files), files=files)


def base_config(n_modes=5, integrator=None, duration=5.):
    cfg = demo_config()
    cfg.motion.xi, cfg.force.x_f = 9.12, 15.12
    cfg.beam.n_modes = n_modes
    cfg.output.t_start, cfg.output.t_end, cfg.output.dt = 0., duration, .001
    for key, value in (integrator or TIGHT).items():
        setattr(cfg.integrator, key, value)
    return cfg


def sample_specs(stage, seed):
    """Fixed row identities, independent of process scheduling and rejected draws."""
    rows = []
    def add(category, split, group=None):
        index = len(rows)
        rows.append(dict(sample_id=f'{stage}-{index:05d}', id=f'{stage}-{index:05d}',
                         index=index, stage=stage, category=category, split=split,
                         group=group or category, master_seed=int(seed)))
    if stage == 'id':
        for split, counts in [('train', (70, 140, 140, 350)), ('val', (15, 30, 30, 75)), ('test', (15, 30, 30, 75))]:
            for category, count in zip(CATEGORIES, counts):
                for _ in range(count):
                    add(category, split)
    elif stage == 'pilot':
        for split, counts in [('train', (2, 4, 4, 12)), ('val', (1, 1, 1, 2)), ('test', (1, 1, 1, 2))]:
            for category, count in zip(CATEGORIES, counts):
                for _ in range(count):
                    add(category, split)
    elif stage in ('ood_amplitude', 'ood_frequency'):
        for factor in ('r', 'P'):
            for _ in range(50):
                add('both', 'test', factor + '_' + stage.removeprefix('ood_'))
    elif stage == 'ood_long':
        for category, count in zip(CATEGORIES, (5, 10, 10, 25)):
            for _ in range(count):
                add(category, 'test')
    elif stage == 'convergence':
        for category in CATEGORIES:
            add(category, 'test')
        for group in ('r_amplitude', 'P_amplitude', 'r_frequency', 'P_frequency', 'nominal_high', 'initial_positive', 'initial_negative', 'long'):
            add('both', 'test', group)
    else:
        raise ValueError(f'unknown stage {stage}')
    # Membership is stratified before deterministic shuffling; ids stay immutable.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), NAMESPACES[stage], 997]))
    if stage != 'convergence':
        rows = [rows[i] for i in rng.permutation(len(rows))]
    return rows


def evaluate_waveform(t, coefficients):
    t = np.asarray(t, dtype=float)
    if coefficients['scale'] == 0:
        return np.zeros_like(t)
    ramp = .5 * (1 - np.cos(np.pi * np.minimum(np.maximum(t, 0) / .2, 1)))
    carrier = np.sin(2 * np.pi * t[:, None] * np.asarray(coefficients['frequencies'])[None, :]
                     + np.asarray(coefficients['phases'])[None, :]).sum(axis=1)
    return ramp * carrier * coefficients['scale']


def _waveform(t, seed_sequence, band, rms_range, active, high=False):
    rng = np.random.default_rng(seed_sequence)
    target = float(rms_range[1] if high else rng.uniform(*rms_range)) if active else 0.
    rejected = []
    if not active:
        return np.zeros_like(t), dict(frequencies=[], phases=[], scale=0., target_rms=0., rms=0., crest=0., rejected=[])
    for attempt in range(100):
        coeff = dict(frequencies=rng.uniform(*band, 8).tolist(), phases=rng.uniform(0, 2 * np.pi, 8).tolist(), scale=1.)
        raw = evaluate_waveform(t, coeff)
        normalization = float(np.sqrt(np.mean(raw[t >= .2] ** 2)))
        coeff['scale'] = target / normalization
        values = evaluate_waveform(t, coeff)
        crest = float(np.max(np.abs(values)) / target)
        if crest <= 4:
            coeff.update(target_rms=target, rms=float(np.sqrt(np.mean(values[t >= .2] ** 2))), crest=crest, band=list(band), rejected=rejected)
            return values, coeff
        rejected.append(dict(attempt=attempt, reason='crest_exceeds_4', crest=crest))
    raise RuntimeError('waveform crest rejection limit reached')


def realization(spec, t, attempt):
    seq = np.random.SeedSequence([spec['master_seed'], NAMESPACES[spec['stage']], spec['index'], int(attempt)])
    streams = dict(zip(('initial', 'r', 'P'), seq.spawn(3)))
    group = spec['group']
    r_range = (1e-4, 1.6e-4) if group == 'r_amplitude' else (2e-5, 8e-5)
    p_range = (250., 400.) if group == 'P_amplitude' else (100., 200.)
    r_band = (12., 20.) if group == 'r_frequency' else (.5, 10.)
    p_band = (35., 60.) if group == 'P_frequency' else (.5, 30.)
    high = spec['stage'] == 'convergence' and group not in CATEGORIES
    r, rw = _waveform(t, streams['r'], r_band, r_range, spec['category'] in ('r_only', 'both'), high)
    p, pw = _waveform(t, streams['P'], p_band, p_range, spec['category'] in ('p_only', 'both'), high)
    rng = np.random.default_rng(streams['initial'])
    delta = {}
    for key, sigma in [('z_c', 1e-4), ('z_m', 1e-4), ('v_c', 1e-3), ('v_m', 1e-3)]:
        value = float(rng.normal())
        while abs(value) > 3:
            value = float(rng.normal())
        if group in ('initial_positive', 'initial_negative'):
            value = 3. if group == 'initial_positive' else -3.
        delta[key] = value * sigma
    return dict(U=np.column_stack((r, p)), initial_delta=delta, waveforms=dict(r=rw, P=pw),
                subseeds={name: dict(entropy=stream.entropy, spawn_key=list(stream.spawn_key)) for name, stream in streams.items()}, attempt=int(attempt))


def _run(cfg, realization_data, t, input_grid=None):
    cfg = copy.deepcopy(cfg)
    knots = t if input_grid is None else input_grid
    inputs = realization_data['U'] if input_grid is None else np.column_stack([
        evaluate_waveform(knots, realization_data['waveforms'][name]) for name in ('r', 'P')])
    cfg.irregularity.type, cfg.irregularity.t, cfg.irregularity.r = 'time_series', knots.tolist(), inputs[:, 0].tolist()
    cfg.force.type, cfg.force.t, cfg.force.P = 'precomputed', knots.tolist(), inputs[:, 1].tolist()
    cfg.initial.delta = realization_data['initial_delta']
    cfg.initial.random_std = {}
    exc = Excitation.from_config(cfg)
    y0 = initial_state(cfg, excitation=exc)
    return simulate(cfg, y0, exc, t)


def trajectory_arrays(traj):
    return (np.column_stack([getattr(traj, k) for k in Y_NAMES]),
            np.column_stack([getattr(traj, k) for k in U_NAMES]))


def quality(traj, times):
    y, u = trajectory_arrays(traj)
    finite = all(np.isfinite(a).all() for a in (traj.state, y, u, traj.gap_raw, traj.current))
    complete = bool(traj.completed and traj.t.shape == times.shape and np.array_equal(traj.t, times))
    return dict(finite=bool(finite), completed=complete, termination=traj.termination_reason,
                gap_min=float(np.min(traj.gap_raw)), gap_max=float(np.max(traj.gap_raw)),
                current_min=float(np.min(traj.current)), current_max=float(np.max(traj.current)),
                saturation_fraction=float(np.mean(traj.saturated)), saturated_any=bool(np.any(traj.saturated)), retries=0)


def aggregate_quality(rows):
    return dict(gap_min=min(r['gap_min'] for r in rows), gap_max=max(r['gap_max'] for r in rows),
                current_min=min(r['current_min'] for r in rows), current_max=max(r['current_max'] for r in rows),
                saturation_fraction_mean=float(np.mean([r['saturation_fraction'] for r in rows])),
                retries=sum(r['retries'] for r in rows))


def compare_arrays(actual, reference):
    """Per-channel tolerances use reference dynamics about its temporal mean."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return dict(passed=False, reason='shape_or_nonfinite')
    error = actual - reference
    max_error = np.max(np.abs(error), axis=0)
    rms_error = np.sqrt(np.mean(error ** 2, axis=0))
    max_tol = np.maximum(1e-6, .01 * np.ptp(reference, axis=0))
    rms_tol = np.maximum(2e-7, .005 * np.sqrt(np.mean((reference - reference.mean(axis=0)) ** 2, axis=0)))
    return dict(passed=bool(np.all(max_error <= max_tol) and np.all(rms_error <= rms_tol)),
                max_error=max_error.tolist(), rms_error=rms_error.tolist(), max_tolerance=max_tol.tolist(), rms_tolerance=rms_tol.tolist())


def _compare_runs(a, b):
    result = compare_arrays(a['values'], b['values'])
    qa, qb = a['quality'], b['quality']
    status = (qa['finite'] and qb['finite'] and qa['completed'] and qb['completed']
              and qa['termination'] == qb['termination'] and qa['saturated_any'] == qb['saturated_any']
              and abs(qa['saturation_fraction'] - qb['saturation_fraction']) <= .001)
    result['status_consistent'] = bool(status)
    result['passed'] = bool(result['passed'] and status)
    return result


def _convergence_case(job):
    spec, directory, signature, refine, *extra = job
    force_nine = bool(extra and extra[0])
    path = Path(directory) / (spec['id'] + ('-common9' if force_nine else '') + '.json')
    if path.exists():
        cached = _read_json(path)
        if cached['signature'] != signature:
            raise ValueError('incompatible convergence checkpoint; use a new output directory')
        return cached
    started = time.perf_counter()
    duration = 10. if spec['group'] == 'long' else 5.
    t = np.arange(round(duration / .001) + 1) * .001
    realization_data = realization(spec, t, 0)
    runs = {}
    def run(modes, tight):
        key = f'{modes}_{tight}'
        if key not in runs:
            cache = Path(directory) / (spec['id'] + '-' + key + '.npz')
            cache_meta = cache.with_suffix('.meta.json')
            if cache.exists() and cache_meta.exists():
                meta = _read_json(cache_meta)
                if meta['signature'] != signature or file_hash(cache) != meta['sha256']:
                    raise ValueError('incompatible convergence run cache')
                with np.load(cache, allow_pickle=False) as data:
                    runs[key] = dict(values=data['values'], quality=meta['quality'])
                return runs[key]
            cfg = base_config(modes, TIGHT if tight else COARSE, duration)
            traj = _run(cfg, realization_data, t)
            y, u = trajectory_arrays(traj)
            gap = cfg.magnet.delta_0 + y[:, 1] - u[:, 0] - y[:, 2]
            runs[key] = dict(values=np.column_stack((y, gap)), quality=quality(traj, t))
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, values=runs[key]['values'])
            os.replace(temp, cache)
            _write_json(cache_meta, dict(signature=signature, sha256=file_hash(cache), quality=runs[key]['quality']))
        return runs[key]
    comparisons = {}
    comparisons['integrator_5'] = _compare_runs(run(5, False), run(5, True))
    comparisons['integrator_7'] = _compare_runs(run(7, False), run(7, True))
    comparisons['modes_5_7'] = _compare_runs(run(5, True), run(7, True))
    selected_modes = 7
    if force_nine or not comparisons['modes_5_7']['passed']:
        comparisons['integrator_9'] = _compare_runs(run(9, False), run(9, True))
        comparisons['modes_7_9'] = _compare_runs(run(7, True), run(9, True))
        selected_modes = 9
    modal_pass = comparisons[f'modes_{selected_modes - 2}_{selected_modes}']['passed']
    integrator_pass = all(value['passed'] for key, value in comparisons.items() if key.startswith('integrator'))
    diagnostic = None
    if refine:
        cfg = base_config(selected_modes, TIGHT, duration)
        fine = np.arange(round(duration / .0005) + 1) * .0005
        traj = _run(cfg, realization_data, t, input_grid=fine)
        y, _ = trajectory_arrays(traj)
        diagnostic = _compare_runs(run(selected_modes, True), dict(values=np.column_stack((y, traj.gap_raw)), quality=quality(traj, t)))
    record = dict(signature=signature, sample=spec, passed=bool(modal_pass and integrator_pass),
                  selected_modes=selected_modes, comparisons=comparisons, input_grid_diagnostic=diagnostic,
                  realization={k: v for k, v in realization_data.items() if k != 'U'},
                  qualities={k: v['quality'] for k, v in runs.items()}, runtime_seconds=time.perf_counter() - started)
    _write_json(path, record)
    print(f"convergence {spec['id']} passed={record['passed']} modes={selected_modes}", flush=True)
    return record


def _map(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, jobs))


def ensure_convergence(out_dir, seed, workers=1, refine=False):
    out_dir = Path(out_dir)
    path = out_dir / 'convergence.json'
    provenance = source_provenance()
    contract = dict(version=VERSION, seed=seed, source=provenance, config=base_config().to_dict(), refine=refine)
    signature = _hash(contract)
    if path.exists():
        report = _read_json(path)
        if report['signature'] != signature:
            raise ValueError('incompatible convergence report; use a new output directory')
        if not report['passed']:
            raise RuntimeError('convergence gate previously failed; inspect convergence.json')
        return report
    started = time.perf_counter()
    jobs = [(spec, str(out_dir / '.convergence'), signature, refine) for spec in sample_specs('convergence', seed)]
    cases = _map(_convergence_case, jobs, workers)
    if any(case['selected_modes'] == 9 for case in cases):
        # A single shared production configuration must pass every representative case.
        extra_jobs = [(*job, True) for job, case in zip(jobs, cases) if case['selected_modes'] != 9]
        upgraded = {case['sample']['id']: case for case in _map(_convergence_case, extra_jobs, workers)}
        cases = [upgraded.get(case['sample']['id'], case) for case in cases]
    passed = all(case['passed'] for case in cases)
    selected = base_config(max(case['selected_modes'] for case in cases), TIGHT).to_dict()
    report = dict(signature=signature, contract=contract, passed=passed, selected_config=selected,
                  selected_config_hash=_hash(selected), channels=list(Y_NAMES) + ['reconstructed_gap'], cases=cases,
                  runtime_seconds=time.perf_counter() - started)
    _write_json(path, report)
    if not passed:
        raise RuntimeError('convergence failed at required tolerances (up to 9 modes); inspect convergence.json')
    return report


def _sample_job(job):
    spec, cfg_dict, directory, signature, max_attempts = job
    directory = Path(directory)
    path, meta_path = directory / (spec['id'] + '.npz'), directory / (spec['id'] + '.json')
    if path.exists() and meta_path.exists():
        meta = _read_json(meta_path)
        if meta['signature'] != signature or file_hash(path) != meta['sha256']:
            raise ValueError(f'incompatible/corrupt checkpoint {spec["id"]}')
        return meta
    started = time.perf_counter()
    cfg = Config.from_dict(cfg_dict)
    if spec['stage'] == 'ood_long':
        cfg.output.t_end = 10.
    t = cfg.output_times()
    failure_path = directory / (spec['id'] + '.failures.json')
    failures = _read_json(failure_path) if failure_path.exists() else []
    for attempt in range(len(failures), max_attempts):
        try:
            data = realization(spec, t, attempt)
            traj = _run(cfg, data, t)
            stats = quality(traj, t)
            if not stats['finite'] or not stats['completed']:
                raise RuntimeError(f"incomplete/nonfinite trajectory: {stats['termination']}")
            y, u = trajectory_arrays(traj)
            # This exact equality proves the exported input knots are those integrated.
            if not np.array_equal(u, data['U']):
                raise RuntimeError('exported input does not match integrated input knots')
            stats['retries'] = attempt
            directory.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f, Y=y.astype(np.float32), U=u.astype(np.float32), t=t)
            os.replace(temp, path)
            meta = dict(spec, signature=signature, quality=stats, failures=failures,
                        **{k: v for k, v in data.items() if k != 'U'},
                        sha256=file_hash(path), runtime_seconds=time.perf_counter() - started)
            _write_json(meta_path, meta)
            print(f"accepted {spec['id']} attempt={attempt}", flush=True)
            return meta
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            failures.append(dict(attempt=attempt, reason=str(exc)))
            _write_json(failure_path, failures)
    raise RuntimeError(f'{spec["id"]}: exhausted {max_attempts} attempts; failures retained in {failure_path}')


def generate_stage(stage, out_dir, seed=20260922, workers=1, max_attempts=20, convergence=None):
    out_dir = Path(out_dir)
    convergence = convergence or ensure_convergence(out_dir, seed, workers)
    if not convergence['passed']:
        raise RuntimeError('formal generation requires passed convergence')
    specs = sample_specs(stage, seed)
    cfg = copy.deepcopy(convergence['selected_config'])
    if stage == 'ood_long':
        cfg['output']['t_end'] = 10.
    contract = dict(version=VERSION, stage=stage, master_seed=seed, config=cfg,
                    source=source_provenance(), convergence_signature=convergence['signature'],
                    selected_config_hash=convergence['selected_config_hash'], specs=specs, max_attempts=max_attempts)
    signature = _hash(contract)
    path, manifest_path = out_dir / (stage + '.npz'), out_dir / (stage + '.manifest.json')
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing['signature'] != signature or not path.exists() or existing['sha256'] != file_hash(path):
            raise ValueError('incompatible or corrupt dataset; use a new output directory')
        return path
    checkpoint = out_dir / '.shards' / stage
    lock_path = checkpoint / 'contract.json'
    if lock_path.exists() and _read_json(lock_path)['signature'] != signature:
        raise ValueError('incompatible shard contract; use a new output directory')
    _write_json(lock_path, dict(signature=signature, contract=contract))
    started = time.perf_counter()
    metadata = _map(_sample_job, [(s, cfg, str(checkpoint), signature, max_attempts) for s in specs], workers)
    times = Config.from_dict(cfg).output_times()
    y = np.empty((len(specs), len(times), 3), dtype=np.float32)
    u = np.empty((len(specs), len(times), 2), dtype=np.float32)
    seen = set()
    for row, spec in enumerate(specs):
        shard = checkpoint / (spec['id'] + '.npz')
        with np.load(shard, allow_pickle=False) as data:
            y[row], u[row] = data['Y'], data['U']
        content_hash = hashlib.sha256(y[row].tobytes() + u[row].tobytes()).hexdigest()
        if content_hash in seen:
            raise RuntimeError('duplicate trajectory content detected')
        seen.add(content_hash)
        metadata[row]['content_sha256'] = content_hash
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, Y=y, U=u, t=times.astype(np.float64), dt=np.float64(.001),
                            split=np.asarray([s['split'] for s in specs], dtype='U5'),
                            Y_names=np.asarray(Y_NAMES), U_names=np.asarray(U_NAMES))
    os.replace(temp, path)
    manifest = dict(signature=signature, contract=contract, config=cfg, source=contract['source'],
                    samples=metadata, quality=aggregate_quality([m['quality'] for m in metadata]),
                    sha256=file_hash(path), runtime_seconds=time.perf_counter() - started,
                    sample_runtime_seconds=sum(m['runtime_seconds'] for m in metadata),
                    shape=dict(Y=list(y.shape), U=list(u.shape)), convergence_signature=convergence['signature'])
    _write_json(manifest_path, manifest)
    print(f'wrote {path} sha256={manifest["sha256"]}', flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'convergence', 'id', 'ood', 'all'), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=20)
    parser.add_argument('--input-grid-diagnostics', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_attempts < 1:
        parser.error('workers and max-attempts must be positive')
    convergence = ensure_convergence(args.out_dir, args.seed, args.workers, args.input_grid_diagnostics)
    stages = {'convergence': [], 'pilot': ['pilot'], 'id': ['id'],
              'ood': ['ood_amplitude', 'ood_frequency', 'ood_long'],
              'all': ['pilot', 'id', 'ood_amplitude', 'ood_frequency', 'ood_long']}[args.stage]
    for stage in stages:
        generate_stage(stage, args.out_dir, args.seed, args.workers, args.max_attempts, convergence)


if __name__ == '__main__':
    main()
