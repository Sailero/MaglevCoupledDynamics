"""Extend the locked benchmark with independent trajectories, without changing its core.

Batch zero is the original ID dataset. Later batches use independent master seeds.
Run from any directory; default paths are relative to this repository.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from maglev_sim import benchmark as b
from maglev_sim.due import NPZ_KEYS
from maglev_sim.plant.config import Config

OOD_NAMES = ('ood_amplitude', 'ood_frequency', 'ood_long')


def lock_contract(out_dir, contract):
    """A changed seed, source, script or reference artifact requires a new directory."""
    path = Path(out_dir) / 'large_generation_contract.json'
    signature = b._hash(contract)
    if path.exists():
        old = b._read_json(path)
        if old.get('signature') != signature or b._hash(old.get('contract')) != signature:
            raise ValueError('incompatible extension resume contract; use a new output directory')
    else:
        b._write_json(path, dict(signature=signature, contract=contract))
    return signature


def _manifest(path):
    path = Path(path)
    meta_path = path.with_suffix('.manifest.json')
    meta = b._read_json(meta_path)
    if b._hash(meta['contract']) != meta['signature']:
        raise ValueError(f'manifest contract hash mismatch: {meta_path}')
    if b.file_hash(path) != meta['sha256']:
        raise ValueError(f'dataset hash mismatch: {path}')
    return meta, dict(path=str(path.resolve()), sha256=meta['sha256'],
                      manifest_path=str(meta_path.resolve()), manifest_sha256=b.file_hash(meta_path))


def _check_stratification(samples, count):
    counts = Counter(s['split'] for s in samples)
    expected = dict(train=round(.7 * count), val=round(.15 * count), test=round(.15 * count))
    if dict(counts) != expected or len(samples) != count:
        raise ValueError(f'invalid trajectory split counts: {dict(counts)} != {expected}')
    for split, n in expected.items():
        actual = Counter(s['category'] for s in samples if s['split'] == split)
        target = dict(zip(b.CATEGORIES, [round(n * f) for f in (.1, .2, .2, .5)]))
        if dict(actual) != target:
            raise ValueError(f'invalid category stratification in {split}: {dict(actual)}')
    return expected


def _quality_valid(stats):
    keys = ('gap_min', 'gap_max', 'current_min', 'current_max', 'saturation_fraction', 'retries')
    if (not stats.get('finite') or not stats.get('completed') or stats.get('termination') != 'completed'
            or not all(np.isfinite(stats[k]) for k in keys)
            or not 0 <= stats['saturation_fraction'] <= 1
            or stats['gap_min'] > stats['gap_max'] or stats['current_min'] > stats['current_max']
            or stats['retries'] < 0):
        raise ValueError('invalid or incomplete trajectory quality')


def _input_record(path, seed, contract):
    meta, record = _manifest(path)
    if (meta['source'] != contract['source'] or b._hash(meta['config']) != b._hash(contract['config'])
            or meta['convergence_signature'] != contract['convergence_signature']
            or meta['contract']['master_seed'] != seed or meta['contract']['stage'] != 'id'):
        raise ValueError(f'incompatible input batch: {path}')
    _check_stratification(meta['samples'], contract['batch_size'])
    ids = [s['id'] for s in meta['samples']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate source trajectory ID')
    if any(s['master_seed'] != seed for s in meta['samples']):
        raise ValueError('sample master seed does not match its batch')
    record['master_seed'] = seed
    return meta, record


def merge_batches(batch_paths, out_dir, contract, progress=None):
    """Validate, prefix identities and merge through resumable on-disk arrays."""
    out_dir = Path(out_dir)
    lock_contract(out_dir, contract)
    seeds = contract['seeds']
    if len(batch_paths) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('batch count and unique seed list must agree')
    metadata, sources = [], []
    for path, seed in zip(batch_paths, seeds):
        meta, record = _input_record(path, seed, contract)
        metadata.append(meta)
        sources.append(record)
    merge_contract = dict(contract, merge_inputs=sources, dataset='merged_id_trajectories')
    signature = b._hash(merge_contract)
    output = out_dir / 'id.npz'
    manifest_path = output.with_suffix('.manifest.json')
    if manifest_path.exists():
        existing = b._read_json(manifest_path)
        if (existing['signature'] != signature or b._hash(existing['contract']) != signature
                or not output.exists() or existing['sha256'] != b.file_hash(output)):
            raise ValueError('incompatible or corrupt merged dataset on resume')
        return output
    started = time.perf_counter()
    count = len(seeds) * contract['batch_size']
    cfg = Config.from_dict(contract['config'])
    t = np.asarray(cfg.output_times(), dtype=np.float64)
    scratch = out_dir / '.merge'
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint = scratch / 'progress.json'
    shape_y, shape_u = (count, len(t), 3), (count, len(t), 2)
    completed = 0
    if checkpoint.exists():
        saved = b._read_json(checkpoint)
        if saved['signature'] != signature:
            raise ValueError('incompatible merge checkpoint')
        completed = saved['completed_batches']
        if not 0 <= completed <= len(seeds):
            raise ValueError('invalid merge checkpoint progress')
        y = np.lib.format.open_memmap(scratch / 'Y.npy', mode='r+')
        u = np.lib.format.open_memmap(scratch / 'U.npy', mode='r+')
        if y.shape != shape_y or u.shape != shape_u or y.dtype != np.float32 or u.dtype != np.float32:
            raise ValueError('corrupt merge memmap shape/dtype')
    else:
        y = np.lib.format.open_memmap(scratch / 'Y.npy', mode='w+', dtype=np.float32, shape=shape_y)
        u = np.lib.format.open_memmap(scratch / 'U.npy', mode='w+', dtype=np.float32, shape=shape_u)
        y.flush()
        u.flush()
        b._write_json(checkpoint, dict(signature=signature, completed_batches=0))
    samples, seen_ids, seen_content = [], set(), set()
    try:
        for batch_index, (path, seed, meta, source) in enumerate(zip(batch_paths, seeds, metadata, sources)):
            offset = batch_index * contract['batch_size']
            end = offset + contract['batch_size']
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != set(NPZ_KEYS):
                    raise ValueError('input dataset must have exactly seven NPZ fields')
                by, bu = data['Y'], data['U']
                if (by.shape != (contract['batch_size'], len(t), 3) or bu.shape != (contract['batch_size'], len(t), 2)
                        or by.dtype != np.float32 or bu.dtype != np.float32
                        or data['t'].dtype != np.float64 or not np.array_equal(data['t'], t)
                        or data['dt'].shape != () or float(data['dt']) != cfg.output.dt
                        or data['split'].dtype.kind != 'U' or data['split'].shape != (contract['batch_size'],)
                        or tuple(data['Y_names']) != b.Y_NAMES or tuple(data['U_names']) != b.U_NAMES
                        or not np.isfinite(by).all() or not np.isfinite(bu).all()):
                    raise ValueError('incompatible input shapes, time axis, dtypes or finite values')
                if list(data['split']) != [s['split'] for s in meta['samples']]:
                    raise ValueError('manifest and NPZ split order disagree')
                for row, original in enumerate(meta['samples']):
                    _quality_valid(original['quality'])
                    content_hash = hashlib.sha256(by[row].tobytes() + bu[row].tobytes()).hexdigest()
                    if content_hash != original['content_sha256']:
                        raise ValueError('trajectory content hash mismatch')
                    if content_hash in seen_content:
                        raise ValueError('duplicate trajectory content across merged batches')
                    seen_content.add(content_hash)
                    sample = copy.deepcopy(original)
                    sample['source_id'] = original['id']
                    sample['source_sample_id'] = original.get('sample_id', original['id'])
                    sample['source_dataset'] = source['path']
                    sample['source_dataset_sha256'] = source['sha256']
                    sample['source_row'] = row
                    sample['batch_index'] = batch_index
                    sample['id'] = sample['sample_id'] = f"seed{seed}:{original['id']}"
                    if sample['id'] in seen_ids:
                        raise ValueError('duplicate global trajectory ID')
                    seen_ids.add(sample['id'])
                    samples.append(sample)
                if batch_index < completed:
                    if not np.array_equal(y[offset:end], by) or not np.array_equal(u[offset:end], bu):
                        raise ValueError('corrupt completed merge checkpoint data')
                else:
                    y[offset:end], u[offset:end] = by, bu
                    y.flush()
                    u.flush()
                    b._write_json(checkpoint, dict(signature=signature, completed_batches=batch_index + 1))
            if progress:
                progress(stage='merging', merge_completed_batches=batch_index + 1, merge_rows=end)
        split_counts = _check_stratification(samples, count)
        if progress:
            progress(stage='compressing', merge_rows=count)
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, Y=y, U=u, t=t, dt=np.float64(cfg.output.dt),
                                split=np.asarray([s['split'] for s in samples], dtype='U5'),
                                Y_names=np.asarray(b.Y_NAMES), U_names=np.asarray(b.U_NAMES))
        os.replace(temporary, output)
        manifest = dict(signature=signature, contract=merge_contract, config=contract['config'],
                        source=contract['source'], samples=samples, source_datasets=sources,
                        convergence_signature=contract['convergence_signature'],
                        quality=b.aggregate_quality([s['quality'] for s in samples]),
                        split_counts=split_counts, sha256=b.file_hash(output),
                        shape=dict(Y=list(shape_y), U=list(shape_u)),
                        runtime_seconds=time.perf_counter() - started,
                        sample_runtime_seconds=sum(s.get('runtime_seconds', 0.) for s in samples),
                        unique_trajectory_count=len(seen_content), duplicate_trajectory_count=0)
        b._write_json(manifest_path, manifest)
        return output
    finally:
        # Explicitly close Windows mappings so interruption/resume can reopen them.
        y.flush()
        u.flush()
        y._mmap.close()
        u._mmap.close()


def _copy_locked(source, destination, expected_hash):
    source, destination = Path(source), Path(destination)
    if b.file_hash(source) != expected_hash:
        raise ValueError(f'locked source changed: {source}')
    if destination.exists():
        if b.file_hash(destination) != expected_hash:
            raise ValueError(f'incompatible copied reference: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, temp)
    if b.file_hash(temp) != expected_hash:
        raise ValueError(f'copy hash verification failed: {source}')
    os.replace(temp, destination)


class Progress:
    def __init__(self, out_dir, signature, batches, seeds):
        self.path = Path(out_dir) / 'large_generation_status.json'
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.data = dict(signature=signature, status='running', stage='validating', total_batches=batches,
                         seeds=seeds, total_trajectories=batches * 1000, completed_batches=0,
                         generated_trajectories=0, reused_trajectories=1000,
                         ready_trajectories=1000, started_at=datetime.now(timezone.utc).isoformat())
        self()

    def __call__(self, **changes):
        with self.lock:
            self.data.update(changes)
            self.data['updated_at'] = datetime.now(timezone.utc).isoformat()
            self.data['elapsed_seconds'] = time.perf_counter() - self.started
            b._write_json(self.path, self.data)

    def watch(self, stop, directory, completed_batches):
        while not stop.wait(10):
            shard_dir = Path(directory) / '.shards' / 'id'
            accepted = sum(not p.name.endswith('.failures.json') for p in shard_dir.glob('id-*.json'))
            self(current_batch_accepted=accepted,
                 ready_trajectories=completed_batches * 1000 + accepted,
                 generated_trajectories=(completed_batches - 1) * 1000 + accepted)


def build_contract(base_dir, out_dir, batches, base_seed, max_attempts):
    base_dir, out_dir = Path(base_dir).resolve(), Path(out_dir).resolve()
    if base_dir == out_dir or base_dir in out_dir.parents or out_dir in base_dir.parents:
        raise ValueError('base and extension output must be separate, non-nested directories')
    convergence_path = base_dir / 'convergence.json'
    convergence = b._read_json(convergence_path)
    source = b.source_provenance()
    if (not convergence['passed'] or not all(case['passed'] for case in convergence['cases'])
            or convergence['signature'] != b._hash(convergence['contract'])
            or convergence['contract']['source'] != source
            or convergence['selected_config_hash'] != b._hash(convergence['selected_config'])):
        raise ValueError('convergence must be passed, intact and match the frozen current source')
    base_meta, base_record = _manifest(base_dir / 'id.npz')
    seeds = [base_meta['contract']['master_seed']] + [base_seed + i for i in range(1, batches)]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError('extension needs distinct nonnegative master seeds')
    references = []
    for name in OOD_NAMES:
        meta, record = _manifest(base_dir / (name + '.npz'))
        if (meta['source'] != source or meta['convergence_signature'] != convergence['signature']
                or any(s['split'] != 'test' for s in meta['samples'])):
            raise ValueError(f'incompatible locked OOD reference: {name}')
        record['name'] = name
        references.append(record)
    contract = dict(version=1, kind='independent_trajectory_extension', base_dir=str(base_dir), out_dir=str(out_dir),
                    source=source, script_hash=b.file_hash(__file__), convergence_signature=convergence['signature'],
                    convergence_sha256=b.file_hash(convergence_path), selected_config_hash=convergence['selected_config_hash'],
                    config=convergence['selected_config'], base_dataset=base_record, locked_ood=references,
                    seeds=seeds, batch_size=1000, max_attempts=max_attempts)
    _input_record(base_dir / 'id.npz', seeds[0], contract)
    return contract, convergence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=ROOT / 'data' / 'due_v2_benchmark')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data' / 'due_large_10000')
    parser.add_argument('--batches', type=int, default=10)
    parser.add_argument('--base-seed', type=int, default=2026092300)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-attempts', type=int, default=20)
    args = parser.parse_args(argv)
    if args.batches < 1 or args.workers < 1 or args.max_attempts < 1 or args.base_seed < 0:
        parser.error('batches/workers/max-attempts must be positive, base-seed nonnegative')
    contract, convergence = build_contract(args.base_dir, args.out_dir, args.batches, args.base_seed, args.max_attempts)
    out_dir, base_dir = Path(contract['out_dir']), Path(contract['base_dir'])
    signature = lock_contract(out_dir, contract)
    progress = Progress(out_dir, signature, args.batches, contract['seeds'])
    try:
        _copy_locked(base_dir / 'convergence.json', out_dir / 'convergence.json', contract['convergence_sha256'])
        for record in contract['locked_ood']:
            _copy_locked(record['path'], out_dir / (record['name'] + '.npz'), record['sha256'])
            _copy_locked(record['manifest_path'], out_dir / (record['name'] + '.manifest.json'), record['manifest_sha256'])
        paths = [base_dir / 'id.npz']
        progress(stage='generating', completed_batches=1)
        for index, seed in enumerate(contract['seeds'][1:], start=1):
            directory = out_dir / '.batches' / f'seed{seed}'
            progress(stage='generating', current_batch=index, current_seed=seed,
                     current_batch_accepted=0, completed_batches=index, ready_trajectories=index * 1000)
            stop = threading.Event()
            watcher = threading.Thread(target=progress.watch, args=(stop, directory, index), daemon=True)
            watcher.start()
            try:
                path = b.generate_stage('id', directory, seed=seed, workers=args.workers,
                                        max_attempts=args.max_attempts, convergence=convergence)
                _input_record(path, seed, contract)
                paths.append(path)
            finally:
                stop.set()
                watcher.join()
            progress(completed_batches=index + 1, ready_trajectories=(index + 1) * 1000,
                     generated_trajectories=index * 1000, current_batch_accepted=1000)
        output = merge_batches(paths, out_dir, contract, progress)
        progress(status='complete', stage='complete', output=str(output), completed_batches=args.batches,
                 ready_trajectories=args.batches * 1000, generated_trajectories=(args.batches - 1) * 1000,
                 output_sha256=b.file_hash(output))
        print(f'Completed {args.batches * 1000} unique trajectories: {output}', flush=True)
    except BaseException as exc:
        progress(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
