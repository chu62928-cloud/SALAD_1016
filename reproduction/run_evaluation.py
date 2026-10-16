"""Run the author's metrics with immutable bindings and resumable repetitions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

from runtime import ROOT, load_runtime
from audit_assets import sha256
import numpy as np
import torch
from utils.eval_t2m import test_denoiser, test_vae
from audit_metrics import MetricAudit
import importlib.metadata as metadata


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def summarize(rows, targets):
    result = {}
    for key in rows[0]:
        if key in ['repeat', 'elapsed_seconds'] or rows[0][key] is None:
            continue
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        mean = float(values.mean())
        ci = float(values.std(ddof=0) * 1.96 / np.sqrt(len(values)))
        result[key] = {'mean': mean, 'ci95_half_width': ci}
        if key in targets:
            target, target_ci = targets[key]
            result[key].update(paper_mean=target, paper_ci95_half_width=target_ci,
                               difference=mean-target,
                               confidence_intervals_overlap=abs(mean-target) <= ci+target_ci)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['t2m', 'kit'], required=True)
    parser.add_argument('--kind', choices=['denoiser', 'vae'], default='denoiser')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--cal-mm', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    os.chdir(ROOT)
    torch.set_num_threads(2)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise RuntimeError('Commit tracked source changes before an official run: ' + dirty)
    audit_path = ROOT / 'reproduction/artifacts' / ('assets_' + args.dataset + '.json')
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed':
        raise RuntimeError('Dataset and assets audit must pass first')
    for relative, info in audit['asset_files'].items():
        if sha256(ROOT / relative) != info['sha256']:
            raise ValueError('Asset changed after audit: ' + relative)
    for name, info in audit['clip']['files'].items():
        if sha256(info['path']) != info['sha256']:
            raise ValueError('CLIP asset changed after audit: ' + name)
    protocol_path = ROOT / 'reproduction/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    packages = ['torch', 'numpy', 'scipy', 'diffusers', 'transformers', 'huggingface-hub']
    binding = {'commit': commit, 'protocol_sha256': sha256(protocol_path),
               'asset_audit_sha256': sha256(audit_path),
               'dataset': args.dataset, 'kind': args.kind, 'seed': args.seed,
               'repeats': args.repeats, 'cal_mm': args.cal_mm,
               'steps': protocol['inference_steps'], 'scale': protocol['guidance_scale'],
               'python': platform.python_version(),
               'packages': {p: metadata.version(p) for p in packages},
               'gpu': torch.cuda.get_device_name(0)}
    binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    state_path = args.output / 'resume.pt'
    rows = []
    state = None
    if state_path.is_file():
        state = torch.load(state_path, map_location='cpu')
        if state['binding_sha256'] != binding_hash:
            raise ValueError('Resume binding differs; use a new output directory')
        rows = state['rows']
    binding_path = args.output / 'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Output directory belongs to a different experiment')
    atomic_json(binding_path, binding)
    if len(rows) == args.repeats:
        print('Already complete:', args.output)
        return
    opt, vae_opt, trainer, wrapper, loader = load_runtime(
        args.dataset, args.seed, binding['steps'], binding['scale'])
    trainer.denoiser.eval()
    trainer.vae.eval()
    if state is not None:
        restore_rng(state['rng'])
    dataset_size = len(loader.dataset)
    coverage = {'loaded_entries': dataset_size, 'evaluated_per_repeat': len(loader)*32,
                'dropped_per_repeat': dataset_size % 32,
                'batch_size': 32, 'shuffle': True, 'drop_last': True}
    target_key = args.dataset if args.kind == 'denoiser' else args.dataset + '_vae'
    targets = protocol['paper_targets'].get(target_key, {})
    progress = {'status': 'running', 'pid': os.getpid(), 'binding_sha256': binding_hash,
                'coverage': coverage, 'requested_repeats': args.repeats,
                'completed_repeats': len(rows), 'table1_evaluation_complete': False}
    atomic_json(args.output / 'summary.json', progress)
    for repeat in range(len(rows), args.repeats):
        start = time.monotonic()
        audit_context = MetricAudit() if repeat == 0 else nullcontext()
        with audit_context as metric_audit:
            if args.kind == 'denoiser':
                output = test_denoiser(loader, trainer.generate, repeat, wrapper, opt.joints_num,
                                       cal_mm=args.cal_mm)
                _, fid, div, precision, matching, _, mm, _, _ = output
                mpjpe = None
            else:
                fid, div, precision, matching, mpjpe, mm = test_vae(
                    loader, trainer.vae, repeat, wrapper, opt.joints_num, cal_mm=args.cal_mm)
                mpjpe = float(mpjpe)
        if metric_audit is not None:
            atomic_json(args.output / 'metric_audit.json', {'status': 'passed',
                        'repeat': repeat, 'checks': metric_audit.results})
        row = {'repeat': repeat, 'fid': float(fid), 'diversity': float(div),
               'top1': float(precision[0]), 'top2': float(precision[1]),
               'top3': float(precision[2]), 'matching': float(matching),
               'mpjpe': mpjpe, 'multimodality': float(mm) if args.cal_mm else None,
               'elapsed_seconds': time.monotonic()-start}
        if any(not np.isfinite(v) for v in row.values() if v is not None):
            raise ValueError('Nonfinite metric: ' + str(row))
        rows.append(row)
        temporary = state_path.with_suffix('.pt.tmp')
        torch.save({'binding_sha256': binding_hash, 'rows': rows, 'rng': capture_rng()}, temporary)
        os.replace(temporary, state_path)
        atomic_json(args.output / 'repetitions.json', rows)
        progress.update(completed_repeats=len(rows), metrics=summarize(rows, targets),
                        updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / 'summary.json', progress)
        print(json.dumps(row), flush=True)
    progress.update(status='complete',
                    table1_evaluation_complete=args.repeats == 20 and args.kind == 'denoiser' and args.cal_mm,
                    comparison_note='Interval overlap is descriptive, not proof of equality or validation of training/editing.')
    atomic_json(args.output / 'summary.json', progress)


if __name__ == '__main__':
    main()
