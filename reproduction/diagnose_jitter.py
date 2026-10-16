"""Deterministic temporal diagnostics for the delivered raw SALAD motion and mesh."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '2'

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.fft import dct, idct

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def high_frequency(values, fps=20, cutoff=3):
    values = np.asarray(values, dtype=np.float64)
    spectrum = dct(values, axis=0, norm='ortho')
    spectrum[np.arange(len(values))*fps/(2*len(values)) < cutoff] = 0
    return idct(spectrum, axis=0, norm='ortho')


def temporal(points):
    values = np.asarray(points, dtype=np.float64)
    return dict(acceleration_rms_mps2=float(np.sqrt(np.mean(np.diff(values, n=2, axis=0)**2))*20**2),
                jerk_rms_mps3=float(np.sqrt(np.mean(np.diff(values, n=3, axis=0)**2))*20**3),
                above3hz_rms_m=float(np.sqrt(np.mean(high_frequency(values)**2))))


def measure(source, name):
    path = source/'arrays'/(name+'.npz')
    with np.load(path) as cache:
        xyz = cache['xyz'].copy()
    root = xyz[:, :1].astype(np.float64)
    length = np.linalg.norm(xyz[:, 1:] - xyz[:, PARENTS[1:]], axis=-1)
    hf = high_frequency(length)
    fraction = np.sqrt(np.mean(hf**2, axis=0))/length.mean(axis=0)
    result = dict(name=name, source_sha256=sha(path), joints=temporal(xyz-root),
        bone_above3hz_cv_mean=float(fraction.mean()), bone_above3hz_cv_max=float(fraction.max()),
        bone_frame_step_relative_rms=float(np.sqrt(np.mean((np.diff(length, axis=0)/length.mean(axis=0))**2))))
    mesh = source/'mesh_v1/server_only'/(name+'.npz')
    if mesh.exists():
        with np.load(mesh) as cache:
            vertices = cache['vertices'].astype(np.float64)
        result['mesh'] = temporal(vertices-root)
        result['mesh_sha256'] = sha(mesh)
        result['mesh_to_joint_jerk_ratio'] = result['mesh']['jerk_rms_mps3']/result['joints']['jerk_rms_mps3']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--clip', default='walk_20261016')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--assert-bone-jitter', action='store_true')
    args = parser.parse_args()
    names = [p.stem for p in sorted((args.source/'arrays').glob('*.npz')) if p.stem != 'validation_xyz'] if args.all else [args.clip]
    result = dict(fps=20, high_frequency_cutoff_hz=3,
        diagnostic_limit=.005, limit_is_paper_metric_or_global_quality_gate=False,
        description='Rapid bone-length oscillation: mean per-bone RMS above 3Hz / mean length; diagnostic tolerance 0.5%',
        clips=[measure(args.source, name) for name in names])
    print(json.dumps(result, indent=2))
    if args.assert_bone_jitter:
        worst = max(v['bone_above3hz_cv_mean'] for v in result['clips'])
        if worst > result['diagnostic_limit']:
            raise AssertionError('Rapid bone-length oscillation reproduced: %.3f%% exceeds diagnostic 0.5%%' % (worst*100))
