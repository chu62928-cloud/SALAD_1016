"""Audit generation replay and author BVH conversion without changing frozen results."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '2'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest import mock
import numpy as np
import torch

from diagnose_jitter import measure, temporal, high_frequency, PARENTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def kinematics(xyz):
    values = np.asarray(xyz, dtype=np.float64)
    lengths = np.linalg.norm(values[:, 1:]-values[:, PARENTS[1:]], axis=-1)
    result = temporal(values-values[:, :1])
    result['bone_cv_mean'] = float((lengths.std(axis=0)/lengths.mean(axis=0)).mean())
    result['bone_above3hz_cv_mean'] = float((np.sqrt(np.mean(high_frequency(lengths)**2, axis=0))/lengths.mean(axis=0)).mean())
    return result


def replay(args):
    from runtime import load_runtime
    from t2m import Text2Motion
    from utils.fixseed import fixseed
    from utils.motion_process import recover_from_rot, recover_from_ric
    from common.skeleton import Skeleton
    from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    original = read(args.source/'binding.json')
    for name, digest in original['assets'].items():
        if sha(args.root/name) != digest:
            raise ValueError('Generation asset changed: '+name)
    opt, vae_opt, trainer, _, loader = load_runtime('t2m', seed=20261016, steps=50, scale=7.5, split='val')
    trainer.vae.eval()
    examples = read(args.source/'examples.json')
    result = dict(status='running', assets_unchanged=True, batches=[], notebook_comparison=[], rotation_branch=[])
    for seed in [20261016, 20261017, 20261018]:
        selected = [r for r in examples if r['seed'] == seed]
        fixseed(seed)
        captured = []
        random_like = torch.randn_like
        def noise(*a, **kw):
            value = random_like(*a, **kw)
            if value.ndim == 4 and value.shape[0] == 6:
                captured.append(value.detach().clone())
            return value
        with mock.patch('torch.randn_like', side_effect=noise):
            prediction, _ = trainer.generate(([r['text'] for r in selected], torch.zeros(6, 120, 263), torch.full((6,), 120, dtype=torch.long)))
        pred = prediction.cpu().numpy()
        max_error = 0.
        for i, r in enumerate(selected):
            with np.load(args.source/'arrays'/r['array']) as cache:
                max_error = max(max_error, float(np.max(np.abs(pred[i]-cache['normalized']))))
        if max_error != 0:
            raise ValueError('Exact batch replay differs: '+str(max_error))
        result['batches'].append(dict(seed=seed, clips=6, normalized_max_error=max_error))
        if seed == 20261016:
            generator = Text2Motion.__new__(Text2Motion)
            generator.device, generator.vae, generator.denoiser = opt.device, trainer.vae, trainer.denoiser
            generator.vae_opt, generator.scheduler = vae_opt, copy.deepcopy(trainer.noise_scheduler)
            initial = captured[-1]*trainer.noise_scheduler.init_noise_sigma
            for index in [1, 5]:
                _, single, _ = generator.generate(selected[index]['text'], 120, 7.5, 50, init_noise=initial[index:index+1].clone())
                raw_a = prediction[index].cpu().numpy()*loader.dataset.std+loader.dataset.mean
                raw_b = single[0].cpu().numpy()*loader.dataset.std+loader.dataset.mean
                a = recover_from_ric(torch.from_numpy(raw_a).float(), 22).numpy()
                b = recover_from_ric(torch.from_numpy(raw_b).float(), 22).numpy()
                result['notebook_comparison'].append(dict(id=selected[index]['id'],
                    normalized_max_error=float((single[0]-prediction[index]).abs().max()),
                    joint_max_error_m=float(np.abs(a-b).max()), joint_mean_error_m=float(np.linalg.norm(a-b, axis=-1).mean())))
        save(args.output/'generation_audit.json', result)
        print('REPLAY', seed, max_error, flush=True)
    raw_reference = np.load(args.root/'dataset/humanml3d/new_joint_vecs/000021.npy')
    reference_xyz = recover_from_ric(torch.from_numpy(raw_reference).float(), 22)
    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    skeleton.get_offsets_joints(reference_xyz[0])
    reference_rot = recover_from_rot(torch.from_numpy(raw_reference).float(), 22, skeleton)
    result['rotation_ground_truth_consistency_m'] = float(torch.linalg.vector_norm(reference_xyz-reference_rot, dim=-1).mean())
    for r in examples:
        with np.load(args.source/'arrays'/r['array']) as cache:
            raw = cache['normalized']*loader.dataset.std+loader.dataset.mean
            xyz = cache['xyz']
        rotated = recover_from_rot(torch.from_numpy(raw).float(), 22, skeleton).numpy()
        result['rotation_branch'].append(dict(id=r['id'], seed=r['seed'],
            position_branch=kinematics(xyz), rotation_branch=kinematics(rotated),
            mean_joint_difference_m=float(np.linalg.norm(rotated-xyz, axis=-1).mean())))
    result['status'] = 'complete'
    save(args.output/'generation_audit.json', result)


def convert(args):
    from visualization.joints2bvh import Joint2BVHConvertor
    import visualization.BVH_mod as BVH
    from mesh_surface import JointDrivenBody
    examples = read(args.source/'examples.json')
    template = BVH.load(str(args.output/'assets/template.bvh'), need_quater=True)
    with mock.patch('visualization.joints2bvh.BVH.load', return_value=template):
        converter = Joint2BVHConvertor()
    if template.shape[1] != 22 or list(template.parents) != converter.parents:
        raise ValueError('BVH template topology differs from the author converter')
    body_path = Path('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz')
    body = JointDrivenBody(body_path)
    names = ['stand_20261016', 'walk_20261016', 'run_20261016', 'jump_20261016', 'jump_20261017']
    paired = np.load(args.source/'arrays/validation_xyz.npz')
    references = read(args.source/'reference.json')
    real = next(r for r in references if 'walk' in r['text'].lower() and r['frames'] >= 120)
    cases = []
    for name in names:
        r = next(r for r in examples if '%s_%d' % (r['id'], r['seed']) == name)
        with np.load(args.source/'arrays'/r['array']) as cache:
            xyz = cache['xyz'].copy()
        cases.append((name, r['label'], r['text'], xyz, r['sha256']))
    cases.append(('real_walk_control', '真实动作对照', real['text'], paired['real_'+real['id']][:120].copy(), sha(args.source/'arrays/validation_xyz.npz')))
    records = []
    server = args.output/'server_only'
    server.mkdir(exist_ok=True)
    for name, label, text, xyz, digest in cases:
        original = xyz.copy()
        record = dict(name=name, label=label, text=text, source_sha256=digest, stages={})
        for stage in ['raw', 'fixed_bones', 'author_foot_ik']:
            path = server/(name+'_'+stage+'.npz')
            if path.exists():
                with np.load(path) as cache:
                    joints, vertices = cache['xyz'].copy(), cache['vertices'].copy()
            else:
                if stage == 'raw':
                    joints = xyz.copy()
                else:
                    _, joints = converter.convert(xyz.copy(), None, iterations=10, foot_ik=stage == 'author_foot_ik')
                    # A fixed skeleton can only remove bone-length oscillation, not all motion noise.
                    if kinematics(joints)['bone_above3hz_cv_mean'] > 1e-8:
                        raise ValueError('Author IK did not preserve constant bone lengths')
                vertices, _ = body.motion(joints)
                np.savez_compressed(path, xyz=joints, vertices=vertices)
            record['stages'][stage] = dict(joints=kinematics(joints), mesh=temporal(vertices-joints[:, :1]),
                source_joint_error_mean_m=float(np.linalg.norm(joints-original, axis=-1).mean()), sha256=sha(path))
            print('CONVERTED', name, stage, flush=True)
        if not np.array_equal(xyz, original):
            raise ValueError('Author conversion changed source array')
        records.append(record)
        save(args.output/'conversion_audit.json', dict(status='running', cases=records))
    calibration = [kinematics(paired['real_'+r['id']]) for r in references]
    result = dict(status='complete', cases=records,
        template_source='https://github.com/EricGuo5513/momask-codes/blob/94a6636c9c463b7a9414c3401a6f1b67e6c51824/visualization/data/template.bvh',
        template_sha256=sha(args.output/'assets/template.bvh'),
        conversion_source_sha256=sha(args.root/'visualization/joints2bvh.py'),
        real_reference_count=len(calibration), real_reference_mean={key:float(np.mean([r[key] for r in calibration])) for key in calibration[0]},
        surface_model_sha256=sha(body_path), source_files_unchanged=all(sha(args.source/'arrays'/r['array']) == r['sha256'] for r in examples))
    save(args.output/'conversion_audit.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['replay', 'convert'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.root)
    sys.path.insert(0, str(args.root))
    sys.path.insert(0, str(args.root/'reproduction'))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    {'replay': replay, 'convert': convert}[args.stage](args)
