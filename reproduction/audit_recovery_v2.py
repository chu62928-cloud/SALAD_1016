"""Separate HumanML position recovery, rotation recovery, and SMPL skinning."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '2'

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class FixedSMPLSurface:
    """Fit outgoing bone orientations, then use standard fixed-rest SMPL-H LBS.

    This is geometric swing estimation, not recovery of the missing true twists.
    HumanML incoming-bone rotations must not be assigned as SMPL joint rotations.
    """
    def __init__(self, path):
        from mesh_surface import JointDrivenBody
        self.original = JointDrivenBody(path)
        self.rest = self.original.rest
        self.parents = self.original.parents
        self.faces = self.original.faces
        with np.load(path, allow_pickle=True) as data:
            self.arguments = [torch.as_tensor(np.array(data[k]), dtype=torch.float64) for k in
                              ['v_template', 'shapedirs', 'posedirs', 'J_regressor']]
        self.arguments[2] = self.arguments[2].reshape(-1, 459).T.contiguous()
        self.weights = torch.as_tensor(self.original.weights, dtype=torch.float64)
        self.parent_tensor = torch.as_tensor(self.parents, dtype=torch.long)

    def fit(self, xyz):
        from mesh_surface import rotation_between
        xyz = np.asarray(xyz, dtype=np.float64)
        total = np.tile(np.eye(3), (len(xyz), 52, 1, 1))
        for t, points in enumerate(xyz):
            for j in range(52):
                p = self.parents[j]
                inherited = total[t, p] if p >= 0 else np.eye(3)
                children = self.original.children.get(j, [])
                if len(children) == 1:
                    child = children[0]
                    total[t, j] = rotation_between(inherited @ (self.rest[child] - self.rest[j]),
                                                   points[child] - points[j]) @ inherited
                elif len(children) > 1:
                    a = self.rest[children] - self.rest[j]
                    b = points[children] - points[j]
                    a /= np.linalg.norm(a, axis=1, keepdims=True)
                    b /= np.linalg.norm(b, axis=1, keepdims=True)
                    u, _, vt = np.linalg.svd(a.T @ b)
                    correction = np.eye(3)
                    correction[2, 2] = np.linalg.det(vt.T @ u.T)
                    total[t, j] = vt.T @ correction @ u.T
                else:
                    total[t, j] = inherited
        local = total.copy()
        for j in range(1, 52):
            local[:, j] = total[:, self.parents[j]].transpose(0, 2, 1) @ total[:, j]
        return local

    def independent_lbs(self, local, root):
        total = local.copy()
        joints = np.empty((len(local), 52, 3))
        joints[:, 0] = root
        for j in range(1, 52):
            p = self.parents[j]
            joints[:, j] = joints[:, p] + np.einsum('tab,b->ta', total[:, p], self.rest[j] - self.rest[p])
            total[:, j] = total[:, p] @ local[:, j]
        feature = (local[:, 1:] - np.eye(3)).reshape(len(local), -1)
        posed = self.original.template + (feature @ self.original.posedirs.T).reshape(len(local), -1, 3)
        blend = np.einsum('vj,tjab->tvab', self.original.weights, total)
        shift = joints - np.einsum('tjab,jb->tja', total, self.rest)
        vertices = np.einsum('tvab,tvb->tva', blend, posed) + np.einsum('vj,tja->tva', self.original.weights, shift)
        return vertices, joints

    def skin(self, local, root):
        from smplx.lbs import lbs
        vertices, joints = [], []
        template, shapes, poses, regressor = self.arguments
        with torch.no_grad():
            for start in range(0, len(local), 12):
                rotations = torch.as_tensor(local[start:start+12], dtype=torch.float64)
                beta = torch.zeros((len(rotations), shapes.shape[-1]), dtype=torch.float64)
                v, j = lbs(beta, rotations, template, shapes, poses, regressor,
                           self.parent_tensor, self.weights, pose2rot=False)
                shift = torch.as_tensor(root[start:start+12] - self.rest[0], dtype=torch.float64)
                vertices.append((v + shift[:, None]).numpy())
                joints.append((j + shift[:, None]).numpy())
        return np.concatenate(vertices), np.concatenate(joints)

    def audit(self):
        local = np.tile(np.eye(3), (3, 52, 1, 1))
        local[1, 0] = Rotation.from_euler('y', 37, degrees=True).as_matrix()
        local[2, 18] = Rotation.from_euler('z', 65, degrees=True).as_matrix()
        root = np.tile(self.rest[0], (3, 1))
        root[1] += [1.25, .15, -2.5]
        a, aj = self.skin(local, root)
        b, bj = self.independent_lbs(local, root)
        error = float(np.abs(a-b).max())
        joint_error = float(np.abs(aj-bj).max())
        neutral = float(np.abs(a[0]-self.original.template).max())
        rigid = float(np.abs(a[1]-((self.original.template-self.rest[0]) @ local[1, 0].T + root[1])).max())
        if max(error, joint_error, neutral, rigid) > 1e-6:
            raise ValueError('Standard SMPL-H LBS disagrees with independent FK/LBS')
        return dict(lbs_max_error_m=error, fk_max_error_m=joint_error,
                    neutral_max_error_m=neutral, rigid_max_error_m=rigid)


def recover(raw, skeleton):
    from utils.motion_process import recover_from_ric, recover_from_rot, recover_root_rot_pos
    from common.quaternion import cont6d_to_matrix, quaternion_to_cont6d
    x = torch.as_tensor(raw, dtype=torch.float32)
    a = recover_from_ric(x.clone(), 22).numpy()
    b = recover_from_rot(x.clone(), 22, skeleton).numpy()
    q, root = recover_root_rot_pos(x)
    six = torch.cat([quaternion_to_cont6d(q), x[:, 67:193]], -1).reshape(-1, 22, 6)
    matrices = cont6d_to_matrix(six).numpy()
    angles = np.r_[0., np.cumsum(np.asarray(raw[:-1, 0], dtype=np.float64))]
    inverse = Rotation.from_euler('y', -2*angles).as_matrix()
    independent = np.einsum('tab,tjb->tja', inverse, np.asarray(raw[:, 4:67]).reshape(-1, 21, 3))
    independent[..., [0, 2]] += root.numpy()[:, None, [0, 2]]
    independent = np.concatenate([root.numpy()[:, None], independent], 1)
    expected = np.zeros_like(b)
    expected[:, 0] = root.numpy()
    offsets = skeleton.offset().numpy()
    for chain in skeleton.kinematic_tree():
        total = matrices[:, 0]
        for j, p in zip(chain[1:], chain[:-1]):
            total = total @ matrices[:, j]
            expected[:, j] = expected[:, p] + np.einsum('tab,b->ta', total, offsets[j])
    checks = dict(position_independent_max_m=float(np.abs(a-independent).max()),
                  rotation_independent_max_m=float(np.abs(b-expected).max()),
                  matrix_orthogonality_max=float(np.abs(matrices.transpose(0,1,3,2) @ matrices-np.eye(3)).max()),
                  matrix_det_min=float(np.linalg.det(matrices).min()))
    if checks['position_independent_max_m'] > 1e-5 or checks['rotation_independent_max_m'] > 2e-6:
        raise ValueError('Independent recovery disagrees with author functions')
    return a, b, matrices, checks


def metrics(xyz):
    from audit_jitter_pipeline import kinematics
    from diagnose_jitter import temporal
    return dict(**kinematics(xyz), root_height=temporal(np.asarray(xyz)[:, 0, 1]),
                horizontal_follow=temporal(np.asarray(xyz)-np.asarray(xyz)[:, :1]*[1,0,1]))


def source_audit(root):
    names = ['utils/motion_process.py', 'common/skeleton.py', 'common/quaternion.py',
             'visualization/joints2bvh.py', 'visualization/InverseKinematics.py',
             'visualization/remove_fs.py', 'utils/plot_script.py',
             'models/denoiser/trainer.py', 'playground.ipynb', 'visualize_attn.ipynb']
    records = {}
    for name in names:
        original = subprocess.check_output(['git','-C',str(root),'show','1166b90:'+name])
        current = (root/name).read_bytes()
        # Git on Windows can check out CRLF; compare Git's canonical source bytes.
        canonical = current.replace(b'\r\n', b'\n')
        records[name] = dict(sha256=sha(root/name), imported_sha256=hashlib.sha256(original).hexdigest(),
                             matches_imported_source=canonical == original.replace(b'\r\n', b'\n'))
        if name == 'models/denoiser/trainer.py':
            def exported_method(code):
                tree = ast.parse(code.decode())
                return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='test'))
            records[name]['export_method_matches_imported_source'] = exported_method(current)==exported_method(original)
            records[name]['known_other_changes'] = 'Previously recorded CFG-off text handling and strict optimizer/scheduler restore; cached generation is not rerun here'
    if not all(r.get('export_method_matches_imported_source',r['matches_imported_source']) for r in records.values()):
        raise ValueError('Official source differs from the imported SALAD source')
    return records


def run(args):
    from common.skeleton import Skeleton
    from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
    from utils.motion_process import recover_from_ric
    from diagnose_jitter import temporal
    from mesh_surface import JointDrivenBody
    from visualization.joints2bvh import Joint2BVHConvertor
    import visualization.BVH_mod as BVH

    output = args.output
    if (output/'audit.json').exists():
        raise ValueError('Use a new output; completed evidence is immutable')
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output/'server_only'
    cache_dir.mkdir(exist_ok=True)
    source = args.source
    records = source_audit(args.root)
    examples = json.loads((source/'examples.json').read_text())
    refs = json.loads((source/'reference.json').read_text())
    source_files = {str(source/'arrays'/r['array']):r['sha256'] for r in examples}
    source_files[str(source/'arrays/validation_xyz.npz')] = sha(source/'arrays/validation_xyz.npz')
    body_path = Path('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz')
    body = FixedSMPLSurface(body_path)
    body_audit = body.audit()
    reference_path = args.root/'dataset/humanml3d/new_joint_vecs/000021.npy'
    reference = recover_from_ric(torch.as_tensor(np.load(reference_path), dtype=torch.float32), 22)
    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    skeleton.get_offsets_joints(reference[0])
    mean_path = args.root/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy'
    std_path = args.root/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy'
    mean, std = np.load(mean_path), np.load(std_path)
    source_files.update({str(p):sha(p) for p in [reference_path, mean_path, std_path, body_path]})
    template_path = source/'jitter_v1/assets/template.bvh'
    template = BVH.load(str(template_path), need_quater=True)
    with mock.patch('visualization.joints2bvh.BVH.load', return_value=template):
        converter = Joint2BVHConvertor()
    source_files[str(template_path)] = sha(template_path)
    selected_refs = {next(r['id'] for r in refs if word in r['text'].lower() and r['frames'] >= 120):word
                     for word in ['walk', 'jump']}
    rows, cases = [], []
    chosen_generated = {'walk_20261016', 'jump_20261016'}

    def analyze(name, kind, raw, original_xyz, metadata):
        a, b, matrices, checks = recover(raw, skeleton)
        cache_error = float(np.abs(a-original_xyz).max())
        if cache_error > 2e-6:
            raise ValueError('Recovered positions differ from saved coordinates: '+name)
        six = raw[:, 67:193].reshape(-1, 21, 6)
        normx = np.linalg.norm(six[..., :3], axis=-1)
        normcross = np.linalg.norm(np.cross(six[..., :3], six[..., 3:]), axis=-1)
        row = dict(name=name, kind=kind, frames=len(raw),
                   position=metrics(a), rotation=metrics(b), checks=checks,
                   cached_position_max_error_m=cache_error,
                   position_rotation_mean_error_m=float(np.linalg.norm(a-b, axis=-1).mean()),
                   sixd_first_axis_min_norm=float(normx.min()), sixd_cross_min_norm=float(normcross.min()),
                   root_path_difference_m=float(np.abs(a[:, 0]-b[:, 0]).max()))
        rows.append(row)
        chosen = name in chosen_generated or (kind == 'real' and metadata['id'] in selected_refs)
        if not chosen:
            return
        a, b, matrices, raw = a[:120], b[:120], matrices[:120], raw[:120]
        name = name if kind == 'generated' else 'real_'+selected_refs[metadata['id']]+'_control'
        stages = {}
        data = dict(raw=raw, position_xyz=a, rotation_xyz=b)
        for branch, xyz in [('position', a), ('rotation', b)]:
            local = body.fit(xyz)
            vertices, joints = body.skin(local, xyz[:, 0])
            va, ja = body.independent_lbs(local[:2], xyz[:2, 0])
            independent_error = max(float(np.abs(vertices[:2]-va).max()), float(np.abs(joints[:2]-ja).max()))
            if independent_error > 1e-6:
                raise ValueError('Frame LBS differs from independent implementation')
            data[branch+'_local'] = local.astype(np.float32)
            data[branch+'_smpl_vertices'] = vertices.astype(np.float32)
            data[branch+'_smpl_xyz'] = joints[:, :22].astype(np.float32)
            delta = local[:-1].transpose(0,1,3,2) @ local[1:]
            angle = np.rad2deg(np.arccos(np.clip((np.trace(delta,axis1=-2,axis2=-1)-1)/2,-1,1)))
            stages[branch+'_smpl'] = dict(joints=metrics(joints[:, :22]),
                surface_horizontal_follow=temporal(vertices-xyz[:, :1]*[1,0,1]),
                surface_root_relative=temporal(vertices-xyz[:, :1]),
                input_joint_change_mean_m=float(np.linalg.norm(joints[:, :22]-xyz,axis=-1).mean()),
                angle_step_p99_degrees=float(np.quantile(angle[:,:22], .99)),
                angle_step_max_degrees=float(angle[:,:22].max()), independent_lbs_max_error_m=independent_error)
        original_vertices, _ = JointDrivenBody(body_path).motion(a)
        data['original_vertices'] = original_vertices
        stages['original'] = dict(joints=metrics(a),
            surface_horizontal_follow=temporal(original_vertices-a[:, :1]*[1,0,1]),
            surface_root_relative=temporal(original_vertices-a[:, :1]))
        _, author_xyz = converter.convert(a.copy(), None, iterations=10, foot_ik=True)
        author_vertices, _ = JointDrivenBody(body_path).motion(author_xyz)
        data.update(author_xyz=author_xyz.astype(np.float32), author_vertices=author_vertices)
        stages['author_bvh'] = dict(joints=metrics(author_xyz),
            surface_horizontal_follow=temporal(author_vertices-author_xyz[:, :1]*[1,0,1]),
            input_joint_change_mean_m=float(np.linalg.norm(author_xyz-a,axis=-1).mean()))
        naive = np.tile(np.eye(3), (len(a),52,1,1))
        naive[:,:22] = matrices
        _, naive_xyz = body.skin(naive, a[:,0])
        path = cache_dir/(name+'.npz')
        np.savez_compressed(path, **data)
        cases.append(dict(name=name, kind=kind, metadata=metadata, sha256=sha(path),
            position=metrics(a), rotation=metrics(b), stages=stages,
            naive_humanml_as_smpl_joint_error_m=float(np.linalg.norm(naive_xyz[:,:22]-a,axis=-1).mean()),
            position_rotation_mean_error_m=float(np.linalg.norm(a-b,axis=-1).mean())))
        save(output/'progress.json',dict(status='running', case_count=len(cases), analyzed=len(rows)))
        print('CASE',name,flush=True)

    for r in examples:
        path = source/'arrays'/r['array']
        if sha(path) != r['sha256']:
            raise ValueError('Original sample hash mismatch')
        with np.load(path) as data:
            raw = data['normalized']*std+mean
            analyze(Path(r['array']).stem, 'generated', raw, data['xyz'], r)
    with np.load(source/'arrays/validation_xyz.npz') as cached:
        for r in refs:
            path = args.root/'dataset/humanml3d/new_joint_vecs'/(r['id']+'.npy')
            if sha(path) != r['source_motion_sha256']:
                raise ValueError('GT source changed')
            source_files[str(path)] = sha(path)
            full = np.load(path)
            saved = cached['real_'+r['id']]
            candidates = []
            for start in range(len(full)-len(saved)+1):
                raw = full[start:start+len(saved)]
                xyz = recover_from_ric(torch.as_tensor(raw,dtype=torch.float32),22).numpy()
                candidates.append((float(np.abs(xyz-saved).max()), start))
            error, start = min(candidates)
            if error > 2e-6:
                raise ValueError('Could not match exact original GT crop: '+r['id'])
            analyze(r['id'], 'real', full[start:start+len(saved)], saved, dict(r,crop_start=start))
    if len(rows)!=146 or len(cases)!=4:
        raise ValueError('Incomplete fixed sample coverage')
    if any(sha(path)!=digest for path,digest in source_files.items()):
        raise ValueError('Frozen source changed during diagnostic')
    summary = {}
    for kind in ['generated','real']:
        selected = [r for r in rows if r['kind']==kind]
        summary[kind] = dict(count=len(selected),
            position_bone_cv=float(np.mean([r['position']['bone_cv_mean'] for r in selected])),
            rotation_bone_cv=float(np.mean([r['rotation']['bone_cv_mean'] for r in selected])),
            position_rotation_mean_error_m=float(np.mean([r['position_rotation_mean_error_m'] for r in selected])),
            position_jerk=float(np.mean([r['position']['jerk_rms_mps3'] for r in selected])),
            rotation_jerk=float(np.mean([r['rotation']['jerk_rms_mps3'] for r in selected])))
    audit = dict(status='complete', source_audit=records, source_sha256=source_files,
        model_sha256=sha(body_path), template_sha256=sha(template_path),
        lbs_audit=body_audit, rows=rows, cases=cases, summary=summary,
        fps=20, high_frequency_cutoff_hz=3,
        source_motion_unchanged=True, naturalness_fixed=False,
        rotation_mapping='HumanML incoming-bone 6D -> author FK -> geometric outgoing-bone SMPL rotation fit -> standard SMPL-H LBS; true twist is not available',
        selection='All 18 existing examples and 128 fixed GT crops; videos use walk/jump first seed and first eligible real walk/jump in reference order')
    save(output/'audit.json',audit)
    flat=[]
    for r in rows:
        flat.append(dict(name=r['name'],kind=r['kind'],frames=r['frames'],
            position_rotation_mean_error_m=r['position_rotation_mean_error_m'],
            position_bone_cv=r['position']['bone_cv_mean'], rotation_bone_cv=r['rotation']['bone_cv_mean'],
            position_jerk=r['position']['jerk_rms_mps3'],rotation_jerk=r['rotation']['jerk_rms_mps3'],
            position_bone_hf_cv=r['position']['bone_above3hz_cv_mean'],
            position_recovery_max_m=r['checks']['position_independent_max_m'],
            rotation_recovery_max_m=r['checks']['rotation_independent_max_m']))
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    save(output/'progress.json',dict(status='complete',case_count=4,analyzed=len(rows)))
    print(json.dumps(summary),flush=True)


def official(args):
    from utils.plot_script import plot_3d_motion
    from utils.paramUtil import t2m_kinematic_chain
    audit=json.loads((args.output/'audit.json').read_text())
    directory=args.output/'official_skeleton'
    directory.mkdir(exist_ok=True)
    videos=[]
    for row in audit['cases']:
        with np.load(args.output/'server_only'/(row['name']+'.npz')) as data:
            xyz=data['position_xyz'].copy()
        original=xyz.copy()
        path=directory/(row['name']+'.mp4')
        plot_3d_motion(str(path),t2m_kinematic_chain,xyz,title=row['metadata']['text'],fps=20)
        if not np.array_equal(original,xyz):raise ValueError('Plotter mutated source')
        videos.append(dict(name=row['name'],path=str(path.relative_to(args.output)),sha256=sha(path)))
        print('OFFICIAL',row['name'],flush=True)
    save(args.output/'official_videos.json',dict(status='complete',videos=videos,
        source_sha256=sha(args.root/'utils/plot_script.py'),smoothing=False,
        note='Unmodified official plotter subtracts one sequence-wide minimum height and follows root XZ; does not change bone lengths'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['audit','official'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.chdir(args.root)
    sys.path.insert(0,str(args.root));sys.path.insert(0,str(args.root/'reproduction'))
    torch.set_num_threads(2)
    {'audit':run,'official':official}[args.stage](args)
