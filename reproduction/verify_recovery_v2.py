"""Recompute diagnostic metrics and distinguish heading offsets from pose error."""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_key]='2'

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.fft import dct,idct
from scipy.spatial.transform import Rotation

from audit_recovery_v2 import sha,save,FixedSMPLSurface


def yaw_aligned(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    ar=a-a[:,:1];br=b-b[:,:1]
    x=br[...,[0,2]].reshape(-1,2);y=ar[...,[0,2]].reshape(-1,2)
    u,_,vt=np.linalg.svd(x.T@y)
    matrix=u@np.diag([1,np.linalg.det(u@vt)])@vt
    aligned=br.copy();aligned[...,[0,2]]=br[...,[0,2]]@matrix
    error=np.linalg.norm(aligned-ar,axis=-1)
    return dict(yaw_degrees=float(np.rad2deg(np.arctan2(matrix[0,1],matrix[0,0]))),
                aligned_mean_error_m=float(error.mean()),aligned_max_error_m=float(error.max()),
                raw_mean_error_m=float(np.linalg.norm(a-b,axis=-1).mean()))


def independent_metrics(xyz,parents):
    x=np.asarray(xyz,dtype=np.float64)
    length=np.linalg.norm(x[:,1:]-x[:,parents[1:]],axis=-1)
    coef=dct(length,axis=0,norm='ortho')
    coef[np.arange(len(x))*20/(2*len(x))<3]=0
    high=idct(coef,axis=0,norm='ortho')
    relative=x-x[:,:1]
    return dict(bone_cv_mean=float(np.mean(length.std(0)/length.mean(0))),
        bone_above3hz_cv_mean=float(np.mean(np.sqrt((high**2).mean(0))/length.mean(0))),
        jerk_rms_mps3=float(np.sqrt((np.diff(relative,n=3,axis=0)**2).mean())*8000))


def run(args):
    sys.path.insert(0,str(args.root));sys.path.insert(0,str(args.root/'reproduction'))
    from common.skeleton import Skeleton
    from common.quaternion import cont6d_to_matrix,qbetween_np,qmul_np,qinv_np,qfix,quaternion_to_matrix_np
    from utils.motion_process import recover_from_ric,recover_from_rot
    from utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
    from diagnose_jitter import PARENTS
    audit=json.loads((args.output/'audit.json').read_text())
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen source modified')
    reference=np.load(args.root/'dataset/humanml3d/new_joint_vecs/000021.npy')
    first=recover_from_ric(torch.as_tensor(reference,dtype=torch.float32),22)
    skeleton=Skeleton(torch.from_numpy(t2m_raw_offsets),t2m_kinematic_chain,'cpu')
    skeleton.get_offsets_joints(first[0])
    mean=np.load(args.root/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
    std=np.load(args.root/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
    results=[];max_metric_error=0.
    with np.load(args.source/'arrays/validation_xyz.npz') as saved:
        for row in audit['rows']:
            if row['kind']=='generated':
                with np.load(args.source/'arrays'/(row['name']+'.npz')) as data:
                    raw=data['normalized']*std+mean
            else:
                full=np.load(args.root/'dataset/humanml3d/new_joint_vecs'/(row['name']+'.npy'))
                original=saved['real_'+row['name']]
                candidates=[]
                for start in range(len(full)-len(original)+1):
                    xyz=recover_from_ric(torch.as_tensor(full[start:start+len(original)],dtype=torch.float32),22).numpy()
                    candidates.append((float(np.abs(xyz-original).max()),start))
                error,start=min(candidates)
                if error>2e-6:raise ValueError('GT crop mismatch')
                raw=full[start:start+len(original)]
            x=torch.as_tensor(raw,dtype=torch.float32)
            a=recover_from_ric(x.clone(),22).numpy();b=recover_from_rot(x.clone(),22,skeleton).numpy()
            for key,xyz in [('position',a),('rotation',b)]:
                independent=independent_metrics(xyz,PARENTS)
                for name,value in independent.items():
                    max_metric_error=max(max_metric_error,abs(value-row[key][name]))
            framewise=[yaw_aligned(a[t:t+1],b[t:t+1]) for t in range(len(a))]
            results.append(dict(name=row['name'],kind=row['kind'],**yaw_aligned(a,b),
                framewise_aligned_mean_error_m=float(np.mean([r['aligned_mean_error_m'] for r in framewise])),
                framewise_aligned_max_error_m=max(r['aligned_max_error_m'] for r in framewise),
                diagnostic_framewise_yaw_degrees=[r['yaw_degrees'] for r in framewise]))
    if max_metric_error>1e-9:raise ValueError('Independent metric recomputation differs')
    known=Rotation.from_euler('xyz',[[15,-28,48],[-62,37,123]],degrees=True).as_matrix()
    six=np.concatenate([known[:,:,0],known[:,:,1]],-1)
    recovered=cont6d_to_matrix(torch.as_tensor(six)).numpy()
    rotation_error=float(np.abs(recovered-known).max())
    transpose_error=float(np.abs(recovered-known.transpose(0,2,1)).max())
    if rotation_error>1e-12 or transpose_error<.1:
        raise ValueError('6D column/row convention test failed')
    forward=Rotation.from_euler('y',[178,182],degrees=True).apply([0,0,1])
    qs=qbetween_np(forward,np.tile([0.,0.,1.],(2,1)))
    delta=qmul_np(qs[1:],qinv_np(qs[:-1]))
    encoded=np.arcsin(delta[:,2])
    direct=quaternion_to_matrix_np(delta)[0]
    encoded_matrix=Rotation.from_euler('y',2*encoded[0]).as_matrix()
    continuous=qfix(qs[:,None])[:,0]
    delta_fixed=qmul_np(continuous[1:],qinv_np(continuous[:-1]))
    encoded_fixed=Rotation.from_euler('y',2*np.arcsin(delta_fixed[0,2])).as_matrix()
    sign_probe=dict(scope='Synthetic crossing of 180-degree heading; does not establish provenance of the seven GT mismatches',
        original_delta_matrix_error=float(np.abs(direct-encoded_matrix).max()),
        continuous_quaternion_delta_matrix_error=float(np.abs(direct-encoded_fixed).max()))
    if sign_probe['original_delta_matrix_error']<.1 or sign_probe['continuous_quaternion_delta_matrix_error']>1e-5:
        raise ValueError('Quaternion sign negative control was not reproduced')
    body=FixedSMPLSurface('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz')
    case_checks=[]
    for row in audit['cases']:
        path=args.output/'server_only'/(row['name']+'.npz')
        if sha(path)!=row['sha256']:raise ValueError('Cache changed')
        with np.load(path) as data:
            checks=dict(name=row['name'],heading=yaw_aligned(data['position_xyz'],data['rotation_xyz']))
            for branch in ['position','rotation']:
                vertices,joints=body.skin(data[branch+'_local'].astype(np.float64),data['position_xyz'][:,0])
                error=max(float(np.abs(vertices-data[branch+'_smpl_vertices']).max()),
                          float(np.abs(joints[:,:22]-data[branch+'_smpl_xyz']).max()))
                if error>2e-6:raise ValueError('Saved rotations do not reproduce standard SMPL surface')
                checks[branch+'_standard_lbs_replay_max_m']=error
                bone=independent_metrics(data[branch+'_smpl_xyz'],PARENTS)['bone_cv_mean']
                if bone>1e-5:raise ValueError('Standard SMPL bone lengths changed over time')
            checks['raw_bone_hf_cv']=independent_metrics(data['position_xyz'],PARENTS)['bone_above3hz_cv_mean']
            if row['kind']=='generated':
                prior=args.source/'mesh_v1/server_only'/(row['name']+'.npz')
                with np.load(prior) as old:
                    error=float(np.abs(data['original_vertices']-old['vertices']).max())
                if error!=0:raise ValueError('Earlier skinning pipeline was not reproduced exactly')
                checks['previous_direct_skinning_max_m']=error
            case_checks.append(checks)
    summary={}
    for kind in ['generated','real']:
        chosen=[r for r in results if r['kind']==kind]
        summary[kind]=dict(count=len(chosen),
            aligned_mean_error_m=float(np.mean([r['aligned_mean_error_m'] for r in chosen])),
            worst_aligned_max_error_m=max(r['aligned_max_error_m'] for r in chosen),
            framewise_aligned_mean_error_m=float(np.mean([r['framewise_aligned_mean_error_m'] for r in chosen])),
            framewise_aligned_max_error_m=max(r['framewise_aligned_max_error_m'] for r in chosen),
            constant_yaw_mismatches=[r['name'] for r in chosen if r['aligned_max_error_m']>1e-5])
    if summary['real']['framewise_aligned_max_error_m']>1e-5:
        raise ValueError('GT differences include non-yaw pose inconsistency')
    jump=next(c for c in case_checks if c['name']=='jump_20261016')
    if jump['raw_bone_hf_cv']<=.005:raise ValueError('Original rapid bone-length variation disappeared unexpectedly')
    value=dict(status='passed',rows=results,summary=summary,case_checks=case_checks,
        metric_recompute_max_error=max_metric_error,
        sixd_matrix_roundtrip_max_error=rotation_error,wrong_transpose_negative_control_error=transpose_error,
        quaternion_sign_probe=sign_probe,
        ground_truth_constant_heading_check_passed=not summary['real']['constant_yaw_mismatches'],
        framewise_heading_is_diagnostic_only=True,framewise_alignment_applied_to_videos_or_metrics=False,
        original_bone_jitter_still_present=True,diagnostic_limit_is_paper_metric=False,
        standard_lbs_replay_passed=True,source_files_unchanged=True,
        previous_renderer_equals_official=False,naturalness_fixed=False,
        audit_sha256=sha(args.output/'audit.json'),verification_code_sha256=sha(__file__))
    save(args.output/'verification.json',value)
    print(json.dumps(dict(status='passed',summary=summary,metric_recompute_max_error=max_metric_error)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2);run(args)
