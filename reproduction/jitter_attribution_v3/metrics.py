"""Consistent diagnostics on complete 20 Hz joint sequences."""
import numpy as np
from scipy.fft import dct

PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
NAMES = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
         'left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
         'left_collar','right_collar','head','left_shoulder','right_shoulder',
         'left_elbow','right_elbow','left_wrist','right_wrist']


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value,dtype=np.float64)**2)))


def vector_rms(value):
    return float(np.sqrt(np.mean(np.sum(np.asarray(value,dtype=np.float64)**2,axis=-1))))


def spectrum(value, fps=20):
    centered = value-value.mean(axis=0,keepdims=True)
    coeff = dct(centered,axis=0,norm='ortho')
    power = coeff**2
    freq = np.arange(len(value))*fps/(2*len(value))
    total = float(power.sum())
    result = {}
    for cutoff in [2,3,4,5]:
        energy = float(power[freq>=cutoff].sum())
        result[f'hf{cutoff}_rms_m'] = float(np.sqrt(energy/value.size))
        result[f'hf{cutoff}_energy_fraction'] = energy/total if total>1e-20 else 0.
    return result


def fixed_directions(points, offsets):
    result = np.empty_like(points,dtype=np.float64)
    result[:,0] = points[:,0]
    directions = points[:,1:]-points[:,PARENTS[1:]]
    lengths = np.linalg.norm(directions,axis=-1)
    if (lengths<1e-8).any():
        raise ValueError('Degenerate input bone')
    desired = np.linalg.norm(offsets[1:],axis=-1)
    steps = directions/lengths[...,None]*desired[None,:,None]
    for j in range(1,22):
        result[:,j] = result[:,PARENTS[j]]+steps[:,j-1]
    return result


def measures(points, protocol):
    from physics_metrics import measure
    p = np.asarray(points,dtype=np.float64)
    local = p-p[:,:1]
    horizontal = p-p[:,:1]*[1,0,1]
    fps = protocol['fps']
    bones = np.linalg.norm(p[:,1:]-p[:,PARENTS[1:]],axis=-1)
    deriv = {name:np.diff(p,n=order,axis=0)*fps**order for order,name in enumerate(['speed','acceleration','jerk'],1)}
    local_deriv = {name:np.diff(local,n=order,axis=0)*fps**order for order,name in enumerate(['speed','acceleration','jerk'],1)}
    result = measure(p,protocol)
    for name in deriv:
        suffix = {'speed':'mps','acceleration':'mps2','jerk':'mps3'}[name]
        result[f'world_{name}_rms_{suffix}'] = vector_rms(deriv[name])
        result[f'local_{name}_rms_{suffix}'] = vector_rms(local_deriv[name])
        result[f'root_{name}_rms_{suffix}'] = vector_rms(deriv[name][:,0])
    result['interior_local_jerk_rms_mps3'] = vector_rms(local_deriv['jerk'][4:-4])
    for phase in range(4):
        result[f'local_jerk_mod4_{phase}_mps3'] = vector_rms(local_deriv['jerk'][phase::4])
    result.update({'local_'+k:v for k,v in spectrum(local,fps).items()})
    result.update({'horizontal_'+k:v for k,v in spectrum(horizontal,fps).items()})
    result.update({'root_height_'+k:v for k,v in spectrum(p[:,0,1],fps).items()})
    coeff = dct(bones-bones.mean(0),axis=0,norm='ortho')
    freq = np.arange(len(p))*fps/(2*len(p))
    hf = np.sqrt((coeff[freq>=3]**2).sum(0)/len(p))/bones.mean(0)
    result['bone_hf3_cv'] = float(hf.mean())
    result['bone_cv_max'] = float((bones.std(0)/bones.mean(0)).max())
    bone_rows = [dict(joint=j,name=NAMES[j],parent=int(PARENTS[j]),mean_length_m=float(bones[:,j-1].mean()),
        length_cv=float(bones[:,j-1].std()/bones[:,j-1].mean()),hf3_length_cv=float(hf[j-1]),
        min_length_m=float(bones[:,j-1].min()),max_length_m=float(bones[:,j-1].max())) for j in range(1,22)]
    joint_rows = []
    for j in range(22):
        row = dict(joint=j,name=NAMES[j])
        for name in deriv:
            suffix = {'speed':'mps','acceleration':'mps2','jerk':'mps3'}[name]
            row[f'world_{name}_rms_{suffix}'] = vector_rms(deriv[name][:,j])
            row[f'local_{name}_rms_{suffix}'] = vector_rms(local_deriv[name][:,j])
        joint_rows.append(row)
    series = dict(bone_length_m=bones.astype(np.float32),
        **{f'joint_{name}':np.linalg.norm(values,axis=-1).astype(np.float32) for name,values in deriv.items()})
    return result,bone_rows,joint_rows,series


def paired_error(prediction, target, fps=20):
    p,t = [np.asarray(x,dtype=np.float64) for x in [prediction,target]]
    local_p,local_t = p-p[:,:1],t-t[:,:1]
    return dict(mpjpe_world_m=float(np.linalg.norm(p-t,axis=-1).mean()),
        mpjpe_root_aligned_m=float(np.linalg.norm(local_p-local_t,axis=-1).mean()),
        acceleration_error_world_mps2=vector_rms(np.diff(p-t,n=2,axis=0)*fps**2),
        acceleration_error_local_mps2=vector_rms(np.diff(local_p-local_t,n=2,axis=0)*fps**2),
        jerk_error_local_mps3=vector_rms(np.diff(local_p-local_t,n=3,axis=0)*fps**3))
