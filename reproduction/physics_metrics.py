"""World-space kinematic diagnostics for HumanML3D, with explicit denominators."""
import numpy as np

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                    9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def contact_diagnostics(xyz, fps, height, speed_threshold):
    toes = xyz[:, [10, 11]]
    speed = np.linalg.norm(np.diff(toes[..., [0, 2]], axis=0), axis=-1)*fps
    low = np.maximum(toes[:-1, :, 1], toes[1:, :, 1]) <= height
    sliding = low & (speed > speed_threshold)
    any_low = low.any(axis=1)
    return {
        'near_floor_speed_mps': float(speed[low].mean()) if low.any() else None,
        'near_floor_speed_p95_mps': float(np.quantile(speed[low], .95)) if low.any() else None,
        'near_floor_toe_interval_fraction': float(low.mean()),
        'near_floor_intervals': int(low.sum()),
        'toe_intervals': int(low.size),
        'skating_frame_ratio': float(sliding.any(axis=1).mean()),
        'skating_given_contact_ratio': float(sliding.any(axis=1)[any_low].mean()) if any_low.any() else None,
    }


def measure(xyz, protocol):
    p = np.asarray(xyz, dtype=np.float64)
    fps = float(protocol['fps'])
    if p.ndim != 3 or p.shape[1:] != (22, 3) or len(p)<4 or not np.isfinite(p).all() or fps<=0:
        raise ValueError('Expected finite [T>=4,22,3], positive fps, Y-up metres')
    feet = p[:, [7, 10, 8, 11], 1]
    root_speed = np.linalg.norm(np.diff(p[:, 0, [0, 2]], axis=0), axis=-1)*fps
    bones = np.linalg.norm(p[:, 1:]-p[:, PARENTS[1:]], axis=-1)
    means = bones.mean(axis=0)
    if (means<=1e-8).any():
        raise ValueError('Degenerate mean bone length')
    local = p-p[:, :1]
    acceleration = np.diff(local, n=2, axis=0)*fps**2
    jerk = np.diff(local, n=3, axis=0)*fps**3
    result = contact_diagnostics(p, fps, protocol['contact_height_m'], protocol['slide_speed_mps'])
    result.update(
        frames=len(p),
        foot_penetration_mean_m=float(np.maximum(-feet, 0).mean()),
        foot_penetration_p95_m=float(np.quantile(np.maximum(-feet, 0), .95)),
        foot_penetration_max_m=float(np.maximum(-feet, 0).max()),
        penetration_frame_ratio=float((feet.min(axis=1)<-protocol['penetration_tolerance_m']).mean()),
        airborne_frame_ratio=float((feet.min(axis=1)>protocol['contact_height_m']).mean()),
        bone_length_cv=float((bones.std(axis=0)/means).mean()),
        relative_acceleration_rms_mps2=float(np.sqrt((acceleration**2).sum(axis=-1).mean())),
        relative_jerk_rms_mps3=float(np.sqrt((jerk**2).sum(axis=-1).mean())),
        root_speed_mps=float(root_speed.mean()),
        root_net_displacement_m=float(np.linalg.norm(p[-1, 0, [0, 2]]-p[0, 0, [0, 2]])),
    )
    for height in protocol['sensitivity_heights_m']:
        for threshold in protocol['sensitivity_speeds_mps']:
            key = 'skating_h%02d_v%03d' % (round(height*100),round(threshold*100))
            result[key] = contact_diagnostics(p,fps,height,threshold)['skating_frame_ratio']
    return result
