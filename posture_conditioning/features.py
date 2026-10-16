"""Static, shape-normalized kinematics. World coordinates use z as up."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate skeleton direction')
    return x / norm


class StaticFeatures:
    def __init__(self, model_path):
        with np.load(Path(model_path), allow_pickle=True) as data:
            reg = data['J_regressor']
            if reg.shape == ():
                reg = reg.item()
            self.rest = np.asarray(reg @ data['v_template'], dtype=np.float64)[:22]
            self.parents = data['kintree_table'][0, :22].astype(int)
            self.parents[0] = -1
        right = unit(self.rest[2] - self.rest[1])
        up = self.rest[9] - self.rest[0]
        up = unit(up - (up @ right) * right)
        forward = unit(np.cross(up, right))
        self.basis = np.stack([right, forward, up], 1)
        self.reference = np.tile(np.eye(3), (22, 1, 1))
        self.reference[0] = self.basis.T
        self.reference_joints, self.reference_global = self.fk(self.reference)
        self.length = float(np.mean([np.linalg.norm(self.rest[k]-self.rest[h]) +
                                     np.linalg.norm(self.rest[a]-self.rest[k])
                                     for h,k,a in [(1,4,7),(2,5,8)]]))

    def fk(self, local):
        positions = np.zeros((22, 3))
        global_rot = np.empty((22, 3, 3))
        global_rot[0] = local[0]
        for j in range(1, 22):
            p = self.parents[j]
            global_rot[j] = global_rot[p] @ local[j]
            positions[j] = positions[p] + global_rot[p] @ (self.rest[j] - self.rest[p])
        return positions, global_rot

    def __call__(self, fullpose, up_axis='z'):
        fullpose = np.asarray(fullpose, dtype=np.float64)
        # SMPL-X slots after body joint 21 contain face and hands and are never read.
        allowed = list(range(12)) + [13, 14]
        local = np.tile(np.eye(3), (22, 1, 1))
        local[allowed] = Rotation.from_rotvec(fullpose.reshape(-1,3)[allowed]).as_matrix()
        if up_axis == 'y':
            local[0] = Rotation.from_euler('x',90,degrees=True).as_matrix() @ local[0]
        elif up_axis != 'z':
            raise ValueError('Unknown native up axis: '+str(up_axis))
        forward = local[0] @ self.basis[:,1]
        forward[2] = 0
        forward = unit(forward)
        heading = np.stack([unit(np.cross(forward,[0.,0.,1.])),forward,[0.,0.,1.]])
        local[0] = heading @ local[0]
        joints, global_rot = self.fk(local)
        residual = local[:12] @ self.reference[:12].transpose(0,2,1)
        rotation = (residual[:,:2,:] - np.eye(3)[:2]).reshape(12,6)
        position = (joints[:12] - self.reference_joints[:12]) / self.length
        direction = np.zeros((12,3))
        direction[0] = global_rot[0] @ self.basis[:,2] - np.array([0.,0.,1.])
        for j in range(1,12):
            p = self.parents[j]
            direction[j] = unit(joints[j]-joints[p]) - unit(self.reference_joints[j]-self.reference_joints[p])
        shoulder = ((joints[16,2]-joints[17,2]) -
                    (self.reference_joints[16,2]-self.reference_joints[17,2])) / self.length
        core = np.concatenate([rotation,position,direction],1).astype(np.float32)
        if not np.isfinite(core).all() or not np.isfinite(shoulder):
            raise ValueError('Nonfinite posture feature')
        return core, np.array([shoulder],dtype=np.float32)
