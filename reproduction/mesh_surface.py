"""Joint-driven neutral SMPL-H surface, without modifying motion coordinates.

Adapted from the project's existing official_a_mesh_examples.JointDrivenBody.
The surface is an estimate: exact joint attachments do not imply mesh accuracy.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_between(source, target):
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(source @ target, -1, 1))
    sine = np.linalg.norm(cross)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()


class JointDrivenBody:
    def __init__(self, path):
        with np.load(Path(path), allow_pickle=True) as asset:
            self.template = np.asarray(asset['v_template'], dtype=np.float64)
            self.rest = np.asarray(asset['J_regressor'], dtype=np.float64) @ self.template
            self.weights = np.asarray(asset['weights'], dtype=np.float64)
            self.parents = np.asarray(asset['kintree_table'], dtype=np.int64)[0].copy()
            self.parents[0] = -1
            self.posedirs = np.asarray(asset['posedirs'], dtype=np.float64).reshape(-1, 459)
            self.faces = np.asarray(asset['f'], dtype=np.int64)
        self.children = {j: [k for k in range(1, 22) if self.parents[k] == j] for j in range(22)}

    def frame(self, xyz):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape != (22, 3) or not np.isfinite(xyz).all():
            raise ValueError('Expected finite 22-joint world coordinates')
        lengths = np.linalg.norm(xyz[1:] - xyz[self.parents[1:22]], axis=1)
        if lengths.min() < 1e-9:
            raise ValueError('Degenerate source bone')
        total = np.repeat(np.eye(3)[None], 52, axis=0)
        target = np.empty((52, 3))
        target[:22] = xyz
        for joint in range(22):
            children = self.children[joint]
            parent = int(self.parents[joint])
            inherited = total[parent] if parent >= 0 else np.eye(3)
            if len(children) == 1:
                child = children[0]
                total[joint] = rotation_between(inherited @ (self.rest[child] - self.rest[joint]),
                                                xyz[child] - xyz[joint]) @ inherited
            elif len(children) > 1:
                a = self.rest[children] - self.rest[joint]
                b = xyz[children] - xyz[joint]
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                b = b / np.linalg.norm(b, axis=1, keepdims=True)
                u, _, vt = np.linalg.svd(a.T @ b)
                sign = np.eye(3)
                sign[2, 2] = np.linalg.det(vt.T @ u.T)
                total[joint] = vt.T @ sign @ u.T
            else:
                total[joint] = inherited
        for joint in range(22, 52):
            parent = int(self.parents[joint])
            total[joint] = total[parent]
            target[joint] = target[parent] + total[parent] @ (self.rest[joint] - self.rest[parent])
        local = total.copy()
        for joint in range(1, 52):
            local[joint] = total[self.parents[joint]].T @ total[joint]
        corrective = ((local[1:] - np.eye(3)).reshape(-1) @ self.posedirs.T).reshape(-1, 3)
        blend = np.einsum('vj,jab->vab', self.weights, total)
        translations = target - np.einsum('jab,jb->ja', total, self.rest)
        posed = np.einsum('vab,vb->va', blend, self.template + corrective) + self.weights @ translations
        recovered = np.einsum('jab,jb->ja', total, self.rest) + translations
        error = float(np.max(np.abs(recovered - target)))
        if error > 1e-9 or not np.isfinite(posed).all():
            raise ValueError('Skin attachment audit failed')
        return posed.astype(np.float32), error

    def motion(self, xyz):
        source = np.asarray(xyz).copy()
        frames = [self.frame(frame) for frame in xyz]
        if not np.array_equal(source, xyz):
            raise ValueError('Skinning changed its input')
        return np.stack([v for v, _ in frames]), max(e for _, e in frames)

    def audit(self):
        neutral, _ = self.frame(self.rest[:22])
        error = float(np.max(np.abs(neutral - self.template)))
        rotation = Rotation.from_euler('y', 37, degrees=True).as_matrix()
        shift = np.array([1.25, .15, -2.5])
        transformed, _ = self.frame(self.rest[:22] @ rotation.T + shift)
        rigid_error = float(np.max(np.abs(transformed - (self.template @ rotation.T + shift))))
        if error > 2e-6 or rigid_error > 2e-6:
            raise ValueError('Neutral template or rigid transform check failed')
        return dict(neutral_template_max_m=error, rigid_transform_max_m=rigid_error)
