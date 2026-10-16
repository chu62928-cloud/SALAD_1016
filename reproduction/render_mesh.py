"""Render the original SALAD examples as neutral human surfaces over a fixed floor."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '2'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mesh_surface import JointDrivenBody


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


class WorldRenderer:
    def __init__(self, faces):
        import pyrender
        import trimesh
        self.p, self.t, self.faces = pyrender, trimesh, faces
        self.renderer = pyrender.OffscreenRenderer(620, 620)
        floor = trimesh.creation.box(extents=[50, .006, 50])
        floor.apply_translation([0, -.003, 0])
        floor.visual.vertex_colors = [220, 225, 226, 255]
        pieces = [floor]
        for v in np.arange(-20, 20.1, .5):
            for axis in [0, 2]:
                line = trimesh.creation.box(extents=[40, .001, .007] if axis == 2 else [.007, .001, 40])
                location = [0, .001, 0]
                location[axis] = float(v)
                line.apply_translation(location)
                line.visual.vertex_colors = [166, 177, 179, 255]
                pieces.append(line)
        self.ground = pyrender.Mesh.from_trimesh(trimesh.util.concatenate(pieces), smooth=False)
        self.scenes = {}

    def draw(self, vertices, root, side, camera):
        p = self.p
        colors = [[.24, .55, .62, 1], [.72, .40, .40, 1]]
        if side not in self.scenes:
            scene = p.Scene(bg_color=[.965, .975, .975, 1], ambient_light=[.48, .48, .48])
            scene.add(self.ground)
            node = scene.add(p.OrthographicCamera(xmag=camera['ymag'], ymag=camera['ymag'], znear=.1, zfar=100))
            light = np.eye(4)
            light[:3, 3] = [3, 5, 8]
            scene.add(p.DirectionalLight(color=np.ones(3), intensity=2), pose=light)
            self.scenes[side] = [scene, node, None]
        scene, node, body = self.scenes[side]
        if body is not None:
            scene.remove_node(body)
        surface = self.t.Trimesh(vertices=vertices, faces=self.faces, process=False)
        material = p.MetallicRoughnessMaterial(baseColorFactor=colors[side], metallicFactor=0, roughnessFactor=.85)
        self.scenes[side][2] = scene.add(p.Mesh.from_trimesh(surface, material=material, smooth=True))
        yaw, pitch = np.deg2rad([0 if side == 0 else 60, 8])
        back = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
        right = np.cross([0, 1, 0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        center = np.array([root[0], camera['ycenter'], root[2]])
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, up, back], axis=1)
        pose[:3, 3] = center + 20*back
        scene.set_pose(node, pose=pose)
        camera_points = (vertices - pose[:3, 3]) @ pose[:3, :3]
        if np.abs(camera_points[:, :2]).max() > camera['ymag']:
            raise ValueError('Human surface leaves camera bounds')
        rgb, _ = self.renderer.render(scene)
        colored = ((rgb[..., 2].astype(float)-rgb[..., 0] > 12) if side == 0 else
                   (rgb[..., 0].astype(float)-rgb[..., 2] > 14))
        count = int(colored.sum())
        if count < 1000:
            raise ValueError('Human surface is blank')
        return Image.fromarray(rgb), count

    def close(self):
        self.renderer.delete()


def run(args):
    source = args.source.resolve()
    output = source/'mesh_v1'
    output.mkdir(exist_ok=True)
    lock = (output/'render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    examples = read(source/'examples.json')
    protocol = read(source/'protocol.json')
    fps = protocol['fps']
    if len(examples) != 18 or fps != 20:
        raise ValueError('Expected the original 18 SALAD examples at 20 fps')
    contract = dict(version='salad-joint-driven-mesh-v1',
        examples_sha256=sha(source/'examples.json'), protocol_sha256=sha(source/'protocol.json'),
        body_model_sha256=sha(args.body), skinning_sha256=sha(Path(__file__).with_name('mesh_surface.py')),
        renderer_sha256=sha(__file__), fps=fps, frames=protocol['example_frames'],
        method='Neutral SMPL-H, pose corrective shapes, joint-driven linear blend skinning; estimated surface',
        processing='No new generation, IK fitting, smoothing, floor snapping, foot locking or bone length correction',
        floor_y=0, grid_m=.5, camera='Fixed common scale and height; horizontal root following only',
        scientific_metrics='Unchanged: original recovered joints, not the estimated mesh', arrays_server_only=True)
    if (output/'binding.json').exists() and read(output/'binding.json') != contract:
        raise ValueError('Existing surface binding differs')
    save(output/'binding.json', contract)
    body = JointDrivenBody(args.body)
    audit = body.audit()
    (output/'server_only').mkdir(exist_ok=True)
    (output/'videos').mkdir(exist_ok=True)
    cases, bounds = [], []
    save(output/'status.json', dict(status='running', stage='surface', pid=os.getpid()))
    for record in examples:
        path = source/'arrays'/record['array']
        if sha(path) != record['sha256']:
            raise ValueError('Original motion cache differs')
        with np.load(path) as archive:
            xyz = archive['xyz'].copy()
        if xyz.shape != (120, 22, 3):
            raise ValueError('Unexpected motion shape')
        name = '%s_%d' % (record['id'], record['seed'])
        cache = output/'server_only'/(name+'.npz')
        receipt = cache.with_suffix('.json')
        if receipt.exists():
            previous = read(receipt)
            if previous['source_sha256'] != record['sha256'] or previous['sha256'] != sha(cache):
                raise ValueError('Surface cache differs')
            with np.load(cache) as archive:
                vertices = archive['vertices'].copy()
            error = previous['attachment_max_m']
        else:
            vertices, error = body.motion(xyz)
            np.savez_compressed(cache, vertices=vertices)
            save(receipt, dict(source_sha256=record['sha256'], sha256=sha(cache), attachment_max_m=error))
        local = vertices - xyz[:, None, 0]*[1, 0, 1]
        bounds.append((float(local[..., 1].min()), float(local[..., 1].max()),
                       float(np.linalg.norm(local[..., [0, 2]], axis=-1).max())))
        cases.append((record, xyz, vertices, error, name))
        print('SURFACE', name, error, flush=True)
    ymin, ymax, radius = min(b[0] for b in bounds), max(b[1] for b in bounds), max(b[2] for b in bounds)
    camera = dict(ycenter=(ymin+ymax)/2,
        ymag=max((ymax-ymin)/2 + radius*np.sin(np.deg2rad(8)) + .13, radius+.13))
    save(output/'camera.json', camera)
    audit.update(joint_attachment_max_m=max(c[3] for c in cases), source_files_unchanged=False)
    fonts = [args.font, Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    font_path = next(p for p in fonts if p and p.exists())
    title = ImageFont.truetype(str(font_path), 25)
    small = ImageFont.truetype(str(font_path), 19)
    renderer = WorldRenderer(body.faces)
    manifest = dict(status='running', binding_sha256=sha(output/'binding.json'), camera=camera,
                    geometry_audit=audit, videos=[])
    try:
        for record, xyz, vertices, error, name in cases[:args.limit]:
            video, poster = output/'videos'/(name+'.mp4'), output/'videos'/(name+'.png')
            receipt = output/'videos'/(name+'.json')
            if receipt.exists():
                entry = read(receipt)
                if entry['binding_sha256'] != manifest['binding_sha256'] or sha(video) != entry['sha256'] or sha(poster) != entry['poster_sha256']:
                    raise ValueError('Rendered media differs')
            else:
                counts = []
                proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                    '-s', '1280x800', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
                    '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
                    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    for frame in range(len(xyz)):
                        picture = Image.new('RGB', (1280, 800), 'white')
                        draw = ImageDraw.Draw(picture)
                        draw.text((20, 12), '%s | 种子 %d | %.2f 秒' % (record['label'], record['seed'], frame/fps), font=title, fill='#243836')
                        draw.text((20, 49), record['text'], font=small, fill='#4c5d5b')
                        draw.text((20, 80), '正面', font=small, fill='#327f91')
                        draw.text((660, 80), '侧前方', font=small, fill='#a15757')
                        pair = []
                        for side, left in [(0, 10), (1, 650)]:
                            image, count = renderer.draw(vertices[frame], xyz[frame, 0], side, camera)
                            picture.paste(image, (left, 112))
                            pair.append(count)
                        counts.append(pair)
                        draw.text((20, 746), '滑步 %.1f%%    双脚离地 %.1f%%    骨长变异 %.2f%%' % (
                            100*record['metrics']['skating_frame_ratio'], 100*record['metrics']['airborne_frame_ratio'],
                            100*record['metrics']['bone_length_cv']), font=small, fill='#3b4b49')
                        if frame == len(xyz)//2:
                            picture.save(poster)
                        proc.stdin.write(picture.tobytes())
                    proc.stdin.close()
                    stderr = proc.stderr.read().decode()
                    if proc.wait() != 0:
                        raise RuntimeError(stderr)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
                entry = dict(id=record['id'], label=record['label'], seed=record['seed'],
                    path='videos/'+video.name, poster='videos/'+poster.name, sha256=sha(video),
                    poster_sha256=sha(poster), source_sha256=record['sha256'], binding_sha256=manifest['binding_sha256'],
                    attachment_max_m=error, min_body_pixels=np.min(counts, axis=0).tolist(),
                    width=1280, height=800, frames=len(xyz), fps=fps)
                save(receipt, entry)
            manifest['videos'].append(entry)
            save(output/'videos.json', manifest)
            save(output/'status.json', dict(status='running', stage='render', completed=len(manifest['videos']), requested=args.limit, pid=os.getpid()))
            print('RENDERED', name, flush=True)
    finally:
        renderer.close()
    for record, *_ in cases:
        if sha(source/'arrays'/record['array']) != record['sha256']:
            raise ValueError('Source motion changed while rendering')
    audit['source_files_unchanged'] = True
    manifest['status'] = 'complete' if len(manifest['videos']) == 18 else 'preview'
    manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
    save(output/'videos.json', manifest)
    save(output/'status.json', dict(status=manifest['status'], videos=len(manifest['videos']), pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--body', type=Path, required=True)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--limit', type=int, default=18)
    args = parser.parse_args()
    if not 1 <= args.limit <= 18:
        parser.error('limit must be between 1 and 18')
    try:
        run(args)
    except Exception as error:
        save(args.source/'mesh_v1/status.json', dict(status='failed', error=repr(error), pid=os.getpid()))
        raise
