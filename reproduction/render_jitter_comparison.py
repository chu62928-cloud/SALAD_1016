"""Side-by-side raw, constant-skeleton and author foot-IK diagnostics."""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '2'
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run(args):
    sys.path.insert(0, str(args.root/'reproduction'))
    from render_mesh import WorldRenderer
    from mesh_surface import JointDrivenBody
    from diagnose_jitter import temporal
    from physics_metrics import measure
    lock = (args.output/'render.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    audit = json.loads((args.output/'conversion_audit.json').read_text())
    if audit['status'] != 'complete' or not audit['source_files_unchanged']:
        raise ValueError('Completed conversion audit required')
    protocol = json.loads((args.source/'protocol.json').read_text())
    body = JointDrivenBody(Path('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz'))
    stages = ['raw', 'fixed_bones', 'author_foot_ik']
    stage_labels = ['原始关节直接蒙皮', '固定骨架拟合', '固定骨架 + 足部拟合']
    cases, low, high, radius = [], float('inf'), -float('inf'), 0.
    for row in audit['cases']:
        values = []
        for stage in stages:
            path = args.output/'server_only'/(row['name']+'_'+stage+'.npz')
            if sha(path) != row['stages'][stage]['sha256']:
                raise ValueError('Diagnostic surface changed')
            with np.load(path) as cache:
                xyz, vertices = cache['xyz'].copy(), cache['vertices'].copy()
            local = vertices - xyz[:, :1]*[1, 0, 1]
            low = min(low, float(local[..., 1].min()))
            high = max(high, float(local[..., 1].max()))
            radius = max(radius, float(np.linalg.norm(local[..., [0, 2]], axis=-1).max()))
            root_height = temporal(xyz[:, 0, 1])
            camera_surface = temporal(local)
            stats = dict(root_height=root_height, camera_follow_surface=camera_surface, physics=measure(xyz, protocol))
            values.append((xyz, vertices, stats))
        cases.append((row, values))
    camera = dict(ycenter=(low+high)/2,
        ymag=max((high-low)/2+radius*np.sin(np.deg2rad(8))+.13, radius+.13))
    font_path = Path('/root/autodl-tmp/data_preprocess_motion/staging/standing_v2/assets/msyh.ttc')
    title, small = ImageFont.truetype(str(font_path), 25), ImageFont.truetype(str(font_path), 19)
    renderer = WorldRenderer(body.faces)
    video_dir = args.output/'videos'
    video_dir.mkdir(exist_ok=True)
    manifest = dict(status='running', videos=[], camera=camera, renderer_sha256=sha(__file__),
        conversion_audit_sha256=sha(args.output/'conversion_audit.json'), fps=20,
        stages=stages, stage_labels=stage_labels,
        notes='Same input motion, same fixed floor and camera. Both fitted variants modify the display joints. No additional temporal filter.')
    try:
        for row, values in cases:
            name = row['name']
            video, poster = video_dir/(name+'.mp4'), video_dir/(name+'.png')
            proc = subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                '-s','1920x900','-r','20','-i','-','-an','-c:v','libx264','-threads','2',
                '-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
                stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            pixels_min = np.full(3, 10**9)
            try:
                for frame in range(120):
                    picture = Image.new('RGB', (1920,900), 'white')
                    draw = ImageDraw.Draw(picture)
                    draw.text((20,14), row['label']+' | '+name+' | %.2f 秒' % (frame/20),font=title,fill='#253c38')
                    draw.text((20,54), '输入描述：'+row['text'],font=small,fill='#52665f')
                    for column, (stage, label, (xyz,vertices,stats)) in enumerate(zip(stages,stage_labels,values)):
                        left = column*640
                        draw.text((left+20,94),label,font=title,fill='#337c87' if column == 0 else '#a35e5b')
                        image, count = renderer.draw(vertices[frame],xyz[frame,0],1,camera)
                        pixels_min[column] = min(pixels_min[column],count)
                        picture.paste(image,(left+10,140))
                        relative_hf = row['stages'][stage]['mesh']['above3hz_rms_m']*1000
                        draw.text((left+20,778),'高频表面位移 %.2f 毫米' % relative_hf,font=small,fill='#354c43')
                        draw.text((left+20,810),'骨长变异 %.2f%% | 滑步 %.1f%%' % (
                            stats['physics']['bone_length_cv']*100,stats['physics']['skating_frame_ratio']*100),font=small,fill='#354c43')
                        draw.text((left+20,842),'距原关节平均 %.1f 厘米' % (row['stages'][stage]['source_joint_error_mean_m']*100),font=small,fill='#65736a')
                    if frame == 60:
                        picture.save(poster)
                    proc.stdin.write(picture.tobytes())
                proc.stdin.close()
                error = proc.stderr.read().decode()
                if proc.wait() != 0:
                    raise RuntimeError(error)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            entry = dict(name=name,label=row['label'],text=row['text'],path='videos/'+video.name,poster='videos/'+poster.name,
                sha256=sha(video),poster_sha256=sha(poster),min_body_pixels=pixels_min.tolist(),width=1920,height=900,
                frames=120,fps=20,stats={stage: values[i][2] for i,stage in enumerate(stages)})
            manifest['videos'].append(entry)
            save(args.output/'videos.json',manifest)
            print('RENDERED',name,flush=True)
    finally:
        renderer.close()
    manifest['status'] = 'complete'
    save(args.output/'videos.json',manifest)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
