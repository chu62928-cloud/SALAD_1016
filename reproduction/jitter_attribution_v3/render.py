"""Synchronized unsmoothed views using the previously audited renderer and LBS."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'reproduction'),str(ROOT)]
from audit_recovery_v2 import FixedSMPLSurface,sha,save
from render_recovery_v2 import skeleton_image
from render_mesh import WorldRenderer
from physics_metrics import PARENTS

OUT=ROOT/'reproduction/artifacts/jitter_attribution_v3'
BODY=Path('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz')
FONT=Path('/root/autodl-tmp/data_preprocess_motion/staging/standing_v2/assets/msyh.ttc')


def archive(path):
    with np.load(path) as data:return {k:data[k].copy() for k in data.files}


def main():
    output=OUT/'media';output.mkdir(exist_ok=True)
    assert not (output/'complete.json').exists()
    subject=json.loads((OUT/'vae/selected_examples.json').read_text())['sample_ids'][0]
    sample=archive(OUT/'vae/examples'/f'data_self_ground_truth_-1_{subject}.npz')
    mean=archive(OUT/'vae/examples'/f'data_self_posterior_mean_0_{subject}.npz')
    posterior=archive(OUT/'vae/examples'/f'data_self_posterior_sample_0_{subject}.npz')
    gen=archive(OUT/'sampling/examples/humanml_probe_steps50_cfg7.5_20261016_walk.npz')
    variants=[archive(OUT/'generated/examples'/f'data_self_{name}_0_{subject}.npz') for name in ['official','adapted','paired']]
    cases=[
        dict(name='vae_reconstruction',title='真实走路与 VAE 重建 | '+subject,
            labels=['真实动作','VAE 后验均值重建','VAE 后验采样重建'],
            paths=[d['position_xyz'] for d in [sample,mean,posterior]]),
        dict(name='recovery_paths',title='同一 SALAD 生成：三条恢复路径 | a person walks forward.',
            labels=['位置通道直接恢复','仅固定骨长，保留骨段方向','独立 6D 旋转通道 → 固定骨架 FK'],
            paths=[gen[k+'_xyz'] for k in ['position','position_fixed','rotation_fk']]),
        dict(name='finetuning',title='同一文本、相同有效帧 | '+subject,
            labels=['原始 SALAD','数据微调','数据微调 + 对应体态条件'],
            paths=[d['position_xyz'] for d in variants])]
    body=FixedSMPLSurface(BODY)
    audit=body.audit()
    renderer=WorldRenderer(body.faces)
    font=ImageFont.truetype(str(FONT),23);small=ImageFont.truetype(str(FONT),19)
    records=[]
    try:
        for case in cases:
            points=case['paths'];frames=len(points[0]);assert all(len(x)==frames for x in points)
            surfaces=[]
            for xyz in points:
                vertices,_=body.skin(body.fit(xyz),xyz[:,0]);surfaces.append(vertices)
            roots=[xyz[:,0] for xyz in points]
            low,high,radius=float('inf'),-float('inf'),0.
            for xyz,root in zip(points+surfaces,roots+roots):
                local=xyz-root[:,None]*[1,0,1]
                low=min(low,float(local[...,1].min()));high=max(high,float(local[...,1].max()))
                radius=max(radius,float(np.linalg.norm(local[...,[0,2]],axis=-1).max()))
            camera=dict(ycenter=(high+low)/2,ymag=max((high-low)/2+radius*np.sin(np.deg2rad(8))+.15,radius+.15))
            # WorldRenderer retains camera nodes, so apply each case's common scale explicitly.
            for _,node,_ in renderer.scenes.values():node.camera.xmag=node.camera.ymag=camera['ymag']
            video=output/(case['name']+'.mp4');poster=video.with_suffix('.png')
            process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24',
                '-s','1920x1430','-r','20','-i','-','-an','-c:v','libx264','-threads','2',
                '-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
                stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            counts=np.full(6,10**9)
            try:
                for frame in range(frames):
                    image=Image.new('RGB',(1920,1430),'white');draw=ImageDraw.Draw(image)
                    draw.text((15,10),case['title']+' | %.2f 秒'%(frame/20),font=font,fill='#253f40')
                    for col,label in enumerate(case['labels']):
                        draw.text((col*640+12,52),label,font=small,fill='#253f40')
                        panel=skeleton_image(points[col][frame],roots[col][frame],camera,PARENTS)
                        pixels=int(((np.asarray(panel)[...,0].astype(float)-np.asarray(panel)[...,2])>18).sum())
                        counts[col]=min(counts[col],pixels)
                        image.paste(panel,(col*640+10,82))
                        surface,pixels=renderer.draw(surfaces[col][frame],roots[col][frame],1,camera)
                        counts[col+3]=min(counts[col+3],pixels)
                        image.paste(surface,(col*640+10,735))
                    draw.text((15,708),'上：22 关节骨架；下：沿用已核验的固定中性人体标准蒙皮，表面为几何估计。',font=small,fill='#445455')
                    draw.text((15,1380),'20 帧/秒；同一相机尺度，各列根轨迹跟随；无平滑、锁脚、逐帧贴地或逐帧身高缩放。',font=small,fill='#445455')
                    if frame==frames//2:image.save(poster)
                    process.stdin.write(image.tobytes())
                process.stdin.close();error=process.stderr.read().decode()
                assert process.wait()==0,error
            finally:
                if process.poll() is None:process.kill();process.wait()
            assert counts.min()>400
            decoded=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True)
            assert decoded.returncode==0,decoded.stderr.decode()
            records.append(dict(name=case['name'],title=case['title'],labels=case['labels'],frames=frames,fps=20,
                camera=camera,minimum_foreground_pixels=counts.tolist(),file=video.name,sha256=sha(video),
                poster=poster.name,poster_sha256=sha(poster),all_frames_decoded=True))
            print('RENDERED',case['name'],frames,flush=True)
    finally:renderer.close()
    save(output/'complete.json',dict(status='complete',videos=records,body_sha256=sha(BODY),body_audit=audit,
        source_sha256={name:sha(ROOT/'reproduction'/name) for name in ['audit_recovery_v2.py','render_recovery_v2.py','render_mesh.py']},
        camera_policy='Common scale and height; same horizontal-root-follow policy applied to each column. Recovery-path roots are exactly identical.',
        surface_method='Positions -> estimated outgoing-bone rotations -> standard neutral SMPL-H LBS; axial twist is not identifiable.',
        scientific_metrics_on_original_joints=True,temporal_smoothing=False))


if __name__=='__main__':main()
