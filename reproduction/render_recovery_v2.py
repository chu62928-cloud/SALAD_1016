"""Six synchronized diagnostic panels; skeleton panels never use SMPL."""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_key]='2'
os.environ.setdefault('PYOPENGL_PLATFORM','egl')

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from audit_recovery_v2 import sha,save


def skeleton_image(xyz, root, camera, parents):
    yaw,pitch=np.deg2rad([60,8])
    back=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    right=np.cross([0,1,0],back);right/=np.linalg.norm(right)
    up=np.cross(back,right)
    basis=np.stack([right,up,back],axis=1)
    center=np.array([root[0],camera['ycenter'],root[2]])
    def project(points):
        coords=(np.asarray(points)-center)@basis
        if np.abs(coords[...,:2]).max()>camera['ymag']:
            raise ValueError('Skeleton outside common camera')
        return np.stack([310+coords[...,0]*310/camera['ymag'],310-coords[...,1]*310/camera['ymag']],-1)
    image=Image.new('RGB',(620,620),(246,249,249))
    draw=ImageDraw.Draw(image)
    # Clip ground lines at image bounds; only joint projection asserts framing.
    for value in np.arange(-3,3.01,.5):
        for line in [np.array([[root[0]-3,0,root[2]+value],[root[0]+3,0,root[2]+value]]),
                     np.array([[root[0]+value,0,root[2]-3],[root[0]+value,0,root[2]+3]])]:
            coords=(line-center)@basis
            uv=np.stack([310+coords[:,0]*310/camera['ymag'],310-coords[:,1]*310/camera['ymag']],-1)
            draw.line([tuple(p) for p in uv],fill=(198,207,208),width=1)
    uv=project(xyz)
    for j in range(1,22):
        draw.line([tuple(uv[parents[j]]),tuple(uv[j])],fill=(169,61,57),width=6)
    for x,y in uv:draw.ellipse((x-4,y-4,x+4,y+4),fill=(30,85,95))
    return image


def run(args):
    sys.path.insert(0,str(args.root/'reproduction'))
    from render_mesh import WorldRenderer
    from mesh_surface import JointDrivenBody
    from diagnose_jitter import PARENTS
    audit=json.loads((args.output/'audit.json').read_text())
    if audit['status']!='complete':raise ValueError('Complete numerical audit required')
    stages=['position_xyz','original_vertices','position_smpl_vertices',
            'rotation_xyz','rotation_smpl_vertices','author_vertices']
    labels=['位置恢复：22 关节，无蒙皮','此前流程：变长骨架直接蒙皮','位置 → SMPL 旋转 → 固定骨长蒙皮',
            '旋转恢复：6D → FK 骨架','6D → FK → SMPL 旋转 → 蒙皮','作者 BVH 足部拟合 → 估计表面']
    caches=[]
    low,high,radius=float('inf'),-float('inf'),0.
    for row in audit['cases']:
        path=args.output/'server_only'/(row['name']+'.npz')
        if sha(path)!=row['sha256']:raise ValueError('Diagnostic cache changed')
        with np.load(path) as cache:values={k:cache[k].copy() for k in cache.files}
        for key in stages:
            v=values[key]-values['position_xyz'][:,:1]*[1,0,1]
            low=min(low,float(v[...,1].min()));high=max(high,float(v[...,1].max()))
            radius=max(radius,float(np.linalg.norm(v[...,[0,2]],axis=-1).max()))
        caches.append((row,values))
    camera=dict(ycenter=(low+high)/2,ymag=max((high-low)/2+radius*np.sin(np.deg2rad(8))+.15,radius+.15))
    body=JointDrivenBody('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz')
    renderer=WorldRenderer(body.faces)
    directory=args.output/'videos';directory.mkdir(exist_ok=True)
    font='/root/autodl-tmp/data_preprocess_motion/staging/standing_v2/assets/msyh.ttc'
    title=ImageFont.truetype(font,24);small=ImageFont.truetype(font,20)
    manifest=dict(status='running',videos=[],stages=stages,labels=labels,fps=20,width=1920,height=1460,
                  camera=camera,audit_sha256=sha(args.output/'audit.json'))
    try:
        for row,values in caches:
            name=row['name'];path=directory/(name+'.mp4');poster=directory/(name+'.png')
            command=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1460','-r','20',
                     '-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            minimum=np.full(6,10**9)
            try:
                for frame in range(120):
                    image=Image.new('RGB',(1920,1460),'white');draw=ImageDraw.Draw(image)
                    label='真实动作' if row['kind']=='real' else 'SALAD 生成'
                    draw.text((20,12),label+' | '+name+' | %.2f 秒'%(frame/20),font=title,fill='#263c39')
                    for i,key in enumerate(stages):
                        left=(i%3)*640;top=60+(i//3)*700
                        draw.text((left+12,top),labels[i],font=small,fill='#354d4a')
                        if key.endswith('_xyz'):
                            panel=skeleton_image(values[key][frame],values['position_xyz'][frame,0],camera,PARENTS)
                            count=int(((np.asarray(panel)[...,0].astype(float)-np.asarray(panel)[...,2])>18).sum())
                        else:
                            panel,count=renderer.draw(values[key][frame],values['position_xyz'][frame,0],1,camera)
                        minimum[i]=min(minimum[i],count)
                        if count<400:raise ValueError('Empty body/skeleton panel')
                        image.paste(panel,(left+10,top+40))
                        if i in [0,3]:
                            m=row['position' if i==0 else 'rotation']
                            text='骨长变异 %.3f%% | 高频骨长 %.3f%%'%(m['bone_cv_mean']*100,m['bone_above3hz_cv_mean']*100)
                        else:
                            m=row['stages'][{1:'original',2:'position_smpl',4:'rotation_smpl',5:'author_bvh'}[i]]
                            text='表面高频 %.3f 毫米 | 骨长变异 %.3f%%'%(m['surface_horizontal_follow']['above3hz_rms_m']*1000,m['joints']['bone_cv_mean']*100)
                        draw.text((left+12,top+666),text,font=small,fill='#4f625d')
                    if frame==60:image.save(poster)
                    process.stdin.write(image.tobytes())
                process.stdin.close();error=process.stderr.read().decode()
                if process.wait()!=0:raise RuntimeError(error)
            finally:
                if process.poll() is None:process.kill();process.wait()
            manifest['videos'].append(dict(name=name,kind=row['kind'],path='videos/'+path.name,poster='videos/'+poster.name,
                sha256=sha(path),poster_sha256=sha(poster),frames=120,min_pixels=minimum.tolist()))
            save(args.output/'videos.json',manifest)
            print('RENDERED',name,flush=True)
    finally:renderer.close()
    manifest['status']='complete';save(args.output/'videos.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
