"""Render predetermined held-out examples from saved evaluation arrays."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
from render_mesh import WorldRenderer
from mesh_surface import JointDrivenBody
from posture_conditioning.common import config,sha,write_json


def main():
    evaluation=ROOT/config()['run_root']/'evaluation'/str(config()['seed'])
    source=np.load(evaluation/'intervention_swap.npz')
    ids=source['ids'].tolist()
    lengths=source['lengths'].tolist()
    manifest_rows=sorted([json.loads(l) for l in (Path(config()['data_root'])/'manifest.jsonl').read_text().splitlines()],key=lambda r:r['sample_id'])
    out=ROOT/config()['run_root']/'media'
    out.mkdir(parents=True,exist_ok=True)
    body_path=Path('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz')
    body=JointDrivenBody(body_path)
    font_path=Path('/root/autodl-tmp/data_preprocess_motion/staging/standing_v2/assets/msyh.ttc')
    title=ImageFont.truetype(str(font_path),25)
    small=ImageFont.truetype(str(font_path),19)
    variants=['original','fixed','paired']
    labels=['原 SALAD','固定中性条件适配','正确同人配对']
    arrays={v:{} for v in variants}
    array_bindings={}
    for variant in variants:
        for path in sorted((evaluation/variant/'0').glob('*.npz')):
            with np.load(path) as a:
                for k,i in enumerate(a['ids'].tolist()):
                    if i in ids:
                        arrays[variant][i]=a['xyz'][k,:int(a['lengths'][k])].copy()
                        array_bindings[variant+'/'+str(i)]=dict(file=str(path.relative_to(ROOT)),sha256=sha(path))
    assert all(set(arrays[v])==set(ids) for v in variants)
    renderer=WorldRenderer(body.faces)
    videos=[]
    try:
        for case,(i,length) in enumerate(zip(ids,lengths)):
            file=out/('case_%02d.mp4'%case)
            poster=file.with_suffix('.png')
            receipt=file.with_suffix('.json')
            if receipt.exists():
                entry=json.loads(receipt.read_text())
                assert sha(file)==entry['sha256'] and sha(poster)==entry['poster_sha256']
                videos.append(entry)
                continue
            meshes=[];low=float('inf');high=-float('inf');radius=0.
            for variant in variants:
                xyz=arrays[variant][i]
                vertices,error=body.motion(xyz)
                assert error<1e-6
                relative=vertices-xyz[:,:1]*[1,0,1]
                low=min(low,float(relative[...,1].min()))
                high=max(high,float(relative[...,1].max()))
                radius=max(radius,float(np.linalg.norm(relative[...,[0,2]],axis=-1).max()))
                meshes.append(vertices)
            camera=dict(ycenter=(low+high)/2,ymag=max((high-low)/2+radius*np.sin(np.deg2rad(8))+.13,radius+.13))
            if 1 in renderer.scenes:
                renderer.scenes[1][1].camera.xmag=camera['ymag']
                renderer.scenes[1][1].camera.ymag=camera['ymag']
            proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x880',
                '-r','20','-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','19',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(file)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            counts=np.full(3,10**9)
            try:
                for frame in range(length):
                    picture=Image.new('RGB',(1920,880),'white')
                    draw=ImageDraw.Draw(picture)
                    r=manifest_rows[i]
                    draw.text((20,12),r['identity_group']+' | %.2f 秒 | 固定种子 %d'%(frame/20,20261101+i),font=title,fill='#253C38')
                    for column,(variant,label,vertices) in enumerate(zip(variants,labels,meshes)):
                        left=column*640
                        draw.text((left+20,58),label,font=title,fill='#355E6B')
                        image,count=renderer.draw(vertices[frame],arrays[variant][i][frame,0],1,camera)
                        counts[column]=min(counts[column],count)
                        picture.paste(image,(left+10,108))
                    caption=(Path(config()['data_root'])/r['text']).read_text().splitlines()[0].split('#')[0]
                    for line,value in enumerate(textwrap.wrap(caption,width=130)):
                        draw.text((20,750+line*27),value,font=small,fill='#42534F')
                    if frame==length//2:
                        picture.save(poster)
                    proc.stdin.write(picture.tobytes())
                proc.stdin.close()
                error=proc.stderr.read().decode()
                if proc.wait()!=0:
                    raise RuntimeError(error)
            finally:
                if proc.poll() is None:
                    proc.kill();proc.wait()
            entry=dict(case=case,sample_id=manifest_rows[i]['sample_id'],identity=manifest_rows[i]['identity_group'],
                       text=caption,seed=20261101+i,frames=length,fps=20,width=1920,height=880,
                       path=file.name,poster=poster.name,sha256=sha(file),poster_sha256=sha(poster),
                       min_body_pixels=counts.tolist(),camera=camera)
            write_json(receipt,entry)
            videos.append(entry)
            print('rendered',case,flush=True)
    finally:
        renderer.close()
    write_json(out/'videos.json',dict(status='complete',videos=videos,selection='first test clip of first two identities per source',
               processing='original evaluation joints; no smoothing, IK, floor adjustment or foot locking',
               body_model_sha256=sha(body_path),source_bindings=array_bindings,renderer_sha256=sha(__file__)))


if __name__=='__main__':
    main()
