"""Render recovered joints over a fixed world grid, without motion correction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio
import numpy as np

ROOT=Path(os.environ.get('SALAD_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT))
from utils.paramUtil import t2m_kinematic_chain


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(xyz,record,protocol,video,poster):
    fps=protocol['fps']
    colors=['#c7354a','#2468b4','#303c3b','#2468b4','#c7354a']
    figure=plt.figure(figsize=(11,7),dpi=100)
    figure.patch.set_facecolor('white')
    figure.suptitle(textwrap.fill(record['text'],width=90),fontsize=14,y=.97)
    figure.text(.5,.905,'SALAD | seed %d | %d fps | %.1f s'%(record['seed'],fps,len(xyz)/fps),ha='center',fontsize=10)
    panels=[]
    bottom=min(-.15,float(xyz[...,1].min())-.05)
    top=max(2.1,float(xyz[...,1].max())+.1)
    for left,azimuth,label in [(.01,-60,'View 1'),(.51,0,'View 2')]:
        axis=figure.add_axes([left,.30,.48,.58],projection='3d')
        axis.view_init(elev=15,azim=azimuth)
        axis.set_box_aspect((1,1,(top-bottom)/3.2))
        axis.set_axis_off()
        axis.set_title(label,fontsize=10,pad=0)
        lines=[axis.plot([],[],[],color=color,lw=2.8)[0] for color in colors]
        trace=axis.plot([],[],[],color='#65736f',lw=1,alpha=.5)[0]
        feet=axis.scatter([],[],[],s=26,depthshade=False)
        grid=Line3DCollection([],colors='#ced6d2',linewidths=.7)
        axis.add_collection3d(grid)
        panels.append((axis,lines,trace,feet,grid))
    seconds=np.arange(len(xyz))/fps
    toes=xyz[:,[10,11]]
    speed=np.linalg.norm(np.diff(toes[..., [0,2]],axis=0),axis=-1)*fps
    low=np.maximum(toes[:-1,:,1],toes[1:,:,1])<=protocol['contact_height_m']
    height_axis=figure.add_axes([.07,.09,.39,.14])
    speed_axis=figure.add_axes([.57,.09,.39,.14])
    for index,color in enumerate(['#c7354a','#2468b4']):
        height_axis.plot(seconds,toes[:,index,1]*100,color=color,lw=1.2,label=['Left toe','Right toe'][index])
        speed_axis.plot(seconds[:-1],np.where(low[:,index],speed[:,index],np.nan),color=color,lw=1.2)
    height_axis.axhline(protocol['contact_height_m']*100,color='#626b66',ls='--',lw=.8)
    height_axis.axhline(0,color='#626b66',lw=.8)
    speed_axis.axhline(protocol['slide_speed_mps'],color='#626b66',ls='--',lw=.8)
    height_axis.set_title('Toe height (cm)',fontsize=10)
    speed_axis.set_title('Low-height toe horizontal speed (m/s)',fontsize=10)
    for axis in [height_axis,speed_axis]:
        axis.set_xlim(0,len(xyz)/fps)
        axis.set_xlabel('Time (s)',fontsize=9)
        axis.tick_params(labelsize=8)
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    height_axis.legend(fontsize=7,loc='upper right',frameon=False)
    cursors=[axis.axvline(0,color='#111111',lw=.9) for axis in [height_axis,speed_axis]]
    with imageio.get_writer(str(video),fps=fps,codec='libx264',quality=8,macro_block_size=2) as writer:
        for frame in range(len(xyz)):
            center=xyz[frame,0,[0,2]]
            xmin,zmin=center-1.6
            xmax,zmax=center+1.6
            grid_segments=[]
            for value in np.arange(np.ceil(xmin*2)/2,xmax,.5):
                grid_segments.append([(value,zmin,0),(value,zmax,0)])
            for value in np.arange(np.ceil(zmin*2)/2,zmax,.5):
                grid_segments.append([(xmin,value,0),(xmax,value,0)])
            for axis,lines,trace,feet,grid in panels:
                axis.set_xlim(xmin,xmax)
                axis.set_ylim(zmin,zmax)
                axis.set_zlim(bottom,top)
                grid.set_segments(grid_segments)
                for chain,line in zip(t2m_kinematic_chain,lines):
                    points=xyz[frame,chain]
                    line.set_data(points[:,0],points[:,2])
                    line.set_3d_properties(points[:,1])
                history=xyz[max(0,frame-40):frame+1,0]
                trace.set_data(history[:,0],history[:,2])
                trace.set_3d_properties(np.zeros(len(history)))
                points=xyz[frame,[10,11]]
                feet._offsets3d=(points[:,0],points[:,2],points[:,1])
                feet.set_color(['#18866b' if y<=protocol['contact_height_m'] else '#818986' for y in points[:,1]])
            for cursor in cursors:cursor.set_xdata([seconds[frame],seconds[frame]])
            figure.canvas.draw()
            pixels=np.asarray(figure.canvas.buffer_rgba())[...,:3].copy()
            writer.append_data(pixels)
            if frame==len(xyz)//2:imageio.imwrite(str(poster),pixels)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    records=json.loads((output/'examples.json').read_text(encoding='utf-8'))
    videos=output/'videos'
    videos.mkdir(exist_ok=True)
    manifest={'display':'Original world-space Y-up metres; floor y=0, world grid 0.5m; cameras follow root horizontally; no floor translation, smoothing or foot locking',
              'renderer_sha256':sha(__file__),'videos':[]}
    for record in records:
        path=output/'arrays'/record['array']
        if sha(path)!=record['sha256']:raise ValueError('Example source changed')
        with np.load(path) as archive:xyz=archive['xyz']
        if not np.isfinite(xyz).all() or xyz.shape!=(protocol['example_frames'],22,3):raise ValueError('Unexpected example shape')
        name='%s_%d'%(record['id'],record['seed'])
        video,poster=videos/(name+'.mp4'),videos/(name+'.png')
        render(xyz,record,protocol,video,poster)
        manifest['videos'].append({'id':record['id'],'seed':record['seed'],'path':'videos/'+video.name,
                                   'poster':'videos/'+poster.name,'sha256':sha(video),'poster_sha256':sha(poster),
                                   'frames':len(xyz),'fps':protocol['fps'],'width':1100,'height':700})
        (output/'videos.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('Rendered',name,flush=True)
    manifest['status']='complete'
    (output/'videos.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
