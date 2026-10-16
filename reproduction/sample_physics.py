"""Generate fixed qualitative examples and paired validation physics diagnostics."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
ROOT=Path(os.environ.get('SALAD_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'reproduction'))
from physics_metrics import measure
from runtime import load_runtime
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
import numpy as np
import torch


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows,protocol):
    real={r['id']:r for r in rows if r['kind']=='real'}
    generated={name:[r for r in rows if r['kind']=='generated' and r['id']==name] for name in real}
    if len(real)!=protocol['reference_count'] or any(len(v)!=len(protocol['seeds']) for v in generated.values()):
        raise ValueError('Incomplete paired reference evaluation')
    keys=[k for k in next(iter(real.values())) if k not in ['id','kind','seed','frames','near_floor_intervals','toe_intervals']]
    result={}
    for key in keys:
        pairs=[]
        for name,reference in real.items():
            values=[r[key] for r in generated[name] if r[key] is not None]
            if reference[key] is not None and values:
                pairs.append((name,float(reference[key]),float(np.mean(values)),len(values)))
        if not pairs:
            result[key]={'valid_pairs':0,'real_mean':None,'generated_mean':None}
            continue
        reference=np.array([p[1] for p in pairs])
        prediction=np.array([p[2] for p in pairs])
        delta=prediction-reference
        clusters={}
        for i,pair in enumerate(pairs):clusters.setdefault(pair[0].lstrip('M'),[]).append(i)
        indices=list(clusters.values())
        sums=np.array([delta[index].sum() for index in indices])
        counts=np.array([len(index) for index in indices])
        rng=np.random.RandomState(protocol['bootstrap_seed'])
        picks=rng.randint(len(indices),size=(protocol['bootstrap_repetitions'],len(indices)))
        boot=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
        result[key]={'valid_pairs':len(pairs),'original_id_clusters':len(indices),
                     'generated_nonmissing_seed_observations':sum(p[3] for p in pairs),
                     'real_mean':float(reference.mean()),'generated_mean':float(prediction.mean()),
                     'difference':float(delta.mean()),'paired_ci95':np.quantile(boot,[.025,.975]).tolist()}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    args.protocol=args.protocol.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'summary.json').exists():raise RuntimeError('Use a new output directory, existing run retained')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    os.chdir(ROOT)
    torch.set_num_threads(2)
    audit=ROOT/'reproduction/artifacts/assets_t2m.json'
    audit_data=json.loads(audit.read_text())
    if audit_data['status']!='passed':raise ValueError('Prior asset audit required')
    assets={name:sha(ROOT/name) for name in audit_data['asset_files']}
    if any(assets[name]!=info['sha256'] for name,info in audit_data['asset_files'].items()):
        raise ValueError('Official assets changed')
    source=Path(__file__).parent
    binding={'status':'bound','created_at':datetime.now(timezone.utc).isoformat(),
             'core_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'auxiliary_commit':os.environ.get('SALAD_AUXILIARY_COMMIT'),
             'protocol_sha256':sha(args.protocol),'asset_audit_sha256':sha(audit),'assets':assets,
             'code_sha256':{name:sha(source/name) for name in ['sample_physics.py','physics_metrics.py']},
             'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)}
    write_json(output/'binding.json',binding)
    write_json(output/'protocol.json',protocol)
    status={'status':'running','pid':os.getpid(),'stage':'loading'}
    write_json(output/'status.json',status)
    opt,_,trainer,_,loader=load_runtime('t2m',seed=protocol['seeds'][0],steps=protocol['steps'],
                                     scale=protocol['guidance_scale'],split='val')
    trainer.vae.eval()
    dataset=loader.dataset
    mean,std=dataset.mean,dataset.std
    def recover(normalized):
        return recover_from_ric(torch.from_numpy(normalized*std+mean).float(),22).numpy()
    archive=output/'arrays'
    archive.mkdir(exist_ok=True)
    examples=[]
    for seed in protocol['seeds']:
        fixseed(seed)
        descriptions=[v['text'] for v in protocol['examples']]
        lengths=torch.full((len(descriptions),),protocol['example_frames'],dtype=torch.long)
        placeholder=torch.zeros(len(descriptions),protocol['example_frames'],263)
        pred,_=trainer.generate((descriptions,placeholder,lengths))
        pred=pred.cpu().numpy()
        for i,item in enumerate(protocol['examples']):
            xyz=recover(pred[i])
            path=archive/('%s_%d.npz'%(item['id'],seed))
            np.savez_compressed(path,normalized=pred[i],xyz=xyz)
            examples.append(dict(item,seed=seed,array=path.name,sha256=sha(path),metrics=measure(xyz,protocol)))
        write_json(output/'examples.json',examples)
    status.update(stage='reference_selection',examples=len(examples))
    write_json(output/'status.json',status)
    full_ids=set((ROOT/'dataset/humanml3d/val.txt').read_text().split())
    candidates=[(i,name) for i,name in enumerate(dataset.name_list[dataset.pointer:]) if name in full_ids]
    candidates.sort(key=lambda pair:hashlib.sha256((protocol['version']+':'+pair[1]).encode()).digest())
    chosen=candidates[:protocol['reference_count']]
    if len(chosen)!=protocol['reference_count']:raise ValueError('Insufficient validation references')
    reference=[]
    rows=[]
    stored={}
    for rank,(index,name) in enumerate(chosen):
        fixseed(900000+rank)
        _,_,caption,_,motion,length,_=dataset[index]
        xyz=recover(motion[:length])
        rows.append(dict(id=name,kind='real',seed=-1,**measure(xyz,protocol)))
        reference.append({'id':name,'text':caption,'frames':int(length),'loader_index':index,
                          'crop_caption_seed':900000+rank,
                          'source_motion_sha256':sha(ROOT/'dataset/humanml3d/new_joint_vecs'/(name+'.npy')),
                          'source_text_sha256':sha(ROOT/'dataset/humanml3d/texts'/(name+'.txt'))})
        stored['real_'+name]=xyz
    write_json(output/'reference.json',reference)
    for seed in protocol['seeds']:
        fixseed(seed)
        for start in range(0,len(reference),protocol['batch_size']):
            batch=reference[start:start+protocol['batch_size']]
            lengths=torch.tensor([v['frames'] for v in batch],dtype=torch.long)
            pred,_=trainer.generate(([v['text'] for v in batch],torch.zeros(len(batch),196,263),lengths))
            for item,features in zip(batch,pred.cpu().numpy()):
                xyz=recover(features[:item['frames']])
                rows.append(dict(id=item['id'],kind='generated',seed=seed,**measure(xyz,protocol)))
                stored['gen_%d_%s'%(seed,item['id'])]=xyz
            status.update(stage='paired_generation',seed=seed,generated=len(rows)-len(reference))
            write_json(output/'status.json',status)
            print(status,flush=True)
    paired_archive=archive/'validation_xyz.npz'
    np.savez_compressed(paired_archive,**stored)
    write_csv(output/'metrics.csv',rows)
    write_csv(output/'example_metrics.csv',[dict(id=v['id'],label=v['label'],seed=v['seed'],**v['metrics']) for v in examples])
    # Re-read retained world-coordinate arrays before accepting the diagnostic run.
    with np.load(paired_archive) as saved:
        for row in rows:
            key='real_'+row['id'] if row['kind']=='real' else 'gen_%d_%s'%(row['seed'],row['id'])
            measured=measure(saved[key],protocol)
            if any(measured[k]!=v for k,v in row.items() if k not in ['id','kind','seed']):
                raise ValueError('Retained array metric replay mismatch')
    summary={'status':'complete','completed_at':datetime.now(timezone.utc).isoformat(),
             'reference_clips':len(reference),'generated_clips':len(rows)-len(reference),'example_clips':len(examples),
             'metrics':aggregate(rows,protocol),'retained_array_metric_replay':'passed',
             'files_sha256':{name:sha(output/name) for name in ['protocol.json','binding.json','examples.json',
                             'reference.json','metrics.csv','example_metrics.csv','arrays/validation_xyz.npz']}}
    write_json(output/'summary.json',summary)
    if args.render:
        status.update(stage='rendering')
        write_json(output/'status.json',status)
        subprocess.run([sys.executable,str(source/'render_physics.py'),'--output',str(output)],check=True)
    status.update(status='complete',stage='complete')
    write_json(output/'status.json',status)
    print('Completed physics diagnostics',flush=True)


if __name__=='__main__':main()
