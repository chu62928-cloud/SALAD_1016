"""Frozen held-out-subject evaluation and actual input interventions."""
import argparse
from collections import defaultdict
import copy
import csv
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
from .common import ROOT,config,seed_all,sha,write_json
from .data import Data
from .runtime import runtime
from .sample import generate
from .train import denoise_evaluation,backbone_hash
from reproduction.physics_metrics import measure
from reproduction.diagnose_jitter import high_frequency
from utils.motion_process import recover_from_ric

VARIANTS=['original','fixed','paired','shuffled']
PHYSICS=dict(fps=20,contact_height_m=.05,slide_speed_mps=.2,penetration_tolerance_m=.02,
             sensitivity_heights_m=[.03,.05,.1],sensitivity_speeds_mps=[.1,.2,.3])


def csv_write(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def flat_task(task):
    task=task.lower()
    return any(w in task for w in ['walk','gait']) and not any(w in task for w in
            ['stair','step','ramp','slope','uphill','downhill','incline','decline'])


def physical(xyz):
    result=measure(xyz,PHYSICS)
    result.update(root_height_hf_m=float(np.sqrt(np.mean(high_frequency(xyz[:,0,1])**2))),
                  relative_hf_m=float(np.sqrt(np.mean(high_frequency(xyz-xyz[:,:1])**2))))
    return result


def statistics(gt,text,motion,seed):
    from utils.metrics import calculate_frechet_distance
    fid=calculate_frechet_distance(gt.mean(0),np.cov(gt,rowvar=False),
                                   motion.mean(0),np.cov(motion,rowvar=False))
    rng=np.random.RandomState(seed)
    a=rng.choice(len(motion),300,replace=False)
    b=rng.choice(len(motion),300,replace=False)
    return dict(fid=float(fid),diversity=float(np.linalg.norm(motion[a]-motion[b],axis=-1).mean()))


class Evaluation:
    def __init__(self,seed):
        from models.t2m_eval_wrapper import EvaluatorModelWrapper
        from utils.get_opt import get_opt
        from utils.word_vectorizer import WordVectorizer
        self.seed=seed
        seed_all(seed)
        self.data=Data()
        self.data.open_cache()
        self.model,self.vae,self.scheduler=runtime()
        self.data.install_text_cache(self.model)
        self.initial={k:v.detach().cpu().clone() for k,v in self.model.adapter_state().items()}
        self.wrapper=EvaluatorModelWrapper(get_opt(str(ROOT/'checkpoints/t2m/Comp_v6_KLD005/opt.txt'),torch.device('cuda')))
        self.vectorizer=WordVectorizer(str(ROOT/'glove'),'our_vab')
        self.mean=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
        self.std=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
        self.out=ROOT/config()['run_root']/'evaluation'/str(seed)
        self.out.mkdir(parents=True,exist_ok=True)

    def load(self,variant):
        self.model.load_adapter(self.initial)
        self.model.eval()
        if variant=='original':
            return dict(variant=variant,checkpoint=None)
        path=ROOT/config()['run_root']/variant/str(self.seed)/'checkpoints/best.pt'
        status=json.loads((path.parents[1]/'status.json').read_text())
        assert status['status']=='complete' and status['step']==config()['steps']
        state=torch.load(path,map_location='cpu')
        assert state['bindings']==self.data.bindings
        self.model.load_adapter(state['adapter'])
        return dict(variant=variant,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),
                    selected_step=state['step'],validation=state['best'])

    def embed(self,ids,motions):
        tokens=[];words=[];positions=[];cap_lens=[]
        for i in ids:
            t=['sos/OTHER']+self.data.rows[i]['tokens'][0][:20]+['eos/OTHER']
            cap_lens.append(len(t))
            t=t+['unk/OTHER']*(22-len(t))
            pairs=[self.vectorizer[x] for x in t]
            words.append(np.stack([p[0] for p in pairs]))
            positions.append(np.stack([p[1] for p in pairs]))
        order=np.argsort(-np.array(cap_lens),kind='stable')
        ids=[ids[k] for k in order]
        lengths=torch.tensor([self.data.rows[i]['length'] for i in ids],device='cuda')
        text,motion=self.wrapper.get_co_embeddings(torch.tensor(np.array(words)[order],device='cuda'),
                   torch.tensor(np.array(positions)[order],device='cuda'),
                   torch.tensor(np.array(cap_lens)[order],device='cuda'),motions[order],lengths)
        aligned=np.argsort(lengths.cpu().numpy())[::-1].copy()
        return text.cpu().numpy(),motion.cpu().numpy(),[ids[k] for k in aligned]

    def generation(self,variant):
        target=self.out/variant
        target.mkdir(exist_ok=True)
        summary=target/'generation.json'
        if summary.exists():
            return json.loads(summary.read_text())
        rows=self.data.rows
        test=[i for i,r in enumerate(rows) if r['split']=='test']
        all_metrics=[]
        for repeat in range(config()['evaluation_repeats']):
            folder=target/str(repeat)
            folder.mkdir(exist_ok=True)
            indices=np.random.RandomState(930401+repeat).permutation(test).tolist()
            for start in range(0,len(indices),32):
                file=folder/('%04d.npz'%start)
                if file.exists():
                    continue
                ids=indices[start:start+32]
                lengths=[rows[i]['length'] for i in ids]
                core,shoulder=self.data.conditions(ids,variant)
                text=[rows[i]['captions'][0] for i in ids]
                seeds=[20261101+repeat*100003+i for i in ids]
                generated=generate(self.model,self.vae,self.scheduler,text,lengths,core,shoulder,seeds,
                                   scale=0 if variant=='original' else 1.)
                gt=np.zeros((len(ids),generated.shape[1],263),dtype='float32')
                p_rows=[]
                xyzs=[]
                for k,i in enumerate(ids):
                    gt[k,:lengths[k]]=(self.data.motion(i)-self.mean)/self.std
                    xyz=recover_from_ric((generated[k,:lengths[k]]*torch.tensor(self.std,device='cuda')+
                                           torch.tensor(self.mean,device='cuda')).float(),22).cpu().numpy()
                    p=physical(xyz)
                    r=rows[i]
                    p_rows.append(dict(sample_id=r['sample_id'],identity=r['identity_group'],source=r['source_dataset'],
                                       flat_ground=flat_task(r['task']),repeat=repeat,**p))
                    padded=np.zeros((196,22,3),dtype='float32');padded[:len(xyz)]=xyz;xyzs.append(padded)
                embedding_text,embedding_motion,aligned=self.embed(ids,generated)
                _,embedding_gt,aligned_gt=self.embed(ids,torch.tensor(gt,device='cuda'))
                assert aligned==aligned_gt
                distance=cdist(embedding_text.astype('float64'),embedding_motion.astype('float64'))
                ranks=np.argsort(distance,axis=1,kind='stable')
                hit=(ranks[:,:3]==np.arange(len(ids))[:,None]).any(1)
                np.savez_compressed(file,ids=np.array(ids),aligned=np.array(aligned),lengths=np.array(lengths),
                                    text=embedding_text,motion=embedding_motion,gt=embedding_gt,
                                    r3=hit,matching=np.diag(distance),xyz=np.stack(xyzs),
                                    normalized_motion=generated.cpu().numpy())
                write_json(file.with_suffix('.json'),p_rows)
                write_json(self.out/'status.json',dict(status='generation',variant=variant,repeat=repeat,
                                                       completed=min(start+32,len(indices)),total=len(indices)))
                print(variant,'generation',repeat,start,len(indices),flush=True)
            files=sorted(folder.glob('*.npz'))
            gt=[];text=[];motion=[];hits=[];matching=[];phys=[]
            for f in files:
                with np.load(f) as a:
                    gt.append(a['gt']);text.append(a['text']);motion.append(a['motion'])
                    hits.extend(a['r3'].tolist());matching.extend(a['matching'].tolist())
                phys+=json.loads(f.with_suffix('.json').read_text())
            gt=np.concatenate(gt);text=np.concatenate(text);motion=np.concatenate(motion)
            assert len(gt)==len(test)
            metric=statistics(gt,text,motion,930401+repeat)
            metric.update(repeat=repeat,r3=float(np.mean(hits)),matching=float(np.mean(matching)))
            subject_metrics={}
            for key in ['bone_length_cv','root_height_hf_m','relative_hf_m','skating_frame_ratio',
                        'near_floor_toe_interval_fraction','airborne_frame_ratio','penetration_frame_ratio']:
                grouped=defaultdict(list)
                for p in phys:
                    if key in ['skating_frame_ratio','near_floor_toe_interval_fraction','airborne_frame_ratio',
                               'penetration_frame_ratio'] and not p['flat_ground']:
                        continue
                    grouped[p['identity']].append(p[key])
                subject_metrics[key]=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None
            metric.update(subject_metrics)
            metric['flat_ground_clips']=sum(p['flat_ground'] for p in phys)
            all_metrics.append(metric)
            csv_write(folder/'physics.csv',phys)
        result=dict(variant=variant,seed=self.seed,repeats=all_metrics,
                    mean={k:float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]
                          if k not in ['repeat'] and all(m[k] is not None for m in all_metrics)})
        write_json(summary,result)
        return result

    def interventions(self):
        self.load('paired')
        rows=self.data.rows
        # Predetermined coverage: first test clip of each of the first two identities
        # per source. Selection uses identifiers only, never generated appearance.
        identities=defaultdict(list)
        for identity in self.data.split_ids['test']:
            identities[identity.split(':')[0]].append(identity)
        ids=[self.data.groups[x][0] for source in sorted(identities) for x in identities[source][:2]]
        core,shoulder=self.data.conditions(ids)
        donors=self.data.donor_map(930401,'test')
        wrong_core,wrong_shoulder=self.data.conditions(ids,donors=donors)
        text=[rows[i]['captions'][0] for i in ids]
        lengths=[rows[i]['length'] for i in ids]
        seeds=[20261101+i for i in ids]
        base=generate(self.model,self.vae,self.scheduler,text,lengths,core,shoulder,seeds,capture=True)
        diagnostics={}
        for layer,block in enumerate(self.model.blocks):
            diagnostics[str(layer)]=block.posture_adapter.diagnostics
        torch.save(diagnostics,self.out/'attention_gate_residual.pt')
        changes={'swap':(wrong_core,wrong_shoulder)}
        for group,joints in [('pelvis',[0,1,2]),('spine',[3,6,9]),('left_leg',[1,4,7,10]),('right_leg',[2,5,8,11])]:
            modified=core.clone();modified[:,joints]=0
            changes[group]=(modified,shoulder)
        changes['shoulders']=(core,torch.zeros_like(shoulder))
        result=[]
        for group,(c,s) in changes.items():
            other=generate(self.model,self.vae,self.scheduler,text,lengths,c,s,seeds)
            for k,i in enumerate(ids):
                xyz0=recover_from_ric((base[k,:lengths[k]]*torch.tensor(self.std,device='cuda')+
                                       torch.tensor(self.mean,device='cuda')).float(),22).cpu().numpy()
                xyz1=recover_from_ric((other[k,:lengths[k]]*torch.tensor(self.std,device='cuda')+
                                       torch.tensor(self.mean,device='cuda')).float(),22).cpu().numpy()
                delta=np.linalg.norm(xyz1-xyz0,axis=-1)
                for phase,part in enumerate(np.array_split(np.arange(len(delta)),4)):
                    result.append(dict(sample_id=rows[i]['sample_id'],group=group,quarter=phase,
                                       mean_joint_change_m=float(delta[part].mean()),
                                       pelvis_change_m=float(delta[part,0].mean()),
                                       trunk_change_m=float(delta[part][:,[3,6,9]].mean()),
                                       legs_change_m=float(delta[part][:,[1,2,4,5,7,8,10,11]].mean())))
            np.savez_compressed(self.out/('intervention_'+group+'.npz'),ids=ids,lengths=lengths,
                                baseline=base.cpu().numpy(),changed=other.cpu().numpy())
        csv_write(self.out/'interventions.csv',result)

    def run(self):
        bindings=[]
        original_hash=backbone_hash(self.model)
        for variant in VARIANTS:
            binding=self.load(variant)
            bindings.append(binding)
            target=self.out/variant
            target.mkdir(exist_ok=True)
            if not (target/'denoise.json').exists():
                mean,results=denoise_evaluation(self.data,self.model,self.scheduler,'test',variant,
                                               repeats=config()['evaluation_repeats'])
                csv_write(target/'denoise.csv',results)
                write_json(target/'denoise.json',dict(subject_mean=mean,repeats=config()['evaluation_repeats']))
            self.generation(variant)
        self.interventions()
        assert original_hash==backbone_hash(self.model)
        write_json(self.out/'bindings.json',dict(models=bindings,data=self.data.bindings,
                   evaluation_source_sha256=sha(__file__),sample_source_sha256=sha(ROOT/'posture_conditioning/sample.py')))
        write_json(self.out/'status.json',dict(status='complete',seed=self.seed,
                                              frozen_backbone_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=20261017)
    args=parser.parse_args()
    Evaluation(args.seed).run()
