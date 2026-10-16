"""Deterministic adapter training with identity-balanced sampling and exact resume."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .common import ROOT,config,seed_all,write_json,rng_state,restore_rng,sha
from .data import Data,prepare
from .runtime import runtime


def backbone_hash(model):
    digest=hashlib.sha256()
    for name,p in model.backbone.named_parameters():
        if '.posture_adapter.' not in name:
            digest.update(name.encode())
            digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class Engine:
    def __init__(self,data,model,scheduler,variant,seed,steps):
        self.data,self.model,self.scheduler=data,model,scheduler
        self.variant,self.seed,self.steps=variant,seed,steps
        self.cfg=config()
        self.step=0
        self.best=float('inf')
        self.optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=self.cfg['learning_rate'],weight_decay=self.cfg['weight_decay'])
        self.model.train()

    def update(self):
        cfg=self.cfg
        seed=self.seed+self.step*100003
        # All variants use the same subject, task, caption, posterior sample, t and noise.
        seed_all(seed)
        rng=np.random.RandomState(seed % 2**32)
        indices=self.data.sample_indices(rng,cfg['microbatch']*cfg['accumulation'])
        captions=[]
        for i in indices:
            choices=self.data.rows[i]['captions']
            caption=choices[rng.randint(len(choices))]
            captions.append('' if rng.rand()<cfg['text_drop'] else caption)
        donors=None
        if self.variant=='shuffled':
            epoch=self.step*len(indices)//5299
            donors=self.data.donor_map(self.seed+epoch)
        self.optimizer.zero_grad(set_to_none=True)
        step=self.step+1
        lr=cfg['learning_rate']*min(step/cfg['warmup'],1.)
        if step>cfg['warmup']:
            progress=(step-cfg['warmup'])/max(1,self.steps-cfg['warmup'])
            lr=1e-5+(cfg['learning_rate']-1e-5)*.5*(1+math.cos(math.pi*progress))
        for group in self.optimizer.param_groups:
            group['lr']=lr
        losses=[]
        for start in range(0,len(indices),cfg['microbatch']):
            ids=indices[start:start+cfg['microbatch']]
            z,mask,text=self.data.batch(ids,captions[start:start+len(ids)])
            noise=torch.randn_like(z)*mask[...,None,None]
            times=torch.randint(0,1000,(len(ids),),device='cuda')
            target=self.scheduler.get_velocity(z,noise,times)
            noisy=self.scheduler.add_noise(z,noise,times)
            core,shoulder=self.data.conditions(ids,self.variant,donors)
            prediction=self.model(noisy,times,text,core,shoulder,mask)
            loss=masked_error(prediction,target,mask).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            (loss/cfg['accumulation']).backward()
            losses.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        self.step=step
        return dict(step=step,loss=float(np.mean(losses)),gradient_norm=float(norm),lr=lr)

    def state(self):
        return dict(adapter=self.model.adapter_state(),optimizer=self.optimizer.state_dict(),
                    step=self.step,best=self.best,seed=self.seed,steps=self.steps,variant=self.variant,
                    rng=rng_state(),bindings=self.data.bindings,config_sha256=sha(ROOT/'configs/posture_v1.json'),
                    schedule='500-step linear warmup; cosine to 1e-5',
                    sample_position=self.step*self.cfg['microbatch']*self.cfg['accumulation'])

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp')
        torch.save(self.state(),temp)
        os.replace(temp,path)

    def load(self,path):
        s=torch.load(path,map_location='cpu')
        assert s['bindings']==self.data.bindings and s['seed']==self.seed and s['variant']==self.variant
        assert s['steps']==self.steps and s['config_sha256']==sha(ROOT/'configs/posture_v1.json')
        self.model.load_adapter(s['adapter'])
        self.optimizer.load_state_dict(s['optimizer'])
        self.step,self.best=s['step'],s['best']
        restore_rng(s['rng'])


def masked_error(prediction,target,mask):
    per_time=(prediction-target).square().mean((-1,-2))
    return (per_time*mask).sum(1)/mask.sum(1)


@torch.no_grad()
def denoise_evaluation(data,model,scheduler,split,variant,seed=930401,repeats=1,intervene=False):
    model.eval()
    indices=[i for i,r in enumerate(data.rows) if r['split']==split]
    donor=data.donor_map(seed,split) if intervene and split!='validation' else None
    results=[]
    for start in range(0,len(indices),32):
        ids=indices[start:start+32]
        errors=[]
        for repeat in range(repeats):
            # Per-sample seeds avoid dependence on batch order and model variant.
            mu=torch.from_numpy(np.array(data.mu[ids])).cuda()
            std=torch.from_numpy(np.array(data.std[ids])).cuda()
            zs=[];ns=[];ts=[]
            for row,i in enumerate(ids):
                gen=torch.Generator(device='cuda').manual_seed(seed+repeat*100003+i)
                zs.append(mu[row]+std[row]*torch.randn(mu[row].shape,device='cuda',generator=gen))
                ns.append(torch.randn(mu[row].shape,device='cuda',generator=gen))
                ts.append(torch.randint(0,1000,(1,),device='cuda',generator=gen))
            z=torch.stack(zs);noise=torch.stack(ns);times=torch.cat(ts)
            lengths=torch.tensor([data.rows[i]['length']//4 for i in ids],device='cuda')
            mask=torch.arange(49,device='cuda')[None]<lengths[:,None]
            z=z*mask[...,None,None];noise=noise*mask[...,None,None]
            target=scheduler.get_velocity(z,noise,times)
            noisy=scheduler.add_noise(z,noise,times)
            text=[data.rows[i]['captions'][repeat%len(data.rows[i]['captions'])] for i in ids]
            core,shoulder=data.conditions(ids,variant,donor)
            pred=model(noisy,times,text,core,shoulder,mask,scale=0 if variant=='original' else 1)
            errors.append(masked_error(pred,target,mask).cpu().numpy())
        errors=np.mean(errors,axis=0)
        for i,error in zip(ids,errors):
            r=data.rows[i]
            results.append(dict(sample_id=r['sample_id'],identity=r['identity_group'],
                                source=r['source_dataset'],error=float(error)))
    model.train()
    subjects={}
    for row in results:
        subjects.setdefault(row['identity'],[]).append(row['error'])
    return float(np.mean([np.mean(v) for v in subjects.values()])),results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['fixed','paired','shuffled'],required=True)
    parser.add_argument('--seed',type=int,default=20261017)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    seed_all(args.seed)
    data=Data()
    model,vae,scheduler=runtime()
    initial=backbone_hash(model)
    prepare(data,model,vae)
    del vae
    data.install_text_cache(model)
    engine=Engine(data,model,scheduler,args.variant,args.seed,args.steps)
    folder=ROOT/config()['run_root']/args.variant/str(args.seed)
    folder.mkdir(parents=True,exist_ok=True)
    if args.resume:
        engine.load(folder/'checkpoints/last.pt')
    elif (folder/'checkpoints/last.pt').exists():
        raise FileExistsError('Existing run retained; use --resume')
    started=time.time()
    write_json(folder/'bindings.json',dict(data=data.bindings,backbone_sha256=initial))
    write_json(folder/'source_binding.json',dict(
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'posture_conditioning').glob('*.py'))
               if p.name in ['train.py','adapter.py','data.py','features.py','runtime.py','common.py']},
        transformer_sha256=sha(ROOT/'models/denoiser/transformer.py'),
        config_sha256=sha(ROOT/'configs/posture_v1.json')))
    while engine.step<args.steps:
        entry=engine.update()
        if engine.step%100==0 or engine.step==1:
            entry.update(elapsed_sec=time.time()-started,variant=args.variant,status='training')
            write_json(folder/'status.json',entry)
            print(json.dumps(entry),flush=True)
        if engine.step%1000==0 or engine.step==args.steps:
            validation,rows=denoise_evaluation(data,model,scheduler,'validation',args.variant)
            entry['validation_subject_mean']=validation
            with (folder/'validation.jsonl').open('a') as f:
                f.write(json.dumps(entry)+'\n')
            if validation<engine.best:
                engine.best=validation
                engine.save(folder/'checkpoints/best.pt')
            engine.save(folder/'checkpoints/last.pt')
            print('validation',json.dumps(entry),flush=True)
    final=backbone_hash(model)
    assert final==initial,'Frozen backbone changed'
    write_json(folder/'status.json',dict(status='complete',step=engine.step,best_validation=engine.best,
                                        elapsed_sec=time.time()-started,backbone_sha256=final,
                                        frozen_backbone_unchanged=True))


if __name__=='__main__':
    main()
