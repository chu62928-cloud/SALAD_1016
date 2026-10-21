"""Read-only layer-by-layer SALAD jitter attribution on frozen sequences."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent),str(ROOT/'reproduction'),str(ROOT)]
from metrics import PARENTS,measures,fixed_directions,paired_error,vector_rms
from utils.motion_process import recover_from_ric,recover_from_rot,recover_root_rot_pos
from common.skeleton import Skeleton
from common.quaternion import cont6d_to_matrix,quaternion_to_cont6d
from utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
from utils.get_opt import get_opt
from models.vae.model import VAE

OUT = ROOT/'reproduction/artifacts/jitter_attribution_v3'
FULL = ROOT/'reproduction/artifacts/posture_after_ft_full_v2'
PHYSICS = ROOT/'reproduction/artifacts/physics_v1_20261016'
PROTOCOL = json.loads((Path(__file__).parent/'protocol.json').read_text())
PATHS = PROTOCOL['recovery_paths']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): digest.update(block)
    return digest.hexdigest()


def save(path,data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


class Campaign:
    def __init__(self,stage):
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.stage = stage
        self.output = OUT/stage
        assert not (self.output/'complete.json').exists(), 'Completed diagnostic is immutable'
        assert not (self.output/'clip_metrics.csv').exists(), 'Partial diagnostic exists; preserve it and investigate before retrying'
        self.output.mkdir(parents=True,exist_ok=True)
        self.started = time.time()
        self.inputs = {}
        self.checks = dict(position_cache_max_m=0.,position_independent_max_m=0.,fk_independent_max_m=0.,
            fk_root_difference_max_m=0.,fixed_direction_root_difference_max_m=0.,fixed_length_max_m=0.)
        self.manifest = [json.loads(line) for line in (FULL/'full_data_root/manifest.jsonl').read_text().splitlines()]
        self.manifest.sort(key=lambda row:row['sample_id'])
        self.target = [dict(row,index=index) for index,row in enumerate(self.manifest) if row['split']=='test']
        assert len(self.target)==1327 and len({r['identity_group'] for r in self.target})==57
        self.target_by_index = {row['index']:row for row in self.target}
        self.freeze(FULL/'full_data_root/manifest.jsonl')
        self.freeze(PHYSICS/'reference.json')
        self.freeze(PHYSICS/'protocol.json')
        self.freeze(FULL/'adaptation/backbone_binding.json')
        self.freeze(ROOT/'posture_after_ft_full_v2/config.json')
        frozen = json.loads((FULL/'full_campaign_status.json').read_text())['sources']
        assert len(frozen)==30
        for name,digest in frozen.items(): assert self.freeze(ROOT/name)==digest
        for name in ['utils/motion_process.py','common/skeleton.py','common/quaternion.py',
                     'models/vae/model.py','models/vae/encdec.py','models/vae/trainer.py',
                     'models/denoiser/trainer.py','reproduction/physics_metrics.py']:
            self.freeze(ROOT/name)
        self.mean = np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
        self.std = np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
        for name in ['mean.npy','std.npy']: self.freeze(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta'/name)
        refpath = ROOT/'dataset/humanml3d/new_joint_vecs/000021.npy'
        self.freeze(refpath)
        ref = recover_from_ric(torch.from_numpy(np.load(refpath)).float(),22).numpy()
        self.skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets),t2m_kinematic_chain,'cpu')
        self.offsets = self.skeleton.get_offsets_joints(torch.from_numpy(ref[0])).numpy()
        save(self.output/'skeleton.json',dict(offsets=self.offsets.tolist(),parents=PARENTS.tolist(),
            rest_leg_length_m=float(np.mean(np.linalg.norm(self.offsets[[4,5]],axis=-1)+np.linalg.norm(self.offsets[[7,8]],axis=-1))),
            reference_sha256=sha(refpath),height_rescaling=False,post_processing=False))
        candidates = [r for r in self.target if 'walk' in r['original_task'].lower() or 'gait' in r['original_task'].lower()]
        candidates = sorted(candidates,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
        self.selected = {r['sample_id'] for r in candidates[:2]}
        if not self.selected: self.selected = {self.target[0]['sample_id']}
        save(self.output/'selected_examples.json',dict(rule='First two walk/gait held-out IDs in SHA256 order, before computing diagnostics.',sample_ids=sorted(self.selected)))
        self.clip_rows = []
        self.handles = []
        self.writers = {}
        for label in ['bone','joint','clip']:
            stream = (self.output/(label+'_metrics.csv')).open('w',encoding='utf-8',newline='')
            self.handles.append(stream)
            self.writers[label] = [stream,None]

    def freeze(self,path):
        path = Path(path); value=sha(path); self.inputs[str(path)]=value; return value

    def load(self,path):
        content = Path(path).read_bytes()
        self.inputs[str(path)] = hashlib.sha256(content).hexdigest()
        return np.load(io.BytesIO(content))

    def row(self,label,record):
        stream,writer = self.writers[label]
        if writer is None:
            writer = csv.DictWriter(stream,fieldnames=list(record)); writer.writeheader(); self.writers[label][1]=writer
        writer.writerow(record)

    def paths(self,raw,cached=None):
        x = torch.as_tensor(raw,dtype=torch.float32)
        position = recover_from_ric(x,22).numpy()
        rotation = recover_from_rot(x,22,self.skeleton).numpy()
        fixed = fixed_directions(position,self.offsets)
        if cached is not None:
            self.checks['position_cache_max_m'] = max(self.checks['position_cache_max_m'],float(np.abs(position-cached).max()))
            assert np.abs(position-cached).max()<2e-5
        q,root = recover_root_rot_pos(x)
        six = torch.cat([quaternion_to_cont6d(q),x[:,67:193]],-1).reshape(-1,22,6)
        matrices = cont6d_to_matrix(six).numpy()
        angles = np.r_[0.,np.cumsum(np.asarray(raw[:-1,0],dtype=float))]
        inverse = Rotation.from_euler('y',-2*angles).as_matrix()
        independent = np.einsum('tab,tjb->tja',inverse,np.asarray(raw[:,4:67]).reshape(-1,21,3))
        independent[...,[0,2]] += root.numpy()[:,None,[0,2]]
        independent = np.concatenate([root.numpy()[:,None],independent],1)
        expected = np.zeros_like(rotation); expected[:,0]=root.numpy()
        for chain in t2m_kinematic_chain:
            total=matrices[:,0]
            for j,p in zip(chain[1:],chain[:-1]):
                total=total @ matrices[:,j]
                expected[:,j] = expected[:,p]+np.einsum('tab,b->ta',total,self.offsets[j])
        self.checks['position_independent_max_m']=max(self.checks['position_independent_max_m'],float(np.abs(position-independent).max()))
        self.checks['fk_independent_max_m']=max(self.checks['fk_independent_max_m'],float(np.abs(rotation-expected).max()))
        self.checks['fk_root_difference_max_m']=max(self.checks['fk_root_difference_max_m'],float(np.abs(position[:,0]-rotation[:,0]).max()))
        self.checks['fixed_direction_root_difference_max_m']=max(self.checks['fixed_direction_root_difference_max_m'],float(np.abs(position[:,0]-fixed[:,0]).max()))
        for points in [fixed,rotation]:
            lengths=np.linalg.norm(points[:,1:]-points[:,PARENTS[1:]],axis=-1)
            self.checks['fixed_length_max_m']=max(self.checks['fixed_length_max_m'],float(np.abs(lengths-np.linalg.norm(self.offsets[1:],axis=-1)).max()))
        assert self.checks['position_independent_max_m']<3e-5
        assert self.checks['fk_independent_max_m']<2e-6
        assert self.checks['fixed_length_max_m']<3e-6
        return dict(position=position,position_fixed=fixed,rotation_fk=rotation)

    def analyze(self,raw,meta,cached=None,gt=None,store=False):
        paths=self.paths(raw,cached)
        series={}
        error_keys = ['mpjpe_world_m','mpjpe_root_aligned_m','acceleration_error_world_mps2','acceleration_error_local_mps2','jerk_error_local_mps3']
        discrepancy = paired_error(paths['rotation_fk'],paths['position'])
        six=np.asarray(raw[:,67:193],dtype=np.float64).reshape(-1,21,6)
        axis1,axis2=six[...,:3],six[...,3:]
        norm1=np.linalg.norm(axis1,axis=-1)
        perpendicular=axis2-(axis1*axis2).sum(-1,keepdims=True)*axis1/np.maximum(norm1[...,None]**2,1e-30)
        six_diagnostics=dict(rotation6d_first_axis_min_norm=float(norm1.min()),
            rotation6d_perpendicular_min_norm=float(np.linalg.norm(perpendicular,axis=-1).min()))
        for name,xyz in paths.items():
            values,bones,joints,trace=measures(xyz,PROTOCOL)
            errors = paired_error(xyz,gt[name]) if gt is not None else {key:None for key in error_keys}
            row=dict(meta,path=name,**values,**errors,position_rotation_discrepancy_m=discrepancy['mpjpe_root_aligned_m'],
                     position_rotation_acceleration_discrepancy_mps2=discrepancy['acceleration_error_local_mps2'],**six_diagnostics)
            self.row('clip',row); self.clip_rows.append(row)
            for bone in bones: self.row('bone',dict(meta,path=name,**bone))
            for joint in joints: self.row('joint',dict(meta,path=name,**joint))
            series.update({name+'_'+key:value for key,value in trace.items()})
        if store or meta['sample_id'] in self.selected:
            filename=f"{meta['dataset']}_{meta['variant']}_{meta['repeat']}_{meta['sample_id']}.npz"
            folder=self.output/'examples';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/filename,raw=raw,**{key+'_xyz':value for key,value in paths.items()},**series)
        return paths,series

    def store_series(self,name,metadata,series):
        folder=self.output/'server_only';folder.mkdir(exist_ok=True)
        arrays={}
        for key in series[0]:
            size=max(len(item[key]) for item in series)
            values=np.full((len(series),size,series[0][key].shape[-1]),np.nan,dtype=np.float32)
            for index,item in enumerate(series): values[index,:len(item[key])]=item[key]
            arrays[key]=values
        np.savez_compressed(folder/(name+'.npz'),sample_ids=np.asarray([r['sample_id'] for r in metadata]),
            lengths=np.asarray([r['frames'] for r in metadata]),**arrays)

    def finish(self,extra=None):
        for handle in self.handles: handle.close()
        assert all(sha(path)==digest for path,digest in self.inputs.items()),'Frozen input changed'
        save(self.output/'input_binding.json',self.inputs)
        outputs={str(p.relative_to(self.output)):dict(bytes=p.stat().st_size,sha256=sha(p))
                 for p in self.output.rglob('*') if p.is_file() and p.name not in ['complete.json']}
        save(self.output/'complete.json',dict(status='complete',stage=self.stage,rows=len(self.clip_rows),
            elapsed_seconds=time.time()-self.started,checks=self.checks,input_count=len(self.inputs),
            source_sha256={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
            files=outputs,extra=extra or {},training=False))
        print('COMPLETE',self.stage,len(self.clip_rows),round(time.time()-self.started,1),flush=True)


def generated():
    campaign=Campaign('generated')
    for variant in PROTOCOL['target_generated_variants']:
        for repeat in PROTOCOL['generation_repeats']:
            folder=FULL/'evaluation/20261017'/variant/str(repeat)
            seen=set()
            for bi,path in enumerate(sorted(folder.glob('[0-9][0-9][0-9][0-9].npz'))):
                metadata=[];traces=[]
                with campaign.load(path) as archive:
                    ids,lengths=archive['ids'],archive['lengths']
                    normalized,xyz=archive['normalized_motion'],archive['xyz']
                    for k,index in enumerate(ids):
                        record=campaign.target_by_index[int(index)]
                        assert int(index) not in seen;seen.add(int(index))
                        length=int(lengths[k]);assert 40<=length<=196 and length%4==0
                        raw=normalized[k,:length]*campaign.std+campaign.mean
                        meta=dict(dataset='data_self',variant=variant,repeat=repeat,sample_id=record['sample_id'],
                            subject=record['identity_group'],source=record['source_dataset'],task=record['original_task'],
                            frames=length)
                        _,trace=campaign.analyze(raw,meta,cached=xyz[k,:length])
                        metadata.append(meta);traces.append(trace)
                campaign.store_series(f'{variant}_{repeat}_{bi:04}',metadata,traces)
                if bi%10==0: print('generated',variant,repeat,len(seen),flush=True)
            assert len(seen)==1327
    # Preserve the original-domain 128 x 3 generation position-only reference.
    reference=json.loads((PHYSICS/'reference.json').read_text())
    with campaign.load(PHYSICS/'arrays/validation_xyz.npz') as archive:
        for record in reference:
            for seed in [20261016,20261017,20261018]:
                p=archive[f"gen_{seed}_{record['id']}"]
                values,bones,joints,_=measures(p,PROTOCOL)
                meta=dict(dataset='humanml',variant='official_cached',repeat=seed,sample_id=record['id'],
                    subject=record['id'].lstrip('M'),source='HumanML3D',task=record['text'],frames=len(p))
                errors={key:None for key in ['mpjpe_world_m','mpjpe_root_aligned_m','acceleration_error_world_mps2','acceleration_error_local_mps2','jerk_error_local_mps3']}
                row=dict(meta,path='position',**values,**errors,position_rotation_discrepancy_m=None,
                         position_rotation_acceleration_discrepancy_mps2=None,
                         rotation6d_first_axis_min_norm=None,rotation6d_perpendicular_min_norm=None)
                campaign.row('clip',row);campaign.clip_rows.append(row)
                for bone in bones: campaign.row('bone',dict(meta,path='position',**bone))
                for joint in joints: campaign.row('joint',dict(meta,path='position',**joint))
    assert len(campaign.clip_rows)==1327*3*5*3+384
    campaign.finish()


def hml_references(campaign):
    reference=json.loads((PHYSICS/'reference.json').read_text())
    result=[]
    with campaign.load(PHYSICS/'arrays/validation_xyz.npz') as archive:
        for record in reference:
            file=ROOT/'dataset/humanml3d/new_joint_vecs'/(record['id']+'.npy')
            assert campaign.freeze(file)==record['source_motion_sha256']
            full=np.load(file);saved=archive['real_'+record['id']];length=len(saved)
            best=(float('inf'),None)
            for start in range(0,len(full)-length+1,32):
                offsets=list(range(start,min(start+32,len(full)-length+1)))
                values=np.stack([full[left:left+length] for left in offsets])
                recovered=recover_from_ric(torch.from_numpy(values).float(),22).numpy()
                errors=np.abs(recovered-saved).max(axis=(1,2,3))
                index=int(errors.argmin())
                if errors[index]<best[0]: best=(float(errors[index]),offsets[index])
            assert best[0]<2e-5,(record['id'],best)
            raw=full[best[1]:best[1]+length]
            result.append((raw,dict(dataset='humanml',sample_id=record['id'],subject=record['id'].lstrip('M'),
                source='HumanML3D',task=record['text'],frames=length)))
    return result


def load_vae(campaign):
    opt=get_opt(str(ROOT/'checkpoints/t2m/t2m_vae_gelu/opt.txt'),torch.device('cuda'))
    opt.is_train=False
    path=ROOT/'checkpoints/t2m/t2m_vae_gelu/model/net_best_fid.tar'
    campaign.freeze(path)
    vae=VAE(opt).cuda()
    vae.load_state_dict(torch.load(path,map_location='cpu')['vae']);vae.freeze()
    return vae


@torch.no_grad()
def vae():
    campaign=Campaign('vae')
    model=load_vae(campaign)
    data=hml_references(campaign)
    for record in campaign.target:
        file=FULL/'full_data_root'/record['motion']
        assert campaign.freeze(file)==record['motion_sha256']
        raw=np.load(file)
        data.append((raw,dict(dataset='data_self',sample_id=record['sample_id'],subject=record['identity_group'],
            source=record['source_dataset'],task=record['original_task'],frames=len(raw))))
    grouped=defaultdict(list)
    for raw,meta in data: grouped[(meta['dataset'],len(raw))].append((raw,meta))
    latent_rows=[];batch_index=0
    for (dataset,length),items in sorted(grouped.items()):
        for start in range(0,len(items),32):
            batch=items[start:start+32]
            x=torch.from_numpy(np.stack([(raw-campaign.mean)/campaign.std for raw,_ in batch])).float().cuda()
            capture=[]
            handle=model.dist.register_forward_hook(lambda module,args,out:capture.append(out.detach()))
            torch.manual_seed(PROTOCOL['vae_seed']+batch_index)
            z,_=model.encode(x);handle.remove()
            mu,logvar=capture[0].chunk(2,-1)
            outputs={mode:model.decode(latent).cpu().numpy() for mode,latent in [('posterior_mean',mu),('posterior_sample',z)]}
            for mode,latent in [('posterior_mean',mu),('posterior_sample',z)]:
                prediction=torch.from_numpy(outputs[mode]).cuda()
                reencoded=model.dist(model.conv_enc(model.motion_enc(prediction)))
                reencoded_mu,_=reencoded.chunk(2,-1)
                for k,(_,meta) in enumerate(batch):
                    values=latent[k].cpu().numpy()
                    latent_rows.append(dict(meta,mode=mode,mean=float(values.mean()),std=float(values.std()),
                        rms=float(np.sqrt(np.mean(values**2))),temporal_step_rms=float(np.sqrt(np.mean(np.diff(values,axis=0)**2))),
                        decode_reencode_rms=float(torch.mean((latent[k]-reencoded_mu[k])**2).sqrt().item()),
                        posterior_std_mean=float((logvar[k]*.5).exp().mean().item())))
            traces_by_mode={mode:[] for mode in outputs}
            gt_traces=[]
            metadata=[]
            for k,(raw,meta) in enumerate(batch):
                gt,gt_trace=campaign.analyze(raw,dict(meta,variant='ground_truth',repeat=-1))
                gt_traces.append(gt_trace)
                metadata.append(meta)
                for mode,predicted in outputs.items():
                    prediction=predicted[k]*campaign.std+campaign.mean
                    _,trace=campaign.analyze(prediction,dict(meta,variant=mode,repeat=0),gt=gt)
                    traces_by_mode[mode].append(trace)
            for mode,traces in traces_by_mode.items(): campaign.store_series(f'{dataset}_{mode}_{batch_index:04}',metadata,traces)
            campaign.store_series(f'{dataset}_ground_truth_{batch_index:04}',metadata,gt_traces)
            batch_index+=1
            if batch_index%5==0: print('vae',dataset,length,batch_index,flush=True)
    save(campaign.output/'latent_statistics.json',latent_rows)
    assert len(campaign.clip_rows)==(1327+128)*3*3
    campaign.finish(dict(vae_weights_shared_by_all_finetuned_models=True,unpadded_exact_lengths=True))


@torch.no_grad()
def smoke():
    campaign=Campaign('smoke')
    model=load_vae(campaign)
    for record in campaign.target[:2]:
        file=FULL/'full_data_root'/record['motion']
        assert campaign.freeze(file)==record['motion_sha256']
        raw=np.load(file)
        meta=dict(dataset='data_self',variant='ground_truth',repeat=-1,sample_id=record['sample_id'],
            subject=record['identity_group'],source=record['source_dataset'],task=record['original_task'],frames=len(raw))
        gt,_=campaign.analyze(raw,meta,store=True)
        x=torch.from_numpy((raw-campaign.mean)/campaign.std).float().unsqueeze(0).cuda()
        z,_=model.encode(x)
        prediction=model.decode(z)[0].cpu().numpy()*campaign.std+campaign.mean
        campaign.analyze(prediction,dict(meta,variant='posterior_sample',repeat=0),gt=gt,store=True)
    with campaign.load(FULL/'evaluation/20261017/official/0/0000.npz') as archive:
        for k,index in enumerate(archive['ids'][:2]):
            record=campaign.target_by_index[int(index)]
            length=int(archive['lengths'][k])
            raw=archive['normalized_motion'][k,:length]*campaign.std+campaign.mean
            meta=dict(dataset='data_self',variant='official',repeat=0,sample_id=record['sample_id'],
                subject=record['identity_group'],source=record['source_dataset'],task=record['original_task'],frames=length)
            campaign.analyze(raw,meta,cached=archive['xyz'][k,:length],store=True)
    campaign.finish()


@torch.no_grad()
def sampling():
    from diffusers import DDIMScheduler
    from models.denoiser.model import Denoiser
    from models.denoiser.trainer import DenoiserTrainer
    from utils.fixseed import fixseed
    campaign=Campaign('sampling')
    vae=load_vae(campaign)
    opt=get_opt(str(ROOT/'checkpoints/t2m/t2m_denoiser_vpred_vaegelu/opt.txt'),torch.device('cuda'))
    opt.is_train=False
    denoiser=Denoiser(opt,vae.opt.latent_dim).cuda()
    weight=ROOT/'checkpoints/t2m/t2m_denoiser_vpred_vaegelu/model/net_best_fid.tar'
    campaign.freeze(weight)
    missing,unexpected=denoiser.load_state_dict(torch.load(weight,map_location='cpu')['denoiser'],strict=False)
    assert not unexpected and all(key.startswith('clip_model.') for key in missing)
    scheduler=DDIMScheduler(num_train_timesteps=opt.num_train_timesteps,beta_start=opt.beta_start,
        beta_end=opt.beta_end,beta_schedule=opt.beta_schedule,prediction_type=opt.prediction_type,clip_sample=False)
    trainer=DenoiserTrainer(opt,denoiser,vae,scheduler)
    prompts=json.loads((PHYSICS/'protocol.json').read_text())['examples']
    noise={};final_latents={};baseline_error=0.;latent_rows=[]
    for steps,scale in PROTOCOL['sampling_probe']['settings']:
        opt.num_inference_timesteps=steps;opt.cond_scale=scale;opt.classifier_free_guidance=scale>1
        variant=f'steps{steps}_cfg{scale:g}'
        for seed in [20261016,20261017,20261018]:
            fixseed(seed)
            final=[];initial=[]
            decode_hook=vae.conv_dec.register_forward_pre_hook(lambda module,args:final.append(args[0].detach().cpu().clone()))
            def first_noise(module,args):
                if not initial: initial.append(args[0][:len(prompts)].detach().cpu().numpy().copy())
            noise_hook=denoiser.register_forward_pre_hook(first_noise)
            # The author calls denoiser.forward directly, so instrument its first
            # transformer block via a temporary forward wrapper instead of a module hook.
            original=denoiser.forward
            def wrapped(*args,**kwargs):
                first_noise(denoiser,args)
                return original(*args,**kwargs)
            denoiser.forward=wrapped
            pred,_=trainer.generate(([p['text'] for p in prompts],torch.zeros(len(prompts),120,263),torch.full((len(prompts),),120)))
            denoiser.forward=original;noise_hook.remove();decode_hook.remove()
            initial_digest=hashlib.sha256(initial[0].tobytes()).hexdigest()
            if seed in noise: assert noise[seed]==initial_digest
            else: noise[seed]=initial_digest
            raw=pred.cpu().numpy()*campaign.std+campaign.mean
            z=final[-1].cuda()
            encoded=vae.dist(vae.conv_enc(vae.motion_enc(pred)))
            mu,_=encoded.chunk(2,-1)
            for k,prompt in enumerate(prompts):
                meta=dict(dataset='humanml_probe',variant=variant,repeat=seed,sample_id=prompt['id'],
                    subject=prompt['id'],source='HumanML3D',task=prompt['text'],frames=120)
                campaign.analyze(raw[k],meta,store=True)
                values=z[k].cpu().numpy()
                latent_rows.append(dict(meta,latent_mean=float(values.mean()),latent_std=float(values.std()),
                    latent_rms=float(np.sqrt(np.mean(values**2))),latent_step_rms=float(np.sqrt(np.mean(np.diff(values,axis=0)**2))),
                    decode_reencode_rms=float(torch.mean((z[k]-mu[k])**2).sqrt().item())))
                if steps==50 and scale==7.5:
                    with campaign.load(PHYSICS/'arrays'/f"{prompt['id']}_{seed}.npz") as archive:
                        baseline_error=max(baseline_error,float(np.abs(pred[k].cpu().numpy()-archive['normalized']).max()))
            final_latents[variant+'_'+str(seed)]=final[-1].numpy()
            print('sampling',variant,seed,flush=True)
    folder=campaign.output/'server_only';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'final_latents.npz',**final_latents)
    save(campaign.output/'latent_statistics.json',latent_rows)
    assert len(campaign.clip_rows)==5*3*6*3
    assert baseline_error<2e-5,baseline_error
    campaign.finish(dict(paired_initial_noise_sha256=noise,baseline_normalized_replay_max=baseline_error,
                         author_sampling_configuration=dict(beta_schedule=opt.beta_schedule,prediction_type=opt.prediction_type,
                             beta_start=opt.beta_start,beta_end=opt.beta_end,train_timesteps=opt.num_train_timesteps)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','vae','generated','sampling'])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    save(OUT/'protocol.json',PROTOCOL)
    globals()[args.stage]()
