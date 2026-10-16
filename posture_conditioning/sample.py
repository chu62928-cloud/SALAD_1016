"""Noise-only sampling with separately controlled text and posture guidance."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,config,seed_all,write_json,sha
from .data import Data
from .runtime import runtime


@torch.no_grad()
def generate(model,vae,scheduler,text,lengths,core,shoulder,seeds,scale=1.,capture=False):
    model.eval()
    lengths=torch.as_tensor(lengths,device='cuda',dtype=torch.long)
    if any(int(x)%4 or not 40<=int(x)<=196 for x in lengths):
        raise ValueError('Lengths must be multiples of four between 40 and 196')
    n=len(text)
    max_len=int(lengths.max())//4
    mask=torch.arange(max_len,device='cuda')[None]<(lengths//4)[:,None]
    latents=[]
    for seed in seeds:
        g=torch.Generator(device='cuda').manual_seed(int(seed))
        latents.append(torch.randn((49,7,32),device='cuda',generator=g)[:max_len])
    z=torch.stack(latents)*mask[...,None,None]
    schedule=copy.deepcopy(scheduler)
    schedule.set_timesteps(config()['inference_steps'])
    z=z*schedule.init_noise_sigma
    for step,t in enumerate(schedule.timesteps.to('cuda')):
        # Two base branches share a forward call. The extra conditioned branch gives
        # an independent posture increment rather than multiplying it by text CFG.
        base=model(torch.cat([z,z]),t,['']*n+list(text),mask=torch.cat([mask,mask]),scale=0)
        unconditional,conditional=base.chunk(2)
        prediction=unconditional+config()['text_scale']*(conditional-unconditional)
        if scale!=0:
            conditioned=model(z,t,text,core,shoulder,mask,scale=1.,
                              capture=capture and step==len(schedule.timesteps)//2)
            prediction=prediction+scale*(conditioned-conditional)
        z=schedule.step(prediction,t,z).prev_sample*mask[...,None,None]
    motion=vae.decode(z)
    if isinstance(motion,(tuple,list)):
        motion=motion[0]
    assert torch.isfinite(motion).all()
    return motion


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--text',required=True)
    parser.add_argument('--identity',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--seed',type=int,default=20261017)
    parser.add_argument('--posture-scale',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    seed_all(args.seed)
    data=Data()
    model,vae,scheduler=runtime()
    state=torch.load(args.checkpoint,map_location='cpu')
    model.load_adapter(state['adapter'])
    identity=data.identity_index[args.identity]
    core=torch.tensor(data.core[[identity]],device='cuda')
    shoulder=torch.tensor(data.shoulder[[identity]],device='cuda')
    motion=generate(model,vae,scheduler,[args.text],[args.frames],core,shoulder,[args.seed],args.posture_scale)
    mean=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
    std=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
    raw=motion[0,:args.frames].cpu().numpy()*std+mean
    from utils.motion_process import recover_from_ric
    xyz=recover_from_ric(torch.from_numpy(raw),22).numpy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,motion=raw,xyz=xyz)
    write_json(args.output.with_suffix('.json'),dict(text=args.text,identity=args.identity,
               frames=args.frames,seed=args.seed,posture_scale=args.posture_scale,
               checkpoint_sha256=sha(args.checkpoint),array_sha256=sha(args.output),ground_truth_input=False))


if __name__=='__main__':
    main()
