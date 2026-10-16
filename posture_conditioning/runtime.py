"""Load released weights only, without constructing an official test loader."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from diffusers import DDIMScheduler
from utils.get_opt import get_opt
from models.vae.model import VAE
from models.denoiser.model import Denoiser
from .common import ROOT
from .adapter import PostureModel


def runtime():
    device = torch.device('cuda:0')
    opt = get_opt(str(ROOT/'checkpoints/t2m/t2m_denoiser_vpred_vaegelu/opt.txt'), device)
    opt.is_train = False
    vo = get_opt(str(ROOT/'checkpoints/t2m/t2m_vae_gelu/opt.txt'), device)
    vae = VAE(vo)
    state = torch.load(ROOT/'checkpoints/t2m/t2m_vae_gelu/model/net_best_fid.tar', map_location='cpu')
    vae.load_state_dict(state['vae'])
    vae.freeze()
    denoiser = Denoiser(opt, vo.latent_dim)
    state = torch.load(ROOT/'checkpoints/t2m/t2m_denoiser_vpred_vaegelu/model/net_best_fid.tar', map_location='cpu')
    missing, unexpected = denoiser.load_state_dict(state['denoiser'], strict=False)
    assert not unexpected and all(k.startswith('clip_model.') for k in missing)
    scheduler = DDIMScheduler(num_train_timesteps=opt.num_train_timesteps,
                              beta_start=opt.beta_start,beta_end=opt.beta_end,
                              beta_schedule=opt.beta_schedule,prediction_type=opt.prediction_type,
                              clip_sample=False)
    return PostureModel(denoiser).to(device), vae.to(device), scheduler


@torch.no_grad()
def posterior(vae, motion):
    encoded = vae.dist(vae.conv_enc(vae.motion_enc(motion)))
    return encoded.chunk(2,-1)
