"""Construct the unmodified released networks and evaluation components."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from diffusers import DDIMScheduler
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.denoiser.trainer import DenoiserTrainer
from test_denoiser import load_vae, load_denoiser

NAMES = {'t2m': 't2m_denoiser_vpred_vaegelu',
         'kit': 'kit_denoiser_vpred_vaegelu_bsz16'}


def load_runtime(dataset, seed=1234, steps=50, scale=7.5, split='test'):
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    from utils.fixseed import fixseed
    fixseed(seed)
    opt = get_opt(str(ROOT / 'checkpoints' / dataset / NAMES[dataset] / 'opt.txt'), device)
    opt.seed = seed
    opt.num_inference_timesteps = steps
    opt.cond_scale = scale
    opt.classifier_free_guidance = scale > 1.0
    vae_opt = get_opt(str(ROOT / 'checkpoints' / dataset / opt.vae_name / 'opt.txt'), device)
    eval_path = str(ROOT / 'checkpoints' / dataset / 'Comp_v6_KLD005/opt.txt')
    wrapper = EvaluatorModelWrapper(get_opt(eval_path, device))
    loader, data = get_dataset_motion_loader(eval_path, 32, split, device=device)
    vae = load_vae(vae_opt).to(device)
    denoiser = load_denoiser(opt, vae_opt.latent_dim).to(device)
    scheduler = DDIMScheduler(num_train_timesteps=opt.num_train_timesteps,
                             beta_start=opt.beta_start, beta_end=opt.beta_end,
                             beta_schedule=opt.beta_schedule,
                             prediction_type=opt.prediction_type, clip_sample=False)
    trainer = DenoiserTrainer(opt, denoiser, vae, scheduler)
    return opt, vae_opt, trainer, wrapper, loader
