"""Exercise the released training objectives and checkpoint APIs on train only."""
import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

from runtime import ROOT, NAMES
from run_evaluation import atomic_json
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from data.t2m_dataset import MotionDataset, Text2MotionDataset
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from test_denoiser import load_vae
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
import numpy as np
import torch


def exact(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first.cpu(), second.cpu())
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(exact(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(exact(a, b) for a, b in zip(first, second))
    return first == second


def exercise(trainer, loader, parameters, steps, scheduler_name, output, name):
    trainer.optim = torch.optim.AdamW(parameters, lr=trainer.opt.lr, betas=(0.9, 0.99),
                                      weight_decay=trainer.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(trainer.optim,
                milestones=trainer.opt.milestones, gamma=trainer.opt.gamma)
    setattr(trainer, scheduler_name, scheduler)
    before = parameters[0].detach().clone()
    losses = []
    iterator = iter(loader)
    for step in range(1, steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if step < trainer.opt.warm_up_iter:
            trainer.update_lr_warm_up(step, trainer.opt.warm_up_iter, trainer.opt.lr)
        trainer.optim.zero_grad(set_to_none=True)
        loss = trainer.train_forward(batch)[0]
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        trainer.optim.step()
        if step >= trainer.opt.warm_up_iter:
            scheduler.step()
        losses.append(float(loss.detach()))
    assert not torch.equal(before, parameters[0])
    checkpoint_path = output / (name + '_smoke.pt')
    trainer.save(str(checkpoint_path), 0, steps)
    model = trainer.vae if name == 'vae' else trainer.denoiser
    state = copy.deepcopy(model.state_dict() if name == 'vae' else model.state_dict_without_clip())
    optim_state = copy.deepcopy(trainer.optim.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    with torch.no_grad():
        parameters[0].add_(1)
    trainer.optim.state.clear()
    assert trainer.resume(str(checkpoint_path)) == (0, steps)
    restored = model.state_dict() if name == 'vae' else model.state_dict_without_clip()
    assert exact(state, restored)
    assert exact(optim_state, trainer.optim.state_dict())
    assert exact(scheduler_state, scheduler.state_dict())
    trainer.logger.close()
    result = {'status': 'passed', 'updates': steps, 'batch_size': loader.batch_size,
              'loaded_train_entries': len(loader.dataset), 'losses': losses,
              'checkpoint_model_optimizer_scheduler_exact': True,
              'full_training_convergence_tested': False}
    print(name, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=NAMES, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch size must be positive')
    os.chdir(ROOT)
    torch.set_num_threads(2)
    fixseed(20261016)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    opt = get_opt(str(ROOT/'checkpoints'/args.dataset/NAMES[args.dataset]/'opt.txt'), device)
    vae_opt = get_opt(str(ROOT/'checkpoints'/args.dataset/opt.vae_name/'opt.txt'), device)
    mean = np.load(ROOT/'checkpoints'/args.dataset/'Comp_v6_KLD005/meta/mean.npy')
    std = np.load(ROOT/'checkpoints'/args.dataset/'Comp_v6_KLD005/meta/std.npy')
    train_split = Path(opt.data_root)/'train.txt'
    result = {'status': 'running', 'dataset': args.dataset, 'split': 'train',
              'seed': 20261016, 'scope': 'Engineering checks; fresh trainable networks, no convergence claim',
              'resume_limit': 'Original checkpoints omit random generator and data iterator states; exact trajectory resumption is not claimed.'}
    result_path = args.output/'summary.json'
    atomic_json(result_path, result)
    vae_opt.is_train = True
    vae_opt.log_dir = str(args.output/'vae_logs')
    vae = VAE(vae_opt).to(device).train()
    vae_trainer = VAETrainer(vae_opt, vae)
    dataset = MotionDataset(vae_opt, mean, std, train_split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    result['vae'] = exercise(vae_trainer, loader, list(vae.parameters()), args.steps,
                             'scheduler', args.output, 'vae')
    atomic_json(result_path, result)
    del vae_trainer, vae, loader, dataset
    torch.cuda.empty_cache()

    opt.is_train = True
    opt.log_dir = str(args.output/'denoiser_logs')
    frozen_vae = load_vae(vae_opt).to(device)
    denoiser = Denoiser(opt, vae_opt.latent_dim).to(device).train()
    frozen = {name: value.detach().cpu().clone() for name, value in frozen_vae.state_dict().items()}
    scheduler = DDIMScheduler(num_train_timesteps=opt.num_train_timesteps,
                beta_start=opt.beta_start, beta_end=opt.beta_end, beta_schedule=opt.beta_schedule,
                prediction_type=opt.prediction_type, clip_sample=False)
    trainer = DenoiserTrainer(opt, denoiser, frozen_vae, scheduler)
    dataset = Text2MotionDataset(opt, mean, std, train_split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    result['denoiser'] = exercise(trainer, loader, denoiser.parameters_without_clip(), args.steps,
                                  'lr_scheduler', args.output, 'denoiser')
    assert all(torch.equal(value.cpu(), frozen[name]) for name, value in frozen_vae.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in frozen_vae.parameters())
    assert all(p.grad is None and not p.requires_grad for p in denoiser.clip_model.parameters())
    result['frozen_vae_parameters_exact'] = True
    result['vae_clip_gradients_absent'] = True
    result['status'] = 'passed'
    atomic_json(result_path, result)


if __name__ == '__main__':
    main()
