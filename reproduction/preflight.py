"""Exercise released weights, conditioning, gradients and all editing modes."""
import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

from runtime import ROOT, load_runtime, NAMES
from run_evaluation import atomic_json
from audit_assets import sha256
import numpy as np
import torch
from models.vae.model import VAE
from models.vae.trainer import VAETrainer
from models.denoiser.trainer import DenoiserTrainer
from utils.fixseed import fixseed
from utils.metrics import calculate_R_precision
from t2m import Text2Motion


def finite(value):
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite tensor')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['t2m', 'kit'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opt, vae_opt, trainer, wrapper, loader = load_runtime(args.dataset, split='val')
    result = {'status': 'running', 'dataset': args.dataset, 'checks': {},
              'scope': 'Engineering smoke tests, not full training or human perceptual validation',
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}

    def passed(name, value=True):
        result['checks'][name] = value
        atomic_json(args.output, result)
        print(name, value, flush=True)

    batch = next(iter(loader))
    words, pos, captions, cap_lengths, motion, lengths, tokens = batch
    trainer.vae.eval()
    with torch.no_grad():
        reconstructed, _ = trainer.vae(motion.to(opt.device))
        finite(reconstructed)
        assert reconstructed.shape == motion.shape
        et, em = wrapper.get_co_embeddings(words, pos, cap_lengths, motion, lengths)
        finite(et)
        finite(em)
        order = np.argsort(lengths.tolist())[::-1].copy()
        text_original = wrapper.text_encoder(words.to(opt.device).float(), pos.to(opt.device).float(), cap_lengths)
        assert torch.equal(et, text_original[order])
        movements = wrapper.movement_encoder(motion[order].to(opt.device).float()[..., :-4])
        independent_motion = wrapper.motion_encoder(movements, lengths[order]//4)
        assert torch.equal(em, independent_motion)
    passed('vae_reconstruction_shape_finite', list(reconstructed.shape))
    passed('evaluator_text_motion_pairing_exact')
    passed('vae_parameter_count', sum(p.numel() for p in trainer.vae.parameters()))

    sample_motion = motion[:2]
    sample_lengths = lengths[:2]
    sample_captions = captions[:2]
    fixseed(321)
    generated, _ = trainer.generate((sample_captions, sample_motion, sample_lengths))
    finite(generated)
    fixseed(321)
    repeat, _ = trainer.generate((sample_captions, sample_motion, sample_lengths))
    assert torch.equal(generated, repeat)
    fixseed(321)
    without_gt, _ = trainer.generate((sample_captions, torch.zeros_like(sample_motion), sample_lengths))
    assert torch.equal(generated, without_gt)
    passed('sampling_50_steps_finite', list(generated.shape))
    passed('same_seed_repeat_exact')
    passed('generation_independent_of_gt_motion_values')
    trainer.opt.classifier_free_guidance = False
    trainer.opt.cond_scale = 1.0
    fixseed(321)
    plain, _ = trainer.generate((sample_captions, sample_motion, sample_lengths))
    fixseed(321)
    empty, _ = trainer.generate((['']*len(sample_captions), sample_motion, sample_lengths))
    finite(plain)
    finite(empty)
    assert not torch.equal(plain, empty)
    passed('released_weights_text_effect_without_guidance', float(torch.max(torch.abs(plain-empty))))
    trainer.opt.classifier_free_guidance = True
    trainer.opt.cond_scale = 7.5
    passed('normalization_roundtrip_max_abs', float(np.max(np.abs(
        (loader.dataset.inv_transform(motion.numpy())-loader.dataset.mean)/loader.dataset.std-motion.numpy()))))

    # These optimizer steps run in this disposable process only.
    vae_train_opt = copy.copy(vae_opt)
    vae_train_opt.is_train = True
    vae_train_opt.log_dir = str(args.output.parent / 'preflight_vae_logs')
    fresh_vae = VAE(vae_train_opt).to(opt.device)
    vae_trainer = VAETrainer(vae_train_opt, fresh_vae)
    optimizer = torch.optim.AdamW(fresh_vae.parameters(), lr=2e-4)
    before = next(fresh_vae.parameters()).detach().clone()
    loss, _ = vae_trainer.train_forward(motion[:2, :64])
    finite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in fresh_vae.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, next(fresh_vae.parameters()))
    passed('fresh_vae_backward_optimizer', float(loss))
    vae_trainer.logger.close()

    denoiser_train_opt = copy.copy(opt)
    denoiser_train_opt.is_train = True
    denoiser_train_opt.log_dir = str(args.output.parent / 'preflight_denoiser_logs')
    denoiser_trainer = DenoiserTrainer(denoiser_train_opt, trainer.denoiser, trainer.vae, trainer.noise_scheduler)
    trainer.denoiser.train()
    parameters = trainer.denoiser.parameters_without_clip()
    optimizer = torch.optim.AdamW(parameters, lr=opt.lr)
    before = parameters[0].detach().clone()
    loss, _, _ = denoiser_trainer.train_forward((sample_captions, sample_motion, sample_lengths))
    finite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert all(p.grad is None for p in trainer.vae.parameters())
    assert all(p.grad is None for p in trainer.denoiser.clip_model.parameters())
    optimizer.step()
    assert not torch.equal(before, parameters[0])
    passed('denoiser_backward_optimizer_frozen_vae_clip', float(loss))
    denoiser_trainer.logger.close()
    del denoiser_trainer, trainer, optimizer, fresh_vae, vae_trainer
    torch.cuda.empty_cache()

    playground = Text2Motion(NAMES[args.dataset], args.dataset)
    fixseed(456)
    source = 'a person waves the left hand.'
    noise, original, attention = playground.generate(source, 64)
    finite(original)
    samples = {'original': original.detach().cpu().numpy()}
    original_attention = [value.clone() for value in attention]
    for mode, target, options in [
        ('word_swap', 'a person waves the right hand.', {'swap_src_proportion': 0.2}),
        ('refine', 'a person waves the left hand while walking.', {}),
        ('reweight', source, {'tgt_word': 'waves', 'reweight_scale': 2.0}),
        ('mirror', 'a person waves the right hand.', {'mirror_mode': 'upper'}),
    ]:
        edited = playground.edit(noise, src_text=source, edit_text=target, edit_mode=mode,
                                 src_sa=attention[0], src_ta=attention[1], src_ca=attention[2], **options)
        finite(edited)
        assert edited.shape == original.shape
        assert all(torch.equal(value, saved) for value, saved in zip(attention, original_attention))
        difference = float(torch.max(torch.abs(edited-original)))
        assert difference > 0
        passed('editing_' + mode, {'shape': list(edited.shape), 'max_abs_change': difference})
        samples[mode] = edited.detach().cpu().numpy()
    sample_path = args.output.with_suffix('.samples.npz')
    np.savez_compressed(sample_path, **samples)
    passed('normalized_sample_archive', {'path': str(sample_path), 'sha256': sha256(sample_path),
                                        'source_text': source, 'frames': 64, 'seed': 456})
    result['status'] = 'passed'
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
