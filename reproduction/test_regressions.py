"""Regression checks for text conditioning and the evaluation contract."""
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from models.denoiser.trainer import DenoiserTrainer
from utils.get_opt import get_opt
from utils.metrics import calculate_R_precision, calculate_frechet_distance


class SmallVAE(torch.nn.Module):
    def encode(self, motion):
        return torch.zeros(len(motion), 2, 7, 2), {}

    def decode(self, latent):
        return latent


class TextDenoiser(torch.nn.Module):
    def forward(self, latent, timestep, text, **kwargs):
        value = torch.tensor([float(bool(t)) for t in text]).reshape(-1, 1, 1, 1)
        return torch.ones_like(latent) * value, (None, None, None)

    def remove_clip_cache(self):
        pass


class SmallScheduler:
    init_noise_sigma = 1

    def set_timesteps(self, steps):
        self.timesteps = torch.arange(steps)

    def step(self, prediction, timestep, latent):
        return Namespace(prev_sample=latent-prediction)


class Regressions(unittest.TestCase):
    def generate(self, guided, text):
        opt = Namespace(device='cpu', is_train=False, classifier_free_guidance=guided,
                        num_inference_timesteps=2, cond_scale=7.5)
        trainer = DenoiserTrainer(opt, TextDenoiser(), SmallVAE(), SmallScheduler())
        torch.manual_seed(7)
        return trainer.generate(([text], torch.zeros(1, 8, 263), torch.tensor([8])))[0]

    def test_text_survives_without_guidance(self):
        self.assertFalse(torch.equal(self.generate(False, 'walking'), self.generate(False, '')))

    def test_guidance_retains_conditioned_branch(self):
        self.assertFalse(torch.equal(self.generate(True, 'walking'), self.generate(True, '')))

    def test_matching_requires_correct_pairs(self):
        embedding = np.eye(32)
        correct = calculate_R_precision(embedding, embedding, 3, sum_all=True)
        wrong = calculate_R_precision(embedding, np.roll(embedding, 1, axis=0), 3, sum_all=True)
        self.assertEqual(correct[0], 32)
        self.assertEqual(wrong[0], 0)

    def test_identical_gaussians_have_zero_fid(self):
        self.assertAlmostEqual(calculate_frechet_distance(np.zeros(4), np.eye(4),
                                                        np.zeros(4), np.eye(4)), 0)

    def test_saved_options_preserve_colons(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'opt.txt'
            path.write_text('checkpoints_dir: ./checkpoints\ndataset_name: t2m\nname: example\nlabel: value: detail\nlayers: [1, 2]\n')
            opt = get_opt(path, torch.device('cpu'))
            self.assertEqual(opt.label, 'value: detail')
            self.assertEqual(opt.layers, [1, 2])


if __name__ == '__main__':
    unittest.main()
