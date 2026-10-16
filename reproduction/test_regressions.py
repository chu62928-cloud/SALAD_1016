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
from data.t2m_dataset import MotionDataset
from t2m import mirror, refine


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
            path.write_text('checkpoints_dir: ./checkpoints\ndataset_name: t2m\nname: example\nlabel: value: detail\nlayers: [1, 2]\nweight_decay: 1e-06\n')
            opt = get_opt(path, torch.device('cpu'))
            self.assertEqual(opt.label, 'value: detail')
            self.assertEqual(opt.layers, [1, 2])
            self.assertEqual(opt.weight_decay, 1e-6)

    def test_resume_rejects_invalid_optimizer_state(self):
        model = torch.nn.Linear(1, 1)
        trainer = DenoiserTrainer(Namespace(device='cpu', is_train=False), model,
                                   SmallVAE(), SmallScheduler())
        trainer.optim = torch.optim.AdamW(model.parameters())
        trainer.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(trainer.optim, [10])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.pt'
            torch.save({'denoiser': model.state_dict(), 'optim': {}, 'lr_scheduler': {},
                        'epoch': 0, 'total_iter': 1}, path)
            with self.assertRaises(KeyError):
                trainer.resume(path)

    def test_training_windows_cover_exact_and_last_window_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root/'a.npy', np.repeat(np.arange(10, 14)[:, None], 263, axis=1))
            np.save(root/'b.npy', np.repeat(np.arange(20, 26)[:, None], 263, axis=1))
            (root/'train.txt').write_text('a\nb\n')
            opt = Namespace(joints_num=22, window_size=4, motion_dir=str(root))
            dataset = MotionDataset(opt, np.zeros(263), np.ones(263), root/'train.txt')
            self.assertEqual(len(dataset), 4)
            self.assertEqual([dataset[i][0, 0] for i in range(len(dataset))], [10, 20, 21, 22])

    def test_mirror_preserves_source_attention(self):
        source = torch.arange(2*7*3, dtype=torch.float32).reshape(1, 1, 1, 14, 3)
        original = source.clone()
        output = mirror(source, 'upper')
        self.assertTrue(torch.equal(source, original))
        self.assertFalse(torch.equal(output, original))
        self.assertTrue(torch.equal(mirror(output, 'upper'), original))

    def test_refine_aligns_shared_tokens_after_insertion(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return {'source': [0, 1, 2, 3, 9], 'edited': [0, 1, 4, 2, 3, 9]}[text]
        source = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 1, 8)
        edited = source + 100
        original = source.clone()
        output = refine(Tokenizer(), source, edited, 'source', 'edited')
        expected = torch.tensor([0, 1, 102, 2, 3, 4, 6, 7], dtype=torch.float32)
        self.assertTrue(torch.equal(output.flatten(), expected))
        self.assertTrue(torch.equal(source, original))


if __name__ == '__main__':
    unittest.main()
