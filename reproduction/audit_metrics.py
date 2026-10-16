"""Independently recompute metrics on real evaluation inputs without changing RNG."""
import numpy as np
from utils import eval_t2m


class MetricAudit:
    def __init__(self):
        self.originals = {}
        self.results = {}

    def check(self, name, actual, independent, atol=2e-5):
        actual, independent = np.asarray(actual), np.asarray(independent)
        error = float(np.max(np.abs(actual.astype('float64')-independent.astype('float64'))))
        if not np.allclose(actual, independent, atol=atol, rtol=1e-5):
            raise AssertionError('Independent metric mismatch %s: %.9g' % (name, error))
        row = self.results.setdefault(name, {'calls': 0, 'max_abs_error': 0})
        row['calls'] += 1
        row['max_abs_error'] = max(row['max_abs_error'], error)

    def precision(self, first, second, top_k, sum_all=False):
        output = self.originals['calculate_R_precision'](first, second, top_k, sum_all)
        distances = np.linalg.norm(first.astype('float64')[:, None]-second.astype('float64')[None], axis=-1)
        indices = np.argsort(distances, axis=1)
        correct = indices == np.arange(len(first))[:, None]
        independent = np.maximum.accumulate(correct, axis=1)[:, :top_k]
        if sum_all:
            independent = independent.sum(axis=0)
        self.check('retrieval_counts', output, independent, atol=0)
        return output

    def distance(self, first, second):
        output = self.originals['euclidean_distance_matrix'](first, second)
        independent = np.linalg.norm(first.astype('float64')[:, None]-second.astype('float64')[None], axis=-1)
        self.check('matching_distance_matrix', output, independent, atol=1e-4)
        return output

    def statistics(self, activation):
        output = self.originals['calculate_activation_statistics'](activation)
        values = activation.astype('float64')
        center = values.mean(axis=0)
        residual = values-center
        covariance = residual.T @ residual / (len(values)-1)
        self.check('embedding_mean', output[0], center)
        self.check('embedding_covariance', output[1], covariance)
        return output

    def fid(self, mu1, covariance1, mu2, covariance2, eps=1e-6):
        output = self.originals['calculate_frechet_distance'](mu1, covariance1, mu2, covariance2, eps)
        eigenvalues, vectors = np.linalg.eigh(covariance1)
        root = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
        sandwich = root @ covariance2 @ root
        sandwich = (sandwich+sandwich.T)/2
        trace_root = np.sqrt(np.maximum(np.linalg.eigvalsh(sandwich), 0)).sum()
        delta = mu1.astype('float64')-mu2.astype('float64')
        independent = delta @ delta + np.trace(covariance1)+np.trace(covariance2)-2*trace_root
        self.check('fid_symmetric_eigendecomposition', output, independent)
        return output

    def sampled_distance(self, name, activation, count):
        rng = np.random.RandomState()
        rng.set_state(np.random.get_state())
        output = self.originals[name](activation, count)
        size = activation.shape[0] if name == 'calculate_diversity' else activation.shape[1]
        first, second = rng.choice(size, count, replace=False), rng.choice(size, count, replace=False)
        values = activation.astype('float64')
        if name == 'calculate_diversity':
            independent = np.linalg.norm(values[first]-values[second], axis=-1).mean()
        else:
            independent = np.linalg.norm(values[:, first]-values[:, second], axis=-1).mean()
        self.check(name, output, independent)
        return output

    def mpjpe(self, ground_truth, prediction):
        output = self.originals['calculate_mpjpe'](ground_truth, prediction)
        real, generated = ground_truth.numpy().astype('float64'), prediction.numpy().astype('float64')
        real -= real[:, :1]
        generated -= generated[:, :1]
        independent = np.sqrt(np.square(real-generated).sum(axis=-1)).mean(axis=-1)
        self.check('root_aligned_mpjpe', output.numpy(), independent)
        return output

    def __enter__(self):
        replacements = {
            'calculate_R_precision': self.precision,
            'euclidean_distance_matrix': self.distance,
            'calculate_activation_statistics': self.statistics,
            'calculate_frechet_distance': self.fid,
            'calculate_diversity': lambda a, n: self.sampled_distance('calculate_diversity', a, n),
            'calculate_multimodality': lambda a, n: self.sampled_distance('calculate_multimodality', a, n),
            'calculate_mpjpe': self.mpjpe,
        }
        for name, replacement in replacements.items():
            self.originals[name] = getattr(eval_t2m, name)
            setattr(eval_t2m, name, replacement)
        return self

    def __exit__(self, *exception):
        for name, original in self.originals.items():
            setattr(eval_t2m, name, original)
