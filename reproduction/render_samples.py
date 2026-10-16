"""Render the fixed preflight samples using the author's skeleton visualizer."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils import paramUtil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['t2m', 'kit'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(ROOT/'reproduction/artifacts'/('preflight_'+args.dataset+'.samples.npz'))
    normalization = ROOT/'checkpoints'/args.dataset/'Comp_v6_KLD005/meta'
    mean, std = np.load(normalization/'mean.npy'), np.load(normalization/'std.npy')
    humanml = args.dataset == 't2m'
    joints = 22 if humanml else 21
    tree = paramUtil.t2m_kinematic_chain if humanml else paramUtil.kit_kinematic_chain
    captions = {
        'original': 'a person waves the left hand.',
        'word_swap': 'a person waves the right hand.',
        'refine': 'a person waves the left hand while walking.',
        'reweight': 'a person waves the left hand. (waves x2)',
        'mirror': 'a person waves the right hand. (upper mirror)',
    }
    result = {'dataset': args.dataset, 'fps': 20 if humanml else 12.5,
              'display': 'Author visualizer: floor translation and framewise root centering, original samples retained',
              'qualitative_scope': 'Fixed seed 456, 64 frames; examples do not replace quantitative editing evaluation',
              'videos': {}}
    for name in arrays.files:
        normalized = arrays[name][0]
        xyz = recover_from_ric(torch.from_numpy(normalized*std+mean).float(), joints).numpy()
        assert np.isfinite(xyz).all()
        video = args.output/(name+'.mp4')
        plot_3d_motion(str(video), tree, xyz, title=captions[name], figsize=(5, 5),
                       fps=result['fps'], radius=4 if humanml else 1920)
        result['videos'][name] = {'path': video.name, 'frames': len(xyz),
                                 'text': captions[name], 'bytes': video.stat().st_size}
        print(name, video.stat().st_size, flush=True)
    result['status'] = 'passed'
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
