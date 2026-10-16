"""Validate the official benchmark before using its permissive data loader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root, dataset):
    dim, minimum = (263, 40) if dataset == 't2m' else (251, 24)
    splits = {}
    errors = []
    checked = set()
    timed_captions = 0
    filtered = []
    for split in ['train', 'val', 'test']:
        path = root / (split + '.txt')
        if not path.is_file():
            errors.append('Missing split: ' + str(path))
            continue
        ids = path.read_text().splitlines()
        splits[split] = {'count': len(ids), 'sha256': sha256(path)}
        if len(set(ids)) != len(ids):
            errors.append('Duplicate IDs in ' + split)
        for name in ids:
            if name in checked:
                continue
            checked.add(name)
            motion_path = root / 'new_joint_vecs' / (name + '.npy')
            text_path = root / 'texts' / (name + '.txt')
            try:
                motion = np.load(motion_path, allow_pickle=False)
                if motion.ndim != 2 or motion.shape[1] != dim:
                    raise ValueError('Wrong feature shape: ' + str(motion.shape))
                if not np.isfinite(motion).all():
                    raise ValueError('Nonfinite motion')
                if len(motion) < minimum or len(motion) >= 200:
                    filtered.append(name)
                lines = text_path.read_text().splitlines()
                if not lines:
                    raise ValueError('Empty text file')
                for line in lines:
                    fields = line.split('#')
                    if len(fields) != 4 or not fields[0] or not fields[1]:
                        raise ValueError('Malformed caption')
                    start, end = float(fields[2]), float(fields[3])
                    if np.isfinite(start) and np.isfinite(end) and (start or end):
                        timed_captions += 1
            except (OSError, ValueError) as exc:
                errors.append(name + ': ' + str(exc))
    sets = {s: set((root / (s + '.txt')).read_text().splitlines())
            for s in splits}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if a in sets and b in sets and sets[a] & sets[b]:
            errors.append('Overlapping split IDs: ' + a + '/' + b)
    return {'root': str(root.resolve()), 'splits': splits,
            'checked_ids': len(checked), 'length_filtered_ids': filtered,
            'timed_captions': timed_captions,
            'kit_time_crop_note': 'Upstream loader uses 20 fps for time tags even for KIT; audit before interpreting timed KIT segments.' if dataset == 'kit' and timed_captions else None,
            'errors': errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['t2m', 'kit'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / 'dataset' / ('humanml3d' if args.dataset == 't2m' else 'kit-ml')
    result = audit_dataset(root, args.dataset)
    files = list((ROOT / 'checkpoints' / args.dataset).rglob('*.tar'))
    files += list((ROOT / 'checkpoints' / args.dataset).rglob('opt.txt'))
    files += list((ROOT / 'checkpoints' / args.dataset).rglob('*.npy'))
    files += list((ROOT / 'glove').glob('our_vab*'))
    result['asset_files'] = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
                              'sha256': sha256(p)} for p in sorted(files)}
    result['checked_at'] = datetime.now(timezone.utc).isoformat()
    result['status'] = 'passed' if not result['errors'] else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ['asset_files', 'length_filtered_ids']}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
