"""Download only the released models needed by the official evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import gdown

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    't2m': {
        'Comp_v6_KLD005/meta/mean.npy': '1oXE0sHOSu_DsqDqQl2VA5cli_WQ-4v3n',
        'Comp_v6_KLD005/meta/std.npy': '1GcAcDB6Z7PAbGWtZrxDuQfzik9G2nvn-',
        'Comp_v6_KLD005/opt.txt': '1dyJ7PpKLSmgZussLE1pN7n9XY9Kg6v-l',
        't2m_denoiser_vpred_vaegelu/opt.txt': '1C6PPl4qDlYlt_MSLP9PysupAjQxktpVb',
        't2m_denoiser_vpred_vaegelu/model/net_best_fid.tar': '1FKn6uLsVuzP29PCB2u5JeTPEvkpVdRLK',
        't2m_vae_gelu/opt.txt': '1b1SGqjzTp7drzoDoW9zvSCt-kj1z68GB',
        't2m_vae_gelu/model/net_best_fid.tar': '1nHZnfC3Hh3Q1FLoCqJYOH-56XT5nbgbT',
        'text_mot_match/model/finest.tar': '1YDbKNhuuX6boei3RwgWvu4aCR_T39JPA',
    },
    'kit': {
        'Comp_v6_KLD005/meta/mean.npy': '1UZpNFx0SIpoqFYy3QOheZd_p3auYAIXz',
        'Comp_v6_KLD005/meta/std.npy': '1dmj_AE-STnCINkKwBkH4zuwuQ0IsxyxR',
        'Comp_v6_KLD005/opt.txt': '1-d8HHu6o2ULIfhHoz8FEV1vnppn93lNj',
        'kit_denoiser_vpred_vaegelu_bsz16/opt.txt': '1xfGypGRIcz02awfuK4VpxaVMpanJBbr7',
        'kit_denoiser_vpred_vaegelu_bsz16/model/net_best_fid.tar': '1XEl0CeAS9NQSnKFwYqE4swObx1n273gT',
        'kit_vae_gelu/opt.txt': '18qnLffrbnDmkXgJ4W5gtDp0X6eJoCLSu',
        'kit_vae_gelu/model/net_best_fid.tar': '1V5sNA1sOmnz8oqeV4Py9EPfdWg6dJreD',
        'text_mot_match/model/finest.tar': '1wV3yJCQYvsGkS63BBgvA9i0u6uqGtud8',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=FILES, required=True)
    args = parser.parse_args()
    out = ROOT / 'reproduction/artifacts' / ('downloads_' + args.dataset + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'downloading', 'dataset': args.dataset, 'files': {}}
    for name, file_id in FILES[args.dataset].items():
        path = ROOT / 'checkpoints' / args.dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt['current'] = name
        out.write_text(json.dumps(receipt, indent=2))
        if not path.is_file():
            downloaded = gdown.download(id=file_id, output=str(path), resume=True)
            if not downloaded:
                raise RuntimeError('Download failed: ' + name)
        if not path.stat().st_size:
            raise ValueError('Empty asset: ' + str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(2 ** 20), b''):
                digest.update(chunk)
        receipt['files'][name] = {'google_drive_file_id': file_id,
                                 'bytes': path.stat().st_size,
                                 'sha256': digest.hexdigest()}
        print('VERIFIED', name, path.stat().st_size, flush=True)
    receipt['status'] = 'complete'
    receipt['completed_at'] = datetime.now(timezone.utc).isoformat()
    receipt.pop('current', None)
    temporary = out.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2))
    os.replace(temporary, out)


if __name__ == '__main__':
    main()
