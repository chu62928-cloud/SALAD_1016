"""Fetch immutable public assets, verifying their published content hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    'clip': {
        'repo': 'openai/clip-vit-base-patch32',
        'revision': '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        'files': {
            'pytorch_model.bin': ('sha256', 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'),
            'config.json': ('git_blob', 'a2a88b96561196777ca173b15309ea859f4d2ce0'),
            'tokenizer.json': ('git_blob', '564c0ebd5ce29c4ee4864004aee693deadd3128c'),
            'tokenizer_config.json': ('git_blob', '4fdaf6842dd5a725b940c92b6b692490ce59d548'),
            'special_tokens_map.json': ('git_blob', '9bfb42aa97dcd61e89f279ccaee988bccb4fabae'),
            'vocab.json': ('git_blob', '182766ce89b439768edadda342519f33802f5364'),
            'merges.txt': ('git_blob', 'bbfec752c9a675946c6dce106def6f35c882dcc2'),
        },
    },
    'kit': {
        'repo': 'camenduru/KIT-ML',
        'revision': 'f5bf633e1084638a4e00fa92cda55dfab23658c4',
        'files': {
            'new_joint_vecs.rar': ('sha256', '1c5219abbf62812f991536946b933dd6dd7876f8e353fa3fd575ad03d762457f'),
            'texts.rar': ('sha256', '9c882e373206ca13bfbc7b16f2b0379a1f01ff5265f79d1f1f4edfea12f92afb'),
            'Mean.npy': ('sha256', '9e23fac51db2215ab5666324226be48f27efd6a6e7b22ebd17c28e0f056a7c22'),
            'Std.npy': ('sha256', 'd370201d449fcafa106684f8de9aa688399b27db5e363295829085674e713145'),
            'train.txt': ('sha256', '4860e167cfc4c3dd4f3e41d74bff4e2339dafdc67d2c9024331aa2a070a857ea'),
            'val.txt': ('sha256', '1d3bee66f33d8bef65868c099c10e51101bf5a77c40412438c995755cd7159ac'),
            'test.txt': ('sha256', '0e6fd0ee0a0a9a24da1681da56a22d86d95b78c41e59b0f0fa9bd984dbf281db'),
        },
    },
}


def digest(path, kind='sha256'):
    result = hashlib.sha1() if kind == 'git_blob' else hashlib.sha256()
    if kind == 'git_blob':
        result.update(('blob %d\0' % path.stat().st_size).encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def download(session, url, path):
    partial = path.with_name(path.name + '.part')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            headers = {'Range': 'bytes=%d-' % offset} if offset else {}
            with session.get(url, headers=headers, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get(
                        'Content-Range', '').startswith('bytes %d-' % offset)):
                    raise ValueError('Server did not honor the resume range')
                with partial.open('ab' if offset else 'wb') as stream:
                    for block in response.iter_content(2**20):
                        stream.write(block)
            os.replace(partial, path)
            return
        except (requests.RequestException, OSError):
            if attempt == 4:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', choices=MANIFESTS, required=True)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    args = parser.parse_args()
    manifest = MANIFESTS[args.asset]
    if args.asset == 'clip':
        from huggingface_hub.constants import HF_HUB_CACHE
        destination = Path(HF_HUB_CACHE) / 'models--openai--clip-vit-base-patch32' / 'snapshots' / manifest['revision']
    else:
        destination = ROOT / 'dataset/kit-ml_download'
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    receipt = {'repo': manifest['repo'], 'revision': manifest['revision'],
               'endpoint': args.endpoint, 'files': {}, 'status': 'running'}
    receipt_path = ROOT / 'reproduction/artifacts' / ('hf_download_' + args.asset + '.json')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    for name, (kind, expected) in manifest['files'].items():
        path = destination / name
        if not path.is_file():
            url = '%s/%s/resolve/%s/%s?download=true' % (
                args.endpoint.rstrip('/'), manifest['repo'], manifest['revision'], name)
            print('Downloading', name, flush=True)
            download(session, url, path)
        actual = digest(path, kind)
        if actual != expected:
            raise ValueError('Published hash mismatch: ' + name)
        receipt['files'][name] = {'path': str(path), 'bytes': path.stat().st_size,
                                 'sha256': digest(path), 'published_hash': expected,
                                 'published_hash_kind': kind}
        receipt_path.write_text(json.dumps(receipt, indent=2))
        print('Verified', name, path.stat().st_size, flush=True)
    receipt['status'] = 'complete'
    receipt_path.write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
