"""Download only small diagnostic evidence and media, with SHA256 verification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sync_results import remote,HIDDEN,SERVER

DEST=Path(__file__).resolve().parents[1]/'artifacts/jitter_attribution_v3'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    code='''from pathlib import Path
import tarfile,json,hashlib
p=Path('/root/autodl-tmp/SALAD_1016/reproduction/artifacts/jitter_attribution_v3')
files=[p/'protocol.json']
for stage in ['smoke','vae','generated','sampling']:
 for name in ['complete.json','input_binding.json','skeleton.json','selected_examples.json','latent_statistics.json']:
  f=p/stage/name
  if f.is_file():files.append(f)
for folder in ['summary','figures','media','qa']:
 files.extend(f for f in (p/folder).rglob('*') if f.is_file())
for name in ['REPORT_zh.md','index.html','FINAL_VERIFICATION.json']:
 f=p/name
 if f.is_file():files.append(f)
manifest={str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(set(files))}
assert all('server_only' not in k and '/examples/' not in k and not k.endswith('.pt') for k in manifest)
archive=p/'delivery.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name in manifest:tar.add(p/name,arcname=name,recursive=False)
(p/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'files':manifest,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size}))
'''
    snapshot=json.loads(remote(code,timeout=60))
    archive=DEST/'delivery.tar.gz'
    subprocess.run(['scp','-q','motion-data:'+SERVER+'/reproduction/artifacts/jitter_attribution_v3/delivery.tar.gz',str(archive)],
        check=True,creationflags=HIDDEN)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==snapshot['archive_sha256']
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target=(DEST/member.name).resolve()
            assert target.is_relative_to(DEST.resolve()) and member.isfile()
            assert member.name in snapshot['files']
        tar.extractall(DEST,filter='data')
    for name,info in snapshot['files'].items():
        file=DEST/name
        assert file.stat().st_size==info['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==info['sha256']
    (DEST/'transfer_verification.json').write_text(json.dumps(dict(status='passed',**snapshot),indent=2),encoding='utf-8')
    archive.unlink()
    print('TRANSFER PASSED',len(snapshot['files']),snapshot['archive_bytes'],flush=True)


if __name__=='__main__':main()
