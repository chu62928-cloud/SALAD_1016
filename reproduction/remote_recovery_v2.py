"""Deploy and retrieve the isolated recovery/skinning diagnostic."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'reproduction'))
from sync_results import remote,SERVER,HIDDEN

SOURCE=SERVER+'/reproduction/artifacts/physics_v1_20261016'
OUTPUT=SOURCE+'/recovery_v2'
LOCAL=REPO/'reproduction/artifacts/physics_v1_20261016/recovery_v2'
FILES=['audit_recovery_v2.py','render_recovery_v2.py','verify_recovery_v2.py']


def launch(stage):
    remote(f'''from pathlib import Path
p=Path({OUTPUT!r});(p/'code').mkdir(parents=True,exist_ok=True)
''')
    for name in FILES:
        subprocess.run(['scp','-q','-o','BatchMode=yes',str(REPO/'reproduction'/name),'motion-data:'+OUTPUT+'/code/'],
                       check=True,capture_output=True,timeout=60,creationflags=HIDDEN)
    hashes={n:hashlib.sha256((REPO/'reproduction'/n).read_bytes()).hexdigest() for n in FILES}
    python='/root/autodl-tmp/envs/salad_1016/bin/python' if stage=='official' else '/root/miniconda3/envs/mdm5090/bin/python'
    script={'render':'render_recovery_v2.py','verify':'verify_recovery_v2.py'}.get(stage,'audit_recovery_v2.py')
    command=[python,'-u',OUTPUT+'/code/'+script]
    if stage not in ['render','verify']:command+=[stage]
    command+=['--root',SERVER,'--output',OUTPUT]
    if stage!='render':command+=['--source',SOURCE]
    print(remote(f'''from pathlib import Path
import subprocess,json,os,hashlib
p=Path({OUTPUT!r})
expected={hashes!r}
if {{n:hashlib.sha256((p/'code'/n).read_bytes()).hexdigest() for n in expected}}!=expected:raise ValueError('Source transfer mismatch')
for old in p.glob('*_launch.json'):
 record=json.loads(old.read_text());cmd=Path('/proc')/str(record['pid'])/'cmdline'
 if cmd.exists() and b'recovery_v2' in cmd.read_bytes():raise ValueError('Diagnostic stage is still running')
if (p/{(stage+'_launch.json')!r}).exists():raise ValueError('Existing stage receipt retained; inspect before rerunning')
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
with (p/{(stage+'.log')!r}).open('ab') as log:
 proc=subprocess.Popen({command!r},cwd={SERVER!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,command={command!r},code_sha256=expected)
(p/{(stage+'_launch.json')!r}).write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''').decode())


def status():
    print(remote(f'''from pathlib import Path
import json
p=Path({OUTPUT!r})
for stage in ['audit','render','official','verify']:
 f=p/(stage+'_launch.json')
 if f.exists():
  record=json.loads(f.read_text());cmd=Path('/proc')/str(record['pid'])/'cmdline'
  print(stage,json.dumps(dict(pid=record['pid'],alive=cmd.exists() and b'recovery_v2' in cmd.read_bytes())))
 f=p/(stage+'.log')
 if f.exists():print(f.read_text(errors='replace')[-2400:])
if (p/'progress.json').exists():print((p/'progress.json').read_text())
''').decode())


def fetch():
    snapshot=json.loads(remote(f'''from pathlib import Path
import json,base64,hashlib
p=Path({OUTPUT!r});out={{}}
for f in list(p.glob('*.json'))+list(p.glob('*.csv')):
 b=f.read_bytes();out[f.name]=dict(base64=base64.b64encode(b).decode(),sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(out))
'''))
    LOCAL.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,record in snapshot.items():
        content=base64.b64decode(record['base64'])
        if hashlib.sha256(content).hexdigest()!=record['sha256']:raise ValueError('Evidence transfer mismatch')
        (LOCAL/name).write_bytes(content);hashes[name]=record['sha256']
    media=[]
    for name in ['videos.json','official_videos.json']:
        if name in snapshot:
            manifest=json.loads((LOCAL/name).read_text(encoding='utf-8'))
            for record in manifest['videos']:
                media.append((record['path'],record['sha256']))
                if 'poster' in record:media.append((record['poster'],record['poster_sha256']))
    for name,digest in media:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe artifact path')
        target=LOCAL/path;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            subprocess.run(['scp','-q','-o','BatchMode=yes','motion-data:'+OUTPUT+'/'+path.as_posix(),str(target)],
                           check=True,capture_output=True,timeout=60,creationflags=HIDDEN)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Media transfer mismatch')
        hashes[name]=digest
    (LOCAL/'transfer.json').write_text(json.dumps(dict(status='passed',sha256=hashes),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='synced',evidence=len(snapshot),media=len(media))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','render','official','verify','status','fetch'])
    stage=p.parse_args().stage
    if stage=='status':status()
    elif stage=='fetch':fetch()
    else:launch(stage)
