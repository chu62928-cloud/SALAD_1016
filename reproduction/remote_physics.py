"""Deploy isolated SALAD diagnostics and synchronize small deliverable files."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'reproduction'))
from sync_results import remote, HIDDEN, SERVER

RUN=SERVER+'/reproduction/artifacts/physics_v1_20261016'
LOCAL=REPO/'reproduction/artifacts/physics_v1_20261016'
PYTHON='/root/autodl-tmp/envs/salad_1016/bin/python'
SOURCES=['sample_physics.py','physics_metrics.py','physics_protocol.json','test_physics_metrics.py','render_physics.py']


def deploy():
    expected={name:hashlib.sha256((REPO/'reproduction'/name).read_bytes()).hexdigest() for name in SOURCES}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    remote(f'''from pathlib import Path
p=Path({RUN!r})
if (p/'status.json').exists() or (p/'binding.json').exists():raise RuntimeError('Existing diagnostic run retained; inspect status')
(p/'code').mkdir(parents=True,exist_ok=True)
''')
    subprocess.run(['scp','-q','-o','BatchMode=yes',*[str(REPO/'reproduction'/name) for name in SOURCES],
                    'motion-data:'+RUN+'/code/'],check=True,capture_output=True,timeout=60,creationflags=HIDDEN)
    result=remote(f'''from pathlib import Path
import hashlib,json,os,subprocess
p=Path({RUN!r});source=p/'code'
expected={expected!r}
actual={{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in expected}}
if actual!=expected:raise ValueError('Deployed source hash mismatch')
env=dict(os.environ,SALAD_ROOT={SERVER!r},SALAD_AUXILIARY_COMMIT={commit!r},HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
test=subprocess.run([{PYTHON!r},str(source/'test_physics_metrics.py')],env=env,capture_output=True,text=True)
(p/'unit_tests.json').write_text(json.dumps(dict(status='passed' if test.returncode==0 else 'failed',exit_code=test.returncode,output=test.stdout+test.stderr,code_sha256=actual),indent=2))
if test.returncode:raise RuntimeError('Physics tests failed')
with (p/'job.log').open('ab') as log:
 process=subprocess.Popen([{PYTHON!r},'-u',str(source/'sample_physics.py'),'--protocol',str(source/'physics_protocol.json'),'--output',str(p),'--render'],env=env,cwd={SERVER!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=process.pid,auxiliary_commit={commit!r},code_sha256=actual)
(p/'launch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''',timeout=50)
    print(result.decode())


def status():
    print(remote(f'''from pathlib import Path
import json
p=Path({RUN!r})
for name in ['status.json','summary.json']:
 if (p/name).exists():
  v=json.loads((p/name).read_text());v.pop('metrics',None);v.pop('files_sha256',None)
  print(name,json.dumps(v))
if (p/'job.log').exists():print((p/'job.log').read_text(errors='replace')[-2000:])
''').decode())


def fetch():
    snapshot=json.loads(remote(f'''from pathlib import Path
import base64,hashlib,json,datetime
p=Path({RUN!r})
files={{}};hashes={{}};media={{}}
for name in ['summary.json','binding.json','protocol.json','examples.json','reference.json','metrics.csv','example_metrics.csv','status.json','launch.json','unit_tests.json','videos.json']:
 f=p/name
 if f.exists():
  b=f.read_bytes();files[name]=base64.b64encode(b).decode();hashes[name]=hashlib.sha256(b).hexdigest()
if 'videos.json' in files:
 manifest=json.loads(base64.b64decode(files['videos.json']))
 for row in manifest['videos']:
  for key,sha in [('path','sha256'),('poster','poster_sha256')]:
   f=p/row[key];digest=hashlib.sha256(f.read_bytes()).hexdigest()
   if digest!=row[sha]:raise ValueError('Published media changed')
   media[row[key]]=digest
print(json.dumps(dict(files=files,sha256=hashes,media=media,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
'''))
    LOCAL.mkdir(parents=True,exist_ok=True)
    for name,encoded in snapshot.pop('files').items():
        content=base64.b64decode(encoded)
        if hashlib.sha256(content).hexdigest()!=snapshot['sha256'][name]:raise ValueError('Transfer hash mismatch')
        (LOCAL/name).write_bytes(content)
    for name,digest in snapshot['media'].items():
        target=LOCAL/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:continue
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp','-q','-o','BatchMode=yes','motion-data:'+RUN+'/'+name,str(target)],
                       check=True,capture_output=True,timeout=60,creationflags=HIDDEN)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Video checksum mismatch')
    (LOCAL/'transfer.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps({'status':'synced','media_files':len(snapshot['media']),'checked_at':snapshot['checked_at']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['deploy','status','fetch'])
    stage=parser.parse_args().stage
    {'deploy':deploy,'status':status,'fetch':fetch}[stage]()
