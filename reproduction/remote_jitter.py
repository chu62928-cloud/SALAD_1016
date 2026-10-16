"""Isolated temporal diagnostics; never replace the frozen SALAD run."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'reproduction'))
from sync_results import remote, HIDDEN, SERVER

SOURCE = SERVER+'/reproduction/artifacts/physics_v1_20261016'
OUT = SOURCE+'/jitter_v1'
LOCAL = REPO/'reproduction/artifacts/physics_v1_20261016/jitter_v1'
FILES = ['diagnose_jitter.py', 'audit_jitter_pipeline.py', 'render_jitter_comparison.py']


def launch(stage):
    remote(f'''from pathlib import Path
p=Path({OUT!r});(p/'code').mkdir(parents=True,exist_ok=True);(p/'assets').mkdir(exist_ok=True)
''')
    for name in FILES:
        subprocess.run(['scp','-q','-o','BatchMode=yes',str(REPO/'reproduction'/name),'motion-data:'+OUT+'/code/'],
                       capture_output=True,check=True,timeout=60,creationflags=HIDDEN)
    hashes = {name: hashlib.sha256((REPO/'reproduction'/name).read_bytes()).hexdigest() for name in FILES}
    python = '/root/autodl-tmp/envs/salad_1016/bin/python' if stage == 'replay' else '/root/miniconda3/envs/mdm5090/bin/python'
    command = [python,'-u',OUT+'/code/'+('render_jitter_comparison.py' if stage == 'render' else 'audit_jitter_pipeline.py')]
    if stage != 'render':command.append(stage)
    command += ['--root',SERVER,'--source',SOURCE,'--output',OUT]
    print(remote(f'''from pathlib import Path
import hashlib,json,subprocess,os
p=Path({OUT!r})
expected={hashes!r}
if {{n:hashlib.sha256((p/'code'/n).read_bytes()).hexdigest() for n in expected}}!=expected:raise ValueError('Code differs')
if (p/{(stage+'_launch.json')!r}).exists():
 old=json.loads((p/{(stage+'_launch.json')!r}).read_text())
 try:os.kill(old['pid'],0)
 except ProcessLookupError:pass
 else:raise ValueError('Stage already running')
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (p/{(stage+'.log')!r}).open('ab') as log:
 proc=subprocess.Popen({command!r},cwd={SERVER!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,code_sha256=expected)
(p/{(stage+'_launch.json')!r}).write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''').decode())


def status():
    print(remote(f'''from pathlib import Path
import json,os
p=Path({OUT!r})
for stage in ['replay','convert','render']:
 if (p/(stage+'_launch.json')).exists():
  value=json.loads((p/(stage+'_launch.json')).read_text())
  try:os.kill(value['pid'],0);value['alive']=True
  except ProcessLookupError:value['alive']=False
  print(stage,json.dumps(value))
 if (p/(stage+'.log')).exists():print((p/(stage+'.log')).read_text(errors='replace')[-1800:])
for name in ['generation_audit.json','conversion_audit.json']:
 if (p/name).exists():
  value=json.loads((p/name).read_text());value.pop('rotation_branch',None);value.pop('cases',None)
  print(name,json.dumps(value))
''').decode())


def verify():
    print(remote(f'''from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
from scipy.fft import dct,idct
p=Path({OUT!r});source=Path({SOURCE!r});root=Path({SERVER!r})
command=[sys.executable,str(p/'code/diagnose_jitter.py'),'--source',str(source),'--clip','jump_20261016','--assert-bone-jitter']
result=subprocess.run(command,capture_output=True,text=True,timeout=30)
if result.returncode!=1 or 'Rapid bone-length oscillation reproduced' not in result.stderr:
 raise ValueError('Original bone-length symptom was not reproduced')
raw=json.loads(result.stdout)
audit=json.loads((p/'conversion_audit.json').read_text())
parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
fitted=[]
for case in audit['cases']:
 for stage in ['raw','fixed_bones','author_foot_ik']:
  path=p/'server_only'/(case['name']+'_'+stage+'.npz')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=case['stages'][stage]['sha256']:raise ValueError('Cached comparison changed')
  if stage=='raw':continue
  with np.load(path) as cache:xyz=cache['xyz']
  lengths=np.linalg.norm(xyz[:,1:]-xyz[:,parents[1:]],axis=-1)
  spectrum=dct(lengths,axis=0,norm='ortho')
  spectrum[np.arange(len(xyz))*20/(2*len(xyz))<3]=0
  high=idct(spectrum,axis=0,norm='ortho')
  fraction=float((np.sqrt(np.mean(high**2,axis=0))/lengths.mean(axis=0)).mean())
  if fraction>1e-8:raise ValueError('Fixed skeleton still changes bone lengths')
  fitted.append(dict(name=case['name'],stage=stage,bone_hf_cv_mean=fraction))
for row in json.loads((source/'examples.json').read_text()):
 if hashlib.sha256((source/'arrays'/row['array']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Original motion changed')
for name,digest in json.loads((source/'binding.json').read_text())['assets'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Generation asset changed')
if hashlib.sha256((p/'assets/template.bvh').read_bytes()).hexdigest()!=audit['template_sha256']:raise ValueError('Template changed')
receipt=dict(status='passed',original_command=command,original_returncode=result.returncode,
 original_assertion=result.stderr.strip().splitlines()[-1],original_measurement=raw,
 fitted_bone_length_checks=fitted,original_arrays_and_assets_unchanged=True,
 scope='Bone-length oscillation only; root-height and perceptual naturalness are not repaired',naturalness_fixed=False)
(p/'bone_jitter_regression.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(status='passed',original_returncode=result.returncode,fitted_checks=len(fitted),naturalness_fixed=False)))
''', timeout=55).decode())


def fetch():
    snapshot = json.loads(remote(f'''from pathlib import Path
import base64,hashlib,json
p=Path({OUT!r});files={{}}
for name in ['generation_audit.json','conversion_audit.json','replay_launch.json','convert_launch.json','videos.json','render_launch.json','bone_jitter_regression.json']:
 if (p/name).exists():
  content=(p/name).read_bytes()
  files[name]=dict(content=base64.b64encode(content).decode(),sha256=hashlib.sha256(content).hexdigest())
print(json.dumps(files))
'''))
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name, entry in snapshot.items():
        content = base64.b64decode(entry['content'])
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Evidence transfer differs')
        (LOCAL/name).write_bytes(content)
    media = {}
    if 'videos.json' in snapshot:
        manifest = json.loads((LOCAL/'videos.json').read_text(encoding='utf-8'))
        for entry in manifest['videos']:
            for key, digest in [('path','sha256'),('poster','poster_sha256')]:
                relative = Path(entry[key])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid media path')
                target = LOCAL/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != entry[digest]:
                    subprocess.run(['scp','-q','-o','BatchMode=yes','motion-data:'+OUT+'/'+relative.as_posix(),str(target)],
                                   capture_output=True,check=True,timeout=60,creationflags=HIDDEN)
                if hashlib.sha256(target.read_bytes()).hexdigest() != entry[digest]:
                    raise ValueError('Media checksum differs')
                media[relative.as_posix()] = entry[digest]
    (LOCAL/'transfer.json').write_text(json.dumps(dict(status='passed',files_sha256={n:e['sha256'] for n,e in snapshot.items()},media_sha256=media),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='synced', evidence=len(snapshot), media=len(media))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['replay','convert','render','status','fetch','verify'])
    stage = parser.parse_args().stage
    if stage == 'status':status()
    elif stage == 'fetch':fetch()
    elif stage == 'verify':verify()
    else:launch(stage)
