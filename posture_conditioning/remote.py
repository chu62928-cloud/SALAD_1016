"""SSH orchestration; no credentials or datasets are copied into Git."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SERVER = '/root/autodl-tmp/SALAD_posture_v1'
PYTHON = '/root/autodl-tmp/envs/salad_1016/bin/python'
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def remote(code, timeout=55):
    encoded = base64.b64encode(code.encode()).decode()
    command = PYTHON + ' -c "import base64;exec(base64.b64decode(\'' + encoded + '\'))"'
    result = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', 'motion-data', command],
                            capture_output=True, timeout=timeout, creationflags=HIDDEN)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout.decode(errors='replace')


def inspect():
    print(remote('''
from pathlib import Path
import json,numpy as np
r=Path('/root/autodl-tmp/data_preprocess_motion/data_self')
c=json.loads((r/'candidate_manifest_v2.jsonl').open().readline())
m=json.loads((r/c['reference_metadata']['file']).read_text())
p=np.load(r/c['posture']['file'])
print('static', [(k,p[k].shape) for k in p.files])
i=m['fitted_representative_index'];print('pose',p['fullpose'][i,:48].tolist())
print('joints',p['joints'][i,:17].tolist())
print('text', (r/c['text']).read_text() if 'text' in c else (r/('texts/'+c['sample_id']+'.txt')).read_text())
print('split', (r/'metadata/identity_splits.json').read_text()[:1800])
for path in [Path('/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz'),Path('/root/autodl-tmp/ViMo/data/body_models/smplx/SMPLX_NEUTRAL.npz')]:
 print(str(path),path.exists())
 if path.exists():
  a=np.load(path,allow_pickle=True);print([(k,a[k].shape) for k in a.files if k in ['v_template','J_regressor','kintree_table']])
'''))


def status():
    print(remote('''
from pathlib import Path
import json,subprocess
r=Path('/root/autodl-tmp/SALAD_posture_v1')
for f in sorted((r/'runs/posture_v1').rglob('status.json')):
 print(str(f.relative_to(r)),f.read_text())
for f in sorted((r/'runs/posture_v1').glob('*.log')):
 print(str(f.relative_to(r)),f.read_text(errors='replace')[-2200:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True))
'''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'status', 'code'])
    parser.add_argument('--code')
    args = parser.parse_args()
    if args.action == 'code':
        print(remote(args.code))
    else:
        globals()[args.action]()
