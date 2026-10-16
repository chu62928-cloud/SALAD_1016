"""Index reproducible server artifacts and verify the experiment queue stopped."""
import json
from pathlib import Path
import torch
from .common import ROOT,config,sha,write_json


def main():
    run=ROOT/config()['run_root']
    campaign=json.loads((run/'campaign_status.json').read_text())
    assert campaign['status']=='complete' and campaign['training_stopped']
    jobs=json.loads((run/'jobs.json').read_text())
    alive={}
    for variant,pid in jobs.items():
        path=Path('/proc')/str(pid)/'cmdline'
        alive[variant]=path.exists() and b'posture_conditioning.train' in path.read_bytes()
    assert not any(alive.values())
    checkpoints=[]
    for folder in sorted(run.glob('*/*/checkpoints')):
        for path in sorted(folder.glob('*.pt')):
            state=torch.load(path,map_location='cpu')
            checkpoints.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                                    sha256=sha(path),step=state['step'],variant=state['variant'],seed=state['seed']))
    files=[]
    for folder in [run/'evaluation',run/'preflight',run/'media']:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ['.npz','.pt','.json','.csv','.mp4','.png']:
                files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    total=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert total<8*1024**3
    write_json(ROOT/'results/posture_v1/artifact_index.json',dict(status='complete',server_root=str(ROOT),
               training_jobs_alive=alive,training_stopped=True,bytes=total,checkpoints=checkpoints,files=files,
               campaign=campaign,data_and_individual_media_uploaded_to_github=False))
    print('Artifact index complete; stopped jobs; bytes',total,flush=True)


if __name__=='__main__':
    main()
