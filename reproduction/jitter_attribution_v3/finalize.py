"""Freeze final evidence after numerical, rendering, browser, and transfer checks."""
import json
from pathlib import Path
import subprocess

from run import ROOT,OUT,sha,save


def main():
    source=Path(__file__).parent
    phases={s:json.loads((OUT/s/'complete.json').read_text()) for s in ['smoke','vae','generated','sampling']}
    assert {s:r['rows'] for s,r in phases.items()}==dict(smoke=18,vae=13095,generated=60099,sampling=270)
    for s,status in phases.items():
        for name in ['run.py','metrics.py','protocol.json','test_metrics.py']:
            assert sha(source/name)==status['source_sha256'][name]
    jobs={}
    for name in ['vae','generated','sampling','render','summary']:
        pid=int((OUT/(name+'.pid')).read_text());file=Path('/proc',str(pid),'cmdline')
        alive=file.exists() and b'jitter_attribution_v3' in file.read_bytes()
        assert not alive
        jobs[name]=dict(pid=pid,alive=alive)
    assert json.loads((OUT/'summary/independent_verification.json').read_text())['status']=='passed'
    assert json.loads((OUT/'qa/browser.json').read_text())['status']=='passed'
    media=json.loads((OUT/'media/complete.json').read_text())
    for item in media['videos']:
        assert sha(OUT/'media'/item['file'])==item['sha256']
        assert sha(OUT/'media'/item['poster'])==item['poster_sha256']
    tests=subprocess.run(['/root/autodl-tmp/envs/salad_1016/bin/python',
        'reproduction/jitter_attribution_v3/test_metrics.py'],cwd=ROOT,capture_output=True,text=True)
    assert tests.returncode==0
    result=dict(status='complete',phase_rows={s:r['rows'] for s,r in phases.items()},diagnostic_jobs=jobs,
        all_diagnostic_jobs_stopped=True,tests=dict(returncode=tests.returncode,output=tests.stderr.strip()),
        numerical_verification='summary/independent_verification.json',flat_contact_replay='summary/flat_walking.json',
        browser_qa='qa/browser.json',images_png_only=True,training_started=False,model_fixed=False,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        sources={p.name:sha(p) for p in source.iterdir() if p.is_file() and p.suffix in ['.py','.json','.md','.cjs']},
        camera_correction='Initial cross-model shared-root camera made subjects small; now common scale/height with each column following its own root. Motion and metrics unchanged. Initial media archived server-only.',
        files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))
            for folder in ['summary','figures','media','qa'] for p in (OUT/folder).rglob('*') if p.is_file()})
    result['files'].update({name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name))
                            for name in ['index.html','REPORT_zh.md','protocol.json']})
    save(OUT/'FINAL_VERIFICATION.json',result)
    print('FINAL COMPLETE',len(result['files']),len(result['sources']),flush=True)


if __name__=='__main__':main()
