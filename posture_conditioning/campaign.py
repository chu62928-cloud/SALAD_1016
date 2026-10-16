"""Bounded experiment queue: finish the frozen protocol, publish evidence, stop."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT,config,write_json,sha


def check_training(seed):
    run=ROOT/config()['run_root']
    return all((run/v/str(seed)/'status.json').exists() and
               json.loads((run/v/str(seed)/'status.json').read_text()).get('status')=='complete'
               for v in config()['variants'])


def run_command(module,args,log):
    with log.open('ab') as f:
        result=subprocess.run([sys.executable,'-u','-m',module]+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Failed '+module+'; inspect '+str(log))


def main():
    run=ROOT/config()['run_root']
    run.mkdir(parents=True,exist_ok=True)
    lock=(run/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=run/'campaign_status.json'
    seed=config()['seed']
    try:
        jobs=json.loads((run/'jobs.json').read_text())
        while not check_training(seed):
            for variant,pid in jobs.items():
                state=run/variant/str(seed)/'status.json'
                complete=state.exists() and json.loads(state.read_text()).get('status')=='complete'
                proc=Path('/proc')/str(pid)/'cmdline'
                if not complete and (not proc.exists() or b'posture_conditioning.train' not in proc.read_bytes()):
                    raise RuntimeError('Training stopped before completion: '+variant)
            write_json(status,dict(status='training',seed=seed,pid=os.getpid()))
            time.sleep(15)
        write_json(status,dict(status='evaluating',seed=seed,pid=os.getpid()))
        run_command('posture_conditioning.evaluate',['--seed',str(seed)],run/'evaluation.log')
        run_command('posture_conditioning.report',[],run/'report.log')
        decision=json.loads((ROOT/'results/posture_v1/decision.json').read_text())
        if decision['expand_training_seeds']:
            for extra in [20261018,20261019]:
                write_json(status,dict(status='replication_training',seed=extra,pid=os.getpid()))
                procs=[]
                for variant in config()['variants']:
                    log=(run/(variant+'_'+str(extra)+'.log')).open('ab')
                    procs.append(subprocess.Popen([sys.executable,'-u','-m','posture_conditioning.train',
                                                  '--variant',variant,'--seed',str(extra)],cwd=ROOT,
                                                 stdout=log,stderr=subprocess.STDOUT))
                for proc in procs:
                    if proc.wait()!=0:
                        raise RuntimeError('Replication training failed')
                run_command('posture_conditioning.evaluate',['--seed',str(extra)],run/('evaluation_'+str(extra)+'.log'))
        write_json(status,dict(status='complete',pid=os.getpid(),single_seed=not decision['expand_training_seeds'],
                               training_stopped=True,finished_at=time.time(),decision_sha256=sha(ROOT/'results/posture_v1/decision.json')))
    except Exception as error:
        write_json(status,dict(status='failed',error=str(error),pid=os.getpid()))
        raise


if __name__=='__main__':
    main()
