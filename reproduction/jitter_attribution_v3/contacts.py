"""Separate flat-ground walking contact diagnostics from all-task kinematics."""
import json
import numpy as np

from run import OUT,FULL,PROTOCOL,save
from summarize import read_clips,summarize,compare,PRIMARY


def flat(task):
    task=task.lower()
    return any(w in task for w in ['walk','gait']) and not any(w in task for w in
        ['stair','step','ramp','slope','uphill','downhill','incline','decline'])


def main():
    rows=read_clips('vae')+read_clips('generated')
    rows=[r for r in rows if r['dataset']=='data_self' and flat(r['task'])]
    assert rows
    metrics=PRIMARY+[f'skating_h{round(h*100):02d}_v{round(v*100):03d}'
        for h in PROTOCOL['sensitivity_heights_m'] for v in PROTOCOL['sensitivity_speeds_mps']]
    summary,individuals=summarize(rows,metrics)
    comparisons=[]
    for a,b in [('official','adapted'),('adapted','fixed'),('adapted','paired'),('fixed','paired'),('paired','shuffled')]:
        comparisons.append(compare(individuals,'data_self',a,b,'position'))
    # Reproduce the original flat-only summaries exactly from their frozen caches.
    old_errors={}
    for variant in PROTOCOL['target_generated_variants']:
        frozen=[json.loads((FULL/'evaluation/20261017'/variant/str(repeat)/'physical_summary.json').read_text())
            for repeat in PROTOCOL['generation_repeats']]
        reference=float(np.mean([r['skating_frame_ratio'] for r in frozen]))
        current=next(r for r in summary if r['variant']==variant and r['path']=='position')['metrics']['skating_frame_ratio']['mean']
        old_errors[variant]=abs(current-reference)
        assert abs(current-reference)<1e-10,(variant,current,reference)
    save(OUT/'summary/flat_walking.json',dict(status='complete',
        selection='Same frozen original rule: walk/gait and no stair/step/ramp/slope/uphill/downhill/incline/decline.',
        subjects=len({r['subject'] for r in rows}),clips=len({r['sample_id'] for r in rows}),
        summary=summary,comparisons=comparisons,original_flat_summary_replay_max_errors=old_errors,
        limit='Foot contact is a height-based proxy. Stairs, slopes and other tasks are not valid flat-floor sliding comparisons.'))
    for r in summary:
        if r['path']=='position':print(r['variant'],r['subjects'],r['clips'],r['metrics']['skating_frame_ratio']['mean'])


if __name__=='__main__':main()
