"""Independent formulas, cache pairing, split, and media checks."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from run import OUT,FULL,PROTOCOL,save,sha


def main():
    checks={}
    bindings=json.loads((FULL/'evaluation/20261017/bindings.json').read_text())
    backbone=json.loads((FULL/'adaptation/backbone_binding.json').read_text())
    assert sha(backbone['checkpoint'])==backbone['checkpoint_sha256']
    assert bindings['data']['manifest']==sha(FULL/'full_data_root/manifest.jsonl')
    assert len(bindings['models'])==5
    adapted=next(r for r in bindings['models'] if r['variant']=='adapted')
    for item in bindings['models']:
        if item['variant'] in ['fixed','paired','shuffled']:
            file=FULL/item['variant']/'20261017/checkpoints/best.pt'
            assert sha(file)==item['checkpoint_sha256']
            assert item['frozen_backbone_sha256']==adapted['frozen_backbone_sha256']
    checks['model_bindings']=dict(selected_backbone_sha256=backbone['checkpoint_sha256'],
        models=bindings['models'],binding_sha256=sha(FULL/'evaluation/20261017/bindings.json'))
    manifest=[json.loads(line) for line in (FULL/'full_data_root/manifest.jsonl').read_text().splitlines()]
    groups={s:{r['identity_group'] for r in manifest if r['split']==s} for s in ['train','validation','test']}
    assert [len(groups[s]) for s in groups]==[256,56,57]
    assert not any(groups[a]&groups[b] for a,b in [('train','test'),('train','validation'),('validation','test')])
    checks['subject_disjoint_split']={s:len(v) for s,v in groups.items()}
    metadata={}
    for variant in PROTOCOL['target_generated_variants']:
        count=0
        for repeat in PROTOCOL['generation_repeats']:
            folder=FULL/'evaluation/20261017'/variant/str(repeat)
            for file in sorted(folder.glob('[0-9][0-9][0-9][0-9].npz')):
                with np.load(file) as a:
                    for index,length,seed,text in zip(a['ids'],a['lengths'],a['seeds'],a['captions']):
                        key=(repeat,int(index));value=(int(length),int(seed),str(text))
                        if variant=='official':
                            assert key not in metadata;metadata[key]=value
                        else: assert metadata[key]==value
                        count+=1
        assert count==1327*3
    checks['same_generated_ids_lengths_seeds_texts_across_five_models']=True
    checks['generated_paired_clip_seed_records']=len(metadata)
    parents=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
    errors=defaultdict(float)
    sample_checks=0
    for stage in ['vae','sampling','generated']:
        selected=sorted((OUT/stage/'examples').glob('*.npz'))[:12]
        rows=list(csv.DictReader((OUT/stage/'clip_metrics.csv').open()))
        lookup={f"{r['dataset']}_{r['variant']}_{r['repeat']}_{r['sample_id']}":r for r in rows if r['path']=='position'}
        for file in selected:
            with np.load(file) as a:p=a['position_xyz'].astype(np.float64)
            row=lookup[file.stem];local=p-p[:,:1]
            lengths=np.sqrt(np.sum((p[:,1:]-p[:,parents[1:]])**2,axis=-1))
            cv=np.mean(np.std(lengths,axis=0)/np.mean(lengths,axis=0))
            jerk=(local[3:]-3*local[2:-1]+3*local[1:-2]-local[:-3])*8000
            acceleration=(local[2:]-2*local[1:-1]+local[:-2])*400
            jr=np.sqrt(np.sum(jerk**2,axis=-1).mean())
            ar=np.sqrt(np.sum(acceleration**2,axis=-1).mean())
            # Independent cosine matrix, not scipy's DCT implementation.
            n=len(p);k=np.arange(n)
            transform=np.sqrt(2/n)*np.cos(np.pi/n*k[:,None]*(np.arange(n)[None,:]+.5))
            transform[0]/=np.sqrt(2)
            coefficients=np.einsum('kt,tjc->kjc',transform,local-local.mean(0))
            hf=np.sqrt(np.sum(coefficients[k*20/(2*n)>=3]**2)/local.size)
            for metric,value in [('bone_length_cv',cv),('local_jerk_rms_mps3',jr),
                ('local_acceleration_rms_mps2',ar),('local_hf3_rms_m',hf)]:
                difference=abs(value-float(row[metric]));errors[metric]=max(errors[metric],difference)
                assert difference<1e-8,(file.name,metric,difference)
            sample_checks+=1
    checks['independent_analytic_formula_examples']=sample_checks
    checks['independent_formula_max_absolute_error']=dict(errors)
    # Independently group one core endpoint from the clip table to check aggregation order.
    rows=list(csv.DictReader((OUT/'generated/clip_metrics.csv').open()))
    endpoint='local_jerk_rms_mps3'
    for variant in PROTOCOL['target_generated_variants']:
        by_clip=defaultdict(list);by_subject=defaultdict(list)
        for r in rows:
            if r['dataset']=='data_self' and r['variant']==variant and r['path']=='position':
                by_clip[(r['subject'],r['sample_id'])].append(float(r[endpoint]))
        for (subject,_),values in by_clip.items():assert len(values)==3;by_subject[subject].append(np.mean(values))
        independent=float(np.mean([np.mean(values) for values in by_subject.values()]))
        summary=json.loads((OUT/'summary/summary.json').read_text())
        selected=next(r for r in summary if r['dataset']=='data_self' and r['variant']==variant and r['path']=='position')
        assert abs(selected['metrics'][endpoint]['mean']-independent)<1e-10
    checks['independent_subject_equal_weight_aggregation']=True
    media=json.loads((OUT/'media/complete.json').read_text())
    for row in media['videos']:
        file=OUT/'media'/row['file'];assert sha(file)==row['sha256']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(file)]))['streams'][0]
        assert (probe['width'],probe['height'],probe['r_frame_rate'],int(probe['nb_read_frames']))==(1920,1430,'20/1',row['frames'])
    checks['three_videos_valid_dimensions_fps_frames']=True
    save(OUT/'summary/independent_verification.json',dict(status='passed',checks=checks,
        limits='Numerical and media checks do not certify naturalness or causally isolate the training loss.'))
    print('INDEPENDENT VERIFICATION PASSED',flush=True)


if __name__=='__main__':main()
