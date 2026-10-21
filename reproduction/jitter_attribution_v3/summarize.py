"""Subject-balanced summaries and paired uncertainty for the frozen diagnostics."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

from run import OUT, PROTOCOL, save, sha

META = {'dataset','variant','repeat','sample_id','subject','source','task','frames','path'}
PRIMARY = ['bone_length_cv','bone_hf3_cv','local_jerk_rms_mps3','local_acceleration_rms_mps2',
    'local_hf3_rms_m','local_hf3_energy_fraction','skating_frame_ratio','skating_given_contact_ratio',
    'near_floor_toe_interval_fraction','near_floor_speed_mps','penetration_frame_ratio',
    'mpjpe_root_aligned_m','acceleration_error_local_mps2','position_rotation_discrepancy_m',
    'position_rotation_acceleration_discrepancy_mps2','interior_local_jerk_rms_mps3',
    'world_jerk_rms_mps3','root_jerk_rms_mps3']


def read_clips(stage):
    rows = list(csv.DictReader((OUT/stage/'clip_metrics.csv').open()))
    for row in rows:
        for key in row.keys()-META:
            row[key] = float(row[key]) if row[key] else None
    return rows


def reduce_rows(rows, keys, metrics):
    sums=defaultdict(lambda:np.zeros(len(metrics),dtype=np.float64))
    counts=defaultdict(lambda:np.zeros(len(metrics),dtype=np.int64))
    for row in rows:
        key=tuple(row[k] for k in keys)
        for i,metric in enumerate(metrics):
            value=row.get(metric)
            if value is not None and np.isfinite(value):
                sums[key][i]+=value;counts[key][i]+=1
    output=[]
    for key,total in sums.items():
        record=dict(zip(keys,key))
        record.update({m:float(total[i]/counts[key][i]) if counts[key][i] else None for i,m in enumerate(metrics)})
        output.append(record)
    return output


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def subjects(rows, metrics):
    clips=reduce_rows(rows,['dataset','variant','path','subject','sample_id'],metrics)
    return reduce_rows(clips,['dataset','variant','path','subject'],metrics)


def summarize(rows, metrics):
    individuals=subjects(rows,metrics)
    groups=defaultdict(list)
    for row in individuals:groups[(row['dataset'],row['variant'],row['path'])].append(row)
    output=[]
    for (dataset,variant,path),group in sorted(groups.items()):
        rng=np.random.default_rng(PROTOCOL['bootstrap_seed'])
        draw=rng.integers(len(group),size=(PROTOCOL['bootstrap_repetitions'],len(group)))
        info={}
        for metric in metrics:
            x=np.array([r[metric] if r[metric] is not None else np.nan for r in group])
            if not np.isfinite(x).any(): continue
            valid_x=x[np.isfinite(x)]
            metric_draw=rng.integers(len(valid_x),size=(PROTOCOL['bootstrap_repetitions'],len(valid_x)))
            boot=valid_x[metric_draw].mean(axis=1)
            info[metric]=dict(mean=float(np.nanmean(x)),ci95=np.quantile(boot,[.025,.975]).tolist(),
                              subjects_nonmissing=int(np.isfinite(x).sum()))
        selected=[r for r in rows if (r['dataset'],r['variant'],r['path'])==(dataset,variant,path)]
        output.append(dict(dataset=dataset,variant=variant,path=path,subjects=len(group),
            clips=len({r['sample_id'] for r in selected}),repeats=len({r['repeat'] for r in selected}),metrics=info))
    return output,individuals


def compare(individuals, dataset, a, b, path):
    pairs={variant:{r['subject']:r for r in individuals if r['dataset']==dataset and r['variant']==variant and r['path']==path}
           for variant in [a,b]}
    ids=sorted(pairs[a].keys() & pairs[b].keys())
    assert ids and pairs[a].keys()==pairs[b].keys()
    rng=np.random.default_rng(PROTOCOL['bootstrap_seed'])
    draw=rng.integers(len(ids),size=(PROTOCOL['bootstrap_repetitions'],len(ids)))
    info={}
    for metric in PRIMARY:
        x=np.array([pairs[a][s].get(metric) if pairs[a][s].get(metric) is not None else np.nan for s in ids])
        y=np.array([pairs[b][s].get(metric) if pairs[b][s].get(metric) is not None else np.nan for s in ids])
        valid=np.isfinite(x)&np.isfinite(y)
        if not valid.any():continue
        difference=y-x
        valid_difference=difference[valid]
        metric_draw=rng.integers(len(valid_difference),size=(PROTOCOL['bootstrap_repetitions'],len(valid_difference)))
        boot=valid_difference[metric_draw].mean(axis=1)
        mean_a,mean_b=float(np.nanmean(x[valid])),float(np.nanmean(y[valid]))
        info[metric]=dict(a=mean_a,b=mean_b,delta_b_minus_a=float(np.nanmean(difference)),
            delta_ci95=np.quantile(boot,[.025,.975]).tolist(),
            percent_change=100*(mean_b/mean_a-1) if abs(mean_a)>1e-6 else None,
            subjects_nonmissing=int(valid.sum()))
    return dict(dataset=dataset,a=a,b=b,path=path,subjects=len(ids),metrics=info)


def path_compare(individuals,variant,a,b):
    selected=[dict(r,variant=r['path'],path='comparison') for r in individuals if r['variant']==variant]
    record=compare(selected,'data_self',a,b,'comparison')
    record.update(variant=variant,comparison='recovery_path')
    return record


def anatomical(stage,kind):
    path=OUT/stage/(kind+'_metrics.csv')
    index='joint'
    fields= ['length_cv','hf3_length_cv','mean_length_m'] if kind=='bone' else [
        'world_speed_rms_mps','world_acceleration_rms_mps2','world_jerk_rms_mps3','local_jerk_rms_mps3']
    sums=defaultdict(lambda:np.zeros(len(fields)))
    counts=defaultdict(int)
    names={}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key=tuple(row[k] for k in ['dataset','variant','path','subject',index])
            sums[key]+=np.array([float(row[m]) for m in fields]);counts[key]+=1
            names[row[index]]=row['name']
    grouped=defaultdict(list)
    for key,total in sums.items():grouped[(key[0],key[1],key[2],key[4])].append(total/counts[key])
    output=[]
    for key,values in sorted(grouped.items()):
        record=dict(zip(['dataset','variant','path','joint'],key),name=names[key[3]],subjects=len(values))
        record.update(dict(zip(fields,np.mean(values,axis=0).tolist())))
        output.append(record)
    return output


def main():
    for stage,expected in [('vae',13095),('generated',60099),('sampling',270)]:
        status=json.loads((OUT/stage/'complete.json').read_text())
        assert status['status']=='complete' and status['rows']==expected
        # All recorded outputs and frozen inputs must still match the completion snapshot.
        for name,entry in status['files'].items():assert sha(OUT/stage/name)==entry['sha256']
        inputs=json.loads((OUT/stage/'input_binding.json').read_text())
        for name,digest in inputs.items():assert sha(name)==digest
    output=OUT/'summary';output.mkdir(exist_ok=True)
    raw={stage:read_clips(stage) for stage in ['vae','generated','sampling']}
    all_rows=sum(raw.values(),[])
    metrics=sorted(all_rows[0].keys()-META)
    summaries,individuals=summarize(all_rows,metrics)
    comparisons=[]
    for dataset in ['humanml','data_self']:
        for mode in ['posterior_mean','posterior_sample']:
            for path in PROTOCOL['recovery_paths']:
                comparisons.append(compare(individuals,dataset,'ground_truth',mode,path))
    for a,b in [('official','adapted'),('adapted','fixed'),('adapted','paired'),('fixed','paired'),('paired','shuffled')]:
        for path in PROTOCOL['recovery_paths']:
            comparisons.append(compare(individuals,'data_self',a,b,path))
    for variant in ['ground_truth','posterior_mean','posterior_sample','official','adapted','fixed','paired','shuffled']:
        for other in ['position_fixed','rotation_fk']:
            comparisons.append(path_compare(individuals,variant,'position',other))
    target=raw['generated']
    for variant in PROTOCOL['target_generated_variants']:
        subset=[r for r in target if r['dataset']=='data_self' and r['variant']==variant]
        assert len(subset)==1327*3*3
        assert len({r['sample_id'] for r in subset})==1327
        assert len({r['subject'] for r in subset})==57
        assert all(r['mpjpe_root_aligned_m'] is None for r in subset)
    save(output/'summary.json',summaries)
    save(output/'paired_comparisons.json',comparisons)
    write_csv(output/'subject_metrics.csv',individuals)
    for kind in ['bone','joint']:
        for stage in ['vae','generated']:
            write_csv(output/(stage+'_'+kind+'_summary.csv'),anatomical(stage,kind))
    flat=[]
    for group in summaries:
        record={k:group[k] for k in ['dataset','variant','path','subjects','clips','repeats']}
        record.update({k:v['mean'] for k,v in group['metrics'].items()})
        flat.append(record)
    fields=sorted(set().union(*(r.keys() for r in flat)))
    with (output/'group_means.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    # VAE error differences have an aligned reference; generated sequences do not.
    save(output/'verification.json',dict(status='passed',stage_rows={k:len(v) for k,v in raw.items()},
        shared_subjects=57,shared_clips=1327,input_hashes_unchanged=True,output_hashes_verified=True,
        bootstrap_repetitions=PROTOCOL['bootstrap_repetitions'],generation_framewise_gt_error=False,
        definition='Repeats within clip, clips within subject, equal subject weights; paired subject bootstrap.'))
    print('SUMMARY COMPLETE',flush=True)


if __name__=='__main__':main()
