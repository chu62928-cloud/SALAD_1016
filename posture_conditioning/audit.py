"""Recompute evidence from persisted arrays, independently of reported summaries."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import torch
from .common import ROOT,config,sha,write_json
from .data import Data
from .evaluate import VARIANTS,physical


def main():
    cfg=config()
    seed=cfg['seed']
    data=Data()
    run=ROOT/cfg['run_root']
    evaluation=run/'evaluation'/str(seed)
    bindings=json.loads((evaluation/'bindings.json').read_text())
    assert bindings['data']==data.bindings
    assert bindings['evaluation_source_sha256']==sha(ROOT/'posture_conditioning/evaluate.py')
    assert bindings['sample_source_sha256']==sha(ROOT/'posture_conditioning/sample.py')
    checks={}
    train_source=[]
    for variant in cfg['variants']:
        folder=run/variant/str(seed)
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='complete' and status['step']==10000 and status['frozen_backbone_unchanged']
        source=json.loads((folder/'source_binding.json').read_text())
        for name,value in source['files'].items():
            assert sha(ROOT/name)==value
        assert sha(ROOT/'models/denoiser/transformer.py')==source['transformer_sha256']
        assert sha(ROOT/'configs/posture_v1.json')==source['config_sha256']
        train_source.append(source)
        history=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
        best=min(history,key=lambda r:r['validation_subject_mean'])
        state=torch.load(folder/'checkpoints/best.pt',map_location='cpu')
        assert state['step']==best['step'] and state['best']==best['validation_subject_mean']
        assert state['bindings']==data.bindings and state['sample_position']==state['step']*64
    assert train_source[0]==train_source[1]==train_source[2]
    checks['training_completed_frozen_sources_and_validation_selection']=True
    test={i for i,r in enumerate(data.rows) if r['split']=='test'}
    counts={};largest_physics_error=0.;fid_error=0.;xyz_error=0.
    for variant in VARIANTS:
        summary=json.loads((evaluation/variant/'generation.json').read_text())
        for repeat in range(cfg['evaluation_repeats']):
            seen=[];gt=[];text=[];motion=[];hits=[];matching=[]
            for path in sorted((evaluation/variant/str(repeat)).glob('*.npz')):
                p_rows=json.loads(path.with_suffix('.json').read_text())
                with np.load(path) as a:
                    ids=a['ids'].tolist();seen+=ids
                    assert len(ids)==len(p_rows)
                    assert set(a['aligned'].tolist())==set(ids)
                    assert np.isfinite(a['normalized_motion']).all()
                    distance=cdist(a['text'].astype('float64'),a['motion'].astype('float64'))
                    ranked=np.argsort(distance,axis=1,kind='stable')
                    hit=(ranked[:,:3]==np.arange(len(ids))[:,None]).any(1)
                    assert np.array_equal(hit,a['r3']) and np.array_equal(np.diag(distance),a['matching'])
                    hits.extend(hit.tolist());matching.extend(np.diag(distance).tolist())
                    gt.append(a['gt']);text.append(a['text']);motion.append(a['motion'])
                    for k,(i,p) in enumerate(zip(ids,p_rows)):
                        assert p['sample_id']==data.rows[i]['sample_id']
                        assert p['identity']==data.rows[i]['identity_group']
                        xyz=a['xyz'][k,:int(a['lengths'][k])]
                        measured=physical(xyz)
                        for key,value in measured.items():
                            if value is None:
                                assert p[key] is None
                            else:
                                largest_physics_error=max(largest_physics_error,abs(p[key]-value))
                    # One whole batch per variant/repeat rechecks decoding from saved
                    # full motion features, rather than trusting xyz alone.
                    if path.name=='0000.npz':
                        from utils.motion_process import recover_from_ric
                        mean=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
                        std=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
                        for k in range(len(ids)):
                            length=int(a['lengths'][k])
                            raw=torch.tensor(a['normalized_motion'][k,:length],device='cuda')*torch.tensor(std,device='cuda')+torch.tensor(mean,device='cuda')
                            recovered=recover_from_ric(raw.float(),22).cpu().numpy()
                            xyz_error=max(xyz_error,float(np.max(np.abs(recovered-a['xyz'][k,:length]))))
            assert set(seen)==test and len(seen)==len(test)
            gt=np.concatenate(gt).astype('float64');motion=np.concatenate(motion).astype('float64')
            a_mean,b_mean=gt.mean(0),motion.mean(0)
            a_cov,b_cov=np.cov(gt,rowvar=False),np.cov(motion,rowvar=False)
            product=sqrtm(a_cov@b_cov)
            independent=float(np.sum((a_mean-b_mean)**2)+np.trace(a_cov+b_cov-2*product.real))
            reported=summary['repeats'][repeat]
            fid_error=max(fid_error,abs(independent-reported['fid']))
            assert abs(np.mean(hits)-reported['r3'])<1e-12
            assert abs(np.mean(matching)-reported['matching'])<1e-12
            counts[variant+'/'+str(repeat)]=len(seen)
    assert largest_physics_error<1e-12
    assert fid_error<1e-4
    assert xyz_error<1e-6
    checks.update(full_test_generation_coverage=True,retrieval_recomputed=True,independent_fid=True,
                  all_physics_recomputed=True,motion_feature_recovery_checked=True)
    preflight=json.loads((run/'preflight/summary.json').read_text())
    geometry=json.loads((run/'preflight/geometry.json').read_text())
    assert preflight['status']==geometry['status']=='passed'
    assert geometry['subjects']==369
    checks.update(preflight['checks'])
    checks['native_coordinate_convention_all_subjects']=True
    artifact_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert artifact_bytes<8*1024**3
    receipt=dict(status='passed',checks=checks,generation_counts=counts,physics_max_error=largest_physics_error,
                 independent_fid_max_error=fid_error,feature_to_xyz_max_error=xyz_error,
                 native_geometry_max_error=geometry['max_error_m'],
                 external_reference_asset_identical_to_fitting_asset=False,
                 external_reference_asset_max_difference_m=geometry['external_asset_max_error_m'],
                 artifact_bytes=artifact_bytes,training_source_binding=train_source[0],
                 audit_source_sha256=sha(__file__),naturalness_fixed=False)
    write_json(ROOT/'results/posture_v1/verification.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
