"""Check fitted native rotations against exported joints independently of features."""
from collections import defaultdict
import copy
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT,config,write_json
from .data import Data


def main():
    data=Data()
    with np.load(config()['body_model'],allow_pickle=True) as model:
        template=model['v_template'].astype('float64')
        shapedirs=model['shapedirs'].astype('float64')
        reg=model['J_regressor']
        if reg.shape==():
            reg=reg.item()
    subjects=[json.loads(line) for line in (data.root/'metadata/subjects.jsonl').read_text().splitlines()]
    results=[]
    for subject in subjects:
        ref=next(iter(subject['references'].values()))
        meta=json.loads((data.root/ref['file']).read_text())
        with np.load(data.root/meta['fitted_parameters']['file']) as a:
            index=meta['fitted_representative_index']
            beta=a['betas'].reshape(-1)
            vertices=template+np.einsum('vck,k->vc',shapedirs[...,:len(beta)],beta)
            rest=np.asarray(reg@vertices)[:22]
            feature=copy.deepcopy(data.features)
            feature.rest=rest
            local=Rotation.from_rotvec(a['fullpose'][index,:66].reshape(22,3)).as_matrix()
            fk,_=feature.fk(local)
            prediction=fk+rest[0]+a['trans'][index]
            observed=a['joints'][index,:22]
            errors=np.linalg.norm(prediction-observed,axis=-1)
            # The external reference asset is not proven identical to the fitting
            # asset. Infer constant bone offsets from one frame, then verify every
            # other frame independently to audit the native rotation convention.
            _,global_rotation=feature.fk(local)
            inferred=np.empty((22,3))
            inferred[0]=observed[0]-a['trans'][index]
            for j in range(1,22):
                p=feature.parents[j]
                offset=global_rotation[p].T@(observed[j]-observed[p])
                inferred[j]=inferred[p]+offset
            feature.rest=inferred
            temporal_error=0.
            for frame in range(len(a['fullpose'])):
                local_frame=Rotation.from_rotvec(a['fullpose'][frame,:66].reshape(22,3)).as_matrix()
                predicted,_=feature.fk(local_frame)
                predicted+=inferred[0]+a['trans'][frame]
                temporal_error=max(temporal_error,float(np.linalg.norm(predicted-a['joints'][frame,:22],axis=-1).max()))
            if temporal_error>1e-5:
                raise AssertionError('Native rotation convention failed: '+subject['identity_group']+' '+str(temporal_error))
            results.append(dict(identity=subject['identity_group'],up_axis=meta['native_up_axis'],
                                external_asset_mean_error_m=float(errors.mean()),external_asset_max_error_m=float(errors.max()),
                                inferred_bones_all_frames_error_m=temporal_error))
    write_json(ROOT/config()['run_root']/'preflight/geometry.json',dict(status='passed',
               subjects=len(results),max_error_m=max(r['inferred_bones_all_frames_error_m'] for r in results),
               external_asset_identical_to_fit=False,
               external_asset_max_error_m=max(r['external_asset_max_error_m'] for r in results),rows=results))
    print('Native FK audit passed',len(results),max(r['inferred_bones_all_frames_error_m'] for r in results),flush=True)


if __name__=='__main__':
    main()
