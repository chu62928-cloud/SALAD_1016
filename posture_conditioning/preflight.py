"""Real-weight engineering checks before any scientific training run."""
import copy
import json
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .common import ROOT,config,seed_all,write_json
from .data import Data,prepare
from .runtime import runtime
from .train import Engine,backbone_hash


def main():
    seed_all(20261017)
    data=Data()
    model,vae,scheduler=runtime()
    folder=ROOT/config()['run_root']/'preflight'
    folder.mkdir(parents=True,exist_ok=True)
    checks={}
    feature=data.features
    pose=np.zeros(165)
    pose[:3]=Rotation.from_matrix(feature.reference[0]).as_rotvec()
    core,shoulder=feature(pose)
    assert np.max(np.abs(core))<1e-6 and np.max(np.abs(shoulder))<1e-6
    checks['neutral_reference_zero']=True
    pose[3:36]=np.random.RandomState(42).normal(0,.08,33)
    expected=feature(pose)
    changed=pose.copy().reshape(-1,3)
    excluded=[12,15]+list(range(16,55))
    changed[excluded]=np.random.RandomState(99).normal(size=(len(excluded),3))
    actual=feature(changed.reshape(-1))
    assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
    checks['head_neck_arms_hands_invariant']=True
    changed=pose.copy()
    changed[:3]=Rotation.from_matrix(Rotation.from_euler('z',.72).as_matrix() @ Rotation.from_rotvec(pose[:3]).as_matrix()).as_rotvec()
    actual=feature(changed)
    assert all(np.allclose(a,b,atol=1e-6) for a,b in zip(expected,actual))
    checks['global_yaw_invariant']=True
    # Verify left/right interchange with a symmetric synthetic skeleton and reflection.
    symmetry=np.array([0,2,1,3,5,4,6,8,7,9,11,10,12,14,13,15,17,16,19,18,21,20])
    synthetic=copy.deepcopy(feature)
    reflection=np.diag([-1.,1.,1.])
    # SMPL local x points opposite to the canonical right axis.
    synthetic.rest=(feature.rest+(feature.rest[symmetry]@reflection))*.5
    synthetic.basis=np.array([[-1.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
    synthetic.reference=np.tile(np.eye(3),(22,1,1))
    synthetic.reference[0]=synthetic.basis.T
    synthetic.reference_joints,synthetic.reference_global=synthetic.fk(synthetic.reference)
    p=np.zeros((55,3));p[0]=Rotation.from_matrix(synthetic.reference[0]).as_rotvec()
    p[1:12]=np.random.RandomState(55).normal(0,.1,(11,3))
    local=Rotation.from_rotvec(p[:22]).as_matrix()
    mirrored=reflection@local[symmetry]@reflection
    # Root is in world coordinates, so reflect world x and local x.
    mirrored[0]=reflection@local[0]@reflection
    other=p.copy();other[:22]=Rotation.from_matrix(mirrored).as_rotvec()
    a,b=synthetic(p.reshape(-1)),synthetic(other.reshape(-1))
    assert np.allclose(a[0][:,6:9][symmetry[:12]]@reflection,b[0][:,6:9],atol=1e-5)
    assert np.allclose(a[1],-b[1],atol=1e-5)
    checks['left_right_mirror']=True
    prepare(data,model,vae)
    del vae
    data.install_text_cache(model)
    z,mask,text=data.batch([0,1],sample_posterior=False)
    core,shoulder=data.conditions([0,1])
    timestep=torch.tensor([25,500],device='cuda')
    model.eval()
    with torch.no_grad():
        original=model.backbone(z,timestep,text,len_mask=mask)[0]
        zero=model(z,timestep,text,core,shoulder,mask,scale=0)
        initialized=model(z,timestep,text,core,shoulder,mask)
    assert torch.equal(original,zero) and torch.equal(original,initialized)
    checks['zero_scale_and_initialization_exact']=True
    initial=backbone_hash(model)
    engine=Engine(data,model,scheduler,'paired',20261017,10000)
    entries=[]
    started=time.time()
    for _ in range(3):
        entries.append(engine.update())
    assert model.encoder.input.weight.grad.abs().sum()>0
    assert model.blocks[0].posture_adapter.query.weight.grad.abs().sum()>0
    checks['encoder_attention_gate_gradients']=True
    engine.save(folder/'resume.pt')
    reference=engine.update()
    reference_state={k:v.clone() for k,v in model.adapter_state().items()}
    engine.load(folder/'resume.pt')
    resumed=engine.update()
    assert reference==resumed
    assert all(torch.equal(reference_state[k],v) for k,v in model.adapter_state().items())
    checks['optimizer_rng_schedule_exact_resume']=True
    for _ in range(96):
        entry=engine.update()
        entries.append(entry)
        if engine.step%20==0:
            print('short training',json.dumps(entry),flush=True)
    assert backbone_hash(model)==initial
    checks['frozen_backbone_unchanged']=True
    assert all(p.grad is None for n,p in model.backbone.named_parameters() if '.posture_adapter.' not in n)
    checks['frozen_parameters_no_gradients']=True
    secs=time.time()-started
    write_json(folder/'summary.json',dict(status='passed',checks=checks,data=data.bindings,
                                         completed_steps=engine.step,elapsed_sec=secs,
                                         optimizer_seconds=secs/101,
                                         peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                         loss_first=entries[0]['loss'],loss_last=entries[-1]['loss']))
    print((folder/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    main()
