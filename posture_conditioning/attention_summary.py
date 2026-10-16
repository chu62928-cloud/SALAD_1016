"""Export anonymous summaries of attention, gates and actual injected residuals."""
import numpy as np
import torch
from .common import ROOT,config,write_json,sha


def main():
    folder=ROOT/config()['run_root']/'evaluation'/str(config()['seed'])
    values=torch.load(folder/'attention_gate_residual.pt',map_location='cpu')
    with np.load(folder/'intervention_swap.npz') as a:
        lengths=a['lengths']//4
    attention=[];gates=[];residuals=[]
    for layer,v in values.items():
        b,heads,tj,groups=v['alpha'].shape
        alpha=v['alpha'].reshape(b,heads,tj//7,7,groups).numpy()
        gate=v['gate'].squeeze(-1).numpy()
        residual=v['residual_norm'].numpy()
        for subject,length in enumerate(lengths):
            parts=np.array_split(np.arange(int(length)),4)
            attention.append(np.stack([alpha[subject,:,part].mean((0,1,2)) for part in parts],axis=1))
            gates.append(np.stack([gate[subject,part].mean(0) for part in parts],axis=1))
            residuals.append(np.stack([residual[subject,part].mean(0) for part in parts],axis=1))
    result=dict(subjects=len(lengths),layers=len(values),diffusion_step_index=25,
                alpha=np.mean(attention,axis=0).tolist(),gate=np.mean(gates,axis=0).tolist(),
                residual_norm=np.mean(residuals,axis=0).tolist(),
                attention_sha256=sha(folder/'attention_gate_residual.pt'),
                interpretation='Descriptive model internals; attention and gate are not causal effect estimates; motion latent regions overlap')
    assert np.allclose(np.array(result['alpha']).sum(0),1.,atol=1e-6)
    write_json(ROOT/'results/posture_v1/attention_summary.json',result)


if __name__=='__main__':
    main()
