"""Recompute frozen caches from original train inputs and released weights."""
import numpy as np
import torch
from .common import ROOT,config,seed_all,write_json
from .data import Data
from .runtime import runtime,posterior


@torch.no_grad()
def main():
    seed_all(20261017)
    data=Data();data.open_cache()
    model,vae,_=runtime()
    indices=[i for i,r in enumerate(data.rows) if r['split']=='train']
    indices=np.random.RandomState(20261017).choice(indices,32,replace=False).tolist()
    mean=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy')
    std=np.load(ROOT/'checkpoints/t2m/Comp_v6_KLD005/meta/std.npy')
    motion=np.zeros((32,196,263),dtype='float32')
    for k,i in enumerate(indices):
        x=data.motion(i)
        motion[k,:len(x)]=(x-mean)/std
    mu,logvar=posterior(vae,torch.tensor(motion,device='cuda'))
    errors={'posterior_mean':float(np.max(np.abs(mu.cpu().numpy()-data.mu[indices]))),
            'posterior_std':float(np.max(np.abs((.5*logvar).exp().cpu().numpy()-data.std[indices])))}
    captions=[data.rows[i]['captions'][0] for i in indices]
    positions=[data.text_index[c] for c in captions]
    words,masks,token_pos=model.backbone.clip_model.encode_text(captions)
    errors['text_hidden']=float(np.max(np.abs(words.cpu().numpy()-data.words[positions])))
    print('Raw cache audit errors',errors,flush=True)
    assert np.array_equal(masks.cpu().numpy(),data.word_masks[positions])
    assert np.array_equal(token_pos.cpu().numpy(),data.word_pos[positions])
    assert errors['posterior_mean']==errors['posterior_std']==0
    same_batch_max=0.
    for start in range(0,len(data.captions),64):
        expected,mask,pos=model.backbone.clip_model.encode_text(data.captions[start:start+64])
        stop=start+len(expected)
        difference=float(np.max(np.abs(expected.cpu().numpy()-data.words[start:stop])))
        same_batch_max=max(same_batch_max,difference)
        assert np.array_equal(mask.cpu().numpy(),data.word_masks[start:stop])
        assert np.array_equal(pos.cpu().numpy(),data.word_pos[start:stop])
    assert same_batch_max==0,'Same-batch cache differs from released frozen encoder'
    write_json(ROOT/config()['run_root']/'preflight/cache_audit.json',dict(status='passed',samples=32,
               posterior_input_split='train',text_scope='entire frozen cache',errors=errors,text_entries_recomputed=len(data.captions),
               same_batch_text_max_error=same_batch_max,
               description='Posterior exact; entire text cache exact at original batch sizes. Different-batch float32 difference is descriptive, not a relaxed cache tolerance.'))
    print(errors,flush=True)


if __name__=='__main__':
    main()
