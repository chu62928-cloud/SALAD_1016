"""Isolate the published generated-caption length behavior without regenerating."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
import numpy as np
import torch
from audit_gt import ROOT,save_json,sha256


def retrieval(text,motion):
    distance=np.sqrt(np.sum((text[:,None,:]-motion[None,:,:])**2,axis=-1))
    ordering=np.argsort(distance,axis=1)
    return dict(matching=float(np.diag(distance).mean()),
                **{"top%d"%k:float(np.any(ordering[:,:k]==np.arange(len(text))[:,None],axis=1).mean()) for k in [1,2,3]})


def run(args):
    torch.set_num_threads(2)
    os.chdir(ROOT)
    # Construct only the official evaluation wrapper on CPU, without a denoiser.
    from utils.get_opt import get_opt
    from models.t2m_eval_wrapper import EvaluatorModelWrapper
    options=get_opt(str(ROOT/"checkpoints/t2m/Comp_v6_KLD005/opt.txt"),torch.device("cpu"))
    wrapper=EvaluatorModelWrapper(options)
    rows=[]
    for index in [0,72,144]:
        path=args.root/"salad_t2m/server_only/repeat_00"/("batch_%03d.npz"%index)
        with np.load(path) as cache:
            word,pos,valid,lengths=[torch.from_numpy(cache[k]) for k in ["word","pos","sent_len","lengths"]]
            padded=torch.full_like(valid,word.shape[1])
            for model,key in [("GT","gt"),("SALAD","generated")]:
                motion=torch.from_numpy(cache[key])
                normal,movement=wrapper.get_co_embeddings(word,pos,valid,motion,lengths)
                replaced,repeated=wrapper.get_co_embeddings(word,pos,padded,motion,lengths)
                assert torch.equal(movement,repeated)
                rows.append(dict(batch=index,model=model,input_sha256=sha256(path),
                                 effective_lengths=valid.tolist(),padded_length=int(word.shape[1]),
                                 valid=retrieval(normal.numpy(),movement.numpy()),
                                 forced_padded=retrieval(replaced.numpy(),movement.numpy()),
                                 embedding_l2_mean=float(torch.norm(normal-replaced,dim=1).mean())))
    result=dict(status="complete",scope="Three archived GT/SALAD batches, not FlowMotion generated motions",
                rows=rows,conclusion="Sentence length is changed in isolation; this does not establish complete FlowMotion score correction or causality for its FID.",
                no_generation=True,no_training=True)
    save_json(args.root/"text_length_control.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    run(parser.parse_args())
