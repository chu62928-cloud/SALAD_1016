"""Evaluate the official KIT CLIP/1000-step MDM release on paired test inputs."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np
import torch
from audit_gt import ROOT, extract, save_json, sha256, diagnostic_metrics


def main(args):
    import fcntl
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "binding.json").exists():
        raise ValueError("Use a fresh run directory")
    checkpoint = args.checkpoint.resolve()
    original = json.loads(checkpoint.with_name("args.json").read_text())
    if original["dataset"] != "kit" or original["diffusion_steps"] != 1000:
        raise ValueError("Expected official KIT/1000-step checkpoint")
    source = args.salad_run.resolve()
    reference = json.loads((source / "binding.json").read_text())
    if reference["dataset"] != "kit" or reference["repeats"] != 3:
        raise ValueError("Expected paired three-repeat KIT reference")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("KIT reference generation incomplete")
    mdm = args.mdm_root.resolve()
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    sys.path = [str(mdm)] + [p for p in sys.path if p != str(ROOT)]
    os.chdir(args.flow_root)
    from utils import parser_util
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.sampler_util import ClassifierFreeSampleModel
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.humanml.utils.metrics import (calculate_R_precision, euclidean_distance_matrix,
                                                   calculate_activation_statistics, calculate_frechet_distance)
    from utils.fixseed import fixseed
    parser = argparse.ArgumentParser()
    for add in [parser_util.add_data_options, parser_util.add_model_options, parser_util.add_diffusion_options]:
        add(parser)
    defaults = vars(parser.parse_args([]))
    defaults.update(original)
    config = SimpleNamespace(**defaults)
    if config.text_encoder_type != "clip" or config.arch != "trans_enc":
        raise ValueError("Unexpected official KIT model architecture")
    fixseed(1234)
    model, diffusion = create_model_and_diffusion(config, SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
    load_saved_model(model, str(checkpoint), use_avg=config.use_ema)
    model = ClassifierFreeSampleModel(model).cuda().eval()
    wrapper = EvaluatorMDMWrapper("kit", torch.device("cuda:0"))
    mean = np.load(ROOT / "checkpoints/kit/Comp_v6_KLD005/meta/mean.npy")
    std = np.load(ROOT / "checkpoints/kit/Comp_v6_KLD005/meta/std.npy")
    training_mean = np.load(ROOT / "dataset/kit-ml/Mean.npy")
    training_std = np.load(ROOT / "dataset/kit-ml/Std.npy")
    namespace = dict(torch=torch)
    extract(args.flow_root / "eval_humanml.py", ["pred_jitter"], namespace)
    jitter = namespace["pred_jitter"]
    binding = dict(model="MDM_CLIP_1000", dataset="kit", repeats=3, clips_per_repeat=704,
                   seed=1234, sampling_steps=1000, cfg=2.5, use_ema=config.use_ema,
                   sampler="Official p_sample_loop; all 1000 original trained diffusion steps",
                   checkpoint_sha256=sha256(checkpoint), args_sha256=sha256(checkpoint.with_name("args.json")),
                   effective_config=vars(config), prompts_binding_sha256=sha256(source / "binding.json"),
                   driver_sha256=sha256(__file__), mdm_source_sha256={str(p.relative_to(mdm)):sha256(p) for p in mdm.rglob("*.py")},
                   no_training=True, no_postprocessing=True,
                   padding="Native full 196 model frames; additional zero-normalized-padding comparison recorded")
    save_json(output / "binding.json", binding)
    repetitions = []
    stream = (output / "per_clip.csv").open("w", newline="", encoding="utf-8")
    writer = None
    try:
        with torch.no_grad():
            for repeat in range(3):
                rows, gt_embeddings, generated_embeddings = [], [], []
                precision, matching = np.zeros(3), 0.
                for index in range(22):
                    cache = source / "server_only" / ("repeat_%02d"%repeat) / ("batch_%03d.npz"%index)
                    start = time.monotonic()
                    with np.load(cache) as inputs:
                        lengths = torch.from_numpy(inputs["lengths"]).cuda()
                        captions = inputs["captions"].tolist()
                        y = dict(text=captions, lengths=lengths,
                                 scale=torch.full((32,),2.5,device="cuda"),
                                 mask=(torch.arange(196,device="cuda")[None]<lengths[:,None])[:,None,None,:])
                        generated = diffusion.p_sample_loop(model, (32,251,1,196), clip_denoised=False,
                                                            model_kwargs={"y":y}, progress=False, noise=None, const_noise=False)
                        physical = generated[:,:,0].permute(0,2,1).cpu().numpy()*training_std+training_mean
                        normalized = (physical-mean)/std
                        word,pos,sent_len = [torch.from_numpy(inputs[k]) for k in ["word","pos","sent_len"]]
                        et,em = wrapper.get_co_embeddings(word,pos,sent_len,torch.from_numpy(normalized),lengths)
                        _,gt = wrapper.get_co_embeddings(word,pos,sent_len,torch.from_numpy(inputs["gt"]),lengths)
                        et,em = et.cpu().numpy(),em.cpu().numpy()
                        gt_embeddings.append(gt.cpu().numpy())
                        generated_embeddings.append(em)
                        precision += calculate_R_precision(et,em,top_k=3,sum_all=True)
                        matching += euclidean_distance_matrix(et,em).trace()
                        shard = output / "server_only" / ("repeat_%02d"%repeat)
                        shard.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(shard/cache.name,normalized_eval=normalized,lengths=inputs["lengths"],input_cache_sha256=sha256(cache))
                        for i,length in enumerate(lengths.cpu().tolist()):
                            p22 = recover_from_ric(torch.from_numpy(physical[i]).float(),22)
                            values = diagnostic_metrics(p22[:length,:21].numpy(),"kit")
                            masked = normalized[i].copy()
                            masked[length:]=0
                            masked_p = recover_from_ric(torch.from_numpy(masked*std+mean).float(),22)
                            values.update(flow_published_padded=float(jitter(p22*.00073)),
                                          flow_valid_actual_joints=float(jitter(p22[:length,:21]*.00073)),
                                          flow_zero_normalized_padding=float(jitter(masked_p*.00073)))
                            row = dict(repeat=repeat,batch=index,within_batch=i,valid_frames=length,**values)
                            if writer is None:
                                writer=csv.DictWriter(stream,fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(values)
                    save_json(output/"status.json",dict(status="running",repeat=repeat,completed_clips=len(rows),pid=os.getpid()))
                    print(json.dumps(dict(repeat=repeat,batch=index,completed=len(rows),batch_seconds=time.monotonic()-start)),flush=True)
                gt_embeddings,generated_embeddings=np.concatenate(gt_embeddings),np.concatenate(generated_embeddings)
                gt_mu,gt_cov=calculate_activation_statistics(gt_embeddings)
                mu,cov=calculate_activation_statistics(generated_embeddings)
                metrics={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
                metrics.update(repeat=repeat,clips=len(rows),fid=float(calculate_frechet_distance(gt_mu,gt_cov,mu,cov)),
                               top1=float(precision[0]/len(rows)),top2=float(precision[1]/len(rows)),top3=float(precision[2]/len(rows)),matching=float(matching/len(rows)))
                repetitions.append(metrics)
                np.savez_compressed(shard/"embeddings.npz",gt=gt_embeddings,generated=generated_embeddings)
                save_json(output/"repetitions.json",repetitions)
                stream.flush()
                print(json.dumps(metrics),flush=True)
    finally:
        stream.close()
    summary={}
    for key in repetitions[0]:
        if key in ["repeat","clips"]:
            continue
        x=np.array([r[key] for r in repetitions])
        summary[key]=dict(mean=float(x.mean()),ci95_half_width=float(1.96*x.std()/np.sqrt(3)))
    save_json(output/"summary.json",dict(status="complete",dataset="kit",model="MDM_CLIP_1000",repeats=3,metrics=summary))
    save_json(output/"status.json",dict(status="complete",pid=os.getpid(),per_clip_sha256=sha256(output/"per_clip.csv")))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ["mdm-root","flow-root","salad-run","checkpoint","output"]:
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
