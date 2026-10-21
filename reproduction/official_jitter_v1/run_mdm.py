"""Evaluate the available MDM BERT/50-step release on SALAD's frozen prompts."""
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from audit_gt import ROOT, FLOW_COMMIT, extract, save_json, sha256, diagnostic_metrics


def ready(source, repeat, index):
    while True:
        status = json.loads((source / "status.json").read_text())
        if status["status"] == "complete" or status.get("repeat", -1) > repeat or (
            status.get("repeat", -1) == repeat and status.get("completed_clips", 0) >= (index+1)*32):
            return
        time.sleep(1)


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
    config = SimpleNamespace(**json.loads(checkpoint.with_name("args.json").read_text()))
    if config.dataset != "humanml" or config.diffusion_steps != 50 or config.text_encoder_type != "bert":
        raise ValueError("Expected the explicitly labelled HumanML3D BERT/50-step checkpoint")
    source_binding = json.loads((args.salad_run / "binding.json").read_text())
    if source_binding["dataset"] != "t2m" or source_binding["repeats"] != 3:
        raise ValueError("Expected three full paired reference repetitions")
    mdm = args.mdm_root.resolve()
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    sys.path = [str(mdm)] + [p for p in sys.path if p != str(ROOT)]
    os.chdir(args.flow_root)
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils.sampler_util import ClassifierFreeSampleModel
    from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from data_loaders.humanml.utils.metrics import (calculate_R_precision, euclidean_distance_matrix,
                                                   calculate_activation_statistics, calculate_frechet_distance)
    from utils.fixseed import fixseed
    fixseed(1234)
    model, diffusion = create_model_and_diffusion(config, SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
    load_saved_model(model, str(checkpoint), use_avg=config.use_ema)
    model = ClassifierFreeSampleModel(model).cuda().eval()
    wrapper = EvaluatorMDMWrapper("humanml", torch.device("cuda:0"))
    mean = np.load(ROOT / "checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy")
    std = np.load(ROOT / "checkpoints/t2m/Comp_v6_KLD005/meta/std.npy")
    namespace = dict(torch=torch)
    extract(args.flow_root / "eval_humanml.py", ["pred_jitter"], namespace)
    jitter = namespace["pred_jitter"]
    training_mean = np.load(ROOT / "dataset/humanml3d/Mean.npy")
    training_std = np.load(ROOT / "dataset/humanml3d/Std.npy")
    binding = dict(model="MDM_BERT_50", scope="Additional version-labelled baseline; not the 1000-step CLIP model in FlowMotion's paper table",
                   checkpoint_sha256=sha256(checkpoint), args_sha256=sha256(checkpoint.with_name("args.json")),
                   model_source_sha256={str(p.relative_to(mdm)):sha256(p) for p in mdm.rglob("*.py") if ".git" not in p.parts},
                   prompts_binding_sha256=sha256(args.salad_run / "binding.json"),
                   dataset="t2m", repeats=3, clips_per_repeat=4640, seed=1234, sampling_steps=50, cfg=2.5,
                   use_ema=config.use_ema, sampler="Official ancestral p_sample_loop, original 50 trained diffusion steps",
                   padding="Keep model output at all 196 frames as the native MDM/FlowMotion evaluator does; additionally record zero-normalized-padding variant",
                   no_training=True, no_temporal_postprocessing=True)
    save_json(output / "binding.json", binding)
    csv_stream = (output / "per_clip.csv").open("w", newline="", encoding="utf-8")
    writer = None
    repetition_rows = []
    try:
        with torch.no_grad():
            for repeat in range(3):
                rows, gt_embeddings, gen_embeddings = [], [], []
                precision = np.zeros(3)
                matching = 0.
                for index in range(145):
                    ready(args.salad_run, repeat, index)
                    cache = args.salad_run / "server_only" / ("repeat_%02d" % repeat) / ("batch_%03d.npz" % index)
                    with np.load(cache) as inputs:
                        lengths = torch.from_numpy(inputs["lengths"]).cuda()
                        captions = inputs["captions"].tolist()
                        y = dict(text=captions, lengths=lengths, scale=torch.full((len(captions),), 2.5, device="cuda"),
                                 mask=(torch.arange(196,device="cuda")[None] < lengths[:,None])[:,None,None,:])
                        generated = diffusion.p_sample_loop(model, (32,263,1,196), clip_denoised=False,
                                    model_kwargs={"y":y}, progress=False, noise=None, const_noise=False)
                        normalized = generated[:, :, 0].permute(0,2,1).cpu().numpy()
                        physical = normalized * training_std + training_mean
                        eval_normalized = (physical - mean) / std
                        word = torch.from_numpy(inputs["word"])
                        pos = torch.from_numpy(inputs["pos"])
                        sent_len = torch.from_numpy(inputs["sent_len"])
                        et, em = wrapper.get_co_embeddings(word, pos, sent_len, torch.from_numpy(eval_normalized), lengths)
                        _, gt_emb = wrapper.get_co_embeddings(word, pos, sent_len, torch.from_numpy(inputs["gt"]), lengths)
                        et, em = et.cpu().numpy(), em.cpu().numpy()
                        gt_embeddings.append(gt_emb.cpu().numpy())
                        gen_embeddings.append(em)
                        precision += calculate_R_precision(et, em, top_k=3, sum_all=True)
                        matching += euclidean_distance_matrix(et, em).trace()
                        shard = output / "server_only" / ("repeat_%02d" % repeat)
                        shard.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(shard / cache.name, normalized_eval=eval_normalized,
                                            lengths=inputs["lengths"], input_cache_sha256=sha256(cache))
                        for i,length in enumerate(lengths.cpu().tolist()):
                            points = recover_from_ric(torch.from_numpy(physical[i]).float(), 22)
                            values = diagnostic_metrics(points[:length].numpy(), "t2m")
                            masked = eval_normalized[i].copy()
                            masked[length:] = 0
                            masked_points = recover_from_ric(torch.from_numpy(masked * std + mean).float(), 22)
                            values.update(flow_published_padded=float(jitter(points)),
                                          flow_valid_actual_joints=float(jitter(points[:length])),
                                          flow_zero_normalized_padding=float(jitter(masked_points)))
                            row = dict(repeat=repeat, batch=index, within_batch=i, valid_frames=length, **values)
                            if writer is None:
                                writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(values)
                    save_json(output / "status.json", dict(status="running", repeat=repeat, completed_clips=len(rows),pid=os.getpid()))
                    if index % 20 == 0:
                        print(json.dumps(dict(repeat=repeat, batch=index, completed=len(rows))), flush=True)
                gt_mu, gt_cov = calculate_activation_statistics(np.concatenate(gt_embeddings))
                mu, cov = calculate_activation_statistics(np.concatenate(gen_embeddings))
                metrics = {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
                metrics.update(repeat=repeat, clips=len(rows), fid=float(calculate_frechet_distance(gt_mu,gt_cov,mu,cov)),
                               top1=float(precision[0]/len(rows)), top2=float(precision[1]/len(rows)),
                               top3=float(precision[2]/len(rows)), matching=float(matching/len(rows)))
                repetition_rows.append(metrics)
                np.savez_compressed(shard / "embeddings.npz", gt=np.concatenate(gt_embeddings), generated=np.concatenate(gen_embeddings))
                save_json(output / "repetitions.json", repetition_rows)
                csv_stream.flush()
                print(json.dumps(metrics), flush=True)
    finally:
        csv_stream.close()
    result = {}
    for key in repetition_rows[0]:
        if key in ["repeat","clips"]:
            continue
        values = np.array([r[key] for r in repetition_rows])
        result[key] = dict(mean=float(values.mean()),ci95_half_width=float(1.96*values.std()/np.sqrt(3)))
    save_json(output / "summary.json",dict(status="complete",dataset="t2m",model="MDM_BERT_50",repeats=3,metrics=result))
    save_json(output / "status.json",dict(status="complete",pid=os.getpid(),per_clip_sha256=sha256(output / "per_clip.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdm-root",type=Path,required=True)
    parser.add_argument("--flow-root",type=Path,required=True)
    parser.add_argument("--salad-run",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args())
