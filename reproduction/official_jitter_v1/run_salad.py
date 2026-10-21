"""Evaluate released SALAD on each official test split with two metric families."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import torch

from audit_gt import ROOT, FLOW_COMMIT, extract, save_json, sha256, diagnostic_metrics
sys.path.insert(0, str(ROOT / "reproduction"))
from runtime import load_runtime
from utils.motion_process import recover_from_ric
from utils.metrics import (calculate_R_precision, euclidean_distance_matrix,
                          calculate_activation_statistics, calculate_frechet_distance)


def metrics_for_motion(normalized, valid, mean, std, dataset, upstream_jitter):
    raw = np.asarray(normalized) * std + mean
    actual_joints = 22 if dataset == "t2m" else 21
    factor = 1. if dataset == "t2m" else .00073
    points22 = recover_from_ric(torch.from_numpy(raw).float(), 22)
    points = points22[:valid, :actual_joints].numpy()
    result = diagnostic_metrics(points, dataset)
    result.update(flow_published_padded=float(upstream_jitter(points22 * factor)),
                  flow_valid_actual_joints=float(upstream_jitter(points22[:valid, :actual_joints] * factor)))
    return result


def summarize(rows):
    result = {}
    for key in rows[0]:
        if key in ["repeat", "model", "clips", "elapsed_seconds"]:
            continue
        values = np.array([r[key] for r in rows], dtype=np.float64)
        result[key] = dict(mean=float(values.mean()), ci95_half_width=float(1.96 * values.std() / np.sqrt(len(values))))
    return result


def main(args):
    import csv
    import fcntl
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "binding.json").exists():
        raise ValueError("Use a fresh run directory")
    for dataset in [args.dataset]:
        gt_path = ROOT / "reproduction/artifacts/official_jitter_v1" / ("gt_" + dataset) / "summary.json"
        gt = json.loads(gt_path.read_text())
        if not gt["gt_closeness_screen_passed"]:
            raise ValueError("GT compatibility screen did not pass")
    source = args.flow_root.resolve()
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != FLOW_COMMIT:
        raise ValueError("FlowMotion commit changed")
    namespace = dict(torch=torch)
    extract(source / "eval_humanml.py", ["pred_jitter"], namespace)
    jitter = namespace["pred_jitter"]
    opt, vae_opt, trainer, wrapper, loader = load_runtime(args.dataset, args.seed, 50, 7.5)
    trainer.vae.eval()
    trainer.denoiser.eval()
    checkpoints = [Path(opt.model_dir) / "net_best_fid.tar", Path(vae_opt.model_dir) / "net_best_fid.tar",
                   ROOT / "checkpoints" / args.dataset / "text_mot_match/model/finest.tar"]
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint["denoiser"]
        missing, unexpected = trainer.denoiser.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith("clip_model.") for key in missing):
            raise ValueError("Adapted checkpoint schema mismatch")
        checkpoints[0] = args.checkpoint
    mean, std = loader.dataset.mean, loader.dataset.std
    binding = dict(dataset=args.dataset, model=args.model, seed=args.seed, repeats=args.repeats,
                   steps=50, cfg=7.5, scheduler="Author DDIMScheduler", batch_size=32,
                   workers=4, shuffle=True, drop_last=True, loaded=len(loader.dataset),
                   evaluated_per_repeat=len(loader)*32, flow_commit=FLOW_COMMIT,
                   flow_metric_source_sha256=sha256(source / "eval_humanml.py"),
                   metric_protocol="Two separate families; original FlowMotion padded 22-joint fps20 scaled path; native valid-frame physical diagnostics",
                   generation_protocol="Full SALAD official evaluation loader; 3 fixed repetitions; no multimodality sampling. New FID/R share these exact draws; archived 20-repeat publication estimates reported separately.",
                   checkpoints={str(p): sha256(p) for p in checkpoints},
                   source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob("*.py")},
                   gt_compatibility_sha256=sha256(gt_path),
                   split_sha256=sha256(Path(opt.data_root) / "test.txt"),
                   no_training=True, no_postprocessing=True)
    save_json(output / "binding.json", binding)
    repetition_rows = []
    csv_stream = (output / "per_clip.csv").open("w", encoding="utf-8", newline="")
    writer = None
    try:
        with torch.no_grad():
            for repeat in range(args.repeats):
                start = time.monotonic()
                per_model = {"GT": [], args.model: []}
                embeddings = {"GT": [], args.model: []}
                precisions = {"GT": np.zeros(3), args.model: np.zeros(3)}
                matching = {"GT": 0., args.model: 0.}
                count = 0
                shard = output / "server_only" / ("repeat_%02d" % repeat)
                shard.mkdir(parents=True, exist_ok=True)
                for batch_index, batch in enumerate(loader):
                    word, pos, captions, sent_len, motion, lengths, tokens = batch
                    motion = motion.cuda()
                    device_lengths = lengths.cuda()
                    predicted, _ = trainer.generate((captions, motion, device_lengths))
                    if predicted.shape != motion.shape:
                        raise ValueError("Motion shape mismatch")
                    mask = torch.arange(motion.shape[1], device=motion.device)[None] >= device_lengths[:, None]
                    predicted = predicted.masked_fill(mask[..., None], 0)
                    raw_motion, raw_pred = motion.cpu().numpy(), predicted.cpu().numpy()
                    np.savez_compressed(shard / ("batch_%03d.npz" % batch_index), gt=raw_motion, generated=raw_pred,
                                        lengths=lengths.numpy(), captions=np.asarray(captions), tokens=np.asarray(tokens),
                                        word=word.numpy(), pos=pos.numpy(), sent_len=sent_len.numpy())
                    for model, values in [("GT", motion), (args.model, predicted)]:
                        text_emb, motion_emb = wrapper.get_co_embeddings(word, pos, sent_len, values, device_lengths)
                        et, em = text_emb.cpu().numpy(), motion_emb.cpu().numpy()
                        embeddings[model].append(em)
                        precisions[model] += calculate_R_precision(et, em, top_k=3, sum_all=True)
                        matching[model] += euclidean_distance_matrix(et, em).trace()
                        normalized = raw_motion if model == "GT" else raw_pred
                        for i, length in enumerate(lengths.tolist()):
                            metrics = metrics_for_motion(normalized[i], int(length), mean, std, args.dataset, jitter)
                            per_model[model].append(metrics)
                            row = dict(repeat=repeat, model=model, batch=batch_index, within_batch=i,
                                       valid_frames=int(length), **metrics)
                            if writer is None:
                                writer = csv.DictWriter(csv_stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                    count += len(motion)
                    save_json(output / "status.json", dict(status="running", repeat=repeat, completed_clips=count,
                                                           requested_repeats=args.repeats, pid=os.getpid()))
                    if batch_index % 20 == 0:
                        print(json.dumps(dict(repeat=repeat, batch=batch_index, count=count, seconds=time.monotonic()-start)), flush=True)
                if count != len(loader)*32:
                    raise ValueError("Evaluation coverage mismatch")
                joined = {k: np.concatenate(v) for k, v in embeddings.items()}
                gt_mu, gt_cov = calculate_activation_statistics(joined["GT"])
                for model in ["GT", args.model]:
                    row = dict(model=model, repeat=repeat, clips=count)
                    row.update({k: float(np.mean([m[k] for m in per_model[model]])) for k in per_model[model][0]})
                    mu, cov = calculate_activation_statistics(joined[model])
                    row.update(fid=float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov)) if model != "GT" else 0.,
                               top1=float(precisions[model][0]/count), top2=float(precisions[model][1]/count),
                               top3=float(precisions[model][2]/count), matching=float(matching[model]/count),
                               elapsed_seconds=time.monotonic()-start)
                    repetition_rows.append(row)
                    print(json.dumps(row), flush=True)
                np.savez_compressed(shard / "embeddings.npz", **joined)
                csv_stream.flush()
                save_json(output / "repetitions.json", repetition_rows)
    finally:
        csv_stream.close()
    metrics = {model:summarize([r for r in repetition_rows if r["model"]==model]) for model in ["GT",args.model]}
    save_json(output / "summary.json", dict(status="complete", dataset=args.dataset, repeats=args.repeats,
              evaluated_per_repeat=len(loader)*32, metrics=metrics, per_clip_sha256=sha256(output / "per_clip.csv")))
    save_json(output / "status.json", dict(status="complete", requested_repeats=args.repeats, pid=os.getpid()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["t2m", "kit"], required=True)
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="SALAD")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or (args.checkpoint is not None and args.model == "SALAD"):
        parser.error("Positive repeats and an explicit adapted-model name are required")
    main(args)
