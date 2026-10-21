"""Audit the published FlowMotion metric before comparing model outputs."""
import argparse
import ast
from collections import OrderedDict
import codecs as cs
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from os.path import join as pjoin
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy.fft import dct
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.get_opt import get_opt
from utils.word_vectorizer import WordVectorizer

FLOW_COMMIT = "fe4021b9170545b4e9ddc6d13d309bfa0988c83f"
TARGETS = {"t2m": 47.262, "kit": 49.922}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def extract(path, names, namespace):
    """Execute selected upstream definitions unchanged, without optional UI imports."""
    parsed = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("Upstream definitions missing")
    code = ast.Module(body=selected, type_ignores=[])
    exec(compile(code, str(path), "exec"), namespace)
    return {node.name: {"line": node.lineno, "end_line": node.end_lineno} for node in selected}


def diagnostic_metrics(points, dataset):
    spec = {"t2m": (20., 1., 22, [10, 11]), "kit": (12.5, .001, 21, [15, 20])}
    fps, scale, joints, toes = spec[dataset]
    p = np.asarray(points[:, :joints], dtype=np.float64) * scale
    local = p - p[:, :1]
    chains = ([[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
               [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]] if dataset == "t2m" else
              [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4],
               [3, 5, 6, 7], [3, 8, 9, 10]])
    edges = [(a, b) for chain in chains for a, b in zip(chain[:-1], chain[1:])]
    bones = np.stack([np.linalg.norm(p[:, b] - p[:, a], axis=-1) for a, b in edges], axis=1)
    if (bones.mean(0) <= 1e-8).any():
        raise ValueError("Degenerate bone")
    jerk_world = np.diff(p, n=3, axis=0) * fps**3
    jerk_local = np.diff(local, n=3, axis=0) * fps**3
    coeff = dct(local - local.mean(0, keepdims=True), axis=0, norm="ortho")
    frequency = np.arange(len(p)) * fps / (2 * len(p))
    high = coeff[frequency >= 3]
    toe = p[:, toes]
    speed = np.linalg.norm(np.diff(toe[..., [0, 2]], axis=0), axis=-1) * fps
    low = np.maximum(toe[:-1, :, 1], toe[1:, :, 1]) <= .05
    sliding = low & (speed > .2)
    return {
        "world_jerk_rms_mps3": float(np.sqrt(np.mean(np.sum(jerk_world**2, axis=-1)))),
        "local_jerk_rms_mps3": float(np.sqrt(np.mean(np.sum(jerk_local**2, axis=-1)))),
        "bone_cv_percent": float(np.mean(bones.std(0) / bones.mean(0)) * 100),
        "local_hf3_coordinate_rms_mm": float(np.sqrt(np.sum(high**2) / local.size) * 1000),
        "local_hf3_vector_rms_mm": float(np.sqrt(np.sum(high**2) / (len(p) * joints)) * 1000),
        "sliding_frame_percent": float(sliding.any(axis=1).mean() * 100),
        "toe_contact_interval_percent": float(low.mean() * 100),
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(args):
    os.chdir(ROOT)
    torch.set_num_threads(2)
    source = args.flow_root.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != FLOW_COMMIT:
        raise ValueError("Unexpected FlowMotion source version: " + commit)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("FlowMotion tracked source was modified")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "summary.json").exists():
        raise ValueError("Use a new output directory; audits are immutable")

    namespace = dict(torch=torch, np=np, random=random, cs=cs, pjoin=pjoin, tqdm=tqdm,
                     data=data, default_collate=default_collate)
    definitions = extract(source / "data_loaders/humanml/data/dataset.py",
                          ["Text2MotionDatasetV2", "collate_fn"], namespace)
    # Load the exact upstream recovery module under its own package hierarchy.
    sys.path.insert(0, str(source))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    metric_namespace = dict(torch=torch, np=np, OrderedDict=OrderedDict, recover_from_ric=recover_from_ric)
    definitions.update(extract(source / "eval_humanml.py", ["pred_jitter", "evaluate_jitter"], metric_namespace))
    upstream_jitter = metric_namespace["pred_jitter"]
    from tests_metrics import verify_functions
    tests = verify_functions(upstream_jitter, diagnostic_metrics)
    dataset = args.dataset
    evaluator = ROOT / "checkpoints" / dataset / "Comp_v6_KLD005"
    opt = get_opt(str(evaluator / "opt.txt"), torch.device("cpu"))
    mean = np.load(evaluator / "meta/mean.npy")
    std = np.load(evaluator / "meta/std.npy")
    vectorizer = WordVectorizer(str(ROOT / "glove"), "our_vab")
    seed_all(args.seed)
    ds = namespace["Text2MotionDatasetV2"](opt, mean, std, str(Path(opt.data_root) / "test.txt"), vectorizer)
    loader = data.DataLoader(ds, batch_size=32, num_workers=args.workers, shuffle=True,
                             drop_last=True, collate_fn=namespace["collate_fn"])
    binding = {
        "flow_repository": "https://github.com/mccuba/FlowMotion", "flow_commit": commit,
        "flow_source_sha256": {str(p.relative_to(source)): sha256(p) for p in [source / "eval_humanml.py",
            source / "data_loaders/humanml/data/dataset.py", source / "data_loaders/humanml/scripts/motion_process.py"]},
        "definitions_executed_unchanged": definitions, "dataset": dataset,
        "split_sha256": sha256(Path(opt.data_root) / "test.txt"),
        "mean_sha256": sha256(evaluator / "meta/mean.npy"), "std_sha256": sha256(evaluator / "meta/std.npy"),
        "dataset_asset_audit_sha256": sha256(ROOT / "reproduction/artifacts" / ("assets_" + dataset + ".json")),
        "seed": args.seed, "repeats": args.repeats, "batch_size": 32, "workers": args.workers,
        "shuffle": True, "drop_last": True, "loaded_entries": len(ds), "per_repeat": len(loader) * 32,
        "official_metric": "Unmodified pred_jitter/evaluate_jitter; world coordinates; mean norm of third difference; fps=20; recover_from_ric(...,22); full padded 196 frames; KIT factor .00073",
        "diagnostic_metric": "Actual joints 22/21, valid frames only, metres (KIT .001), native fps 20/12.5; clip-equal aggregation; no concatenation, filters or retargeting",
        "gate": "Relative discrepancy <=1% is only a practical closeness screen, not statistical equivalence or proof of reproducing the paper.",
        "candidate_acceleration_not_used": True, "tests": tests,
        "scope": "GT audit only. Do not run model comparisons until GT paper compatibility is established.",
    }
    save_json(output / "binding.json", binding)
    rows, repetition_rows = [], []
    state = {"lengths": [], "cursor": 0, "repeat": 0}
    csv_path = output / "per_clip.csv"
    csv_stream = csv_path.open("w", encoding="utf-8", newline="")
    writer = None

    class CapturedLoader:
        dataset = ds

        def __iter__(self):
            for batch in loader:
                state["lengths"] = batch[5].tolist()
                state["cursor"] = 0
                yield batch

    def capture(points, fps=20):
        nonlocal writer
        value = upstream_jitter(points, fps)
        valid = int(state["lengths"][state["cursor"]])
        state["cursor"] += 1
        factor = .00073 if dataset == "kit" else 1.
        actual_joints = 21 if dataset == "kit" else 22
        native = points.numpy() / factor
        row = {"repeat": state["repeat"], "clip_index": len(rows), "valid_frames": valid,
               "padded_frames": len(points), "flow_published_padded": float(value),
               "flow_valid_22": float(upstream_jitter(points[:valid])),
               "flow_valid_actual_joints": float(upstream_jitter(points[:valid, :actual_joints]))}
        row.update(diagnostic_metrics(native[:valid], dataset))
        # Quantify the boundary/padding contribution without inventing real motion frames.
        magnitude = np.linalg.norm(np.diff(points.numpy(), n=3, axis=0) * 20**3, axis=-1)
        row["flow_boundary_and_padding_sum_fraction"] = float(magnitude[max(valid - 3, 0):].sum() / max(magnitude.sum(), 1e-20))
        if writer is None:
            writer = csv.DictWriter(csv_stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        rows.append(row)
        return value

    metric_namespace["pred_jitter"] = capture
    start = time.monotonic()
    try:
        for repeat in range(args.repeats):
            state["repeat"] = repeat
            rows.clear()
            with redirect_stdout(io.StringIO()):
                result = metric_namespace["evaluate_jitter"]({"ground truth": CapturedLoader()},
                         "kit" if dataset == "kit" else "humanml", io.StringIO())
            if len(rows) != len(loader) * 32:
                raise ValueError("Coverage mismatch")
            summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ["repeat", "clip_index"]}
            if abs(summary["flow_published_padded"] - float(result["ground truth"])) > 1e-3:
                raise ValueError("Original aggregation mismatch")
            summary.update(repeat=repeat, clips=len(rows), original_torch_aggregation=float(result["ground truth"]))
            repetition_rows.append(summary)
            csv_stream.flush()
            save_json(output / "repetitions.json", repetition_rows)
            save_json(output / "status.json", dict(status="running", completed=repeat + 1, requested=args.repeats, pid=os.getpid()))
            print(json.dumps(summary), flush=True)
    finally:
        csv_stream.close()
    result = {}
    for key in repetition_rows[0]:
        if key in ["repeat", "clips"]:
            continue
        values = np.array([r[key] for r in repetition_rows])
        result[key] = {"mean": float(values.mean()), "ci95_half_width": float(1.96 * values.std() / np.sqrt(len(values)))}
    measured = result["original_torch_aggregation"]["mean"]
    relative_difference = (measured - TARGETS[dataset]) / TARGETS[dataset]
    final = dict(status="complete", dataset=dataset, repetitions=args.repeats, clips_per_repeat=len(loader) * 32,
                 metrics=result, paper_gt_target=TARGETS[dataset], relative_difference=relative_difference,
                 gt_closeness_screen_passed=abs(relative_difference) <= .01,
                 model_comparison_authorized_by_gt_check=abs(relative_difference) <= .01,
                 elapsed_seconds=time.monotonic() - start, per_clip_sha256=sha256(csv_path),
                 warning="The published KIT path reads 3 rotation features as a phantom 22nd joint. Published metric includes padding boundaries. Native physical diagnostics do not.")
    save_json(output / "summary.json", final)
    save_json(output / "status.json", dict(status="complete", completed=args.repeats, pid=os.getpid()))
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["t2m", "kit"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    run(args)
