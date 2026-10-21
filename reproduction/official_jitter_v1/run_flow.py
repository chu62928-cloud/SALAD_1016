"""Run the pinned author's FlowMotion entry point with diagnostic observation."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np
import torch

from audit_gt import ROOT, FLOW_COMMIT, sha256, save_json, diagnostic_metrics


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.resolve() != target.resolve():
            raise ValueError("Conflicting asset path: " + str(path))
        return
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main(args):
    torch.set_num_threads(2)
    source = args.flow_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "binding.json").exists():
        raise ValueError("Use a new run directory")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != FLOW_COMMIT:
        raise ValueError("Source revision changed")
    gt = ROOT / "reproduction/artifacts/official_jitter_v1/gt_t2m/summary.json"
    if not json.loads(gt.read_text())["gt_closeness_screen_passed"]:
        raise ValueError("GT compatibility screen failed")
    for name, target in [("dataset/HumanML3D", ROOT / "dataset/humanml3d"),
                         ("dataset/KIT-ML", ROOT / "dataset/kit-ml"), ("glove", ROOT / "glove"),
                         ("t2m", ROOT / "checkpoints/t2m"), ("kit", ROOT / "checkpoints/kit")]:
        link(source / name, target)
    for dataset in ["t2m", "kit"]:
        for name in ["mean", "std"]:
            link(source / "dataset" / (dataset + "_" + name + ".npy"),
                 ROOT / "checkpoints" / dataset / "Comp_v6_KLD005/meta" / (name + ".npy"))
    mdm = Path("/root/autodl-tmp/motion-diffusion-model")
    for name in ["humanml_opt.txt", "kit_opt.txt"]:
        link(source / "dataset" / name, mdm / "dataset" / name)
    link(output / "model000620000.pt", source / "pretrained/model000620000.pt")
    binding = dict(status="prepared", repository="https://github.com/mccuba/FlowMotion", commit=commit,
                   checkpoint_sha256=sha256(source / "pretrained/model000620000.pt"),
                   source_sha256={str(p.relative_to(source)):sha256(p) for p in source.rglob("*.py") if ".git" not in p.parts},
                   gt_gate_sha256=sha256(gt), dataset="humanml", eval_mode="wo_mm", seed=10, repeats=20,
                   sampler="Published tcfm sigma_min=0 Euler step_size=.01 CFG2.5",
                   note="Original eval_humanml.py main body and all metric functions run unchanged. Hydra config is composed via absolute config directory because importing the published decorated entry point fails to resolve its config package. Observation hook records native-frame diagnostics during original evaluate_jitter traversal without consuming RNG or adding loader traversals. Python compatibility TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for trusted pinned checkpoint.")
    save_json(output / "binding.json", binding)
    # Avoid importing SALAD's identically named utils modules into FlowMotion.
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            del sys.modules[key]
    sys.path = [str(source)] + [p for p in sys.path if p != str(ROOT)]
    os.chdir(source)
    spec = importlib.util.spec_from_file_location("flow_eval_observed", source / "eval_humanml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream_evaluate = module.evaluate_jitter
    upstream_jitter = module.pred_jitter
    state = dict(model="", lengths=[], cursor=0, repeat=-1)
    stream = (output / "per_clip.csv").open("w", newline="", encoding="utf-8")
    writer = None
    repetition_rows = []
    current_rows = []

    def observed(points, fps=20):
        nonlocal writer
        value = upstream_jitter(points, fps)
        length = int(state["lengths"][state["cursor"]])
        state["cursor"] += 1
        metrics = diagnostic_metrics(points[:length].numpy(), "t2m")
        row = dict(repeat=state["repeat"], model=state["model"], valid_frames=length,
                   flow_published_padded=float(value), flow_valid_actual_joints=float(upstream_jitter(points[:length])), **metrics)
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        current_rows.append(row)
        return value

    class ObservedLoader:
        def __init__(self, loader, model):
            self.loader, self.model, self.dataset = loader, model, loader.dataset

        def __iter__(self):
            state["model"] = self.model
            for batch in self.loader:
                state["lengths"] = batch[5].tolist()
                state["cursor"] = 0
                yield batch

    def evaluate(loaders, dataset, file):
        state["repeat"] += 1
        current_rows.clear()
        result = upstream_evaluate({key:ObservedLoader(loader, key) for key,loader in loaders.items()}, dataset, file)
        for model in loaders:
            rows = [r for r in current_rows if r["model"] == model]
            values = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ["model", "repeat"]}
            if abs(values["flow_published_padded"] - float(result[model])) > .001:
                raise ValueError("Observation changed metric aggregation")
            values.update(model=model, repeat=state["repeat"], clips=len(rows))
            repetition_rows.append(values)
        stream.flush()
        save_json(output / "repetitions.json", repetition_rows)
        save_json(output / "status.json", dict(status="running", completed=state["repeat"]+1, pid=os.getpid()))
        return result

    module.pred_jitter = observed
    module.evaluate_jitter = evaluate
    overrides = ["dataset=humanml", "dynamic=flow",
                "model_path=" + str(output / "model000620000.pt"), "guidance_param=2.5", "eval_mode=wo_mm",
                "diffusion_steps_sample=-1", "version=tcfm", "sigma_min=0.0", "is_debug=0",
                "hydra.run.dir=" + str(output / "hydra")]
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(source / "config"), version_base=None):
        cfg = compose(config_name="config_base", overrides=overrides)
    try:
        module.main.__wrapped__(cfg)
    finally:
        stream.close()
    if len(repetition_rows) != 40:
        raise ValueError("Expected 20 repetitions for each of GT and FlowMotion")
    save_json(output / "status.json", dict(status="complete", completed=20, pid=os.getpid(),
                                           per_clip_sha256=sha256(output / "per_clip.csv")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
