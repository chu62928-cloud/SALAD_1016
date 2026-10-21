"""Independent checks on saved outputs; never generate new motion or train."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from scipy.linalg import eigh


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_independent(raw, joints):
    x = np.asarray(raw, dtype=np.float64)
    angle = np.concatenate([[0.], np.cumsum(x[:-1,0])]) * 2
    cosine, sine = np.cos(angle), np.sin(angle)
    velocity = np.concatenate([np.zeros((1,2)), x[:-1,1:3]], axis=0)
    root = np.zeros((len(x),3))
    root[:,0] = np.cumsum(cosine*velocity[:,0] - sine*velocity[:,1])
    root[:,2] = np.cumsum(sine*velocity[:,0] + cosine*velocity[:,1])
    root[:,1] = x[:,3]
    position = x[:,4:4+(joints-1)*3].reshape(len(x),joints-1,3).copy()
    px, pz = position[...,0].copy(), position[...,2].copy()
    position[...,0] = cosine[:,None]*px-sine[:,None]*pz+root[:,0:1]
    position[...,2] = sine[:,None]*px+cosine[:,None]*pz+root[:,2:3]
    return np.concatenate([root[:,None],position],axis=1)


def analytic_physical(p, dataset):
    fps = 20 if dataset == "t2m" else 12.5
    local = p-p[:,:1]
    velocity = (local[1:]-local[:-1])*fps
    acceleration = (velocity[1:]-velocity[:-1])*fps
    jerk = (acceleration[1:]-acceleration[:-1])*fps
    grid = np.arange(len(p))
    matrix = np.sqrt(2/len(p))*np.cos(np.pi/len(p)*np.outer(grid,grid+.5))
    matrix[0] /= np.sqrt(2)
    coefficients = matrix@(local-local.mean(0)).reshape(len(p),-1)
    high = coefficients[grid*fps/(2*len(p))>=3]
    parents = ([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
               if dataset == "t2m" else [-1,0,1,2,3,3,5,6,3,8,9,0,11,12,13,14,0,16,17,18,19])
    bone = np.array([np.linalg.norm(p[:,j]-p[:,parents[j]],axis=1) for j in range(1,len(parents))])
    toes = [10,11] if dataset == "t2m" else [15,20]
    foot = p[:,toes]
    sliding = []
    for t in range(len(p)-1):
        sliding.append(any(foot[t,k,1]<=.05 and foot[t+1,k,1]<=.05 and
                           np.linalg.norm(foot[t+1,k,[0,2]]-foot[t,k,[0,2]])*fps>.2 for k in range(2)))
    return dict(local_jerk_rms_mps3=float(np.sqrt(np.mean(np.linalg.norm(jerk,axis=-1)**2))),
                local_hf3_vector_rms_mm=float(np.sqrt(np.sum(high**2)/(len(p)*p.shape[1]))*1000),
                bone_cv_percent=float(np.mean(bone.std(1)/bone.mean(1))*100),
                sliding_frame_percent=float(np.mean(sliding)*100))


def fid_independent(gt, generated):
    a,b = [np.asarray(x,dtype=np.float64) for x in [gt,generated]]
    mean_a, mean_b = a.mean(0), b.mean(0)
    cov_a,cov_b = np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    values,vectors = eigh(cov_a)
    half = (vectors*np.sqrt(np.maximum(values,0)))@vectors.T
    product = half@cov_b@half
    roots = np.sqrt(np.maximum(eigh((product+product.T)/2,eigvals_only=True),0))
    return float(np.sum((mean_a-mean_b)**2)+np.trace(cov_a)+np.trace(cov_b)-2*roots.sum())


def run(args):
    root=args.root.resolve()
    salad_root=Path(__file__).resolve().parents[2]
    checks=[]
    inputs=[("gt_t2m",20,4640),("gt_kit",20,704),("salad_t2m",3,4640),
            ("salad_kit",3,704),("salad_ft_t2m",3,4640),("mdm_bert50_t2m",3,4640),
            ("mdm_clip1000_kit",3,704)]
    if args.runs:
        inputs=[spec for spec in inputs if spec[0] in args.runs]
        assert {spec[0] for spec in inputs}==set(args.runs), "Unknown audit run"
    for name,repeats,clips in inputs:
        directory=root/name
        binding=json.loads((directory/"binding.json").read_text())
        if "source_sha256" in binding:
            for path,expected in binding["source_sha256"].items():
                if Path(path).name in ["audit_gt.py","tests_metrics.py","run_salad.py"]:
                    assert sha(salad_root/path)==expected,(name,path)
        if "driver_sha256" in binding:
            assert sha(Path(__file__).with_name("run_mdm_kit.py"))==binding["driver_sha256"]
        status=json.loads((directory/"status.json").read_text())
        if status["status"]!="complete":
            raise ValueError("Not complete: "+name)
        summary=json.loads((directory/"summary.json").read_text())
        rows=list(csv.DictReader((directory/"per_clip.csv").open(encoding="utf-8")))
        default_model="GT" if name.startswith("gt") else ("MDM_CLIP_1000" if name=="mdm_clip1000_kit" else "MDM_BERT_50")
        models=sorted({r.get("model",default_model) for r in rows})
        for model in models:
            for repeat in range(repeats):
                group=[r for r in rows if int(r["repeat"])==repeat and r.get("model",model)==model]
                assert len(group)==clips,(name,model,repeat,len(group))
        rep=json.loads((directory/"repetitions.json").read_text())
        for row in rep:
            group=[r for r in rows if int(r["repeat"])==int(row["repeat"]) and r.get("model",row.get("model",""))==row.get("model",r.get("model",""))]
            for key in group[0]:
                if key in row and key not in ["model","repeat","clips","clip_index","batch","within_batch"]:
                    assert abs(np.mean([float(r[key]) for r in group])-row[key])<1e-8,(name,key)
        for model in models:
            metrics=summary["metrics"][model] if name.startswith("salad") else summary["metrics"]
            selected=[r for r in rep if r.get("model",model)==model]
            for key,value in metrics.items():
                values=np.array([r[key] for r in selected])
                assert abs(value["mean"]-values.mean())<1e-9,(name,model,key,"summary_mean")
                expected=1.96*values.std()/np.sqrt(len(values))
                assert abs(value["ci95_half_width"]-expected)<1e-9,(name,model,key,"summary_interval")
        checks.append(dict(check="complete_coverage_and_clip_aggregation",run=name,rows=len(rows),models=models))
        if name.startswith("gt"):
            assert summary["gt_closeness_screen_passed"]
            continue
        dataset="kit" if name.endswith("kit") else "t2m"
        mean=np.load(salad_root/"checkpoints"/dataset/"Comp_v6_KLD005/meta/mean.npy")
        std=np.load(salad_root/"checkpoints"/dataset/"Comp_v6_KLD005/meta/std.npy")
        joints,scale=(21,.001) if dataset=="kit" else (22,1.)
        for repeat in range(repeats):
            shards=sorted((directory/"server_only"/("repeat_%02d"%repeat)).glob("batch_*.npz"))
            assert len(shards)==clips//32
            max_jerk_error,max_hf_error,max_bone_error,max_sliding_error=0.,0.,0.,0.
            for shard in [shards[0],shards[len(shards)//2],shards[-1]]:
                batch_index=int(shard.stem.split("_")[-1])
                with np.load(shard) as cache:
                    normalized=cache["generated"] if "generated" in cache else cache["normalized_eval"]
                    for i in [0,15,31]:
                        length=int(cache["lengths"][i])
                        raw=normalized[i,:length]*std+mean
                        p=recover_independent(raw,joints)*scale
                        direct=analytic_physical(p,dataset)
                        model_name="SALAD_full_finetune" if "ft_" in name else ("SALAD" if name.startswith("salad") else default_model)
                        saved=next(r for r in rows if int(r["repeat"])==repeat and int(r["batch"])==batch_index and int(r["within_batch"])==i and r.get("model",model_name)==model_name)
                        max_jerk_error=max(max_jerk_error,abs(direct["local_jerk_rms_mps3"]-float(saved["local_jerk_rms_mps3"])))
                        max_hf_error=max(max_hf_error,abs(direct["local_hf3_vector_rms_mm"]-float(saved["local_hf3_vector_rms_mm"])))
                        max_bone_error=max(max_bone_error,abs(direct["bone_cv_percent"]-float(saved["bone_cv_percent"])))
                        max_sliding_error=max(max_sliding_error,abs(direct["sliding_frame_percent"]-float(saved["sliding_frame_percent"])))
            # Independent double recovery differs from the author's single precision.
            assert max_jerk_error<.02,(name,repeat,max_jerk_error)
            assert max_hf_error<.002,(name,repeat,max_hf_error)
            assert max_bone_error<.0001,(name,repeat,max_bone_error)
            assert max_sliding_error<.001,(name,repeat,max_sliding_error)
            with np.load(directory/"server_only"/("repeat_%02d"%repeat)/"embeddings.npz") as embedded:
                gt=embedded["GT"] if "GT" in embedded else embedded["gt"]
                generated=embedded[model_name] if model_name in embedded else embedded["generated"]
                fid=fid_independent(gt,generated)
            reference=next(r["fid"] for r in rep if r["repeat"]==repeat and r.get("model",model_name)==model_name)
            assert abs(fid-reference)<1e-4,(name,repeat,fid,reference)
            checks.append(dict(check="independent_recovery_derivatives_cosine_spectrum_and_FID",run=name,repeat=repeat,
                               sampled_clips=9,max_jerk_error=max_jerk_error,max_hf_error_mm=max_hf_error,fid_error=abs(fid-reference)))
            checks[-1].update(max_bone_cv_error=max_bone_error,max_sliding_percent_error=max_sliding_error)
    for repeat in range(3) if not args.runs else []:
        a=root/"salad_t2m/server_only"/("repeat_%02d"%repeat)
        b=root/"salad_ft_t2m/server_only"/("repeat_%02d"%repeat)
        paths=sorted(a.glob("batch_*.npz"))
        assert len(paths)==145
        for path in paths:
            with np.load(path) as x,np.load(b/path.name) as y:
                for key in ["gt","lengths","captions","tokens","word","pos","sent_len"]:
                    assert np.array_equal(x[key],y[key]),(repeat,path.name,key)
        checks.append(dict(check="official_finetune_exact_input_pairing",repeat=repeat,batches=145))
        for mdm_name,salad_name,batch_count in [("mdm_bert50_t2m","salad_t2m",145),("mdm_clip1000_kit","salad_kit",22)]:
            source=root/salad_name/"server_only"/("repeat_%02d"%repeat)
            directory=root/mdm_name/"server_only"/("repeat_%02d"%repeat)
            paths=sorted(directory.glob("batch_*.npz"))
            assert len(paths)==batch_count
            for path in paths:
                with np.load(path) as cache:
                    assert str(cache["input_cache_sha256"])==sha(source/path.name)
            checks.append(dict(check="MDM_paired_input_cache_sha256",run=mdm_name,repeat=repeat,batches=batch_count))
    if not args.runs:
        directory=root/"flow_humanml_attempt3"
        assert json.loads((directory/"status.json").read_text())["status"]=="complete"
        rows=list(csv.DictReader((directory/"per_clip.csv").open(encoding="utf-8")))
        repetitions=json.loads((directory/"repetitions.json").read_text())
        assert len(repetitions)==40
        for row in repetitions:
            group=[r for r in rows if int(r["repeat"])==row["repeat"] and r["model"]==row["model"]]
            assert len(group)==(4640 if row["model"]=="ground truth" else 1024)
            for key in group[0]:
                if key not in ["repeat","model"]:
                    assert abs(np.mean([float(r[key]) for r in group])-row[key])<1e-8,("FlowMotion",key)
        checks.append(dict(check="native_FlowMotion_20_repeat_coverage_and_aggregation",rows=len(rows),
                           limitation="No full generated cache was recorded by the native evaluator; physical values independently aggregated, not regenerated."))
    files={str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file() and "server_only" not in p.parts
           and p.name not in ["run.lock","verification.json","verification.partial.json"] and p.suffix in [".json",".csv"]}
    result=dict(status="passed",scope=args.runs or "all_paired_runs",checks=checks,result_sha256=files)
    destination="verification.partial.json" if args.runs else "verification.json"
    (root/destination).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="passed",checks=len(checks),files=len(files))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--runs",nargs="+",help="Verify completed subsets without claiming final completion")
    run(parser.parse_args())
