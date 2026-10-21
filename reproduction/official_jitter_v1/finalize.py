"""Check the completed public artifact, PNGs and runtime provenance."""
import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from audit_gt import save_json, sha256, FLOW_COMMIT


def finish(args):
    root=args.root.resolve()
    public=root/"public"
    verification=json.loads((root/"verification.json").read_text())
    assert verification["status"]=="passed" and verification["scope"]=="all_paired_runs"
    runs=["gt_t2m","gt_kit","salad_t2m","salad_kit","salad_ft_t2m",
          "mdm_bert50_t2m","mdm_clip1000_kit","flow_humanml_attempt3"]
    statuses={name:json.loads((root/name/"status.json").read_text()) for name in runs}
    assert all(state["status"]=="complete" for state in statuses.values())
    manifest=json.loads((public/"file_manifest.json").read_text())
    for name,expected in manifest.items():
        assert sha256(public/name)==expected,name
    pngs=sorted(public.glob("*.png"))
    assert len(pngs)==3
    figures=[]
    for path in pngs:
        with Image.open(path) as opened:
            opened.load()
            assert opened.format=="PNG" and opened.width>=2000 and opened.height>=900
            pixels=np.asarray(opened.convert("RGB"))
            assert np.std(pixels)>15
            # All four diagnostic axes, or all three compatibility panels, must contain marks.
            regions=([(.08,.48,.13,.43),(.56,.97,.13,.43),(.08,.48,.54,.84),(.56,.97,.54,.84)]
                     if "physical" in path.name else [(.06,.32,.20,.76),(.39,.65,.20,.76),(.72,.98,.20,.76)])
            for left,right,top,bottom in regions:
                tile=pixels[int(top*opened.height):int(bottom*opened.height),int(left*opened.width):int(right*opened.width)]
                assert np.mean(np.min(tile,axis=-1)<220)>.025,(path.name,left,top)
            figures.append(dict(file=path.name,width=opened.width,height=opened.height,
                                dpi=opened.info.get("dpi"),all_panels_nonblank=True,sha256=sha256(path)))
    assert not list(public.rglob("*.svg")) and not list(public.rglob("*.pdf")) and not list(public.rglob("*.tiff"))
    flow=args.flow_root.resolve()
    assert subprocess.check_output(["git","-C",str(flow),"rev-parse","HEAD"],text=True).strip()==FLOW_COMMIT
    assert not subprocess.check_output(["git","-C",str(flow),"status","--porcelain","--untracked-files=no"],text=True).strip()
    runtime={}
    for name in ["numpy","scipy","torch","diffusers","transformers","matplotlib","Pillow"]:
        try:
            runtime[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            runtime[name]="not in this evaluation environment"
    comparison_runtime=None
    if args.comparison_python:
        program="import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ['numpy','scipy','torch','transformers','torchdiffeq','hydra-core']}))"
        comparison_runtime=json.loads(subprocess.check_output([str(args.comparison_python),"-c",program],text=True))
    code=Path(__file__).parent
    result=dict(status="complete",scope="Official HumanML3D and KIT-ML benchmark; no new training",
                numerical_verification_sha256=sha256(root/"verification.json"),
                public_manifest_sha256=sha256(public/"file_manifest.json"),
                verified_public_files=len(manifest),completed_runs=statuses,
                sources_sha256={p.name:sha256(p) for p in code.iterdir() if p.is_file()},
                salad_and_verification_environment=runtime,flow_and_MDM_environment=comparison_runtime,
                figures=figures,flow_upstream_tracked_clean=True,
                visual_review="PNG bounds and panel pixels passed; human-visible final image inspection is separate",
                limitations=["ViMoGen unmeasured: official representation/evaluator adaptation unverified",
                             "No KIT-compatible full fine-tuning checkpoint; no LoRA checkpoint",
                             "FlowMotion semantic paper scores not fully reproduced; native sample count differs",
                             "Sampling intervals do not measure training seed or subject uncertainty"])
    save_json(public/"FINAL_VERIFICATION.json",result)
    print(json.dumps(dict(status="complete",files=len(manifest),pngs=len(figures))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--flow-root",type=Path,required=True)
    parser.add_argument("--comparison-python",type=Path,help="Separate FlowMotion/MDM evaluation environment")
    finish(parser.parse_args())
