"""Prepare a small Git record without weights, raw data, or person-level caches."""
import json
from pathlib import Path
import shutil
import hashlib


def main():
    root=Path(__file__).resolve().parents[2]
    source=root/'reproduction/artifacts/jitter_attribution_v3'
    destination=root/'results/jitter_attribution_v3'
    names=['REPORT_zh.md','protocol.json','FINAL_VERIFICATION.json',
        'summary/summary.json','summary/paired_comparisons.json','summary/flat_walking.json',
        'summary/group_means.csv','summary/verification.json','summary/independent_verification.json',
        'summary/generated_bone_summary.csv','summary/generated_joint_summary.csv',
        'summary/vae_bone_summary.csv','summary/vae_joint_summary.csv',
        'figures/figure_contract_and_manifest.json','qa/browser.json','qa/figure_source_preflight.json']
    names.extend(str(file.relative_to(source)).replace('\\','/') for file in sorted((source/'figures').glob('*.png')))
    binding={}
    for name in names:
        file=source/name;target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target)
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        binding[name]=dict(bytes=file.stat().st_size,sha256=digest)
    (destination/'publication_manifest.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print('PUBLICATION PREPARED',len(binding),sum(r['bytes'] for r in binding.values()),flush=True)


if __name__=='__main__':main()
