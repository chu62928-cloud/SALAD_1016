"""Check the portable report, rendered media and their scientific source bindings."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    results=ROOT/'results/posture_v1'
    media=ROOT/'runs/posture_v1/media'
    scientific=read(results/'verification.json')
    assert scientific['status']=='passed' and all(scientific['checks'].values())
    assert scientific['audit_source_sha256']==sha(Path(__file__).with_name('audit.py'))
    source=scientific['training_source_binding']
    for name,digest in source['files'].items():
        assert sha(ROOT/name)==digest
    assert sha(ROOT/'models/denoiser/transformer.py')==source['transformer_sha256']
    assert sha(ROOT/'configs/posture_v1.json')==source['config_sha256']
    bindings=read(results/'bindings.json')
    assert sha(ROOT/'posture_conditioning/report.py')==bindings['report_source_sha256']
    assert sha(ROOT/'posture_conditioning/evaluate.py')==bindings['evaluation']['evaluation_source_sha256']
    assert sha(ROOT/'posture_conditioning/sample.py')==bindings['evaluation']['sample_source_sha256']
    videos=read(media/'videos.json')
    assert videos['status']=='complete' and len(videos['videos'])==8
    assert videos['renderer_sha256']==sha(ROOT/'posture_conditioning/render.py')
    for record in videos['videos']:
        assert sha(media/record['path'])==record['sha256']
        assert sha(media/record['poster'])==record['poster_sha256']
        assert min(record['min_body_pixels'])>500
    browser=read(media/'qa/browser.json')
    assert browser['status']=='passed' and len(browser['videos'])==16
    assert {v['width'] for v in browser['viewports']}=={390,1440}
    assert all(not v['overflow'] and not v['corrupted'] and not v['textOverflow'] for v in browser['viewports'])
    figure=read(results/'figure_qa.json')
    assert figure['status']=='passed' and not figure['visual_inspection_pending']
    for name,digest in figure['source_sha256'].items():
        assert sha(results/name)==digest
    exports=[]
    for stem in ['posture_results','attention_gate']:
        svg=ET.parse(results/(stem+'.svg'))
        assert list(svg.getroot().iter('{http://www.w3.org/2000/svg}text'))
        assert (results/(stem+'.pdf')).read_bytes().startswith(b'%PDF-')
        for suffix in ['png','tiff']:
            with Image.open(results/(stem+'.'+suffix)) as image:
                assert min(image.size)>1000 and min(image.info['dpi'])>=599
                if suffix=='tiff':
                    assert image.info['compression']=='tiff_lzw'
                image.verify()
        exports.extend(stem+'.'+s for s in ['svg','pdf','png','tiff'])
    names=['REPORT_zh.md','DELIVERY_zh.md','metrics.csv','by_source.csv','decision.json','bindings.json',
           'training.json','verification.json','figure_qa.json','attention_summary.json',
           'intervention_summary.csv']+exports
    receipt=dict(status='passed',media_files_sha256_verified=16,browser_playback_checks=16,
                 training_and_evaluation_source_bindings_verified=True,figure_exports_verified=True,
                 files={name:sha(results/name) for name in names},
                 browser_sha256=sha(media/'qa/browser.json'),media_manifest_sha256=sha(media/'videos.json'),
                 offline_page_sha256=sha(media/'index.html'),
                 plot_source_sha256=sha(ROOT/'posture_conditioning/figures.py'),
                 page_source_sha256=sha(ROOT/'posture_conditioning/page.py'),
                 browser_source_sha256=sha(ROOT/'posture_conditioning/qa_browser.cjs'),
                 naturalness_fixed=False,audit_source_sha256=sha(__file__))
    (results/'delivery_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Portable delivery audit passed: source bindings, 16 media hashes, 16 playback checks, 8 figure exports')


if __name__=='__main__':
    main()
