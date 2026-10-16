"""Render and retrieve surfaces for existing SALAD examples; arrays stay on the server."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'reproduction'))
from sync_results import remote, HIDDEN, SERVER

SOURCE = SERVER+'/reproduction/artifacts/physics_v1_20261016'
OUTPUT = SOURCE+'/mesh_v1'
CODE = SOURCE+'/code/mesh_v1'
LOCAL = REPO/'reproduction/artifacts/physics_v1_20261016/mesh_v1'
PYTHON = '/root/miniconda3/envs/mdm5090/bin/python'
BODY = '/root/autodl-tmp/data_preprocess_motion/raw/models/smplh/neutral/model.npz'
FONT = '/root/autodl-tmp/data_preprocess_motion/staging/standing_v2/assets/msyh.ttc'
FILES = ['mesh_surface.py', 'render_mesh.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start(limit):
    remote(f'''from pathlib import Path
import json,os
p=Path({OUTPUT!r});code=Path({CODE!r})
if (p/'launch.json').exists():
 old=json.loads((p/'launch.json').read_text())
 try:
  os.kill(old['pid'],0)
 except ProcessLookupError:pass
 else:raise RuntimeError('A surface renderer is already running')
code.mkdir(parents=True,exist_ok=True)
''')
    expected = {name: sha(REPO/'reproduction'/name) for name in FILES}
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', *[str(REPO/'reproduction'/n) for n in FILES],
                    'motion-data:'+CODE+'/'], check=True, capture_output=True, timeout=60, creationflags=HIDDEN)
    print(remote(f'''from pathlib import Path
import os,json,hashlib,subprocess
source=Path({SOURCE!r});p=Path({OUTPUT!r});code=Path({CODE!r})
expected={expected!r}
if {{n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in expected}}!=expected:raise ValueError('Code transfer differs')
p.mkdir(exist_ok=True)
env=dict(os.environ,PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (p/'render.log').open('ab') as log:
 proc=subprocess.Popen([{PYTHON!r},'-u',str(code/'render_mesh.py'),'--source',str(source),'--body',{BODY!r},'--font',{FONT!r},'--limit',str({limit})],cwd={SERVER!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,limit={limit},code_sha256=expected)
(p/'launch.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''').decode())


def status():
    print(remote(f'''from pathlib import Path
import json,os
p=Path({OUTPUT!r})
for name in ['launch.json','status.json']:
 if (p/name).exists():print(name,(p/name).read_text())
if (p/'render.log').exists():print((p/'render.log').read_text(errors='replace')[-3000:])
''').decode())


def fetch():
    snapshot = json.loads(remote(f'''from pathlib import Path
import base64,hashlib,json
p=Path({OUTPUT!r})
manifest=json.loads((p/'videos.json').read_text())
files={{n:base64.b64encode((p/n).read_bytes()).decode() for n in ['binding.json','camera.json','videos.json','status.json','launch.json']}}
media={{}}
for entry in manifest['videos']:
 for key,digest in [('path','sha256'),('poster','poster_sha256')]:
  relative=Path(entry[key])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid media path')
  actual=hashlib.sha256((p/relative).read_bytes()).hexdigest()
  if actual!=entry[digest]:raise ValueError('Server media differs')
  media[str(relative)]=actual
print(json.dumps(dict(files=files,media=media)))
'''))
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name, encoded in snapshot['files'].items():
        (LOCAL/name).write_bytes(base64.b64decode(encoded))
    for name, digest in snapshot['media'].items():
        target = LOCAL/name
        if target.exists() and sha(target) == digest:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', 'motion-data:'+OUTPUT+'/'+name, str(target)],
                        check=True, capture_output=True, timeout=60, creationflags=HIDDEN)
        if sha(target) != digest:
            raise ValueError('Local media transfer differs')
    (LOCAL/'transfer.json').write_text(json.dumps(dict(status='passed', files_sha256={
        name: sha(LOCAL/name) for name in snapshot['files']}, media_sha256=snapshot['media']), indent=2), encoding='utf-8')
    build_report()
    print(json.dumps(dict(status='synced', media_files=len(snapshot['media']))))


def build_report():
    manifest = json.loads((LOCAL/'videos.json').read_text(encoding='utf-8'))
    examples = json.loads((LOCAL.parent/'examples.json').read_text(encoding='utf-8'))
    available = {(r['id'], r['seed']): r for r in manifest['videos']}
    groups = {}
    for r in examples:
        if (r['id'], r['seed']) in available:
            groups.setdefault(r['id'], []).append(r)
    cards = []
    for key, records in groups.items():
        r = records[0]
        video = available[key, r['seed']]
        choices = ''.join('<option value="%d">%d</option>' % (v['seed'], v['seed']) for v in records)
        cards.append('''<article data-motion="%s"><div class="card-head"><h2>%s</h2><label>种子 <select aria-label="%s的采样种子">%s</select></label></div>
<p class="prompt">%s</p><video controls playsinline preload="metadata" src="%s" poster="%s"></video>
<dl><div><dt>滑步时间</dt><dd class="skating"></dd></div><div><dt>双脚离地</dt><dd class="airborne"></dd></div><div><dt>骨长变异</dt><dd class="bone"></dd></div></dl></article>''' % (
            key, html.escape(r['label']), html.escape(r['label']), choices, html.escape(r['text']), video['path'], video['poster']))
    payload = json.dumps([dict(id=r['id'], seed=r['seed'], metrics=r['metrics'], video=available[r['id'], r['seed']])
                          for values in groups.values() for r in values], ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SALAD 人体网格动作</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f6;color:#243632;font-family:"Microsoft YaHei",sans-serif;letter-spacing:0}
main{max-width:1360px;margin:auto;padding:28px 24px}header{border-bottom:1px solid #cdd8d3;padding-bottom:20px;margin-bottom:24px}
h1{font-size:28px;margin:0 0 10px;line-height:1.4}p{line-height:1.7;margin:8px 0}header p{color:#4c6259}a{color:#176d68;text-underline-offset:3px}
.links{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
article{background:white;border:1px solid #d5ddd9;border-radius:6px;overflow:hidden}.card-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 16px 0}
h2{font-size:20px;margin:0}label{font-size:13px;color:#53685f;display:flex;align-items:center;gap:8px}select{min-height:34px;max-width:140px;border:1px solid #b8c8bf;border-radius:4px;background:white;color:#273d33;padding:4px 8px;font-size:14px}
.prompt{font-size:13px;color:#586960;padding:0 16px;min-height:44px;overflow-wrap:anywhere}video{display:block;width:100%;aspect-ratio:8/5;background:#edf3f2}
dl{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:0;padding:16px;border-top:1px solid #e2e7e4}dt{font-size:12px;color:#64766e;margin-bottom:5px}dd{font-size:19px;margin:0;font-variant-numeric:tabular-nums}
footer{margin-top:24px;font-size:13px;color:#5a6b63;line-height:1.8}@media(max-width:720px){main{padding:20px 12px}.gallery{grid-template-columns:1fr;gap:18px}h1{font-size:24px}.card-head{padding:14px 12px 0}.prompt{padding:0 12px}dl{padding:12px}dd{font-size:17px}}
</style><main><header><h1>SALAD 人体网格动作</h1><p>官方 HumanML3D 权重 · 6 种文字描述 · 18 段固定样例 · 每段 6 秒 · 20 帧/秒</p>
<p>中性人体模板由原始关节驱动；地面高度固定为 0 米。未贴地、锁脚、平滑或修正骨长。网格为估计表面，物理指标仍来自原始关节。</p>
<div class="links"><a href="../index.html">骨架与物理指标</a><a href="REPORT_zh.md">网格记录</a><a href="binding.json">来源绑定</a></div></header><section class="gallery">CARDS</section>
<footer>滑步：相邻两帧脚尖均不高于 5 厘米，且水平速度超过 0.2 米/秒。双脚离地包含正常跳跃。文字描述是模型输入，不代表每段输出均已正确完成该动作。</footer></main>
<script>const records=PAYLOAD;for(const card of document.querySelectorAll('article[data-motion]')){const select=card.querySelector('select');const update=()=>{const r=records.find(r=>r.id===card.dataset.motion&&r.seed===Number(select.value));const v=card.querySelector('video');if(v.getAttribute('src')!==r.video.path){v.pause();v.src=r.video.path;v.poster=r.video.poster;v.load();}card.querySelector('.skating').textContent=(r.metrics.skating_frame_ratio*100).toFixed(1)+'%';card.querySelector('.airborne').textContent=(r.metrics.airborne_frame_ratio*100).toFixed(1)+'%';card.querySelector('.bone').textContent=(r.metrics.bone_length_cv*100).toFixed(2)+'%';};select.addEventListener('change',update);update();}</script></html>'''.replace('CARDS', '\n'.join(cards)).replace('PAYLOAD', payload)
    if (LOCAL.parent/'jitter_v1/index.html').exists():
        page = page.replace('<div class="links">', '<div class="links"><a href="../jitter_v1/index.html">抖动诊断与拟合对照</a>')
    (LOCAL/'index.html').write_text(page, encoding='utf-8')
    audit = manifest['geometry_audit']
    report = '''# SALAD 人体网格记录

已渲染 %d 段原有动作；每段 120 帧、20 帧/秒、1280×800，含正面与侧前方两个视角。没有重新采样或选择输出。

采用中性 SMPL-H 模板、姿态修正形变及关节驱动的线性混合蒙皮。关节附着保留原始关节位置及变长骨骼；未进行逆运动学拟合、平滑、锁脚、贴地或骨长修正。手指未生成独立动作，绕骨轴旋转由关节几何估计。

地面 y=0、0.5 米世界网格；两视角共用固定尺度，相机仅跟随水平根位移。模板表面不是模型直接输出，也不是精确身体重建；表面与脚底可能有估计误差。原物理指标完全沿用原始关节，不用网格重新解释穿地或滑步。

几何核验：中性模板最大误差 %.3g 米；已知刚体变换最大误差 %.3g 米；全部帧最大关节附着误差 %.3g 米。原始动作文件前后 SHA-256 全部一致：%s。模型资产、动作与网格数组均保留服务器，本地只同步视频、海报和小型核验文件。

[动作播放页](index.html) · [绑定](binding.json) · [视频摘要](videos.json) · [传输核验](transfer.json) · [原始物理诊断](../REPORT_zh.md)
''' % (len(manifest['videos']), audit['neutral_template_max_m'], audit['rigid_transform_max_m'],
       audit['joint_attachment_max_m'], audit['source_files_unchanged'])
    (LOCAL/'REPORT_zh.md').write_text(report, encoding='utf-8')


def record():
    manifest = json.loads((LOCAL/'videos.json').read_text(encoding='utf-8'))
    binding = json.loads((LOCAL/'binding.json').read_text(encoding='utf-8'))
    examples = json.loads((LOCAL.parent/'examples.json').read_text(encoding='utf-8'))
    qa = json.loads((LOCAL/'qa/browser.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or len(manifest['videos']) != 18:
        raise ValueError('All 18 surface videos must be complete')
    if {(v['id'], v['seed']) for v in manifest['videos']} != {(v['id'], v['seed']) for v in examples}:
        raise ValueError('Surface examples differ from the original set')
    if binding['examples_sha256'] != sha(LOCAL.parent/'examples.json'):
        raise ValueError('Original example metadata changed')
    if binding['renderer_sha256'] != sha(REPO/'reproduction/render_mesh.py') or binding['skinning_sha256'] != sha(REPO/'reproduction/mesh_surface.py'):
        raise ValueError('Local renderer differs from the executed source')
    if manifest['binding_sha256'] != sha(LOCAL/'binding.json') or not manifest['geometry_audit']['source_files_unchanged']:
        raise ValueError('Source or geometry verification failed')
    if qa['status'] != 'passed' or len(qa['video_checks']) != 36:
        raise ValueError('Desktop and mobile video acceptance required')
    for entry in manifest['videos']:
        source = next(v for v in examples if v['id'] == entry['id'] and v['seed'] == entry['seed'])
        if entry['source_sha256'] != source['sha256']:
            raise ValueError('Surface is bound to another motion')
        for key, digest in [('path', 'sha256'), ('poster', 'poster_sha256')]:
            if sha(LOCAL/entry[key]) != entry[digest]:
                raise ValueError('Delivered media changed')
    verification = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        videos=18, media_checksums=36, browser_checks=36, geometry_audit=manifest['geometry_audit'],
        same_original_motion_set=True)
    if (LOCAL/'verification.json').exists():
        prior = json.loads((LOCAL/'verification.json').read_text(encoding='utf-8'))
        if {k:v for k,v in prior.items() if k != 'checked_at'} != {k:v for k,v in verification.items() if k != 'checked_at'}:
            raise ValueError('Existing mesh verification differs')
        verification = prior
    else:
        (LOCAL/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    destination = REPO/'reproduction/results/mesh_v1'
    for name in ['binding.json', 'camera.json', 'videos.json', 'transfer.json', 'verification.json', 'qa/browser.json']:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (LOCAL/name).read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Archived surface evidence differs: '+name)
        target.write_bytes(content)
    report = (LOCAL/'REPORT_zh.md').read_text(encoding='utf-8')
    report = report.replace('](index.html)', '](artifacts/physics_v1_20261016/mesh_v1/index.html)')
    report = report.replace('](../REPORT_zh.md)', '](PHYSICS_zh.md)')
    for name in ['binding.json', 'videos.json', 'transfer.json']:
        report = report.replace(']('+name+')', '](results/mesh_v1/'+name+')')
    (REPO/'reproduction/MESH_zh.md').write_text(report, encoding='utf-8')
    print(json.dumps(verification))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preview', 'render', 'status', 'fetch', 'report', 'record'])
    stage = parser.parse_args().stage
    if stage in ['preview', 'render']:
        start(1 if stage == 'preview' else 18)
    else:
        {'status': status, 'fetch': fetch, 'report': build_report, 'record': record}[stage]()
