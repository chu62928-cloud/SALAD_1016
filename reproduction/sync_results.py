"""Read-only server evidence sync and local SALAD reproduction reporting."""
import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DESTINATION = REPO / 'reproduction/artifacts/evidence'
SERVER = '/root/autodl-tmp/SALAD_1016'
RUNS = {'t2m': 'eval_t2m_v2', 'kit': 'eval_kit_v3',
        't2m_vae': 'eval_t2m_vae_v2', 'kit_vae': 'eval_kit_vae'}
LABELS = {'top1': '检索率 R1', 'top2': '检索率 R2', 'top3': '检索率 R3',
          'fid': 'FID', 'matching': '匹配距离', 'diversity': '多样性',
          'multimodality': '多模态', 'mpjpe': '平均关节位置误差'}
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
SOURCES = {'t2m': 'bd2b7d5bc143d422629f9dfe3e09838a8c6532c1',
           'kit': '807ff210af0b81e5a0833c2fff0ee95cd9bd0773',
           't2m_vae': 'bd2b7d5bc143d422629f9dfe3e09838a8c6532c1',
           'kit_vae': '18d94fd7ee46d7102b5adda6bd2d24a17eaebd0a'}


def remote(code, timeout=55):
    encoded = base64.b64encode(code.encode()).decode()
    command = ("/root/autodl-tmp/envs/salad_1016/bin/python -c "
               + '"import base64;exec(base64.b64decode(\'' + encoded + '\'))"')
    result = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', 'motion-data', command],
                            capture_output=True, timeout=timeout, creationflags=HIDDEN)
    if result.returncode:
        raise ConnectionError(result.stderr.decode(errors='replace'))
    return result.stdout


def git(*arguments, timeout=90):
    return subprocess.check_output(['git', *arguments], cwd=REPO, text=True,
                                   stderr=subprocess.STDOUT, timeout=timeout, creationflags=HIDDEN).rstrip('\r\n')


def json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def collect():
    code = f'''
from pathlib import Path
import json,base64,hashlib,subprocess,datetime,os
p=Path({SERVER!r});a=p/'reproduction/artifacts'
names=['assets_t2m.json','assets_kit.json','downloads_t2m.json','downloads_kit.json',
 'hf_download_clip.json','hf_download_kit.json','preflight_t2m.json','preflight_kit.json',
 'training_t2m/summary.json','training_kit/summary.json','resume_validation.json',
 'engineering_checks.json']
for folder in {list(RUNS.values())!r}:
 names += [folder+'/'+n for n in ['summary.json','binding.json','repetitions.json','metric_audit.json']]
files={{}};hashes={{}};jobs={{}};tails={{}}
for name in names:
 f=a/name
 if f.is_file():
  b=f.read_bytes();files[name]=base64.b64encode(b).decode();hashes[name]=hashlib.sha256(b).hexdigest()
for folder in {list(RUNS.values())!r}:
 f=a/folder/'summary.json'
 if f.exists():
  v=json.loads(f.read_text());pid=v.get('pid');c=Path('/proc')/str(pid)/'cmdline'
  live=c.exists() and b'reproduction/run_evaluation.py' in c.read_bytes()
  jobs[folder]={{'pid':pid,'alive':live,'status':v['status'],'completed_repeats':v['completed_repeats']}}
 log=a/(folder+'.log')
 if log.exists():tails[folder]=log.read_text(errors='replace')[-1600:]
media={{}}
for folder in ['videos_t2m','videos_kit']:
 for f in (a/folder).glob('*'):
  if f.is_file() and f.suffix in ['.mp4','.json']:
   media[str(f.relative_to(a))]={{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}}
print(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,sha256=hashes,
 jobs=jobs,tails=tails,media=media,head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),
 dirty=subprocess.check_output(['git','-C',str(p),'status','--porcelain','--untracked-files=no'],text=True))))
'''
    snapshot = json.loads(remote(code, timeout=40))
    for name, encoded in snapshot.pop('files').items():
        content = base64.b64decode(encoded)
        if hashlib.sha256(content).hexdigest() != snapshot['sha256'][name]:
            raise ValueError('Evidence transfer checksum differs: ' + name)
        target = DESTINATION / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    json_write(DESTINATION / 'transfer.json', snapshot)
    return snapshot


def load(name):
    return json.loads((DESTINATION/name).read_text(encoding='utf-8'))


def verify(snapshot):
    if snapshot['dirty']:
        raise ValueError('Server tracked source changes detected')
    report = {'status': 'passed', 'runs': {}, 'complete': True}
    protocol_bytes = (REPO/'reproduction/protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    for key, folder in RUNS.items():
        summary = load(folder+'/summary.json')
        binding = load(folder+'/binding.json')
        dataset = key.split('_')[0]
        expected = {'commit': SOURCES[key], 'dataset': dataset,
                    'kind': 'vae' if key.endswith('_vae') else 'denoiser',
                    'seed': protocol['seed'], 'repeats': protocol['repetitions'],
                    'steps': protocol['inference_steps'], 'scale': protocol['guidance_scale'],
                    'cal_mm': not key.endswith('_vae'),
                    'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
                    'asset_audit_sha256': snapshot['sha256']['assets_'+dataset+'.json']}
        if any(binding.get(k) != v for k,v in expected.items()):
            raise ValueError('Frozen evaluation binding differs: '+key)
        binding_hash = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        if summary.get('binding_sha256') != binding_hash or summary.get('requested_repeats') != 20:
            raise ValueError('Summary does not match evaluation binding: '+key)
        coverage = summary['coverage']
        if any(coverage.get(k) != protocol[k] for k in ['batch_size','shuffle','drop_last']):
            raise ValueError('Evaluation loader differs from frozen protocol: '+key)
        if coverage['evaluated_per_repeat'] != (4640 if dataset == 't2m' else 704):
            raise ValueError('Unexpected test coverage: '+key)
        path = DESTINATION/folder/'repetitions.json'
        rows = load(folder+'/repetitions.json') if path.exists() else []
        if len(rows) != summary['completed_repeats']:
            raise RuntimeError('Snapshot changed between reads; retry synchronization')
        if [row['repeat'] for row in rows] != list(range(len(rows))):
            raise ValueError('Repeated or missing evaluation repetition')
        metrics = {'fid','diversity','top1','top2','top3','matching',
                   'mpjpe' if key.endswith('_vae') else 'multimodality'}
        if rows and set(summary.get('metrics', {})) != metrics:
            raise ValueError('Missing or unexpected evaluation metrics: '+key)
        for metric, info in summary.get('metrics', {}).items():
            values = np.asarray([r[metric] for r in rows], dtype=np.float64)
            expected = {'mean': float(values.mean()),
                        'ci95_half_width': float(1.96*values.std(ddof=0)/np.sqrt(len(values)))}
            if not np.isfinite(values).all() or any(abs(info[k]-v)>1e-12 for k,v in expected.items()):
                raise ValueError('Independent summary mismatch: '+key+'/'+metric)
        complete = summary['status'] == 'complete' and len(rows) == 20
        if complete and load(folder+'/metric_audit.json')['status'] != 'passed':
            raise ValueError('Independent runtime metric audit failed')
        for metric, target in protocol['paper_targets'].get(key, {}).items():
            if metric == 'parameters_millions' or not rows:
                continue
            info = summary['metrics'][metric]
            if (info['paper_mean'],info['paper_ci95_half_width']) != tuple(target):
                raise ValueError('Paper target changed: '+key+'/'+metric)
            if abs(info['difference']-(info['mean']-target[0])) > 1e-12:
                raise ValueError('Paper difference mismatch: '+key+'/'+metric)
        if key in ['t2m','kit']:
            if not binding['cal_mm'] or any(row['multimodality'] is None for row in rows):
                raise ValueError('Missing multimodality results')
            if complete and not summary['table1_evaluation_complete']:
                raise ValueError('Incomplete paper-table evaluation')
        report['runs'][key] = {'completed_repeats': len(rows), 'complete': complete,
                              'source_commit': binding['commit'], 'summary_recomputed': bool(rows)}
        report['complete'] &= complete
    for dataset in ['t2m','kit']:
        for name in ['assets_'+dataset+'.json','preflight_'+dataset+'.json',
                     'training_'+dataset+'/summary.json']:
            if load(name)['status'] != 'passed':
                raise ValueError('Required check not passed: '+name)
    if load('resume_validation.json')['status'] != 'passed':
        raise ValueError('Exact resume comparison failed')
    if load('engineering_checks.json')['status'] != 'passed':
        raise ValueError('Dependency and regression checks failed')
    json_write(DESTINATION/'verification.json', report)
    return report


def sync_media(snapshot):
    for name, info in snapshot['media'].items():
        target = REPO/'reproduction/artifacts'/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == info['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['scp','-q','-o','BatchMode=yes', 'motion-data:'+SERVER+'/reproduction/artifacts/'+name,
                        str(target)], check=True, capture_output=True, timeout=60, creationflags=HIDDEN)
        if hashlib.sha256(target.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Media transfer checksum differs: '+name)
    json_write(DESTINATION/'media_transfer.json', {'status':'passed','files':snapshot['media']})


def describe(snapshot, verification):
    local_time = datetime.fromisoformat(snapshot['checked_at']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S %z')
    lines = ['# SALAD 复现结果', '', '核验时间：'+local_time+'。', '',
             '状态：'+('四组评估均已完成 20 次重复。' if verification['complete'] else '正式评估仍在运行，以下未满 20 次的数值仅为阶段性结果。'), '',
             '范围：作者预训练权重的 HumanML3D、KIT-ML 测试集评估；两套 VAE 与去噪网络各 20 步训练短测；四种编辑检查。未执行完整从零训练和人工主观用户实验。', '',
             '代码仓库：https://github.com/chu62928-cloud/SALAD_1016', '',
             '论文：https://openaccess.thecvf.com/content/CVPR2025/html/Hong_SALAD_Skeleton-aware_Latent_Diffusion_for_Text-driven_Motion_Generation_and_Editing_CVPR_2025_paper.html', '']
    tables = []
    for key, title in [('t2m','HumanML3D 文本生成'),('kit','KIT-ML 文本生成'),
                       ('t2m_vae','HumanML3D VAE 重建'),('kit_vae','KIT-ML VAE 重建')]:
        folder = RUNS[key]
        value = load(folder+'/summary.json')
        count = value['completed_repeats']
        lines += ['## '+title,'',f'已完成 {count}/20 次。', '',
                  '| 指标 | 本次均值 ± 95%区间半宽 | 原文 | 差值 |',
                  '| --- | --- | --- | --- |']
        table = []
        for metric in ['top1','top2','top3','fid','matching','diversity','multimodality','mpjpe']:
            info = value.get('metrics',{}).get(metric)
            if info is None:
                continue
            target = f"{info['paper_mean']:.6f} ± {info['paper_ci95_half_width']:.6f}" if 'paper_mean' in info else '未单列对照值'
            difference = f"{info['difference']:+.6f}" if 'difference' in info else '-'
            text = f"{info['mean']:.6f} ± {info['ci95_half_width']:.6f}"
            label = LABELS[metric]
            if key=='t2m_vae' and metric=='mpjpe': label += '（米）'
            if key=='kit_vae' and metric=='mpjpe': label += '（原始毫米）'
            lines.append(f'| {label} | {text} | {target} | {difference} |')
            table.append([label,text,target,difference])
        lines += ['', '指标实现保留作者原口径。区间为 `1.96 × 总体标准差 / √重复次数`；区间重叠不能证明统计等价。', '']
        if key=='t2m_vae':
            lines += ['论文仅报告三位小数：本次 FID 0.003370、关节误差 0.016327 在相同精度下为 0.003、0.016。不能把原文舍入后的 ±0.000 当成精确零方差。', '']
        if key=='kit_vae':
            lines += ['KIT 原始坐标以毫米计，15.162996 毫米约为 0.015163 米；本结果未与 HumanML3D 的米制直接混用。', '']
        if key in ['t2m','kit'] and count == 20:
            compared = [m for m in value['metrics'].values() if 'paper_mean' in m]
            separated = [LABELS[m] for m,v in value['metrics'].items()
                         if 'paper_mean' in v and not v['confidence_intervals_overlap']]
            lines += [f'论文对照：{len(compared)} 个指标中，{len(compared)-len(separated)} 个均值区间与原文区间重叠。'
                      + ('未重叠的指标为：'+ '、'.join(separated)+'。' if separated else '')
                      + '这里没有预先约定的统计等价界值，因此不将区间重叠写成严格等价性检验通过。', '']
        tables.append((title,count,table))
    lines += ['## 数据与协议', '',
              'HumanML3D 审计 29,228 个编号，KIT-ML 审计 6,018 个编号。KIT 原包包含 2 个缺失训练动作和 6 个含断行描述的训练文件，按编号及内容摘要严格登记，保留作者加载行为；验证及测试集没有这些异常。两套资产审计均通过。', '',
              '固定测试划分、种子 1234、批量 32、打乱并丢弃末批、4 个工作进程、50 步 DDIM、无分类器引导强度 7.5。每次 HumanML3D 实评 4,640 条，KIT-ML 实评 704 条。多模态沿用前三批共 96 个描述、每个描述 30 次生成。没有根据测试指标修改配置或挑选重复。', '',
              'CLIP 固定发布版本并核对内容哈希；KIT 镜像的三份划分与官方文件逐字节一致。模型、统计量、评估器、词向量和运行环境均记录在绑定文件中。', '',
              '## 工程验证', '',
              '- 9 项回归测试通过；两套真实权重前向、50 步生成、固定种子精确重放、文本作用及评估配对排序检查通过。',
              '- 两套数据上 VAE 与新初始化去噪网络各更新 20 步，批量 4；梯度有限、参数更新、模型/优化器/调度器恢复检查通过，冻结的 VAE 参数保持逐值一致，VAE 与 CLIP 没有梯度。',
              '- 评估断点恢复后，前两次全部未舍入指标与连续运行逐项一致。原训练检查点缺随机状态与迭代位置，因此不声称训练轨迹可精确续跑。',
              '- 首轮真实嵌入上，检索计数独立复算完全一致，生成 FID 复算差异小于 2e-9。原距离函数的单精度消减误差已记录并按浮点误差界核验，作者指标原值保持不变。所有逐次结果的均值和区间在本地独立复算。',
              '- 四类编辑（词语替换、细化、权重调整、镜像）均能完成，输出有限且有变化，源注意力缓存保持不变。示例用于检查工作流程，不能代替编辑语义正确性的人审或论文主观实验。', '',
              '## 修复与边界', '',
              '已修复无引导采样丢失文本、命令行种子被覆盖、科学计数法与含冒号选项读取、训练窗口重复和遗漏、镜像编辑修改源缓存、插入词语后注意力未对齐、恢复训练静默忽略优化器失败。', '',
              '论文和发布配置的 VAE 批量、去噪网络学习率及 VAE 衰减系数存在差异，完整从零训练前必须明确所用口径。本次短测不能证明最终收敛。KIT 没有带时间段的描述，原加载器固定 20 帧/秒裁剪不影响本次 KIT 基准。', '',
              '## 版本和证据', '']
    for key, info in verification['runs'].items():
        folder=RUNS[key]
        lines += [f"- {key}：源码 `{info['source_commit']}`；[逐次结果](evidence/{folder}/repetitions.json)、[摘要](evidence/{folder}/summary.json)、[指标复算](evidence/{folder}/metric_audit.json)、[运行绑定](evidence/{folder}/binding.json)。"]
    lines += ['', '本地证据的跨端 SHA-256 收据为 [transfer.json](evidence/transfer.json)，独立汇总核验为 [verification.json](evidence/verification.json)。模型与大数据保留在服务器，未推送到 Git。', '', '## 动作示例', '']
    for dataset,title in [('t2m','HumanML3D'),('kit','KIT-ML')]:
        for key,label in [('original','原始生成'),('word_swap','词语替换'),('refine','细化'),('reweight','权重调整'),('mirror','镜像')]:
            lines.append(f'- {title} {label}：[MP4](videos_{dataset}/{key}.mp4)')
    return '\n'.join(lines)+'\n', tables, local_time


def render(snapshot, verification):
    markdown,tables,stamp=describe(snapshot,verification)
    target=REPO/'reproduction/artifacts'
    (target/'RESULTS_zh.md').write_text(markdown,encoding='utf-8')
    parts=[]
    for title,count,rows in tables:
        body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
        parts.append(f'<section><h2>{title}</h2><p>{count} / 20 次</p><div class="table"><table><thead><tr><th>指标</th><th>本次均值 ± 区间半宽</th><th>原文</th><th>差值</th></tr></thead><tbody>{body}</tbody></table></div></section>')
    videos=[]
    for dataset,title in [('t2m','HumanML3D'),('kit','KIT-ML')]:
        items=[]
        for key,label in [('original','原始生成'),('word_swap','词语替换'),('refine','细化'),('reweight','权重调整'),('mirror','镜像')]:
            items.append(f'<figure><video controls muted playsinline preload="metadata" src="videos_{dataset}/{key}.mp4"></video><figcaption>{label}</figcaption></figure>')
        videos.append('<section><h2>'+title+' 动作示例</h2><div class="videos">'+''.join(items)+'</div></section>')
    state='已完成' if verification['complete'] else '评估运行中'
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SALAD 复现结果</title><style>
body{margin:0;color:#202423;background:#fff;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;letter-spacing:0}main{max-width:1140px;padding:28px 24px;margin:auto}header{border-bottom:2px solid #218273;padding-bottom:20px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}p{margin:8px 0}section{border-bottom:1px solid #d5dcda;padding:24px 0}a{color:#146c9b}small{color:#596460}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{text-align:left;border-bottom:1px solid #e0e5e3;padding:9px 12px}th{background:#f0f5f3;font-weight:600}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}figure{margin:0;min-width:0}video{width:100%;aspect-ratio:1;display:block;background:#f6f7f6}figcaption{padding:6px 0;color:#4c5652}@media(max-width:650px){main{padding:20px 16px}.videos{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}h1{font-size:24px}}
</style><main><header><h1>SALAD 复现结果</h1><p>STATE · STAMP</p><p>官方预训练权重评估与训练短测</p><a href="RESULTS_zh.md">完整中文报告</a> · <a href="evidence/verification.json">汇总核验</a></header>CONTENT<section><h2>验证边界</h2><p>未满 20 次的结果为阶段性数值。短测通过不代表完整训练收敛；视频样例不能代替编辑语义人审。全部未舍入结果、模型绑定与独立指标复算见完整报告。</p></section>VIDEOS</main></html>'''
    document=document.replace('STATE',state).replace('STAMP',stamp).replace('CONTENT',''.join(parts)).replace('VIDEOS',''.join(videos))
    (target/'index.html').write_text(document,encoding='utf-8')


def finalize(snapshot, verification, publish):
    if not verification['complete']:
        raise ValueError('Cannot finalize incomplete evaluations')
    if git('branch','--show-current') != 'main':
        raise ValueError('Expected main branch before publishing results')
    marker = REPO/'reproduction/artifacts/completion.json'
    previous = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else None
    dirty = git('status','--porcelain','--untracked-files=all')
    if previous:
        for name, digest in previous['generated_files'].items():
            path = REPO/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Prepared result changed; automatic publication stopped: '+name)
        if any(line[3:] not in previous['generated_files'] for line in dirty.splitlines()):
            raise ValueError('Other local changes detected before publication')
    elif dirty:
        raise ValueError('Local changes detected before result preparation; no tracked files written')
    if not previous:
        markdown,_,_=describe(snapshot,verification)
        final=REPO/'reproduction/results'
        final.mkdir(exist_ok=True)
        def copy_evidence(source, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((DESTINATION/source).read_bytes())
        for key,folder in RUNS.items():
            for name in ['summary.json','binding.json','repetitions.json','metric_audit.json']:
                copy_evidence(folder+'/'+name,final/key/name)
        for name in ['assets_t2m.json','assets_kit.json','downloads_t2m.json','downloads_kit.json',
                     'hf_download_clip.json','hf_download_kit.json','preflight_t2m.json','preflight_kit.json',
                     'training_t2m/summary.json','training_kit/summary.json','resume_validation.json',
                     'engineering_checks.json','verification.json','transfer.json']:
            path=DESTINATION/name
            if path.exists():copy_evidence(name,final/name)
        # The tracked report links to the small evidence committed with it.
        for key,folder in RUNS.items():markdown=markdown.replace('evidence/'+folder+'/', 'results/'+key+'/')
        markdown=markdown.replace('(evidence/transfer.json)','(results/transfer.json)').replace('(evidence/verification.json)','(results/verification.json)')
        markdown=markdown.replace('(videos_t2m/','(artifacts/videos_t2m/').replace('(videos_kit/','(artifacts/videos_kit/')
        (REPO/'reproduction/RESULTS_zh.md').write_text(markdown,encoding='utf-8')
        status=REPO/'reproduction/STATUS_zh.md'
        with status.open('a',encoding='utf-8') as stream:
            stream.write('\n## 完整评估已完成\n\n'+snapshot['checked_at']+
                         '：两套文本生成与两套 VAE 评估均已完成 20 次。独立汇总复算通过，实际数值和限制见 [完整复现报告](RESULTS_zh.md)。\n')
        paths = [REPO/'reproduction/RESULTS_zh.md', REPO/'reproduction/STATUS_zh.md',
                 *sorted(final.rglob('*.json'))]
        json_write(marker, {'status':'complete','checked_at':snapshot['checked_at'],'all_four_runs_20':True,
                            'generated_files': {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in paths}})
    else:
        for key,folder in RUNS.items():
            for name in ['summary.json','binding.json','repetitions.json','metric_audit.json']:
                if (DESTINATION/folder/name).read_bytes() != (REPO/'reproduction/results'/key/name).read_bytes():
                    raise ValueError('Completed evaluation evidence changed after preparation: '+key+'/'+name)
    if publish:
        dirty=git('status','--porcelain','--untracked-files=all')
        generated = json.loads(marker.read_text(encoding='utf-8'))['generated_files']
        if any(line[3:] not in generated for line in dirty.splitlines()):
            raise ValueError('Other local changes detected; results prepared but automatic commit stopped')
        git('add','reproduction/results','reproduction/RESULTS_zh.md','reproduction/STATUS_zh.md')
        staged=git('diff','--cached','--name-only')
        if staged.strip():
            git('-c','user.name=luchu','-c','user.email=chu62928@gmail.com','commit','-m',
                'Record verified SALAD benchmark reproduction results')
        head = git('rev-parse','HEAD')
        git('push','origin','main')
        pull_code=f'''import subprocess,json
p={SERVER!r}
dirty=subprocess.check_output(['git','-C',p,'status','--porcelain','--untracked-files=no'],text=True)
if dirty:raise RuntimeError('Server changes detected; results published but pull stopped')
attempts=[]
for proxy in ['http://127.0.0.1:18979','']:
 try:
  r=subprocess.run(['git','-C',p,'-c','http.proxy='+proxy,'-c','http.lowSpeedLimit=100','-c','http.lowSpeedTime=15','pull','--ff-only'],capture_output=True,text=True,timeout=30)
  attempt=dict(exit_code=r.returncode,output=r.stdout,error=r.stderr,transport='loopback' if proxy else 'direct')
 except subprocess.TimeoutExpired:
  attempt=dict(exit_code=124,output='',error='Git pull timed out',transport='loopback' if proxy else 'direct')
 attempts.append(attempt)
 if attempt['exit_code']==0:break
head=subprocess.check_output(['git','-C',p,'rev-parse','HEAD'],text=True).strip()
print(json.dumps(dict(exit_code=attempt['exit_code'],attempts=attempts,head=head)))
'''
        pull=json.loads(remote(pull_code,timeout=85))
        json_write(DESTINATION/'final_server_pull.json',pull)
        if pull['exit_code'] or pull['head'] != head:
            raise ConnectionError('Results pushed but server pull not verified; inspect final_server_pull.json')
        json_write(DESTINATION/'publication.json', {'status':'complete','commit':head,'server_commit':pull['head'],
                   'checked_at':datetime.now(timezone.utc).isoformat()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['status','sync','watch','finalize'])
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.stage=='watch':
        import msvcrt
        DESTINATION.mkdir(parents=True,exist_ok=True)
        lock=(DESTINATION/'watch.lock').open('a+b')
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        try:
            snapshot=collect()
            result=verify(snapshot)
            if args.stage!='status':
                sync_media(snapshot)
                render(snapshot,result)
            print(json.dumps({'checked_at':snapshot['checked_at'],'runs':result['runs'],'complete':result['complete']},ensure_ascii=True),flush=True)
            stalled=[key for key,job in snapshot['jobs'].items() if job['status']!='complete' and not job['alive']]
            if stalled:raise RuntimeError('Evaluation stopped before completion: '+', '.join(stalled))
            if result['complete'] and args.stage in ['watch','finalize']:
                finalize(snapshot,result,args.publish)
                json_write(DESTINATION/'watch_status.json',{'status':'complete','checked_at':snapshot['checked_at'],'published':args.publish})
                (DESTINATION/'watch_error.json').unlink(missing_ok=True)
                return
            if args.stage=='finalize':raise RuntimeError('Evaluations still running')
            json_write(DESTINATION/'watch_status.json',{'status':'watching' if args.stage=='watch' else 'synced',
                       'checked_at':snapshot['checked_at'],'jobs':snapshot['jobs']})
            (DESTINATION/'watch_error.json').unlink(missing_ok=True)
        except Exception as exc:
            json_write(DESTINATION/'watch_error.json',{'status':'failed','error':str(exc),
                                                      'checked_at':datetime.now(timezone.utc).isoformat()})
            if args.stage!='watch':raise
            print(type(exc).__name__+': '+str(exc),flush=True)
            retry = isinstance(exc,(ConnectionError,subprocess.SubprocessError)) or (
                isinstance(exc,RuntimeError) and 'Snapshot changed' in str(exc))
            json_write(DESTINATION/'watch_status.json', {'status':'retrying' if retry else 'failed',
                       'error':str(exc),'checked_at':datetime.now(timezone.utc).isoformat()})
            if not retry:raise
        if args.stage!='watch':return
        time.sleep(60)


if __name__=='__main__':
    main()
