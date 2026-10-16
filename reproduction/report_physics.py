"""Independently summarize transferred physics rows and build a static viewer."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
DEST=REPO/'reproduction/artifacts/physics_v1_20261016'
METRICS=[('near_floor_speed_mps','低位脚尖水平速度','米/秒',1),
         ('skating_frame_ratio','滑步时间比例','%',100),
         ('near_floor_toe_interval_fraction','低位脚尖接触覆盖率','%',100),
         ('skating_given_contact_ratio','有低位接触时的滑步比例','%',100),
         ('foot_penetration_mean_m','脚部平均穿地深度','毫米',1000),
         ('penetration_frame_ratio','穿地超过 2 厘米的帧比例','%',100),
         ('airborne_frame_ratio','双脚离地超过 5 厘米的帧比例','%',100),
         ('bone_length_cv','骨长变异系数','%',100),
         ('relative_acceleration_rms_mps2','根相对关节加速度均方根','米/秒²',1),
         ('relative_jerk_rms_mps3','根相对关节加加速度均方根','米/秒³',1),
         ('root_speed_mps','根节点水平速度','米/秒',1)]


def load(name):return json.loads((DEST/name).read_text(encoding='utf-8'))


def verify():
    summary,protocol=load('summary.json'),load('protocol.json')
    if summary['status']!='complete' or summary['retained_array_metric_replay']!='passed':raise ValueError('Physics run incomplete')
    for name,digest in summary['files_sha256'].items():
        if not name.startswith('arrays/') and hashlib.sha256((DEST/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Evidence digest mismatch: '+name)
    with (DEST/'metrics.csv').open(encoding='utf-8',newline='') as handle:raw=list(csv.DictReader(handle))
    rows=[{k:v if k in ['id','kind'] else (None if v=='' else float(v)) for k,v in row.items()} for row in raw]
    real=[row for row in rows if row['kind']=='real']
    if len(real)!=128 or len(rows)!=512 or len({r['id'] for r in real})!=128:raise ValueError('Reference count mismatch')
    for key,info in summary['metrics'].items():
        paired=[]
        for reference in real:
            generated=[r for r in rows if r['kind']=='generated' and r['id']==reference['id']]
            if sorted(r['seed'] for r in generated)!=protocol['seeds']:raise ValueError('Seed coverage mismatch')
            values=[r[key] for r in generated if r[key] is not None]
            if values and reference[key] is not None:paired.append((reference['id'],reference[key],np.mean(values)))
        if len(paired)!=info['valid_pairs']:raise ValueError('Valid-pair count mismatch')
        if not paired:continue
        target=np.array([r[1] for r in paired])
        prediction=np.array([r[2] for r in paired])
        expected={'real_mean':target.mean(),'generated_mean':prediction.mean(),'difference':(prediction-target).mean()}
        if any(abs(info[k]-v)>1e-10 for k,v in expected.items()):raise ValueError('Independent mean mismatch: '+key)
        grouped={}
        for name,a,b in paired:grouped.setdefault(name.lstrip('M'),[]).append(b-a)
        groups=[np.asarray(v) for v in grouped.values()]
        rng=np.random.RandomState(protocol['bootstrap_seed'])
        boot=[]
        for _ in range(protocol['bootstrap_repetitions']):
            selection=rng.randint(len(groups),size=len(groups))
            selected=np.concatenate([groups[i] for i in selection])
            boot.append(selected.mean())
        if not np.allclose(np.quantile(boot,[.025,.975]),info['paired_ci95'],rtol=0,atol=1e-10):
            raise ValueError('Independent interval mismatch: '+key)
    examples=load('examples.json')
    expected={(r['id'],seed) for r in protocol['examples'] for seed in protocol['seeds']}
    if {(r['id'],r['seed']) for r in examples}!=expected or len(examples)!=18:raise ValueError('Example coverage mismatch')
    videos=load('videos.json')
    if videos.get('status')!='complete' or len(videos['videos'])!=18:raise ValueError('Video rendering incomplete')
    for record in videos['videos']:
        for path,digest in [('path','sha256'),('poster','poster_sha256')]:
            if hashlib.sha256((DEST/record[path]).read_bytes()).hexdigest()!=record[digest]:raise ValueError('Media digest mismatch')
    receipt={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),'reference_clips':128,
             'generated_clips':384,'examples':18,'independent_means_and_cluster_intervals':True,
             'metric_count':len(summary['metrics']),'media_checksums':36}
    (DEST/'verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return summary,protocol,examples,videos


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',action='store_true')
    args=parser.parse_args()
    summary,protocol,examples,videos=verify()
    manifest={(v['id'],v['seed']):v for v in videos['videos']}
    table=[]
    for key,label,unit,factor in METRICS:
        value=summary['metrics'][key]
        a,b,d=[value[k]*factor for k in ['real_mean','generated_mean','difference']]
        low,high=[v*factor for v in value['paired_ci95']]
        table.append([label+'（'+unit+'）',f'{a:.4f}',f'{b:.4f}',f'{d:+.4f}',f'[{low:+.4f}, {high:+.4f}]',str(value['valid_pairs'])])
    notes=[
        '官方 HumanML3D 预训练权重；50 步 DDIM，引导强度 7.5，20 帧/秒。六类示例各生成三个固定种子，每段 6 秒，共 18 段，未挑选样例。',
        '定量对照使用 128 段官方验证集动作及其描述，每个描述生成三次，共 384 段。真实与生成使用相同描述和裁剪长度，但不是逐帧对应的重建。',
        '先将同一描述的三个生成种子取均值，再按参考动作等权汇总。差值为生成减真实；95% 区间按原编号分组进行 2000 次配对重采样，镜像编号与其原编号同组。没有低位接触时速度记为缺失，并列出有效配对数。',
        '脚尖关节为 10、11；相邻两帧都不高于地面 5 厘米时记为低位接触，定义中不使用速度。低位水平速度超过 0.2 米/秒时记为滑步；滑步时间比例的分母是全部有效帧间隔，并另列有低位接触时的比例和接触覆盖率。',
        '脚部穿地检查踝与脚尖共四关节（7、10、8、11），使用原始世界坐标和地面 y=0；平均深度包含未穿地的零值。两脚最低点高于 5 厘米记为离地，包含正常跳跃，不能直接当作悬浮错误。',
        '骨长变异为逐骨骼时间标准差除以均值后取平均；平滑度使用去除根节点平移后的二阶、三阶差分。导数值依赖帧率与测量噪声，不是统一物理合格门槛。',
        '这些是补充运动学诊断，不是 SALAD 原论文指标，也不是受力仿真或网格碰撞检测。未对模型作物理修正。视频仅水平跟随相机，0.5 米地面网格固定于世界坐标；没有贴地、平滑、锁脚或逐帧减去根节点位移。',
    ]
    lines=['# SALAD 动作与物理检查','','状态：18 段视频和物理对照已完成，跨端文件摘要及独立汇总核验通过。','',
           '## 物理指标对照','','| 指标 | 真实参考 | SALAD | 生成减真实 | 差值95%区间 | 有效配对 |',
           '| --- | ---: | ---: | ---: | --- | ---: |']
    lines+=['| '+' | '.join(row)+' |' for row in table]
    lines+=['','## 协议与边界','']+[paragraph+'\n' for paragraph in notes]
    sensitivity=[]
    lines+=['## 滑步阈值敏感性','','| 高度阈值 | 速度阈值 | 真实滑步比例 | 生成滑步比例 |','| --- | --- | --- | --- |']
    for height in protocol['sensitivity_heights_m']:
        for speed in protocol['sensitivity_speeds_mps']:
            key='skating_h%02d_v%03d'%(round(height*100),round(speed*100))
            info=summary['metrics'][key]
            row=[f'{height*100:.0f} 厘米',f'{speed:.1f} 米/秒',f"{info['real_mean']*100:.3f}%",f"{info['generated_mean']*100:.3f}%"]
            sensitivity.append(row)
            lines.append('| '+' | '.join(row)+' |')
    lines+=['','## 动作示例','']
    cards=[]
    for item in protocol['examples']:
        matches=[r for r in examples if r['id']==item['id']]
        options=''.join(f'<option value="{r["seed"]}">{r["seed"]}</option>' for r in matches)
        first=matches[0]
        media=manifest[(first['id'],first['seed'])]
        cards.append('<article data-motion="'+item['id']+'"><div class="heading"><h2>'+html.escape(item['label'])+
                     '</h2><label>随机种子 <select>'+options+'</select></label></div><p class="caption">'+html.escape(item['text'])+
                     '</p><video controls muted playsinline preload="none" poster="'+media['poster']+'" src="'+media['path']+
                     '"></video><dl></dl></article>')
        lines.append('- '+item['label']+'：'+ '；'.join('[种子 %d](%s)'%(r['seed'],manifest[(r['id'],r['seed'])]['path']) for r in matches))
    lines+=['','[逐动作指标](metrics.csv) · [示例指标](example_metrics.csv) · [独立核验](verification.json) · [运行绑定](binding.json) · [原始协议](protocol.json)','']
    (DEST/'REPORT_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    def html_table(headers,rows):
        return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SALAD 动作与物理检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#202725;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;letter-spacing:0}main{max-width:1250px;padding:28px 24px;margin:auto}header{border-bottom:2px solid #237968;padding-bottom:18px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0}p{margin:8px 0}section{padding:26px 0;border-bottom:1px solid #d6deda}.samples{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.samples article{border:1px solid #d6deda;border-radius:6px;min-width:0;overflow:hidden}.heading{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 14px 0;flex-wrap:wrap}.heading h2{font-size:18px}label{font-size:13px;color:#4f5b56}select{font:inherit;margin-left:4px;border:1px solid #b9c6be;background:#fff;padding:4px;border-radius:3px}.caption{padding:0 14px;color:#57635e;font-size:13px;min-height:44px}video{display:block;width:100%;aspect-ratio:11/7;background:#f6f8f7}dl{margin:0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 14px;padding:12px 14px;font-size:13px}dt{color:#68746e}dd{margin:0;font-variant-numeric:tabular-nums}.table{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #e0e5e2;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3ef}a{color:#16709a}.notes{max-width:1000px}.notes p{margin:12px 0}@media(max-width:700px){main{padding:20px 14px}.samples{grid-template-columns:minmax(0,1fr);gap:18px}h1{font-size:24px}.caption{min-height:0}}
</style><main><header><h1>SALAD 动作与物理检查</h1><p>HumanML3D · 18 段六秒示例 · 128 段真实参考与 384 段生成动作</p><a href="REPORT_zh.md">完整报告</a> · <a href="metrics.csv">逐动作指标</a> · <a href="verification.json">独立核验</a></header><section><div class="samples">CARDS</div></section><section><h2>验证集物理指标</h2><p>按描述配对，生成先对三个种子取均值；差值为生成减真实。</p>TABLE</section><section><h2>滑步阈值敏感性</h2>SENSITIVITY</section><section class="notes"><h2>协议与解释边界</h2>NOTES</section></main><script>
const records=DATA;
for(const article of document.querySelectorAll('article[data-motion]')){
 const select=article.querySelector('select'),video=article.querySelector('video');
 function update(change){
  const record=records.find(r=>r.id===article.dataset.motion&&r.seed===Number(select.value));
  if(change){video.pause();video.src=record.path;video.poster=record.poster;video.load();}
  const m=record.metrics;
  const cells=[['低位脚尖速度',m.near_floor_speed_mps===null?'无低位接触':m.near_floor_speed_mps.toFixed(3)+' 米/秒'],['滑步时间比例',(m.skating_frame_ratio*100).toFixed(1)+'%'],['平均穿地深度',(m.foot_penetration_mean_m*1000).toFixed(2)+' 毫米'],['双脚离地比例',(m.airborne_frame_ratio*100).toFixed(1)+'%']];
  article.querySelector('dl').innerHTML=cells.map(([a,b])=>'<div><dt>'+a+'</dt><dd>'+b+'</dd></div>').join('');
 }
 select.addEventListener('change',()=>update(true));update(false);
}
</script></html>'''
    data=[dict(r,path=manifest[(r['id'],r['seed'])]['path'],poster=manifest[(r['id'],r['seed'])]['poster']) for r in examples]
    document=document.replace('CARDS',''.join(cards)).replace('TABLE',html_table(['指标','真实参考','SALAD','生成减真实','差值95%区间','有效配对'],table))
    document=document.replace('SENSITIVITY',html_table(['高度阈值','速度阈值','真实滑步比例','生成滑步比例'],sensitivity))
    document=document.replace('NOTES',''.join('<p>'+html.escape(p)+'</p>' for p in notes)).replace('DATA',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    if (DEST/'mesh_v1/index.html').exists():
        document=document.replace('<a href="REPORT_zh.md">', '<a href="mesh_v1/index.html">人体网格动作</a> · <a href="REPORT_zh.md">', 1)
    (DEST/'index.html').write_text(document,encoding='utf-8')
    if args.record:
        if load('qa/browser.json')['status']!='passed':raise ValueError('Browser acceptance required before recording')
        repo=REPO
        target=repo/'reproduction/results/physics_v1'
        target.mkdir(parents=True,exist_ok=True)
        for name in ['summary.json','binding.json','protocol.json','examples.json','reference.json','metrics.csv',
                     'example_metrics.csv','verification.json','unit_tests.json','videos.json','transfer.json','qa/browser.json']:
            destination=target/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            content=(DEST/name).read_bytes()
            if destination.exists() and destination.read_bytes()!=content:
                raise ValueError('Existing recorded evidence differs; inspect before replacing: '+name)
            destination.write_bytes(content)
        report='\n'.join(lines)
        for prefix in ['metrics.csv','example_metrics.csv','verification.json','binding.json','protocol.json']:
            report=report.replace(']('+prefix+')','](results/physics_v1/'+prefix+')')
        report=report.replace('](videos/','](artifacts/physics_v1_20261016/videos/')
        destination=repo/'reproduction/PHYSICS_zh.md'
        if destination.exists() and destination.read_text(encoding='utf-8')!=report:
            raise ValueError('Existing recorded report differs; inspect before replacing')
        destination.write_text(report,encoding='utf-8')
    print(json.dumps({'status':'passed','metrics':len(table),'videos':len(examples)}))


if __name__=='__main__':main()
