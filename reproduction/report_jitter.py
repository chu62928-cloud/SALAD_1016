"""Build and verify the isolated jitter diagnosis, preserving original evaluations."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO/'reproduction/artifacts/physics_v1_20261016/jitter_v1'
STAGES = ['raw', 'fixed_bones', 'author_foot_ik']
LABELS = ['原始关节直接蒙皮', '固定骨架拟合', '固定骨架 + 足部拟合']
CASES = ['stand_20261016', 'walk_20261016', 'run_20261016',
         'jump_20261016', 'jump_20261017', 'real_walk_control']
EVIDENCE = ['generation_audit.json', 'conversion_audit.json', 'videos.json',
            'replay_launch.json', 'convert_launch.json', 'render_launch.json', 'transfer.json',
            'bone_jitter_regression.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def verify_evidence():
    generation = read(LOCAL/'generation_audit.json')
    conversion = read(LOCAL/'conversion_audit.json')
    videos = read(LOCAL/'videos.json')
    transfer = read(LOCAL/'transfer.json')
    examples = read(LOCAL.parent/'examples.json')
    mesh_binding = read(LOCAL.parent/'mesh_v1/binding.json')
    regression = read(LOCAL/'bone_jitter_regression.json')
    if regression['status'] != 'passed' or regression['original_returncode'] != 1 or len(regression['fitted_bone_length_checks']) != 12:
        raise ValueError('Original symptom and fixed-bone checks required')
    if not regression['original_arrays_and_assets_unchanged'] or regression['naturalness_fixed']:
        raise ValueError('Regression scope or immutable assets differ')
    if mesh_binding['examples_sha256'] != sha(LOCAL.parent/'examples.json'):
        raise ValueError('Original example metadata changed')
    if any(v['status'] != 'complete' for v in [generation, conversion, videos]):
        raise ValueError('All diagnosis stages must finish')
    if not generation['assets_unchanged'] or not conversion['source_files_unchanged']:
        raise ValueError('Original generation assets or arrays changed')
    if generation['batches'] != [dict(seed=s, clips=6, normalized_max_error=0.)
                                 for s in [20261016, 20261017, 20261018]]:
        raise ValueError('Exact replay of all 18 examples required')
    if len(generation['notebook_comparison']) != 2 or any(
            row['joint_max_error_m'] > 1e-5 for row in generation['notebook_comparison']):
        raise ValueError('Author generation entry point differs')
    if [r['name'] for r in conversion['cases']] != CASES or [r['name'] for r in videos['videos']] != CASES:
        raise ValueError('Unexpected comparison cases')
    if videos['stages'] != STAGES or videos['conversion_audit_sha256'] != sha(LOCAL/'conversion_audit.json'):
        raise ValueError('Rendered diagnosis is bound to another conversion')
    upstream = (REPO/'visualization/joints2bvh.py').read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(upstream).hexdigest() != conversion['conversion_source_sha256']:
        raise ValueError('Author conversion code changed')
    if videos['renderer_sha256'] != sha(REPO/'reproduction/render_jitter_comparison.py'):
        raise ValueError('Executed renderer differs from local source')
    for stage in ['replay', 'convert', 'render']:
        for name, digest in read(LOCAL/(stage+'_launch.json'))['code_sha256'].items():
            if sha(REPO/'reproduction'/name) != digest:
                raise ValueError('Executed diagnosis source changed: '+name)
    if transfer['status'] != 'passed':
        raise ValueError('Transfer checks required')
    for name, digest in transfer['files_sha256'].items():
        if sha(LOCAL/name) != digest:
            raise ValueError('Transferred evidence changed: '+name)
    original = {'%s_%d' % (r['id'], r['seed']): r for r in examples}
    rows = []
    for case, video in zip(conversion['cases'], videos['videos']):
        name = case['name']
        if name != 'real_walk_control' and case['source_sha256'] != original[name]['sha256']:
            raise ValueError('Comparison uses another generated motion')
        if video['frames'] != 120 or video['fps'] != 20 or min(video['min_body_pixels']) < 1000:
            raise ValueError('Incomplete or blank rendered motion')
        for field, digest in [('path', 'sha256'), ('poster', 'poster_sha256')]:
            if sha(LOCAL/video[field]) != video[digest] or transfer['media_sha256'][video[field]] != video[digest]:
                raise ValueError('Delivered media checksum differs')
        for stage, label in zip(STAGES, LABELS):
            converted = case['stages'][stage]
            stats = video['stats'][stage]
            if stage != 'raw' and converted['joints']['bone_above3hz_cv_mean'] > 1e-8:
                raise ValueError('Fitted bone-length invariant failed')
            if stats['root_height'] != video['stats']['raw']['root_height']:
                raise ValueError('This comparison expected unchanged root height')
            rows.append(dict(name=name, label=case['label'], stage=stage, stage_label=label,
                mesh_root_relative_hf_mm=converted['mesh']['above3hz_rms_m']*1000,
                mesh_horizontal_follow_hf_mm=stats['camera_follow_surface']['above3hz_rms_m']*1000,
                root_height_hf_mm=stats['root_height']['above3hz_rms_m']*1000,
                bone_cv_pct=converted['joints']['bone_cv_mean']*100,
                bone_hf_cv_pct=converted['joints']['bone_above3hz_cv_mean']*100,
                joint_change_cm=converted['source_joint_error_mean_m']*100,
                skating_pct=stats['physics']['skating_frame_ratio']*100,
                airborne_pct=stats['physics']['airborne_frame_ratio']*100,
                penetration_frame_pct=stats['physics']['penetration_frame_ratio']*100,
                penetration_max_cm=stats['physics']['foot_penetration_max_m']*100))
    if len(generation['rotation_branch']) != 18:
        raise ValueError('All original examples required for case selection audit')
    worst = max(generation['rotation_branch'], key=lambda r: r['position_branch']['bone_above3hz_cv_mean'])
    if '%s_%d' % (worst['id'], worst['seed']) != 'jump_20261017':
        raise ValueError('The documented stress case is not the maximum bone jitter case')
    return generation, conversion, videos, rows


def make_report(rows, generation, conversion):
    lookup = {(r['name'], r['stage']): r for r in rows}
    table = ['| 片段 | 原始高频位移 | 固定骨架 | 加足部拟合 | 根高度高频位移 | 拟合关节改动 |',
             '|---|---:|---:|---:|---:|---:|']
    physics = ['| 片段 | 滑步比例：原始 → 拟合 | 双脚离地：原始 → 拟合 | 最大脚部穿地：原始 → 拟合 |',
               '|---|---:|---:|---:|']
    for name in CASES:
        a, b, c = [lookup[name, s] for s in STAGES]
        label = a['label'] + ('（第二种子，压力病例）' if name == 'jump_20261017' else '')
        table.append('| %s | %.3f 毫米 | %.3f 毫米 | %.3f 毫米 | %.3f 毫米 | %.2f 厘米 |' % (
            label, a['mesh_horizontal_follow_hf_mm'], b['mesh_horizontal_follow_hf_mm'],
            c['mesh_horizontal_follow_hf_mm'], a['root_height_hf_mm'], c['joint_change_cm']))
        physics.append('| %s | %.1f%% → %.1f%% | %.1f%% → %.1f%% | %.1f → %.1f 厘米 |' % (
            label, a['skating_pct'], c['skating_pct'], a['airborne_pct'], c['airborne_pct'],
            a['penetration_max_cm'], c['penetration_max_cm']))
    worst = max(r['position_branch']['bone_above3hz_cv_mean'] for r in generation['rotation_branch'])*100
    report = '''# SALAD 抖动诊断与网格对照

原始生成动作确有骨长快速波动；此前直接蒙皮把这些变形保留到了人体表面。补上固定骨架与足部拟合能改善部分片段，但全身上下运动仍在，且部分接触指标恶化。当前不能称为动作自然度已修复。

论文检索、分布质量指标接近原文，与单段动作自然、物理合理是不同结论。此前视频能够播放、人体未出画及几何附着正确，只证明渲染流程通过，不能证明动作质量合格。

## 已核实的来源

- 同样三个固定种子、每批六段，18 段归一化输出逐值重放最大误差均为 0；原数据、权重和动作文件摘要不变。
- 同一初始噪声走作者 `Text2Motion.generate`，行走关节最大差约 0.00000149 米、跳跃约 0.00000769 米，未发现足以解释可见抖动的入口差异。
- 原始跳跃首种子的骨长高频相对波动约 1.217%%，18 段最大 %.3f%%；0.5%% 是本次诊断容差，不是论文指标，也不是通用动作质量门槛。行走首种子约 0.404%%，没有超过该容差。
- 作者 `models/denoiser/trainer.py` 导出 BVH 时调用 `Joint2BVHConvertor.convert(..., foot_ik=True)`；此前直接蒙皮没有走该转换。此次使用未修改的转换源码，分别关闭和开启足部拟合，均迭代 10 次。
- 转换器需要的模板在当前仓库资产中缺失；此次补入公开的同拓扑模板，来源与摘要见下文。因此这是作者转换代码的诊断对照，不等同于已验证论文演示视频的完整制作流程。
- 固定骨架消除了骨长随时间伸缩，但足部拟合后的关节平均已改动约 4.4–7.5 厘米。不能将处理后的结果称为模型原始输出。
- 另外检查了模型旋转分支，18 段的加加速度均未改善；真实动作两种恢复方式本身平均有约 1.17 厘米差异，因此未用该分支替换原始评估。

## 保留全身上下运动的比较

下表使用视频相机跟随坐标：只去掉根节点水平平移，保留根节点高度；20 帧/秒，对每个坐标分量取 3 Hz 及以上离散余弦分量，再计算均方根。数值单位为毫米，不是逐帧欧氏位移，也不是动力学或感知质量评分。根高度高频位移在三个阶段完全相同，说明这条拟合流程未修改全身上下轨迹。

TABLE

若完全去掉根平移，跳跃首种子的高频表面位移由 3.344 降到 2.023 毫米，约下降 40%%；但保留上下运动时仅从 3.422 降到 2.777 毫米，约下降 19%%。跑步对应下降仅约 2%%。这解释了为什么局部指标变好之后仍可能看到全身抖动。

高频成分也包含正常步态、落地冲击及端点影响，不能全部解释为错误。真实行走对照也会被拟合改变，且与生成跳跃并非同动作配对；它用于排查显示流程，不能证明生成动作优于真实动作。

## 接触与副作用

使用原物理诊断协议：相邻两帧脚尖均不高于 5 厘米且水平速度大于 0.2 米/秒计滑步；双脚离地包含正常跳跃；穿地统计使用关节而非估计网格脚底。各阶段共用固定世界地面，未单独平移到地面。

PHYSICS

行走和真实行走对照拟合后的悬空比例明显升高，滑步下降不能单独作为改善证据。跑步滑步比例从 23.5%% 上升到 26.1%%；跳跃第二种子的最大脚部穿地从 0 增至约 24.3 厘米。因此不能将作者转换代码直接当成通用物理修复器。

## 对照选择和渲染

五段生成病例来自原 18 段，没有重新采样：站立、行走、跑步、跳跃均取首种子 20261016；另取骨长高频波动最大的跳跃 20261017 作为压力病例，明确按问题严重程度选择。真实行走对照固定取原 128 段参考中首个描述含 walk 且不少于 120 帧的动作的前 120 帧，不是这条生成行走的逐帧目标。

每段视频 120 帧、6 秒、1920×900，三列依次为原始直接蒙皮、固定骨架、固定骨架加足部拟合。使用同一个中性 SMPL-H 模板、同一斜视角和公共固定尺度，地面 y=0；相机仅水平跟随，不跟随上下运动。没有添加独立时间滤波、帧插值或额外贴地。手指及绕骨轴旋转仍为关节驱动的估计，表面不是模型直接预测的精确人体重建。

模板来源：%s

模板 SHA-256：`%s`。原始正式基准及 384 段生成物理诊断均保持原口径，本对照不回填或替换其结果。

[播放六段对照](index.html) · [汇总 CSV](summary.csv) · [生成入口核对](generation_audit.json) · [转换核对](conversion_audit.json) · [视频和根高度指标](videos.json) · [原始网格](../mesh_v1/index.html)
''' % (worst, conversion['template_source'], conversion['template_sha256'])
    return report.replace('TABLE', '\n'.join(table)).replace('PHYSICS', '\n'.join(physics))


def build():
    generation, conversion, videos, rows = verify_evidence()
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (LOCAL/'summary.csv').write_text(output.getvalue(), encoding='utf-8-sig', newline='')
    report = make_report(rows, generation, conversion)
    (LOCAL/'REPORT_zh.md').write_text(report, encoding='utf-8')
    lookup = {(r['name'], r['stage']): r for r in rows}
    articles = []
    order = ['jump_20261016', 'walk_20261016', 'run_20261016', 'stand_20261016', 'jump_20261017', 'real_walk_control']
    for name in order:
        video = next(v for v in videos['videos'] if v['name'] == name)
        a, b, c = [lookup[name, stage] for stage in STAGES]
        title = video['label'] + (' · 第二种子，压力病例' if name == 'jump_20261017' else '')
        cells = ''.join('<tr><th scope="row">%s</th>%s</tr>' % (label, ''.join('<td>%s</td>' % text for text in values))
            for label, values in [
                ('高频表面位移（保留上下运动）', ['%.3f 毫米' % r['mesh_horizontal_follow_hf_mm'] for r in [a,b,c]]),
                ('距原关节平均', ['%.2f 厘米' % r['joint_change_cm'] for r in [a,b,c]]),
                ('骨长变异', ['%.2f%%' % r['bone_cv_pct'] for r in [a,b,c]]),
                ('滑步比例', ['%.1f%%' % r['skating_pct'] for r in [a,b,c]]),
                ('双脚离地', ['%.1f%%' % r['airborne_pct'] for r in [a,b,c]]),
                ('最大脚部穿地', ['%.1f 厘米' % r['penetration_max_cm'] for r in [a,b,c]])])
        articles.append('''<article data-motion="%s"><h2>%s</h2><p class="caption">%s</p>
<video controls playsinline preload="metadata" src="%s" poster="%s"></video>
<div class="table-scroll" tabindex="0"><table><thead><tr><th scope="col">诊断指标</th>%s</tr></thead><tbody>%s</tbody></table></div>
<p class="root-stat">根高度高频位移 %.3f 毫米，三种处理相同。病例：%s</p></article>''' % (
            name, html.escape(title), html.escape(video['text']), video['path'], video['poster'],
            ''.join('<th scope="col">%s</th>' % html.escape(s) for s in LABELS), cells, a['root_height_hf_mm'], name))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SALAD 抖动诊断对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#263b40;font-family:"Microsoft YaHei",sans-serif;letter-spacing:0}
main{max-width:1480px;margin:auto;padding:24px 28px}h1{font-size:27px;line-height:1.4;margin:0 0 12px}h2{font-size:21px;margin:0 0 6px}
p{line-height:1.75;margin:8px 0}.finding{border-left:4px solid #ac5151;padding-left:14px;max-width:1100px}.finding strong{color:#903d3d}
a{color:#176e79;text-underline-offset:3px}.links{display:flex;gap:14px 24px;flex-wrap:wrap;margin:16px 0 8px}
article{padding:24px 0;border-top:1px solid #ced8da;margin-top:22px}.caption,.root-stat{color:#52656a;font-size:13px;overflow-wrap:anywhere}
video{display:block;width:100%;aspect-ratio:32/15;background:#edf1f2;margin:12px 0}
.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;min-width:650px}
th,td{border-bottom:1px solid #dce3e4;padding:10px 12px;text-align:right;line-height:1.5}th:first-child{text-align:left;width:31%}thead{background:#f0f5f5}
th{font-weight:500}details{margin:20px 0;color:#485f64}summary{cursor:pointer;color:#176e79;padding:8px 0;font-size:15px}details p{max-width:1100px;font-size:14px}
@media(max-width:720px){main{padding:20px 12px}h1{font-size:23px}h2{font-size:19px}article{padding-top:20px}.finding{padding-left:10px}th,td{padding:9px}video{margin:10px 0}}
</style><main><header><h1>SALAD 抖动诊断对照</h1>
<p class="finding"><strong>抖动已确认，尚未完全修复。</strong> 原始关节存在骨长快速波动。固定骨架与足部拟合只能改善部分表面变形，全身上下轨迹没有改变，部分片段的悬空或穿地还会加重。</p>
<p>同一批固定动作 · 原始 / 固定骨架 / 加足部拟合 · 6 秒 · 20 帧/秒</p>
<nav class="links"><a href="REPORT_zh.md">完整诊断记录</a><a href="summary.csv">指标表</a><a href="../mesh_v1/index.html">原始 18 段网格</a><a href="../index.html">原始物理诊断</a><a href="generation_audit.json">生成入口核对</a></nav>
</header><details><summary>测量口径与处理边界</summary>
<p>18 段生成输出逐值重放一致；同噪声与作者生成入口的关节差小于 0.01 毫米。对照使用作者未修改的 BVH 转换代码及补入的公开同拓扑模板，固定骨架后关节平均改动约 4.4–7.5 厘米，并不代表论文视频完整制作流程已被验证。</p>
<p>表中高频位移仅去掉水平根平移、保留上下运动，统计 3 Hz 及以上的坐标分量均方根。它也包含正常运动；不是论文指标或自然度评分。视频底部原有标注采用去除完整根平移的口径，完整两套数值均保留在汇总表中。</p>
<p>滑步阈值：脚尖不高于 5 厘米且水平速度超过 0.2 米/秒。悬空会减少可测滑步的机会；跳跃中的离地可以合理。世界地面固定为 0，没有为各阶段单独贴地或追加时间滤波。关节指标与估计网格脚底有差异。</p>
<p>站立、行走、跑步、跳跃取原第一种子；第二段跳跃是原 18 段中骨长高频波动最大的压力病例。真实行走取原参考列表中首个描述含 walk 且足够长的动作，用于检查显示流程；不是生成动作的逐帧目标。</p></details>
ARTICLES</main></html>'''.replace('ARTICLES', '\n'.join(articles))
    (LOCAL/'index.html').write_text(page, encoding='utf-8')
    tracked = report.replace('](index.html)', '](artifacts/physics_v1_20261016/jitter_v1/index.html)')
    tracked = tracked.replace('](../mesh_v1/index.html)', '](artifacts/physics_v1_20261016/mesh_v1/index.html)')
    for name in ['summary.csv', 'generation_audit.json', 'conversion_audit.json', 'videos.json']:
        tracked = tracked.replace(']('+name+')', '](results/jitter_v1/'+name+')')
    (REPO/'reproduction/JITTER_zh.md').write_text(tracked, encoding='utf-8')
    print(json.dumps(dict(status='built', cases=len(CASES), rows=len(rows))))


def record():
    verify_evidence()
    browser = read(LOCAL/'qa/browser.json')
    expected = {(w, n) for w in [1440, 390] for n in CASES}
    if browser['status'] != 'passed' or {(r['width'], r['name']) for r in browser['video_checks']} != expected:
        raise ValueError('All six videos require desktop and mobile playback verification')
    files = EVIDENCE + ['summary.csv', 'qa/browser.json']
    verification = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        original_replay_clips=18, comparison_cases=6, media_files=12, browser_checks=12,
        fixed_bone_invariant=True, root_height_unchanged=True, naturalness_fixed=False,
        report_sha256=sha(LOCAL/'REPORT_zh.md'), page_sha256=sha(LOCAL/'index.html'),
        files_sha256={n:sha(LOCAL/n) for n in files},
        local_source_sha256={n:sha(REPO/'reproduction'/n) for n in [
            'diagnose_jitter.py', 'audit_jitter_pipeline.py', 'render_jitter_comparison.py',
            'remote_jitter.py', 'report_jitter.py', 'mesh_surface.py', 'render_mesh.py', 'physics_metrics.py']})
    path = LOCAL/'verification.json'
    if path.exists():
        old = read(path)
        if {k:v for k,v in old.items() if k != 'checked_at'} != {k:v for k,v in verification.items() if k != 'checked_at'}:
            raise ValueError('Existing diagnosis verification differs')
        verification = old
    else:
        save(path, verification)
    destination = REPO/'reproduction/results/jitter_v1'
    for name in files + ['verification.json']:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (LOCAL/name).read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Archived diagnosis differs: '+name)
        target.write_bytes(content)
    print(json.dumps(dict(status='recorded', naturalness_fixed=False, browser_checks=12)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['report', 'record'])
    stage = parser.parse_args().stage
    {'report':build, 'record':record}[stage]()
