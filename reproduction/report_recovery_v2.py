"""Report the separate recovery/skinning checks without replacing earlier results."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'reproduction/artifacts/physics_v1_20261016/recovery_v2'
TRACKED=REPO/'reproduction/results/recovery_v2'
NAMES={'walk_20261016':'生成行走','jump_20261016':'生成跳跃',
       'real_walk_control':'真实行走','real_jump_control':'真实跳跃'}


def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    audit=load('audit.json');videos=load('videos.json');official=load('official_videos.json');transfer=load('transfer.json')
    verification=load('verification.json')
    if any(x['status']!='complete' for x in [audit,videos,official]) or transfer['status']!='passed':
        raise ValueError('Complete transferred evidence required')
    if verification['status']!='passed':raise ValueError('Independent numerical verification required')
    for name,digest in transfer['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Transferred evidence changed')
    lines=['# SALAD 位置恢复、旋转恢复与蒙皮专项检查','',
        '此前使用的关节驱动人体表面与作者显示流程不完全一致。原始位置骨架已经存在骨长快速波动；显示转换还会改变抖动幅度，必须分别检查。此报告不把固定骨长或高频下降当作动作自然度已经修复。','',
        '## 作者代码与此前流程','',
        '| 入口 | 恢复与显示步骤 | 是否固定骨长或滤波 |','|---|---|---|',
        '| 作者普通示例 playground.ipynb | 263 维反归一化 → recover_from_ric → 骨架动画 | 无时间滤波，无骨长修正 |',
        '| 作者训练器 test(save_motion=True) | 原始位置骨架视频；另导出 Joint2BVHConvertor 的 BVH | BVH 固定骨架并启用 foot_ik，原骨架视频仍用原坐标 |',
        '| 作者 visualize_attn.ipynb 的普通动作视频函数 | 位置恢复后 gaussian_filter1d(joint, 1, axis=0) | 标准差 1 帧的高斯滤波；注意力专用绘图函数另行使用原坐标 |',
        '| 此前人体网格 | 原位置坐标 → 几何估计旋转 → 关节驱动表面 | 保留变长骨骼，非标准固定静止骨架的 SMPL 前向输出 |',
        '| 本次标准蒙皮对照 | 位置或 6D 旋转分支各自恢复骨架 → 相同几何旋转估计 → 标准 SMPL-H 线性混合蒙皮 | 固定中性骨长；无时间滤波、足部锁定或逐帧贴地 |','',
        '恢复、骨架、四元数、BVH 转换和绘图源码均与导入提交 1166b90 一致。训练器文件有先前登记的无引导文本处理和优化器恢复修正，但实际显示导出方法 test 的语法树与原版一致。本次使用已封存输出，没有重新采样或训练。','',
        '官方绘图函数会减去整段动作的最低高度，并跟随根节点水平位置；本次六格对照保留原世界地面和高度。两者均没有逐帧高度矫正，骨长指标不受该固定平移影响。','',
        '## 位置与旋转恢复','',
        '| 范围 | 数量 | 位置骨长变异 % | 旋转恢复骨长变异 % | 原关节差 cm | 整体朝向对齐后差 cm | 位置加加速度 | 旋转加加速度 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,label in [('generated','原 18 段生成动作'),('real','原 128 段真实动作')]:
        r=audit['summary'][key]
        lines.append('| %s | %d | %.4f | %.7f | %.3f | %.6f | %.3f | %.3f |'%(label,r['count'],r['position_bone_cv']*100,
            r['rotation_bone_cv']*100,r['position_rotation_mean_error_m']*100,
            verification['summary'][key]['aligned_mean_error_m']*100,r['position_jerk'],r['rotation_jerk']))
    lines+=['','加加速度单位为米/秒³，计算前去除整条根平移，对每个坐标分量取均方根；不同动作之间不能直接当作感知质量排序。骨长变异是逐骨段时间标准差除以均值后取平均，未做平滑。','',
        '无蒙皮的原始跳跃 jump_20261016，其 3 Hz 及以上骨长相对波动为 1.217%，重现了先前超过 0.5% 诊断容差的现象。0.5% 是定位骨长快速伸缩的诊断阈值，不是论文指标或自然度合格标准。','',
        '真实动作在裁剪后可能留下两种恢复约定间的恒定朝向差。121/128 段在整段只对齐一次朝向后，最大关节差不超过 0.00001 米；其余 7 段仍有随时间变化的整体朝向差，最坏关节差为 %.6f 米。逐帧只对齐绕竖直轴朝向后，全部真实动作的最坏关节差为 %.9f 米，说明这些额外差异主要是整体朝向，而非骨长或身体相对姿势。逐帧对齐仅用于定位原因，没有用于视频或上述原始指标。'%(verification['summary']['real']['worst_aligned_max_error_m'],verification['summary']['real']['framewise_aligned_max_error_m']),'',
        '七段恒定朝向检查未通过：'+', '.join(verification['summary']['real']['constant_yaw_mismatches'])+'。六格视频保留两条原始恢复路径的朝向，因此真实行走两栏可能面向不同方向；该病例的固定朝向差本身不是抖动。','',
        '263 维的第 4:67 列是位置，第 67:193 列是 21 个非根关节的 6D 旋转。HumanML 的正向运动学在旋转后放置当前骨段，并在多条运动链上重新从根旋转开始；标准 SMPL 是由父关节旋转放置子关节偏移。因此这些 6D 矩阵不能直接当作同编号 SMPL 局部旋转。','',
        '本次 6D 对照先调用作者 recover_from_rot，再与位置分支使用相同的几何旋转估计和标准蒙皮。位置本身不能唯一确定绕骨轴扭转，末端关节方向也不完全可辨识；本次没有声称恢复出真实 SMPL 旋转。','',
        '另用合成的 178°→182° 朝向跨界验证原处理代码中的四元数符号风险：直接对相对四元数的 y 分量取 arcsin 可能丢失标量部分的符号，连续化四元数后可避免该合成误差。这证明存在这种编码风险，尚不能据此认定它就是七段真实数据差异的完整来源；本次未重编码原数据。','',
        '## 同动作与真实动作蒙皮对照','',
        '| 片段 | 位置骨长变异 % | 6D 骨长变异 % | 原直接蒙皮高频 mm | 位置→标准蒙皮高频 mm | 6D→标准蒙皮高频 mm | 标准蒙皮距位置关节 cm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in audit['cases']:
        s=r['stages']
        lines.append('| %s | %.3f | %.7f | %.3f | %.3f | %.3f | %.3f |'%(NAMES[r['name']],
            r['position']['bone_cv_mean']*100,r['rotation']['bone_cv_mean']*100,
            s['original']['surface_horizontal_follow']['above3hz_rms_m']*1000,
            s['position_smpl']['surface_horizontal_follow']['above3hz_rms_m']*1000,
            s['rotation_smpl']['surface_horizontal_follow']['above3hz_rms_m']*1000,
            s['position_smpl']['input_joint_change_mean_m']*100))
    lines+=['','表面高频指 20 帧/秒时 3 Hz 及以上离散余弦分量的坐标均方根。只去掉根水平平移，保留上下运动。正常步态、跳跃冲击和端点也可能含有高频，因此高频不等同于错误。','',
        '标准蒙皮将动作重定向到中性人体的固定骨长，关节会移动；表中同时列出改动量。作者 BVH 栏使用原转换代码与先前已登记的公开同拓扑模板，再使用同一个估计表面绘制，不能称为作者论文视频的完整 SMPL 制作流程。','',
        '真实行走和跳跃固定取原参考表中首个描述含 walk/jump 且至少 120 帧的动作，使用原裁剪的前 120 帧；它们是显示管线对照，并非生成动作的逐帧目标。','',
        '## 可以确认与不能确认','',
        '1. 不经过蒙皮的位置骨架已经存在骨长变化，问题不能全部归于 SMPL。','2. 位置与旋转两条路径并不等价；旋转恢复固定了骨长，但没有自动保证时间平滑。真实数据的七段朝向不一致也必须保留，不能把旋转分支直接宣布为全部正确。',
        '3. 此前关节驱动表面会保留骨长伸缩，并可能放大或改变高频外观。两个固定真实病例转入标准蒙皮后，位置与旋转分支的高频幅度几乎相同，没有出现仅在某一恢复分支新增的大幅高频异常；真实跳跃本身的高频较大，不能因此判为抖动。',
        '4. 通过独立正向运动学、独立蒙皮与标准 smplx 库的数值比较，可以检查本次转换的实现一致性；不能证明估计的轴向扭转是真实值，也不能据此排除所有视觉伪影。',
        '5. 本次属于诊断，不涉及训练或模型修复；原论文指标、物理评价、数据划分及微调结果保持原样。','',
        '## 交付与复核','',
        '- 六格同步视频：左上原位置骨架，中上此前直接蒙皮，右上位置转标准蒙皮；左下旋转恢复骨架，中下旋转转标准蒙皮，右下作者 BVH 转换后的估计表面。',
        '- 另附四段直接调用未修改官方绘图函数的骨架视频。',
        '- audit.json 保存 146 条逐动作指标、独立恢复误差、标准蒙皮检查、源文件摘要及固定病例。',
        '- metrics.csv 保存 146 条便于复算的指标；verification.json 与 qa/browser.json 保存数值和播放核验。','']
    text='\n'.join(lines)
    (ROOT/'REPORT_zh.md').write_text(text,encoding='utf-8')
    TRACKED.mkdir(parents=True,exist_ok=True)
    for name in ['audit.json','metrics.csv','official_videos.json','videos.json','verification.json']:
        (TRACKED/name).write_bytes((ROOT/name).read_bytes())
    (REPO/'reproduction/RECOVERY_SKINNING_zh.md').write_text(text+'\n[打开同步对照](artifacts/physics_v1_20261016/recovery_v2/index.html)\n',encoding='utf-8')
    articles=[]
    for r in videos['videos']:
        n=r['name'];official_row=next(v for v in official['videos'] if v['name']==n)
        articles.append('<article data-motion="'+n+'"><h2>'+NAMES[n]+'</h2><video controls playsinline preload="metadata" width="1920" height="1460" poster="'+r['poster']+'" src="'+r['path']+'"></video><p><a href="'+r['path']+'">打开完整对照视频</a> · <a href="'+official_row['path']+'">官方原函数骨架视频</a></p></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SALAD 恢复与蒙皮检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f8f7;color:#223d39;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px;line-height:1.35;margin:0 0 16px}h2{font-size:20px;margin:14px 0}p{max-width:1050px}a{color:#176f87}article{padding:20px 0;border-top:1px solid #cbd8d3}video{display:block;width:100%;height:auto;aspect-ratio:96/73;background:#eef3f1}details{padding:12px 0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}h1,h2,p{overflow-wrap:anywhere}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:16px}h1{font-size:24px}}
</style><main><h1>SALAD 恢复与蒙皮检查</h1><p>同一条动作分别检查位置骨架、旋转骨架和人体表面。四个固定病例共用中性人体、相机尺度、世界地面与 20 帧/秒；同步对照保留根节点上下运动。</p><p><a href="REPORT_zh.md">完整中文报告</a> · <a href="metrics.csv">逐动作指标</a> · <a href="audit.json">数值与来源证据</a></p><details><summary>六格对照与限制</summary><p>左上：原位置骨架；中上：此前直接蒙皮；右上：位置转标准 SMPL 蒙皮。左下：6D 旋转恢复骨架；中下：6D 恢复后转标准蒙皮；右下：作者 BVH 足部拟合后的估计表面。</p><p>标准蒙皮和 BVH 会改变关节坐标。位置无法唯一恢复绕骨轴扭转；高频不等同于错误，固定骨长不等于动作自然度已经修复。官方原函数骨架视频单独提供，不增加时间滤波。</p></details><div class="grid">'''+''.join(articles)+'</div></main></html>'
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    browser=ROOT/'qa/browser.json'
    if browser.exists():
        qa=json.loads(browser.read_text(encoding='utf-8'))
        if qa['status']!='passed' or len(qa['video_checks'])!=16:
            raise ValueError('Incomplete desktop/mobile playback verification')
        (TRACKED/'browser.json').write_bytes(browser.read_bytes())
        paths=[ROOT/'audit.json',ROOT/'verification.json',ROOT/'videos.json',ROOT/'official_videos.json',ROOT/'metrics.csv',browser]
        for name in ['audit_recovery_v2.py','verify_recovery_v2.py','render_recovery_v2.py','remote_recovery_v2.py','report_recovery_v2.py','qa_recovery_v2.cjs']:
            paths.append(REPO/'reproduction'/name)
        delivery=dict(status='passed',numerical_verification=True,playback_checks=16,
            fixed_case_count=4,official_skeleton_videos=4,synchronized_comparison_videos=4,
            ground_truth_constant_heading_check_passed=verification['ground_truth_constant_heading_check_passed'],
            previous_renderer_equals_official=False,naturalness_fixed=False,
            sha256={p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        content=json.dumps(delivery,indent=2,ensure_ascii=False)
        (ROOT/'delivery_verification.json').write_text(content,encoding='utf-8')
        (TRACKED/'delivery_verification.json').write_text(content,encoding='utf-8')
    print(json.dumps(dict(status='reported',rows=len(audit['rows']),cases=len(audit['cases']))))


if __name__=='__main__':main()
