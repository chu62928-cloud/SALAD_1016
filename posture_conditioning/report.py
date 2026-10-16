"""Independent subject aggregation, preregistered gates and Chinese report."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .common import ROOT,config,sha,write_json
from .evaluate import csv_write,VARIANTS


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def compare(subjects,paired,control,repeats=10000):
    identities=sorted(subjects)
    sources=defaultdict(list)
    for i,identity in enumerate(identities):
        sources[identity.split(':')[0]].append(i)
    p=np.array([subjects[i][paired] for i in identities])
    c=np.array([subjects[i][control] for i in identities])
    rng=np.random.RandomState(20261017)
    resampled=np.concatenate([rng.choice(indices,(repeats,len(indices)),replace=True)
                               for indices in sources.values()],axis=1)
    relative=(c[resampled].mean(1)-p[resampled].mean(1))/c[resampled].mean(1)
    low,high=np.quantile(relative,[.0125,.9875])
    point=float((c.mean()-p.mean())/c.mean())
    return dict(control=control,improvement_fraction=point,ci_97_5=[float(low),float(high)],
                point_at_least_one_percent=point>=.01,corrected_ci_supports_improvement=bool(low>0),
                passed=bool(point>=.01 and low>0))


def aggregate(seed=20261017):
    run=ROOT/config()['run_root']
    evaluation=run/'evaluation'/str(seed)
    assert json.loads((evaluation/'status.json').read_text())['status']=='complete'
    out=ROOT/'results/posture_v1'
    out.mkdir(parents=True,exist_ok=True)
    subjects=defaultdict(dict)
    seen=None
    denoise=[];metrics=[];source_rows=[];bindings={}
    for variant in VARIANTS:
        rows=read_csv(evaluation/variant/'denoise.csv')
        current={r['sample_id'] for r in rows}
        assert len(current)==1327 and len(rows)==1327
        if seen is not None:
            assert current==seen
        seen=current
        grouped=defaultdict(list)
        for r in rows:
            grouped[r['identity']].append(float(r['error']))
        assert len(grouped)==57
        for identity,values in grouped.items():
            subjects[identity][variant]=float(np.mean(values))
        mean=float(np.mean([np.mean(v) for v in grouped.values()]))
        recorded=json.loads((evaluation/variant/'denoise.json').read_text())
        assert abs(mean-recorded['subject_mean'])<1e-12
        denoise.append(dict(variant=variant,denoise_subject_mean=mean))
        for source in sorted({x.split(':')[0] for x in grouped}):
            values=[np.mean(v) for x,v in grouped.items() if x.split(':')[0]==source]
            source_rows.append(dict(variant=variant,source=source,subjects=len(values),denoise_mean=float(np.mean(values))))
        generation=json.loads((evaluation/variant/'generation.json').read_text())
        metrics.append(dict(variant=variant,denoise_subject_mean=mean,**generation['mean']))
        bindings[variant]=dict(denoise_sha256=sha(evaluation/variant/'denoise.csv'),
                               generation_sha256=sha(evaluation/variant/'generation.json'))
    assert all(len(v)==4 for v in subjects.values())
    comparisons=[compare(subjects,'paired',control,config()['bootstrap_repeats']) for control in ['fixed','shuffled']]
    by_variant={r['variant']:r for r in metrics}
    paired=by_variant['paired']
    quality=[]
    for reference in ['original','fixed']:
        base=by_variant[reference]
        definitions=[('r3',base['r3']-paired['r3'],.02,'absolute_decrease'),
                     ('fid',paired['fid']/base['fid']-1,.1,'relative_increase'),
                     ('bone_length_cv',paired['bone_length_cv']-base['bone_length_cv'],.005,'absolute_increase'),
                     ('root_height_hf_m',paired['root_height_hf_m']/base['root_height_hf_m']-1,.1,'relative_increase'),
                     ('relative_hf_m',paired['relative_hf_m']/base['relative_hf_m']-1,.1,'relative_increase')]
        if paired.get('flat_ground_clips',0)>0:
            definitions.append(('skating_frame_ratio',paired['skating_frame_ratio']-base['skating_frame_ratio'],.02,'absolute_increase'))
        for metric,value,limit,kind in definitions:
            quality.append(dict(reference=reference,metric=metric,change=float(value),limit=limit,
                                change_kind=kind,passed=bool(value<=limit)))
    main_pass=all(c['passed'] for c in comparisons)
    quality_pass=all(c['passed'] for c in quality) and paired.get('flat_ground_clips',0)>0
    decision=dict(seed=seed,main_gate_passed=main_pass,quality_gate_passed=quality_pass,
                  expand_training_seeds=bool(main_pass and quality_pass),
                  stop_after_single_seed=not(main_pass and quality_pass),comparisons=comparisons,quality=quality)
    write_json(out/'decision.json',decision)
    csv_write(out/'metrics.csv',metrics)
    csv_write(out/'by_source.csv',source_rows)
    csv_write(out/'subjects.csv',[dict(identity=k,**v) for k,v in sorted(subjects.items())])
    write_json(out/'bindings.json',dict(seed=seed,inputs=bindings,
               evaluation=json.loads((evaluation/'bindings.json').read_text()),
               report_source_sha256=sha(__file__)))
    for name in ['interventions.csv']:
        (out/name).write_bytes((evaluation/name).read_bytes())
    intervention_rows=read_csv(evaluation/'interventions.csv')
    intervention_summary=[]
    for group in ['swap','shoulders','spine','pelvis','left_leg','right_leg']:
        for quarter in range(4):
            selected=[r for r in intervention_rows if r['group']==group and int(r['quarter'])==quarter]
            assert len(selected)==8 and len({r['sample_id'] for r in selected})==8
            values={key:float(np.mean([float(r[key]) for r in selected])) for key in
                    ['mean_joint_change_m','pelvis_change_m','trunk_change_m','legs_change_m']}
            intervention_summary.append(dict(group=group,quarter=quarter,subjects=8,**values))
    csv_write(out/'intervention_summary.csv',intervention_summary)
    p=by_variant['paired']
    names={'original':'原 SALAD','fixed':'固定中性条件适配','paired':'正确同人配对','shuffled':'训练配对打乱'}
    lines=['# 静态体态条件动作扩散：第一轮结果','',
           '本实验是 SALAD 启发的新增模型。研究对象为同人独立静态校准姿态与动态运动的关联，不将结果解释为自然站姿异常或生理因果。','',
           '## 结论','',
           ('预设主指标和质量门槛均通过，按冻结方案扩展两个训练种子。' if decision['expand_training_seeds'] else
            '第一轮没有同时通过预设研究与质量门槛。已完成单种子对照，停止扩展训练；当前结果不足以确认个体静态体态带来可靠收益。'),'',
           '## 数据与训练','',
           '保留来源内受试者划分：训练 256 人／5299 段，验证 56 人／1061 段，测试 57 人／1327 段。跨来源真人同一性未核验。三组新增模型各训练 10000 步，有效批量 64，以验证集受试者等权去噪误差选模，测试集不参与调参。GROUNDLINK 没有验证身份。','',
           '静态条件排除头颈、手臂与手，肩带只使用单个左右高低差；核心为骨盆、脊柱及下肢。冻结原 SALAD、VAE 和文本编码器，只训练分组编码器、交叉注意力和门控。使用官方归一化；20 Hz、前 196 帧内四帧单元，尾部 0–3 帧按协议截断。所有静态参考的已知类别都是 calibration，打乱控制按来源及该类别分层，尚无可靠的更细 A/T 校准类别。','',
           '## 定量结果','',
           '| 模型 | 受试者等权去噪误差 ↓ | FID ↓ | R3 ↑ | 骨长变异 % ↓ | 平地脚滑 % ↓ |',
           '|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        lines.append('| %s | %.6f | %.4f | %.4f | %.3f | %.3f |'%(
            names[row['variant']],row['denoise_subject_mean'],row['fid'],row['r3'],
            row['bone_length_cv']*100,row.get('skating_frame_ratio',float('nan'))*100))
    lines+=['','去噪误差为三个固定评价重复的片段均值，再按身份等权。生成质量为三个固定噪声重复；FID 与文本检索来自官方 HumanML3D 评估器，但数据是自有测试集，不能直接与原论文基准表比较。检索候选以固定分组 32 段计算，末批保留；数据有重复文本，检索结果仅作为同协议模型比较。','',
            '## 体态贡献检验','']
    for c in comparisons:
        lines.append('- 相对%s：改善 %.4f%%，经两项比较校正的 97.5%% 区间 [%.4f%%, %.4f%%]；门槛%s。'%(
            names[c['control']],c['improvement_fraction']*100,c['ci_97_5'][0]*100,c['ci_97_5'][1]*100,'通过' if c['passed'] else '未通过'))
    lines+=['','区间按来源分层、受试者为单位进行 10000 次自举，不能将片段当作独立受试者增加样本量。区间只反映当前训练种子下的受试者抽样不确定性，不包含训练随机性。训练配对打乱是训练阶段对照，不等同于仅推理替换条件。','',
            '## 质量门槛','', '| 参照 | 指标及单位 | 变化 | 上限 | 结果 |','|---|---|---:|---:|---|']
    quality_labels={'r3':'R3 下降（百分点）','fid':'FID 增加（%）',
                    'bone_length_cv':'骨长变异增加（百分点）','root_height_hf_m':'根高度高频增加（%）',
                    'relative_hf_m':'根相对高频增加（%）','skating_frame_ratio':'平地脚滑增加（百分点）'}
    for c in quality:
        lines.append('| %s | %s | %.4f | %.4f | %s |'%(names[c['reference']],quality_labels[c['metric']],c['change']*100,c['limit']*100,'通过' if c['passed'] else '未通过'))
    lines+=['','质量门槛只限制预设指标的退化，不是动作自然度或物理正确性的认证。以下接触、离地和穿地统计仅对应每次重复的 %d 段预先识别平地行走；骨长及高频统计覆盖全部测试片段。'%p['flat_ground_clips'],'',
            '| 模型 | 脚部低位覆盖 % | 双脚离地帧 % | 穿地超过 2 cm 帧 % | 根高度高频 mm | 根相对高频 mm |',
            '|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        lines.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f |'%(
            names[row['variant']],row['near_floor_toe_interval_fraction']*100,row['airborne_frame_ratio']*100,
            row['penetration_frame_ratio']*100,row['root_height_hf_m']*1000,row['relative_hf_m']*1000))
    lines+=['','正确配对的脚滑仍有 %.2f%%，平地片段双脚离地 %.2f%%；根高度高频相对原 SALAD 增加 %.2f%%。这些结果不支持本轮已经解决原模型滑步与抖动的说法。固定中性条件已经取得大部分去噪和 FID 改善，不能将一般域适配收益归于个体体态信息。'%(
        p['skating_frame_ratio']*100,p['airborne_frame_ratio']*100,
        (p['root_height_hf_m']/by_variant['original']['root_height_hf_m']-1)*100)]
    lines+=['','脚滑只用于任务名预先识别的平地行走，未把阶梯与斜坡按世界平面评价。接触高度 5 cm，水平脚尖速度阈值 0.2 m/s，分母为全部帧间隔；必须连同接触覆盖、悬空与穿地一起看。骨长与高频指标不依赖地面。高频指高于 3 Hz 的离散余弦分量，包含正常运动成分，不能冒充感知评分。','',
            '## 干预与可视化','',
            'interventions.csv 保存同噪声下体态替换与单组中性化的实际关节变化，分四个归一化时间区间；attention_gate_residual.pt 在服务器保存注意力、门控和实际注入范数。注意力本身不能解释为真实影响强度。单组输入中性化可能产生训练分布外组合，且消息传递会把影响传播到其他令牌，只作为模型敏感性检验。固定视频按身份编号选样，不根据外观挑选。','',
            '匿名汇总 intervention_summary.csv 按八个预选身份等权，保留全部六类干预、四个时间区间及骨盆、躯干、下肢变化。区间仅按序列长度四等分，未经标注不命名下蹲最低点等语义阶段；变化大小不是体态保持准确率。','',
            '## 文件与复现','',
            '- 训练：`python -m posture_conditioning.train --variant paired`，另两组使用 fixed、shuffled。',
            '- 评价：`python -m posture_conditioning.evaluate --seed 20261017`。',
            '- 汇总：`python -m posture_conditioning.report`。',
            '- 服务器检查点：`runs/posture_v1/{fixed,paired,shuffled}/20261017/checkpoints/{best,last}.pt`。',
            '- 协议：`experiments/posture_v1/PROTOCOL_zh.md`；配置：`configs/posture_v1.json`。','',
            '代码与汇总上传 GitHub；原数据、逐人媒体和检查点保留服务器。第一轮阴性或混合结果不意味着所有可能架构均无效，也不能据此声称已经修复原 SALAD 的抖动或滑步。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(decision,indent=2),flush=True)
    return decision


if __name__=='__main__':
    aggregate()
