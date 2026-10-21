"""PNG report figures from complete frozen diagnostics, without motion filtering."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from run import OUT, save, sha
from metrics import NAMES

LABELS={'ground_truth':'真实动作','posterior_mean':'VAE 均值','posterior_sample':'VAE 采样',
    'official':'原始 SALAD','adapted':'数据微调','fixed':'固定体态对照','paired':'对应体态条件','shuffled':'打乱体态对照'}
COLORS=['#65747D','#377A96','#C47544','#8A6A98','#458477','#A75C65']
METHOD_COLORS=dict(ground_truth='#65747D',posterior_mean='#377A96',posterior_sample='#C47544',
    official='#525A8B',adapted='#8A6A98',fixed='#A47C43',paired='#458477',shuffled='#A75C65')
FONT=Path('/root/autodl-tmp/data_preprocess_motion/staging/standing_v2/assets/msyh.ttc')


def main():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_manager.FontProperties(fname=str(FONT)).get_name(),'DejaVu Sans'],
        'font.size':8,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    summaries=json.loads((OUT/'summary/summary.json').read_text())
    groups={(r['dataset'],r['variant'],r['path']):r for r in summaries}
    flat=json.loads((OUT/'summary/flat_walking.json').read_text())
    flat_groups={(r['dataset'],r['variant'],r['path']):r for r in flat['summary']}
    output=OUT/'figures';output.mkdir(exist_ok=True)
    records=[]

    def export(fig,name,claim,source):
        for ax in fig.axes:
            low,high=ax.get_ylim()
            for line in ax.get_lines():
                values=np.asarray(line.get_ydata(),dtype=float)
                values=values[np.isfinite(values)]
                assert not values.size or (values.min()>=low-1e-8 and values.max()<=high+1e-8), (name,low,high,values)
        path=output/(name+'.png')
        fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
        records.append(dict(file=path.name,sha256=sha(path),claim=claim,source=source))

    def points(ax,dataset,variants,path,metric,factor,ylabel):
        for i,variant in enumerate(variants):
            if variant=='ground_truth' and metric=='mpjpe_root_aligned_m':
                ax.plot(i,0,'o',color=COLORS[i%len(COLORS)],ms=5)
                continue
            info=groups[(dataset,variant,path)]['metrics'][metric]
            y=info['mean']*factor;lo,hi=np.array(info['ci95'])*factor
            ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=3,color=METHOD_COLORS.get(variant,COLORS[i%len(COLORS)]),ms=5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([LABELS.get(x,x) for x in variants])
        ax.set_ylabel(ylabel);ax.set_xlim(-.5,len(variants)-.5);ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='#E6E8E9',linewidth=.5)

    fig,axes=plt.subplots(2,3,figsize=(9.2,5.6),constrained_layout=True)
    for row,(dataset,title) in enumerate([('humanml','原始域：128 段参考动作'),('data_self','目标域：57 人，1,327 段测试动作')]):
        for col,(metric,factor,label) in enumerate([('bone_length_cv',100,'骨长变异系数（%）'),
            ('local_jerk_rms_mps3',1,'去根位移 Jerk（米/秒³）'),('mpjpe_root_aligned_m',1000,'去根位置误差（毫米）')]):
            points(axes[row,col],dataset,['ground_truth','posterior_mean','posterior_sample'],'position',metric,factor,label)
            axes[row,col].set_title(title if col==0 else ['','完整有效序列，20 帧/秒','与同帧真实动作比较'][col],fontsize=8)
            axes[row,col].text(-.12,1.08,chr(97+row*3+col),transform=axes[row,col].transAxes,fontweight='bold')
    export(fig,'01_vae_reconstruction','VAE alone introduces bone-length variation; jerk changes depend on domain.',
        'summary/summary.json; root-relative vector RMS; equal subject weights; 95% bootstrap CI, 4000 resamples.')

    fig,axes=plt.subplots(2,2,figsize=(9.2,6),constrained_layout=True)
    variants=['official','adapted','fixed','paired','shuffled']
    specs=[('bone_length_cv',100,'骨长变异系数（%）'),('local_jerk_rms_mps3',1,'去根位移 Jerk（米/秒³）'),
        ('local_hf3_rms_m',1000,'去根位置 ≥3 赫兹 RMS（毫米）'),('skating_frame_ratio',100,'滑动帧比例（%）')]
    for ax,(metric,factor,label),letter in zip(axes.flat,specs,'abcd'):
        full_groups=groups
        if metric=='skating_frame_ratio':groups=flat_groups
        points(ax,'data_self',variants,'position',metric,factor,label)
        ax.tick_params(axis='x',rotation=12)
        baseline=groups[('data_self','ground_truth','position')]['metrics'][metric]['mean']*factor
        ax.axhline(baseline,color='#55636A',ls='--',lw=1,label='真实动作参考')
        ax.text(-.1,1.05,letter,transform=ax.transAxes,fontweight='bold')
        if metric=='skating_frame_ratio':ax.set_title('仅平地走路：20 人，72 段',fontsize=8)
        groups=full_groups
    axes[0,0].legend(loc='best',fontsize=7)
    export(fig,'02_finetuning_and_posture','Adaptation and posture effects must be assessed jointly across physical metrics.',
        'summary/summary.json and flat_walking.json; a-c: 1327 clips, 57 held-out subjects; d: 72 flat-walking clips, 20 subjects; 3 repeats; subject weights; 95% bootstrap CI.')

    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),constrained_layout=True)
    variants=['ground_truth','posterior_mean','official','adapted','paired']
    pathnames=['position','position_fixed','rotation_fk']
    for i,variant in enumerate(variants):
        for ax,(metric,factor,label) in zip(axes,[('bone_length_cv',100,'骨长变异系数（%）'),
            ('local_jerk_rms_mps3',1,'去根位移 Jerk（米/秒³）')]):
            y=[groups[('data_self',variant,p)]['metrics'][metric]['mean']*factor for p in pathnames]
            ax.plot(range(3),y,'o-',label=LABELS[variant],color=METHOD_COLORS[variant],lw=1,ms=4)
            ax.set_xticks(range(3));ax.set_xticklabels(['原始位置','仅固定骨长','旋转通道 FK']);ax.set_ylabel(label)
            ax.grid(axis='y',color='#E6E8E9',linewidth=.5)
    for ax in axes:ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    export(fig,'03_recovery_paths','Fixing lengths and selecting rotation FK affect temporal jitter independently.',
        'summary/summary.json; same target sequences in all three paths; no filtering; mean across subjects.')

    fig,axes=plt.subplots(2,2,figsize=(9.2,5.8),constrained_layout=True)
    settings=['steps25_cfg7.5','steps50_cfg7.5','steps100_cfg7.5','steps50_cfg2.5','steps50_cfg1']
    xlabels=['25 步\n引导 7.5','50 步\n引导 7.5','100 步\n引导 7.5','50 步\n引导 2.5','50 步\n引导 1.0']
    for ax,(metric,factor,label) in zip(axes.flat,specs):
        for i,variant in enumerate(settings):
            g=groups[('humanml_probe',variant,'position')]['metrics'][metric]
            y=g['mean']*factor;lo,hi=np.array(g['ci95'])*factor
            ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=3,color=COLORS[1],ms=5)
        ax.set_xticks(range(5));ax.set_xticklabels(xlabels);ax.set_ylabel(label);ax.set_ylim(bottom=0)
        ax.grid(axis='y',color='#E6E8E9',linewidth=.5)
    export(fig,'04_sampling_sensitivity','Configuration sensitivity is exploratory and not semantic-quality validation.',
        '6 prespecified prompts x 3 paired seeds; same initial noise per seed; equal prompt means; 95% prompt bootstrap CI.')

    sample=OUT/'sampling/examples/humanml_probe_steps50_cfg7.5_20261016_walk.npz'
    with np.load(sample) as data:
        paths={name:{key:data[name+'_'+key] for key in ['bone_length_m','joint_speed','joint_acceleration','joint_jerk']}
               for name in pathnames}
    selected=int(np.argmax(paths['position']['bone_length_m'].std(0)/paths['position']['bone_length_m'].mean(0)))
    joint=int(np.argmax(np.mean(paths['position']['joint_jerk']**2,axis=0)))
    fig,axes=plt.subplots(4,1,figsize=(9.2,8.2),sharex=True,constrained_layout=True)
    for i,(name,label) in enumerate(zip(pathnames,['原始位置','仅固定骨长','独立旋转通道 FK'])):
        values=paths[name]
        for ax,key,order,unit in zip(axes,['bone_length_m','joint_speed','joint_acceleration','joint_jerk'],[0,1,2,3],
            ['骨长（米）','速度模长（米/秒）','加速度模长（米/秒²）','Jerk 模长（米/秒³）']):
            v=values[key][:,selected if order==0 else joint]
            time=(np.arange(len(v))+order/2)/20
            ax.plot(time,v,lw=.85,label=label,color=COLORS[i]);ax.set_ylabel(unit)
            ax.grid(axis='y',color='#E6E8E9',linewidth=.5)
    chinese={'head':'颈至头','left_foot':'左足','right_foot':'右足'}
    axes[0].set_title('固定走路示例；骨段 '+chinese.get(NAMES[selected+1],NAMES[selected+1])+'；关节 '+chinese.get(NAMES[joint],NAMES[joint]),fontsize=9)
    axes[0].legend(ncol=3,fontsize=7);axes[-1].set_xlabel('时间（秒）')
    export(fig,'05_complete_sequence_traces','Full per-bone and per-joint time series distinguish length oscillation from direction jitter.',
        str(sample.relative_to(OUT))+'; diagnostic example only; maximum-CV bone and maximum-RMS-jerk joint selected within this fixed clip, not population evidence.')

    fig,ax=plt.subplots(figsize=(9.2,3.4),constrained_layout=True)
    for i,variant in enumerate(['ground_truth','posterior_mean','posterior_sample','official','adapted','paired']):
        values=[groups[('data_self',variant,'position')]['metrics'][f'local_jerk_mod4_{phase}_mps3']['mean'] for phase in range(4)]
        ax.plot(range(4),values,'o-',label=LABELS[variant],color=METHOD_COLORS[variant],ms=4,lw=1)
    ax.set_xticks(range(4));ax.set_xticklabels(['0','1','2','3'])
    ax.set_xlabel('三阶差分起点帧序号 mod 4');ax.set_ylabel('去根位移 Jerk（米/秒³）')
    ax.set_title('解码步幅相关的周期差异：是线索，尚不能单独归因于某一层',fontsize=9)
    ax.legend(ncol=3,fontsize=7);ax.grid(axis='y',color='#E6E8E9',linewidth=.5)
    export(fig,'06_decoder_phase_diagnostic','Reconstructed and generated motions show decoder-stride-linked jerk modulation absent in target GT.',
        'summary/summary.json; 57 held-out subjects; derivative start index modulo 4; full valid clips; no filtering; observational diagnostic, not a layer ablation.')
    save(output/'figure_contract_and_manifest.json',dict(status='complete',backend='python',archetype='quantitative grid',
        export='PNG only, 300 dpi, white background; no smoothing or data exclusions',
        inference='Descriptive paired bootstrap confidence intervals; no p-value claims or multiple-testing significance claims.',
        data_integrity='All frozen clips included; undefined no-contact statistics remain missing and use available-subject counts.',
        figures=records))
    print('FIGURES COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
