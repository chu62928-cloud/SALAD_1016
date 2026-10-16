"""Quantitative figure from complete subject tables and actual interventions."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                           'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,
                           'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.8,
                           'legend.frameon':False,'savefig.dpi':600})


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    source=ROOT/'results/posture_v1'
    decision=json.loads((source/'decision.json').read_text())
    subjects=rows(source/'subjects.csv')
    interventions=rows(source/'interventions.csv')
    assert len(subjects)==57
    fig=plt.figure(figsize=(7.204724,5.708661),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.1,1])
    primary=fig.add_subplot(grid[0,:])
    names={'fixed':'Fixed-neutral adaptation','shuffled':'Shuffled-pair training'}
    colors=['#356E87','#AC6758']
    for i,c in enumerate(decision['comparisons']):
        point=c['improvement_fraction']*100
        low,high=np.array(c['ci_97_5'])*100
        primary.errorbar(point,i,xerr=np.array([[point-low],[high-point]]),fmt='o',
                         color=colors[i],capsize=3,markersize=5,linewidth=1.3)
    primary.axvline(0,color='#777777',lw=.8)
    primary.axvline(1,color='#777777',lw=.8,ls='--')
    primary.set(yticks=[0,1],yticklabels=[names[c['control']] for c in decision['comparisons']],
                xlabel='Paired-model denoising improvement (%)',ylim=(-.7,1.7))
    primary.set_title('a  Posture contribution on unseen subjects',loc='left',fontweight='bold',fontsize=8)
    primary.text(.99,.97,'57 subjects; stratified subject bootstrap\n97.5% CIs; one training seed',
                 transform=primary.transAxes,ha='right',va='top',fontsize=7)
    individual=fig.add_subplot(grid[1,0])
    for i,control in enumerate(['fixed','shuffled']):
        values=np.array([(float(r[control])-float(r['paired']))/float(r[control])*100 for r in subjects])
        # Deterministic horizontal placement retains every subject without sampling.
        jitter=np.linspace(-.18,.18,len(values))
        order=np.argsort([r['identity'] for r in subjects])
        individual.scatter(i+jitter,values[order],s=8,alpha=.65,color=colors[i],edgecolors='none')
        individual.plot([i-.22,i+.22],[values.mean()]*2,color='#222222',lw=1.2)
    individual.axhline(0,color='#777777',lw=.8)
    individual.set(xticks=[0,1],xticklabels=['Fixed-neutral','Shuffled-pair'],ylabel='Subject improvement (%)')
    individual.set_title('b  All test subjects',loc='left',fontweight='bold',fontsize=8)
    intervention=fig.add_subplot(grid[1,1])
    groups=['swap','shoulders','spine','pelvis','left_leg','right_leg']
    values=np.array([[np.mean([float(r['mean_joint_change_m'])*1000 for r in interventions
                               if r['group']==g and int(r['quarter'])==q])
                      for q in range(4)] for g in groups])
    assert np.isfinite(values).all()
    image=intervention.imshow(values,aspect='auto',cmap='cividis',vmin=0)
    intervention.set(yticks=range(6),yticklabels=['Swap subject','Shoulders','Spine','Pelvis','Left leg','Right leg'],
                     xticks=range(4),xticklabels=['0-25','25-50','50-75','75-100'],xlabel='Normalized time (%)')
    intervention.set_title('c  Actual input interventions',loc='left',fontweight='bold',fontsize=8)
    fig.colorbar(image,ax=intervention,label='Mean joint change (mm)',shrink=.8)
    fig.savefig(source/'posture_results.svg',bbox_inches='tight')
    fig.savefig(source/'posture_results.pdf',bbox_inches='tight')
    fig.savefig(source/'posture_results.tiff',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(source/'posture_results.png',dpi=600,bbox_inches='tight')
    plt.close(fig)
    write_json(source/'figure_qa.json',dict(status='source_checked',backend='python',subjects=len(subjects),
                intervention_subjects=len({r['sample_id'] for r in interventions}),excluded_subjects=0,
                source_sha256={n:sha(source/n) for n in ['decision.json','subjects.csv','interventions.csv']},
                visual_inspection_pending=True))
    if (source/'attention_summary.json').exists():
        diagnostics(source)


def diagnostics(source):
    values=json.loads((source/'attention_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.204724,3.2),layout='constrained')
    keys=['alpha','gate','residual_norm']
    titles=['a  Posture attention','b  Motion-token gate','c  Injected residual norm']
    posture=['Left shoulder','Right shoulder','Spine','Pelvis','Left leg','Right leg']
    motion=['Root','Left lower','Right lower','Spine','Left upper','Right upper','Head']
    for axis,key,title in zip(axes,keys,titles):
        matrix=np.array(values[key])
        image=axis.imshow(matrix,aspect='auto',cmap='cividis',vmin=0,
                          vmax=1 if key in ['alpha','gate'] else None)
        axis.set(xticks=range(4),xticklabels=['Q1','Q2','Q3','Q4'],
                 yticks=range(len(matrix)),yticklabels=posture if key=='alpha' else motion,
                 xlabel='Normalized time quarter')
        axis.set_title(title,loc='left',fontsize=8,fontweight='bold')
        fig.colorbar(image,ax=axis,orientation='horizontal',shrink=.85,pad=.13)
    fig.savefig(source/'attention_gate.svg',bbox_inches='tight')
    fig.savefig(source/'attention_gate.pdf',bbox_inches='tight')
    fig.savefig(source/'attention_gate.png',dpi=600,bbox_inches='tight')
    fig.savefig(source/'attention_gate.tiff',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


if __name__=='__main__':
    main()
