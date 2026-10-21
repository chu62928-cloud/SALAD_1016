"""Build separate official-dataset tables and PNG figures from completed runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from audit_gt import save_json, sha256

METRICS = [
    ("flow_published_padded", "FlowMotion 兼容 Jitter（原生全帧）"),
    ("flow_zero_normalized_padding", "FlowMotion 兼容 Jitter（统一零补齐）"),
    ("local_jerk_rms_mps3", "去根三阶导数 RMS（米/秒³）"),
    ("bone_cv_percent", "骨长变异系数（%）"),
    ("local_hf3_vector_rms_mm", "≥3 赫兹向量 RMS（毫米）"),
    ("sliding_frame_percent", "滑动帧（%）"),
    ("fid", "FID"), ("top1", "检索准确率 R@1"),
    ("top2", "检索准确率 R@2"), ("top3", "检索准确率 R@3"),
    ("matching", "文本动作匹配距离"),
]
LABELS = {"GT":"真实动作", "SALAD":"SALAD", "SALAD_full_finetune":"SALAD＋全参数微调",
          "MDM_BERT_50":"MDM（BERT／50步）", "MDM_CLIP_1000":"MDM（CLIP／1000步）"}
COLORS = {"GT":"#71777D", "SALAD":"#287E9B", "SALAD_full_finetune":"#A65D74",
          "MDM_BERT_50":"#4C8C65", "MDM_CLIP_1000":"#4C8C65"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    x = np.array(values, dtype=float)
    return dict(mean=float(x.mean()), ci95_half_width=float(1.96*x.std()/np.sqrt(len(x))))


def flow_results(directory):
    if read(directory/"status.json")["status"] != "complete":
        raise ValueError("FlowMotion has not completed all 20 repetitions")
    paths = list(directory.glob("eval_humanml_*.log"))
    assert len(paths) == 1
    log = paths[0].read_text()
    repetitions = read(directory/"repetitions.json")
    assert len(repetitions) == 40
    per_repeat = []
    for part in re.split(r"=+ Replication \d+ =+", log)[1:]:
        part = part.split("========== jittering Summary")[0]
        entries = {}
        for model in ["ground truth", "vald"]:
            marker = re.escape(model)
            values = {}
            for key, pattern in [
                ("fid",r"FID: ([\d.eE+-]+)"),
                ("matching",r"matching_score: ([\d.eE+-]+)"),
                ("logged_jitter",r"jitter: ([\d.eE+-]+)"),
            ]:
                found = re.search(r"\["+marker+r"\] "+pattern, part)
                assert found, (model,key)
                values[key] = float(found.group(1))
            found = re.search(r"\["+marker+r"\] R_precision: \(top 1\): ([\d.]+) \(top 2\): ([\d.]+) \(top 3\): ([\d.]+)",part)
            assert found
            values.update({"top%d"%(i+1):float(v) for i,v in enumerate(found.groups())})
            entries[model] = values
        per_repeat.append(entries)
    assert len(per_repeat)==20
    result = {}
    for model in ["ground truth","vald"]:
        group = [r for r in repetitions if r["model"]==model]
        assert len(group)==20
        assert {r["clips"] for r in group}=={4640 if model=="ground truth" else 1024}
        result[model] = {k:stats([r[k] for r in group]) for k in group[0] if k not in ["model","repeat","clips"]}
        for key in ["fid","matching","top1","top2","top3"]:
            result[model][key] = stats([r[model][key] for r in per_repeat])
        for i,row in enumerate(group):
            assert abs(row["flow_published_padded"]-per_repeat[i][model]["logged_jitter"]) < .001
    result["semantic_precision"] = "Per-repeat semantic results read from native log rounded to four decimals; diagnostics retain full precision."
    return result,per_repeat


def cell(value, digits=3):
    return ("%.*f ± %.*f"%(digits,value["mean"],digits,value["ci95_half_width"])) if value else "未测"


def table(dataset,models):
    mdm="MDM_BERT_50" if dataset=="t2m" else "MDM_CLIP_1000"
    order=["GT",mdm,"ViMoGen","SALAD","SALAD_full_finetune"]
    headers=["指标", "真实动作", LABELS[mdm], "ViMoGen", "SALAD", "SALAD＋全参数微调"]
    lines=["| "+" | ".join(headers)+" |", "|"+" --- |"*len(headers)]
    for key,label in METRICS:
        values=[]
        for model in order:
            if dataset=="kit" and model=="SALAD_full_finetune":
                values.append("不适用")
            else:
                model_metrics=models.get(model,{})
                value=model_metrics.get(key)
                if key=="flow_zero_normalized_padding" and model in ["GT","SALAD","SALAD_full_finetune"]:
                    value=model_metrics.get("flow_published_padded")
                values.append(cell(value))
        lines.append("| "+" | ".join([label]+values)+" |")
    return "\n".join(lines)


def configure_font():
    candidates=[Path("/usr/local/share/fonts/noto/NotoSansCJKsc-Regular.otf"),Path("C:/Windows/Fonts/msyh.ttc")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family=font_manager.FontProperties(fname=str(path)).get_name()
            break
    else:
        raise RuntimeError("Chinese font missing")
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":[family,"Arial","DejaVu Sans"],
                         "svg.fonttype":"none","pdf.fonttype":42,"font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.unicode_minus":False,"axes.linewidth":.7,
                         "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white"})


def save_figure(figure,destination):
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    width,height=figure.canvas.get_width_height()
    checked=0
    for item in figure.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box=item.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=width+1 and box.y1<=height+1, item.get_text()
        checked+=1
    figure.savefig(destination,dpi=300)
    save_json(destination.with_suffix(".qa.json"),dict(status="passed",text_objects_within_canvas=checked,
              pixel_width=round(figure.get_figwidth()*300),pixel_height=round(figure.get_figheight()*300),
              dpi=300,format="PNG",static_preflight_exceptions={
                  "EXPORT-VECTOR":"User explicitly requested PNG only; no SVG/PDF/TIFF exported",
                  "FINAL-WIDTH":"Report figure, not a fixed-width journal submission"},
              visual_review="Required after rendering; bounds alone do not establish correct layout"))


def physical_figure(models,dataset,destination):
    mdm="MDM_BERT_50" if dataset=="t2m" else "MDM_CLIP_1000"
    order=[m for m in ["GT",mdm,"SALAD","SALAD_full_finetune"] if m in models]
    figure,axes=plt.subplots(2,2,figsize=(9.2,6.2))
    figure.subplots_adjust(left=.10,right=.97,top=.87,bottom=.16,wspace=.35,hspace=.55)
    keys=[("local_jerk_rms_mps3","去根三阶导数 RMS（米/秒³）"),
          ("bone_cv_percent","骨长变异系数（%）"),
          ("local_hf3_vector_rms_mm","≥3 赫兹向量 RMS（毫米）"),
          ("sliding_frame_percent","滑动帧（%）")]
    for index,(axis,(key,label)) in enumerate(zip(axes.flat,keys)):
        mean=np.array([models[m][key]["mean"] for m in order])
        error=np.array([models[m][key]["ci95_half_width"] for m in order])
        x=np.arange(len(order))
        axis.bar(x,mean,width=.56,color=[COLORS[m] for m in order],zorder=2)
        axis.errorbar(x,mean,yerr=error,fmt="none",ecolor="#222222",elinewidth=.8,capsize=3,zorder=3)
        axis.set_xticks(x)
        axis.set_xticklabels(["真实动作" if m=="GT" else ("全参数微调" if m=="SALAD_full_finetune" else "MDM" if m.startswith("MDM") else "SALAD") for m in order])
        axis.set_ylabel(label)
        axis.set_title(chr(97+index),loc="left",fontweight="bold")
        axis.set_ylim(0,max(mean+error)*1.25+.001)
        axis.grid(axis="y",color="#E3E6E8",linewidth=.6,zorder=0)
        axis.set_axisbelow(True)
        for position,value,spread in zip(x,mean,error):
            axis.text(position,value+spread+axis.get_ylim()[1]*.025,"≈0" if value<.001 else "%.2f"%value,ha="center",fontsize=8)
    name="HumanML3D" if dataset=="t2m" else "KIT-ML"
    figure.suptitle(name+"：几何与时间诊断",x=.10,ha="left",fontsize=13,fontweight="bold")
    figure.text(.10,.035,"每种模型 3 次采样重复；柱为均值，误差线为 1.96×重复间总体标准差/√3。\n仅有效帧；未经平滑或固定骨长。MDM 版本："+("BERT／50 步" if dataset=="t2m" else "CLIP／1000 步")+"。",fontsize=8)
    save_figure(figure,destination)
    plt.close(figure)


def compatibility_figure(gates,flow,destination):
    figure,axes=plt.subplots(1,3,figsize=(10.5,3.5))
    figure.subplots_adjust(left=.07,right=.98,top=.78,bottom=.25,wspace=.45)
    panels=[("HumanML3D 真实动作",47.262,gates["t2m"]["metrics"]["original_torch_aggregation"]),
            ("KIT-ML 真实动作",49.922,gates["kit"]["metrics"]["original_torch_aggregation"]),
            ("FlowMotion：HumanML3D",39.522,flow["vald"]["flow_published_padded"])]
    for index,(axis,(label,target,measurement)) in enumerate(zip(axes,panels)):
        axis.bar([0,1],[target,measurement["mean"]],color=["#B7BEC3","#287E9B"],width=.55,zorder=2)
        axis.errorbar([1],[measurement["mean"]],yerr=[measurement["ci95_half_width"]],fmt="none",ecolor="#222222",capsize=3)
        axis.set_xticks([0,1])
        axis.set_xticklabels(["论文参考","本次实测"])
        axis.set_ylim(0,60)
        axis.set_title(chr(97+index)+"  "+label,loc="left",fontsize=9)
        axis.grid(axis="y",color="#E3E6E8",linewidth=.6,zorder=0)
        axis.set_axisbelow(True)
        for x,value in enumerate([target,measurement["mean"]]):
            axis.text(x,value+2,"%.3f"%value,ha="center",fontsize=8)
    axes[0].set_ylabel("FlowMotion 兼容 Jitter")
    figure.suptitle("官方指标核验：三个面板独立解读",x=.07,ha="left",fontsize=13,fontweight="bold")
    figure.text(.07,.06,"均为 20 次采样重复；误差线同报告定义。HumanML3D 与 KIT 的绝对值不可横向比较。\nFlowMotion 为作者原生 1,024 条生成协议；该图不表示语义指标也已复现。",fontsize=8)
    save_figure(figure,destination)
    plt.close(figure)


def build(args):
    root=args.root.resolve()
    output=root/"public"
    output.mkdir(parents=True,exist_ok=True)
    (output/".gitattributes").write_text("* -text\n",encoding="utf-8")
    verification=read(root/"verification.json")
    assert verification["status"]=="passed" and verification["scope"]=="all_paired_runs"
    specs=[("t2m","salad_t2m","SALAD"),("kit","salad_kit","SALAD"),
           ("t2m","salad_ft_t2m","SALAD_full_finetune"),
           ("t2m","mdm_bert50_t2m","MDM_BERT_50"),("kit","mdm_clip1000_kit","MDM_CLIP_1000")]
    models={"t2m":{},"kit":{}}
    sampling=[]
    for dataset,name,model in specs:
        directory=root/name
        assert read(directory/"status.json")["status"]=="complete",name
        summary=read(directory/"summary.json")
        metrics=summary["metrics"]
        if model.startswith("SALAD"):
            models[dataset][model]=metrics[model]
            if model=="SALAD":
                models[dataset]["GT"]=metrics["GT"]
        else:
            models[dataset][model]=metrics
        for row in read(directory/"repetitions.json"):
            sampling.append(dict(dataset=dataset,run=name,**row))
    gates={ds:read(root/("gt_"+ds)/"summary.json") for ds in ["t2m","kit"]}
    flow,flow_repetitions=flow_results(root/"flow_humanml_attempt3")
    save_json(output/"flow_native_summary.json",flow)
    save_json(output/"flow_native_semantic_repetitions.json",flow_repetitions)
    save_json(output/"results.json",dict(datasets=models,gt_gate=gates,flow_native=flow))
    records=[]
    for dataset,methods in models.items():
        for model,metrics in methods.items():
            for key,value in metrics.items():
                records.append(dict(dataset=dataset,model=model,metric=key,**value))
    with (output/"main_summary.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fields=list(dict.fromkeys(key for row in sampling for key in row))
    with (output/"sampling_repetitions.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(sampling)
    evidence=output/"evidence"
    evidence.mkdir(exist_ok=True)
    for name in ["gt_t2m","gt_kit","salad_t2m","salad_kit","salad_ft_t2m","mdm_bert50_t2m","mdm_clip1000_kit","flow_humanml_attempt3"]:
        for filename in ["binding.json","summary.json","repetitions.json","status.json"]:
            source=root/name/filename
            if source.exists():
                destination=evidence/name/filename
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,destination)
    shutil.copyfile(root/"verification.json",evidence/"verification.json")
    text_control=read(root/"text_length_control.json")
    assert text_control["status"]=="complete" and len(text_control["rows"])==6
    shutil.copyfile(root/"text_length_control.json",evidence/"text_length_control.json")
    archived={}
    for dataset,name in [("t2m","eval_t2m_v2"),("kit","eval_kit_v3")]:
        source=root.parent/name/"summary.json"
        archived[dataset]=read(source)
        shutil.copyfile(source,evidence/("archived_"+name+".json"))
    configure_font()
    physical_figure(models["t2m"],"t2m",output/"01_humanml3d_physical.png")
    physical_figure(models["kit"],"kit",output/"02_kitml_physical.png")
    compatibility_figure(gates,flow,output/"03_official_jitter_check.png")
    text=["# SALAD 官方测试集：抖动、几何和语义评测", "",
          "日期：2026-10-21。所有本轮模型结果均为实测。两套数据集独立报告，没有重新训练或对动作进行后处理。现有微调单列为全参数微调，不称为 LoRA；具体为既有去噪主干全参数适配，VAE和文本编码器仍冻结。", "",
          "## 结论与完成范围", "",
          "已完成两套真实动作各 20 次指标核验；原 SALAD 两套数据、HumanML3D 全参数微调、HumanML3D MDM（BERT／50步）、KIT-ML MDM（CLIP／1000步）各 3 次完整测试集采样；FlowMotion HumanML3D 采用作者原生 20 次评测。ViMoGen 未完成官方骨架、表示转换与评估器核验，因此没有填入未经验证的分数。", "",
          "FlowMotion 的官方仓库为 [mccuba/FlowMotion](https://github.com/mccuba/FlowMotion)，固定提交 `fe4021b9170545b4e9ddc6d13d309bfa0988c83f`。[作者项目主页](https://mccuba.github.io/fm-page/)同时链接该仓库与论文 DOI `10.1016/j.cag.2025.104374`。指标实际为三阶差分模长均值，并非候选的加速度指标。Jitter 的论文参考数值沿用用户提供的表格；本轮直接复核了公开源码与预训练权重运行，出版商正文返回403，未独立读取原文表格。", "",
          "| 数据集 | 本次真实动作 Jitter（20次） | 论文参考 | 相对偏差 |", "| --- | --- | --- | --- |"]
    for ds,label in [("t2m","HumanML3D"),("kit","KIT-ML")]:
        gate=gates[ds]
        text.append("| %s | %s | %.3f | %+.4f%% |"%(label,cell(gate["metrics"]["original_torch_aggregation"]),gate["paper_gt_target"],gate["relative_difference"]*100))
    text += ["", "相对差均通过预先采用的 1% 接近性检查；这不是统计等价证明。下方主表真实动作来自与 SALAD 同轮的 3 次采样，因此与上方 20 次核验略有抽样差别。", "",
             "## HumanML3D", "", "每轮 4,640 条，3 次重复。真实动作、SALAD、全参数微调及 MDM 使用逐批完全相同的文本、长度和真实动作。R 指标按每批 32 条检索计算，越高越好；其余主表指标越低通常越好，但低抖动也可能来自动作幅度变小或过度平滑。", "",table("t2m",models["t2m"]), "",
             "![HumanML3D 诊断](01_humanml3d_physical.png)", "", "## KIT-ML", "",
             "每轮 704 条，3 次重复。MDM 为该数据集公开的 CLIP／1000步原生采样权重，不能与 HumanML3D 的 BERT／50步版本视为同一配置。现有全参数微调是 22关节／263维 HumanML3D 架构；KIT 是 21关节／251维，没有对应权重，不适用。", "",table("kit",models["kit"]), "",
             "![KIT-ML 诊断](02_kitml_physical.png)", "", "## 如何理解这些结果", ""]
    for ds,label in [("t2m","HumanML3D"),("kit","KIT-ML")]:
        gt,salad=models[ds]["GT"],models[ds]["SALAD"]
        text.append("%s：SALAD 的去根三阶导数 RMS 为 %.2f，真实动作为 %.2f 米/秒³；骨长变异系数为 %.2f%%，真实动作约为 %.6f%%；滑动帧为 %.2f%%，真实动作为 %.2f%%。因此应将时间变化、骨架几何与接触分别判断，不能用一个 Jitter 值概括自然度。"%(label,salad["local_jerk_rms_mps3"]["mean"],gt["local_jerk_rms_mps3"]["mean"],salad["bone_cv_percent"]["mean"],gt["bone_cv_percent"]["mean"],salad["sliding_frame_percent"]["mean"],gt["sliding_frame_percent"]["mean"]))
        text.append("")
    text += ["此前官方语义分数已接近 SALAD 论文，但本轮仍测到明显骨长变动。因此，骨长异常不能全部归因于 data_self 的域差异；它在官方测试集的原 SALAD 输出中也存在。这里检查的是直接恢复的关节坐标，没有经过网格蒙皮、逆运动学或渲染，不能把这部分几何异常归因于显示管线。", "",
             "真实动作的骨长近似恒定与官方预处理使用固定骨架有关，不能解释为原始采集完全没有测量误差。三阶导数和高频能量也包含真实快速动作，并非纯粹的噪声；单独数值较高不能自动判断不自然。", ""]
    baseline,adapted=models["t2m"]["SALAD"],models["t2m"]["SALAD_full_finetune"]
    text.append("现有全参数微调在 HumanML3D 上相对原 SALAD：FID %.4f → %.4f，R@3 %.4f → %.4f；去根三阶导数 RMS %.2f → %.2f，骨长变异系数 %.2f%% → %.2f%%，滑动帧 %.2f%% → %.2f%%。这是对官方原域的保持能力检查，不是 data_self 上的适配效果，也不是体态条件模型评测。"%(baseline["fid"]["mean"],adapted["fid"]["mean"],baseline["top3"]["mean"],adapted["top3"]["mean"],baseline["local_jerk_rms_mps3"]["mean"],adapted["local_jerk_rms_mps3"]["mean"],baseline["bone_cv_percent"]["mean"],adapted["bone_cv_percent"]["mean"],baseline["sliding_frame_percent"]["mean"],adapted["sliding_frame_percent"]["mean"]))
    text += ["", "## FlowMotion 原生核验", "",
             "作者原生每轮生成 1,024 条、GT 为 4,640 条，20 次重复；TCFM、Euler 步长 0.01、引导 2.5、种子 10。其 FID 样本数与主表不同，所以单列，不据此作严格同样本量排名。", "",
             "| 指标 | FlowMotion HumanML3D 实测 |", "| --- | --- |"]
    for key,label in METRICS:
        text.append("| %s | %s |"%(label,cell(flow["vald"].get(key))))
    text += ["", "论文提供的 FlowMotion Jitter 参考值为 39.522；本次为 %s。语义匹配与检索表现未达到预期，因此不能宣称完整复现了 FlowMotion 的论文结果。其公开代码在生成样本中将已补齐令牌数组的长度作为 `cap_len`，而真实动作使用有效句长。本轮保留作者原生代码，不用修订后的数值替换原结果。"%cell(flow["vald"]["flow_published_padded"]), ""]
    control=[row for row in text_control["rows"] if row["model"]=="SALAD"]
    before,after=[{key:float(np.mean([row[mode][key] for row in control])) for key in ["matching","top3"]} for mode in ["valid","forced_padded"]]
    text += ["针对句长的单因素对照：对已保存 SALAD 的首、中、末3批（共96条），保持文字令牌、动作和评估器不变，只把有效句长改成统一补齐长度22，匹配距离 %.4f → %.4f，R@3 %.4f → %.4f；动作嵌入逐值不变。这证明句长处理会影响该评估器的语义分数，但这不是 FlowMotion 生成动作上的修正实验，不能量化其完整影响，更不能解释仅依赖动作嵌入的 FID 差异。证据见 `evidence/text_length_control.json`。"%(before["matching"],after["matching"],before["top3"],after["top3"]), "",
             "FlowMotion 原生 FID／检索／匹配分数来自日志，每轮保留 4 位小数；上表由这些逐轮数值重算。物理指标与 Jitter 由观测钩子保存，保留完整精度。公开仓库未找到 KIT 权重，所以 KIT 的 FlowMotion 模型尚未核验。", "",
             "![官方指标核验](03_official_jitter_check.png)", "", "## 两套指标的定义", "",
             "**FlowMotion 兼容 Jitter**：世界坐标三阶差分乘 20³，先取三维模长，再对帧、关节和动作取均值。作者使用完整 196 帧，包括补齐区域；KIT 额外乘 0.00073，并硬编码恢复 22 关节。KIT 实际仅 21 关节，额外关节来自前三个旋转特征。该列为了与作者实现对照保留这些行为，不称为 KIT 的 SI 物理单位。", "",
             "**物理诊断**：仅有效帧，HumanML3D 为 22 关节、20帧/秒、米；KIT 为 21 关节、12.5帧/秒，毫米乘 0.001 转米。去根表示每帧减去骨盆位置。三阶导数 RMS 先对三维平方求和，再对帧和关节平均开方。骨长 CV 为每根骨骼时间标准差除以均值，再对骨骼平均。高频为去根去均值坐标的正交 DCT-II，保留频率≥3赫兹的能量，报告三维向量 RMS。", "",
             "**滑动**：脚尖相邻两帧均不高于 5厘米，且水平速度大于 0.2米/秒；任一脚触发的时间间隔占比。HumanML3D 脚尖为10／11，KIT为15／20。它是高度代理，完整测试集含复杂动作，不能当作真实接触标签或只针对平地行走的认证。真实动作也可能触发较高比例。", "",
             "SALAD 的作者评估把有效长度外的归一化输出置零，MDM／FlowMotion 原生评估保留模型输出；兼容 Jitter 因而受各自补齐行为影响。所有结果还保存有效帧 Jitter；MDM 额外保存统一置零补齐后的 Jitter。有效帧物理指标不受补齐边界影响。动作之间没有拼接差分。", "",
             "| 数据集／模型 | 原生全帧兼容 Jitter | 统一置零补齐 | 仅有效帧、实际关节 |",
             "| --- | --- | --- | --- |"]
    for ds,name in [("t2m","HumanML3D"),("kit","KIT-ML")]:
        for model,m in models[ds].items():
            text.append("| %s／%s | %s | %s | %s |"%(name,LABELS[model],cell(m["flow_published_padded"]),
                         cell(m.get("flow_zero_normalized_padding",m["flow_published_padded"])),cell(m["flow_valid_actual_joints"])))
    text += ["", "上表有效帧列仍使用作者默认20帧/秒及KIT的0.00073缩放，仅用于核查指标口径。KIT有效帧列还移除了虚构的第22关节，变化不能全部归因于补齐。物理时间尺度请以主表诊断指标为准。", "",
             "## 重复、验证与限制", "",
             "每条动作先独立计算，再在一轮内等权平均，最后在重复间取均值。±表示 1.96×重复间总体标准差/√重复数，沿用上游口径，只描述固定权重的采样变化。3 次重复不足以证明稳定的总体优越性；它们不是独立训练种子，也不是独立受试者重复。未计算多模态分数或将生成动作与 GT 逐帧比较的 MPJPE。", "",
             "主表 GT 的 FID=0 为同一批真实动作嵌入与自身的恒等对照，不是两个独立随机GT子集之间的FID；FlowMotion原生流程的随机GT参考另按其代码执行。", "",
             "独立验证检查全部逐片段覆盖及聚合；每模型每轮抽取首、中、末批各3条，以双精度独立恢复、逐级差分、显式余弦矩阵重新检查三阶导数与高频；另独立检查骨长、滑动和特征值法 FID。原 SALAD 与全参数微调全部435批输入逐值配对；MDM 输入缓存通过保存的 SHA-256 绑定。具体误差见 `evidence/verification.json`。抽查不等于全部动作逐值重生成。", "",
             "ViMoGen 现有权重使用不同的276维表示和身体模型；现有工程包含其他任务的用户改动，尚未验证官方21关节／251维KIT及HumanML评估转换。表中保留未测，不填论文数值，也未为完成表格而修改其骨架。LoRA 无对应权重，本轮未测试。", "",
             "## 既有 SALAD 语义复现", "",
             "以下为此前完成的正式20次评测，单独保留溯源，未和本轮物理采样拼成同一次实验。", "",
             "| 数据集 | 既有实测 FID | 论文 FID | 既有实测 R@3 | 论文 R@3 |", "| --- | --- | --- | --- | --- |"]
    for ds,label in [("t2m","HumanML3D"),("kit","KIT-ML")]:
        m=archived[ds]["metrics"]
        text.append("| %s | %s | %.3f | %s | %.3f |"%(label,cell(m["fid"]),m["fid"]["paper_mean"],cell(m["top3"]),m["top3"]["paper_mean"]))
    text += ["", "## 交付文件", "", "- `REPORT_zh.md`：本报告；三个 PNG 为本轮图表。", "- `main_summary.csv`、`sampling_repetitions.csv`、`results.json`：聚合值与逐轮结果。", "- `flow_native_summary.json`：原生 FlowMotion 的物理与语义核验。", "- `evidence/`：权重和源码绑定、状态、逐轮结果、独立验证，以及既有正式语义评测的来源。", "- 对应源码为仓库 `reproduction/official_jitter_v1/`；全量动作、模型权重和逐片段缓存仅服务器保存，不上传 GitHub。", ""]
    (output/"REPORT_zh.md").write_text("\n".join(text),encoding="utf-8")
    save_json(output/"figure_contract.json",dict(
        core_conclusion="Temporal jitter alone does not establish correct geometry or foot contact; datasets and metric conventions are separate.",
        evidence_chain={"01":"HumanML3D full valid-frame diagnostics", "02":"KIT-ML full valid-frame diagnostics", "03":"GT compatibility and separate native FlowMotion check"},
        archetype="quantitative grid",backend="python",export="PNG only, 300dpi, report figures, not a journal submission",
        source_data="main_summary.csv; evidence/*/repetitions.json",
        interval="1.96*population_sd/sqrt(sampling_repeats); not independent training seeds",
        exclusions="ViMoGen and KIT full fine-tuning not plotted because no verified compatible results; omissions explicit in report; FlowMotion separate native protocol",
        transformations="None on motion; no smoothing, resampling, bone correction, IK, frame concatenation or visual image retouching"))
    save_json(output/"file_manifest.json",{str(p.relative_to(output)):sha256(p) for p in output.rglob("*")
              if p.is_file() and p.name not in ["file_manifest.json","FINAL_VERIFICATION.json","git_publication.json"]})
    print(json.dumps(dict(status="complete",files=len(list(output.rglob("*"))),source_rows=len(records))),flush=True)


def preview(root):
    output=root/"draft_previews"
    output.mkdir(parents=True,exist_ok=True)
    configure_font()
    for dataset in ["t2m","kit"]:
        metrics=read(root/("salad_"+dataset)/"summary.json")["metrics"]
        for name,model in ([("salad_ft_t2m","SALAD_full_finetune"),("mdm_bert50_t2m","MDM_BERT_50")]
                           if dataset=="t2m" else [("mdm_clip1000_kit","MDM_CLIP_1000")]):
            summary=root/name/"summary.json"
            if summary.exists() and read(root/name/"status.json")["status"]=="complete":
                result=read(summary)["metrics"]
                metrics[model]=result[model] if model.startswith("SALAD") else result
        physical_figure(metrics,dataset,output/("preview_"+dataset+".png"))
    flow_directory=root/"flow_humanml_attempt3"
    if read(flow_directory/"status.json")["status"]=="complete":
        flow,_=flow_results(flow_directory)
        save_json(output/"flow_native_summary.json",flow)
        gates={dataset:read(root/("gt_"+dataset)/"summary.json") for dataset in ["t2m","kit"]}
        compatibility_figure(gates,flow,output/"preview_compatibility.png")
    print(json.dumps(dict(status="draft_preview_only",output=str(output))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--preview",action="store_true",help="Render completed SALAD results for layout QA; no final report")
    args=parser.parse_args()
    if args.preview:
        preview(args.root.resolve())
    else:
        build(args)
