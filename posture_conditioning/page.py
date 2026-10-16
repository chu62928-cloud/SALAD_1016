"""Offline results and fixed mesh-comparison gallery; no application server."""
import csv
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    results=ROOT/'results/posture_v1'
    media=ROOT/'runs/posture_v1/media'
    videos=json.loads((media/'videos.json').read_text())['videos']
    with (results/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        metrics=list(csv.DictReader(f))
    labels={'original':'原 SALAD','fixed':'固定中性条件适配','paired':'正确同人配对','shuffled':'训练配对打乱'}
    table=''.join('<tr><th>'+labels[r['variant']]+'</th>'+''.join('<td>%.4f</td>'%float(r[k]) for k in
                  ['denoise_subject_mean','fid','r3'])+'<td>%.2f%%</td><td>%.2f%%</td></tr>'%(
                      float(r['bone_length_cv'])*100,float(r['skating_frame_ratio'])*100) for r in metrics)
    gallery=''.join('''<article><h2>%s</h2><p>%s</p><video controls playsinline preload="metadata"
      poster="%s" src="%s"></video><dl><dt>种子</dt><dd>%d</dd><dt>时长</dt><dd>%.2f 秒</dd></dl></article>'''%(
        html.escape(v['identity']),html.escape(v['text']),v['poster'],v['path'],v['seed'],v['frames']/20) for v in videos)
    decision=json.loads((results/'decision.json').read_text())
    conclusion=('已通过首轮预设门槛，后续种子结果见中文报告。' if decision['expand_training_seeds'] else
                '单种子对照未同时通过研究与质量门槛，已停止扩展训练。当前证据不足以确认个体静态体态带来可靠收益。')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>SALAD 静态体态实验结果</title><style>
    *{box-sizing:border-box}body{margin:0;color:#263733;background:#fff;font:15px/1.65 system-ui,sans-serif}
    main{max-width:1240px;padding:28px 24px;margin:auto}h1{font-size:26px;margin:0 0 12px}h2{font-size:18px;margin:18px 0 4px}
    p{max-width:980px;margin:8px 0 18px}a{color:#28687a}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}
    th,td{text-align:right;padding:10px;border-bottom:1px solid #d8e0dd}th:first-child{text-align:left}thead{background:#eef3f1}
    .scroll{overflow:auto;margin:20px 0}section{padding:20px 0;border-top:1px solid #d8e0dd}.figure{display:block;width:100%;max-width:1000px}
    article{padding:16px 0 26px;border-bottom:1px solid #d8e0dd}video{display:block;width:100%;aspect-ratio:1920/880;background:#eef1f0}
    dl{display:flex;gap:12px;font-size:13px;color:#52625b;margin:8px 0}dt,dd{margin:0}article p{font-size:14px;margin:4px 0 10px}
    @media(max-width:600px){main{padding:20px 14px}h1{font-size:22px}video{min-height:180px}table{font-size:13px}}
    </style><main><h1>SALAD 静态体态条件实验</h1><p>CONCLUSION</p>
    <p>来源内受试者划分：训练 256 人，验证 56 人，测试 57 人。静态输入仅骨盆、脊柱、下肢和单个肩带高低差。
    三组各训练 10000 步，另有原 SALAD 对照。<a href="../../../results/posture_v1/REPORT_zh.md">中文报告</a></p>
    <div class="scroll"><table><thead><tr><th>模型</th><th>去噪误差 ↓</th><th>FID ↓</th><th>R3 ↑</th><th>骨长变异 ↓</th><th>平地脚滑 ↓</th></tr></thead>
    <tbody>TABLE</tbody></table></div><section><h2>身份级对照与实际输入干预</h2>
    <img class="figure" src="../../../results/posture_v1/posture_results.png" alt="身份级对照与干预结果图"></section>
    <section><h2>注意力、门控与注入残差</h2><p>八个固定样例、五个去噪块、第 25 个去噪步骤的汇总。
    时间仅分成四个等长区间。头部与上肢标签属于输出动作潜变量区域，不是静态编码器输入，且潜变量区域存在重叠。
    注意力和门控不等于人体实际影响；注入残差范数使用潜变量单位，实际关节变化见上图。</p>
    <img class="figure" src="../../../results/posture_v1/attention_gate.png" alt="体态注意力、动作令牌门控及注入残差"></section>
    <section><h2>固定网格对照</h2><p>每来源按编号取前两位测试身份的首片段，共八段。三列依次为原 SALAD、固定中性条件适配、正确同人配对。
    采用原始关节驱动的估计表面，固定地面与同尺度相机；未平滑、贴地或锁脚。阶梯、斜坡片段的世界平面只作视觉参考。</p>GALLERY</section></main></html>'''
    (media/'index.html').write_text(text.replace('CONCLUSION',conclusion).replace('TABLE',table).replace('GALLERY',gallery),encoding='utf-8')


if __name__=='__main__':
    main()
