# SALAD 抖动分层诊断

独立事后诊断，不修改作者模型、原评价缓存、原训练权重或数据划分。
完整中文报告和同步骨架/网格页面位于
`reproduction/artifacts/jitter_attribution_v3/REPORT_zh.md` 与 `index.html`。
可公开的汇总与 PNG 副本另放于 `results/jitter_attribution_v3/`。

## 已确认的结果

- 官方 VAE 的确定性均值重建已引入骨长变异：原始域 2.778%，目标域 3.016%；后验采样约 3.944%/4.110%。原始真实骨架几乎恒定骨长。
- 目标域原 SALAD、数据微调、对应体态条件的 Jerk 为 97.57、87.11、85.05 米/秒³，骨长 CV 为 5.290%、5.159%、4.800%。
- 同一原 SALAD 结果仅固定骨长后 Jerk 为 105.41，独立旋转通道 FK 为 174.32。固定骨长未解决方向抖动。
- 平地行走滑动帧比例为 41.70%、47.69%、51.22%，与旧正式汇总重放一致。全任务混合的接触指标不用于物理优劣结论。
- 对应/固定/打乱体态模型差异很小，不能把额外适配器训练的变化解释为个人体态控制成功。
- 本次发现解码步幅相关周期误差线索，但未用网络层或训练损失消融确认唯一根因，也未修复模型。

## 运行

服务器 SALAD 根目录执行，使用既有 SALAD 环境；渲染单独使用已有含标准蒙皮及离屏渲染依赖的环境。

```bash
python reproduction/jitter_attribution_v3/test_metrics.py
python reproduction/jitter_attribution_v3/run.py smoke
python reproduction/jitter_attribution_v3/run.py vae
python reproduction/jitter_attribution_v3/run.py generated
python reproduction/jitter_attribution_v3/run.py sampling
python reproduction/jitter_attribution_v3/summarize.py
python reproduction/jitter_attribution_v3/contacts.py
python reproduction/jitter_attribution_v3/render.py
python reproduction/jitter_attribution_v3/verify.py
python reproduction/jitter_attribution_v3/figures.py
python reproduction/jitter_attribution_v3/report.py
```

已完成的测量阶段拒绝覆盖；不要照抄命令重复启动完成任务。复跑应使用新版本输出目录并保留原始证据。
本地 `deliver.py` 只同步小型结果、汇总和媒体，核验 SHA256；完整逐骨长和导数时间序列留服务器。
本地 `qa.cjs` 使用已有浏览器工具检查 1440/390 两个视口的图片和视频。

## 冻结约束

`protocol.json` 定义指标与范围。VAE 128 原始域参考与 1,327 目标测试动作，五模型三生成重复，20 帧/秒。
先片段内平均重复，再人内平均片段，最后人等权；配对自助法 4,000 次。
生成动作没有逐帧对齐的真实动作，不报告生成对真实动作的 MPJPE 或加速度误差。
独立旋转 FK 不经过位置逆运动学；三恢复路径不做时间平滑或足部锁定。
网格为沿用已核验的几何旋转估计与标准中性 SMPL-H 蒙皮，不将 HumanML incoming-bone 旋转直接赋给同编号 SMPL 关节。
仅使用已训练的旧体态条件；新固定世界竖直条件尚无对应权重，没有混配或重训。

图像只输出 PNG，300 dpi、适合网页阅读的报告尺寸。通用科研图验证器的 SVG/PDF/TIFF 默认要求不适用于本次明确的 PNG 限制。
