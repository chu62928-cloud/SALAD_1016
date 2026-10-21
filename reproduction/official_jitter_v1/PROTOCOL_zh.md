# SALAD 官方测试集双口径评测

日期：2026-10-21。HumanML3D 与 KIT-ML 分开报告，不合并绝对 Jitter。

## 评测顺序

1. 固定 FlowMotion 官方仓库 `mccuba/FlowMotion` 的提交 `fe4021b9170545b4e9ddc6d13d309bfa0988c83f`，追踪 `eval_humanml.py` 和数据恢复、加载代码。
2. 按上游原样函数分别完成两套 GT 的 20 次评测。每次 HumanML3D 4,640 条、KIT-ML 704 条；种子 10，批量 32，8 个加载进程，打乱、丢弃末批。
3. GT 均值与论文给定目标的相对偏差小于 1% 后，推进模型评测。该阈值仅为预先采用的接近性检查，不是统计等价检验，不据此断言完整论文复现成功。
4. 原 SALAD 两套官方权重各做 3 次完整测试集重复；HumanML3D 现有全参数微调权重也做 3 次。固定 50 步 DDIM、引导 7.5、种子 1234、批量 32、4 个加载进程，保留作者测试加载器。没有重新训练。
5. 新 FID、R-Precision 与同一次生成结果计算。既有的 20 次正式 FID/R 另列来源，不能和新物理指标冒充同次重复。没有多模态采样，不能报告本轮多模态分数。
6. FlowMotion 公开 HumanML3D 权重采用作者原生 `wo_mm`、20 次、种子 10、TCFM、Euler 步长 0.01、引导 2.5，保留作者生成数量上限 1,000，实际按整批生成 1,024。GT 仍为完整测试加载器。其 FID 的样本量与 SALAD 不同，不能作为严格同数量排名。
7. 可用的 MDM HumanML3D 权重明确标为 BERT／50 步版，复用 SALAD 实际缓存的同一批次文本、长度、GT，3 次完整重复，采用原生 50 步祖先采样、引导 2.5、EMA。不能当成 FlowMotion 论文中的 CLIP／1,000 步版。
8. MDM 的 KIT 官方发布包从 README 链接取得，经 ZIP 全文件完整性检查；服务器直连超时后通过本机网络流式中转，权重未落地本机。压缩包 SHA-256 为 `a7b32c4cd0714840c4fc5b5b28913a605352dd9493af467a626b7e1f24038518`，模型为 `kit_trans_enc_512/model000400000.pt`，SHA-256 为 `f1593a8fd74b5016d8b3ab0997a142b655db237ded0961ef42204f64cb46255a`。原生 CLIP／1,000 步祖先采样、引导2.5，逐批复用 SALAD KIT 文本、长度与GT，3次完整重复；这是独立版本，不与 HumanML3D 的 BERT／50步混称。

## 第一套：FlowMotion 兼容 Jitter

公式为 `mean(norm(Delta^3 p * 20^3))`，对每条动作的时间和关节先取模长均值，再对动作等权平均。它是三阶导数模长均值，不是二阶加速度，也不是平方均值的平方根。

- 反归一化、恢复世界坐标，保留根运动。
- 作者代码对两套数据均用 `recover_from_ric(..., 22)` 和默认 20 帧/秒。
- 读取全部补齐后的 196 帧，没有按 `m_lens` 裁剪；每条动作仍单独差分，不拼接动作。
- HumanML3D 不缩放；KIT-ML 乘 `0.00073`。该系数不等于物理诊断中的毫米转米。
- KIT 实际是 21 关节，硬编码 22 会把旋转通道的前三个数读取为额外关节。兼容列保留此行为以便对照，不能解释为正确人体几何。
- SALAD 的正式评估会把有效长度外的归一化输出置零；MDM／FlowMotion 原生评估保留模型输出。本轮遵守各模型原生行为，并为 MDM 额外计算统一零补齐版本。应结合有效帧列判断补齐的影响。

因此该列标为“FlowMotion 兼容 Jitter”，不把 KIT 的数值直接称为真实 SI 单位抖动。

## 第二套：物理诊断

全部只使用有效动作帧，真实关节数 22／21，世界 Y 轴向上；HumanML3D 20 帧/秒、米制，KIT-ML 12.5 帧/秒、原毫米坐标乘 `0.001`。不改变骨长，不平滑，不贴地，不做逆运动学。

| 指标 | 定义 |
| --- | --- |
| 去根 Jerk RMS，米/秒³ | 减去每帧骨盆位置，三阶差分乘原生帧率三次方，先对三维分量平方求和，再对关节与时间平均开方 |
| 世界 Jerk RMS，米/秒³ | 相同公式，保留根运动，作为附加列 |
| 骨长 CV，% | 每根骨骼有效帧长度的总体标准差／平均长度，再对非根骨骼平均乘 100 |
| ≥3 Hz 高频 RMS，毫米 | 去根坐标去均值后作正交 DCT-II，频率 `k*fps/(2*T)`；保留 ≥3 Hz 能量。主列为向量 RMS，坐标 RMS 附列，二者相差根号 3 |
| 滑动帧，% | 脚尖在相邻两帧均不高于 5 厘米，水平速度大于 0.2 米/秒；任一脚满足的时间间隔比例 |

脚尖索引 HumanML3D 为 10、11，KIT 为 15、20。滑动是地面高度代理，不是实测接触标签；完整官方测试集有复杂动作，不能直接套用此前 `data_self` 平地行走结论。

## 汇总和边界

先逐条动作计算，再对一轮动作等权平均，最后对重复取均值。区间沿用上游 `1.96 * 总体标准差 / sqrt(重复数)`。这些是同一权重的采样重复，不是独立训练模型，也不是独立受试者重复；3 次区间只描述本次采样变动。

GT 对自身的 FID=0 是同一批嵌入的恒等对照，不能当作两个独立 GT 随机子集的 FID。生成不与真实动作逐帧对齐，因此不报告生成对 GT 的 MPJPE 或加速度误差。

现有微调是项目数据上的全参数微调，不是 LoRA，也没有 KIT-ML 架构的对应权重。官方 FlowMotion 仓库只发布了 HumanML3D 权重。其他模型缺少权重或没有完成骨架／特征适配核验时，明确写“未测”或“不适用”，不拿论文值补作实测。

## 文件

- `audit_gt.py`：原样执行上游 GT 加载与 Jitter，保留补齐和有效帧对照。
- `tests_metrics.py`：解析信号检查。
- `run_salad.py`：原 SALAD 与现有全参数微调完整评测。
- `run_flow.py`：原生 FlowMotion 评测和只读指标观察。
- `run_mdm.py`：注明版本的 MDM 对照。
- `run_mdm_kit.py`：KIT 的原生 CLIP／1,000步对照，只用上游参数解析器默认值补齐旧版配置缺失项，不改变已记录训练配置。
- `verify.py`：逐条覆盖与聚合复算、双精度独立恢复与导数／频谱／骨长／滑动抽查、特征值法FID、全部缓存输入绑定核验。
- `report.py`：中文报告、分数据集汇总表和PNG图；图表是报告用途，按用户要求只输出PNG，不输出SVG／PDF／TIFF，静态期刊检查的矢量格式要求不适用。
- `audit_text_lengths.py`：首／中／末3批GT与SALAD缓存的句长单因素对照，CPU评估；只改变有效句长，不改文字令牌／动作／评估器，不等同FlowMotion生成结果修订，也不能解释FID差异。
- `finalize.py`：公开文件摘要、PNG解码与各面板非空检查、运行时版本、FlowMotion源码未改检查。最终PNG仍需目视检查。
- `reproduction/artifacts/official_jitter_v1/`：服务器完整缓存、逐片段 CSV、绑定、逐次结果和状态。
- 公开结果只包含匿名聚合值、PNG、报告和验证记录；原数据、权重和生成缓存不推送 GitHub。

## 重跑入口

在 SALAD 仓库根目录运行，`FLOW_ROOT` 为上述固定提交的 FlowMotion 检出目录；先安装原 SALAD 环境并准备官方数据与评估器。所有运行目录必须为新目录，文件锁及绑定检查阻止覆盖旧实验。

```bash
python reproduction/official_jitter_v1/audit_gt.py --flow-root "$FLOW_ROOT" --dataset t2m --output "$RESULT_ROOT/gt_t2m"
python reproduction/official_jitter_v1/audit_gt.py --flow-root "$FLOW_ROOT" --dataset kit --output "$RESULT_ROOT/gt_kit"
python reproduction/official_jitter_v1/run_salad.py --flow-root "$FLOW_ROOT" --dataset t2m --output "$RESULT_ROOT/salad_t2m"
python reproduction/official_jitter_v1/run_salad.py --flow-root "$FLOW_ROOT" --dataset kit --output "$RESULT_ROOT/salad_kit"
```

全参数微调另加 `--model SALAD_full_finetune --checkpoint <既有权重>`，输出 `salad_ft_t2m`；不更改原始权重。FlowMotion 使用单独隔离环境运行 `run_flow.py`，参数为 `--flow-root`、`--output`。MDM 两入口均需 `--mdm-root`、`--flow-root`、`--salad-run`、`--checkpoint`、`--output`，所需版本与配置见上文，不能互换。

全部评测结束后运行 `verify.py --root "$RESULT_ROOT"`，再运行 `report.py --root "$RESULT_ROOT"`。后者输出 `public/`，GitHub记录该目录的匿名公开副本。报告另读取既有正式20次SALAD语义评测 `../eval_t2m_v2/summary.json` 与 `../eval_kit_v3/summary.json`，二者仅为历史对照，不混入本轮统计。
