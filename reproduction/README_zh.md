# SALAD 复现入口

范围为作者预训练权重的 HumanML3D、KIT-ML 评估、训练短测和四类编辑。完整从零训练与主观用户实验不在本次范围内。

## 环境与资产

服务器工作目录为 `/root/autodl-tmp/SALAD_1016`，独立解释器为 `/root/autodl-tmp/envs/salad_1016/bin/python`。下列示例在仓库根目录运行，先激活该环境。

```bash
python reproduction/download_assets.py --dataset t2m
python reproduction/download_assets.py --dataset kit
python reproduction/download_hf_assets.py --asset clip
python reproduction/download_hf_assets.py --asset kit
```

作者权重脚本使用原 README 所指向网盘的文件编号。CLIP 固定版本，KIT 镜像固定版本及发布摘要；KIT 三划分已与官方网盘内容核对。镜像动作及文本原始压缩包保留，解压至 `dataset/kit-ml/`。HumanML3D 和 GloVe 使用服务器既有官方基准的只读链接，排除体态项目的加工数据。

KIT 发布包有 2 个缺失训练动作、6 个含断行描述的训练文件。审计只接受编号、问题类型及原文本摘要均符合的已知例外；验证、测试划分出现同样问题仍失败。原文件、划分与作者加载行为不修改。KIT 没有带时间段的描述，因此原加载器固定 20 帧/秒裁剪问题不影响本次 KIT 基准。

```bash
python reproduction/test_regressions.py
python reproduction/audit_assets.py --dataset t2m --output reproduction/artifacts/assets_t2m.json
python reproduction/audit_assets.py --dataset kit --output reproduction/artifacts/assets_kit.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/preflight.py --dataset t2m --output reproduction/artifacts/preflight_t2m.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/preflight.py --dataset kit --output reproduction/artifacts/preflight_kit.json
```

资产审计必须通过后才能正式评估。正式运行绑定源码提交、协议、模型、归一化、词向量、CLIP 和数据摘要。所有结果位于被 Git 忽略的 `reproduction/artifacts/`，模型和数据不推入代码仓库。

## 训练短测

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/training_smoke.py --dataset t2m --output reproduction/artifacts/training_t2m
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/training_smoke.py --dataset kit --output reproduction/artifacts/training_kit
```

默认在官方训练划分上，以批量 4 分别更新新初始化 VAE 和去噪网络 20 步。去噪网络使用作者预训练且冻结的 VAE、固定 CLIP。检查有限损失与梯度、参数变化、冻结状态、模型/优化器/调度器保存恢复一致性。短测不能证明完整训练收敛；原训练检查点缺失随机状态和数据迭代位置，不能据此保证轨迹精确续训。

## 正式评估

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/run_evaluation.py --dataset t2m --cal-mm --output reproduction/artifacts/eval_t2m
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/run_evaluation.py --dataset kit --cal-mm --output reproduction/artifacts/eval_kit
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/run_evaluation.py --dataset t2m --kind vae --output reproduction/artifacts/eval_t2m_vae
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/run_evaluation.py --dataset kit --kind vae --output reproduction/artifacts/eval_kit_vae
```

保留作者评估口径：官方测试划分、批量 32、随机抽取描述与片段、打乱、丢弃末批、4 个工作进程、种子 1234、50 步 DDIM、引导强度 7.5、20 次重复。多模态使用前三批共 96 个描述，每描述生成 30 次。`--cal-mm` 会同时计算主质量指标，未计算多模态时输出 `null`。

每轮保存未舍入指标、耗时与随机状态，重复相同命令可继续；源码/资产/协议不匹配时拒绝续跑。首轮用独立公式复算真实嵌入上的检索计数、匹配距离、统计量、FID、多样性及多模态，不消耗额外全局随机数。区间沿用作者的 `1.96 × 总体标准差 / √20`，区间重叠只作描述，不作为统计等价性证明。

Git 维护方式为本地提交及 `git push origin main`，服务器 `git pull --ff-only`。正式任务运行期间固定源码，修改实现后使用新结果目录。

## 本地结果同步

工作站配置好 `motion-data` 的 SSH 别名后，在本仓库运行：

```powershell
python reproduction/sync_results.py sync
python reproduction/test_sync_results.py
python reproduction/sync_results.py watch --publish
```

同步器只读取服务器证据和视频，校验跨端 SHA-256、冻结的源码与协议、测试覆盖数、指标完整性，并独立复算逐次均值和区间。网页入口为 `reproduction/artifacts/index.html`。工作站需要 NumPy；持续监控的文件锁采用 Windows 接口。

`watch --publish` 每 60 秒同步。四组均完成 20 次且验证通过后，生成 `reproduction/RESULTS_zh.md` 和 `reproduction/results/`，提交并推送 GitHub，再快进拉取到服务器并核对提交号。网络故障会重试；证据异常或任务提前停止则记录失败并退出。已有本地改动或生成报告被手工改动时停止自动发布；不会覆盖这些改动。

证据文件按服务器原始字节提交，避免 Windows 换行转换破坏实验绑定摘要。状态与发布收据分别位于 `reproduction/artifacts/evidence/watch_status.json` 和 `publication.json`。持续监控依赖本地工作站与 SSH 连接保持可用，不会唤醒聊天。

本次服务器拉取优先使用仅绑定回环的代理桥 `127.0.0.1:18979`，不可用时尝试直连；两者均有超时，不修改服务器的持久 Git 代理配置。代理桥需要保持运行。全部评估完成并通过检查前，不在服务器执行拉取。
