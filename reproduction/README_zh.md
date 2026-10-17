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

## 动作示例与物理诊断

```bash
python reproduction/test_physics_metrics.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python reproduction/sample_physics.py --protocol reproduction/physics_protocol.json --output reproduction/artifacts/physics_v1 --render
```

本次结果见 [物理诊断报告](PHYSICS_zh.md)，小体积证据保存在 `reproduction/results/physics_v1/`；视频和原始动作数组位于服务器 `reproduction/artifacts/physics_v1_20261016/`。工作站运行 `python reproduction/remote_physics.py fetch` 可校验并同步证据和视频，再运行 `python reproduction/report_physics.py` 生成本地网页 `reproduction/artifacts/physics_v1_20261016/index.html`。同步不会下载模型或动作数组；报告会从逐段指标独立复算均值和配对重采样区间。

独立冻结的 `physics_protocol.json` 定义六种动作、三个种子及物理指标。先生成 18 段六秒动作，再对 128 段官方验证集描述各生成三次，与真实参考动作作相同长度的物理指标对照。沿用官方评估加载器选描述和裁剪，按编号摘要固定抽样，不选择展示结果。输出保留原始归一化特征、世界坐标、逐动作指标、配对汇总及原编号分组重采样区间。镜像编号与其原编号分在同一组。

滑步使用原始世界坐标与固定地面 `y=0`：脚尖在相邻两帧都低于 5 厘米时统计水平速度，超过 0.2 米/秒记为滑步；同时报告低位接触覆盖率、条件滑步比例，以及三档高度/速度阈值敏感性。无低位接触时速度为 `null`，不写零。脚部穿地、腾空、骨长变异及根相对加速度/加加速度是运动学诊断，不是力学仿真、网格碰撞或统一合格门槛；跳跃腾空本身不是错误。这些指标是补充诊断，不是原论文结果表。

视频包含双视角和脚尖曲线。相机水平跟随根节点，0.5 米网格固定于世界坐标；原动作不贴地、不平滑、不锁脚。主要基准评估运行期间，可将新增脚本复制到被忽略的独立目录并设置 `SALAD_ROOT`，不更新正在运行的核心源码。运行绑定记录核心提交、新增代码摘要及资产摘要。

## 人体网格

网格直接使用上述同一批 18 段动作，不重新生成。采用已有中性 SMPL-H 模板、姿态修正形变和关节驱动的线性混合蒙皮，保留原关节位置及骨长变化。它是估计表面，手指和绕骨轴旋转没有独立生成；不能将关节附着精度当作表面重建精度。物理指标继续使用原始关节。

服务器渲染需要 NumPy、SciPy、Pillow、pyrender、trimesh、FFmpeg 和可用的 EGL，以及已获授权的人体模型资产。本次使用既有 `mdm5090` 环境，不修改 SALAD 的评估环境。在 Linux 上可指定自己的模型路径：

```bash
python reproduction/render_mesh.py --source reproduction/artifacts/physics_v1_20261016 --body /path/to/smplh/neutral/model.npz --font /path/to/chinese-font.ttc
```

工作站通过已配置的 SSH 别名同步：

```powershell
python reproduction/remote_mesh.py status
python reproduction/remote_mesh.py fetch
```

本地播放页为 `reproduction/artifacts/physics_v1_20261016/mesh_v1/index.html`，网格记录见 [MESH_zh.md](MESH_zh.md)。原始数组及网格缓存只留服务器，Git 仅保存渲染代码和小型核验证据。相机共用固定尺度及高度，仅跟随水平根位移；地面为固定的 `y=0`，网格间距 0.5 米。没有逆运动学拟合、贴地、平滑、锁脚或骨长修正。

## 位置、旋转与蒙皮专项检查

此前关节驱动表面不等同于作者的完整显示流程。新增 [专项报告](RECOVERY_SKINNING_zh.md) 与 [四组同步对照](artifacts/physics_v1_20261016/recovery_v2/index.html)，分别检查无蒙皮位置骨架、6D 旋转恢复、标准固定骨长 SMPL-H 蒙皮、作者 BVH 转换，并提供未修改官方绘图函数生成的骨架视频。

本次只读复查原 18 段生成动作和 128 段真实动作，不重新采样或训练。生成位置骨长变异平均 3.477%，旋转分支固定骨长但未改善加加速度；真实数据另有七段整体朝向轨迹不一致，保留在报告中，不据此声称旋转分支全部正确或自然度已经修复。

查看和同步已有诊断使用 `python reproduction/remote_recovery_v2.py status` 与 `fetch`。报告入口为 `python reproduction/report_recovery_v2.py`。播放复核脚本为 `node reproduction/qa_recovery_v2.cjs`，需要 Playwright 与 Edge；可通过 `PLAYWRIGHT_MODULE` 指定 Playwright 模块位置。已有 `audit`、`render`、`official`、`verify` 阶段均已结束，不重复启动。
