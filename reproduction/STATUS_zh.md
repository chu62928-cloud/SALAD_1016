# SALAD 复现记录

## 范围与版本

- 用户确认：本次完成预训练权重评估与训练短测，不启动完整从零训练。
- 用户仓库：https://github.com/chu62928-cloud/SALAD_1016
- 初始提交：`6487e98957d1643ac70d77cecdab21f3cda62e2b`。
- 论文：https://openaccess.thecvf.com/content/CVPR2025/html/Hong_SALAD_Skeleton-aware_Latent_Diffusion_for_Text-driven_Motion_Generation_and_Editing_CVPR_2025_paper.html
- 服务器目录：`/root/autodl-tmp/SALAD_1016`。
- 复现采用官方 HumanML3D / KIT-ML 数据划分，与另一个体态项目的划分、测试集和结果分开。

## 已核实

- 2026-10-16：服务器与本地均已克隆，初始提交一致，工作区干净。
- 本地 Git 凭据的推送预检通过；服务器 HTTPS 尚无推送凭据。后续使用本地提交并推送、服务器只拉取的方式同步，不复制凭据。
- 服务器已有 HumanML3D 原始数据及评估词向量，可在核验后只读链接。
- 初次检查磁盘剩余约 27 GB；另有训练任务运行，不控制其进程。

## 待验证

- 原文及补充材料中的准确数值、指标定义和超参数。
- 独立 Python 3.9 / PyTorch 1.13.1+cu117 环境。
- 官方模型、评估器、词向量和数据完整性与校验值。
- 模型加载、前向传播、采样、评估排序、归一化、随机性及重复实验。
- HumanML3D 和 KIT-ML 的完整 20 次评估、动作编辑及从零训练链路。

运行通过与论文数值复现通过分别判断。未计算的指标不得写成零或通过。
