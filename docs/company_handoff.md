# 公司单卡 A800：一次交接完整实验

用户确认一张 A800、80 GB；代码已补齐，真实 GPU 执行待服务器验证。
公司服务器完成四组训练与五端点评估，本机/实验室负责代码和 CPU 检查。
不用先连接实验室取旧数据或旧适配器。

## 直接获取

~~~bash
git clone https://github.com/YiFanWangSCU/skill-annealing-pilot.git
cd skill-annealing-pilot
git rev-parse HEAD
~~~

此公开仓库的 HTTPS clone 不需要 GitHub 登录、token 或 SSH 密钥。
已 clone 本仓库时先保留本地改动，再 git pull --ff-only。
若之前使用私有版本，请在新目录 clone 本公开仓库并保留旧目录和产物，
不要把无共同历史的两个仓库直接混合。服务器仍需能够访问 GitHub。
公司网络和 AI 工具的使用须遵守公司政策。

## 交给 Claude Code 的任务

把下面一段交给服务器上的 Claude Code，并填入本地模型目录和已分配 GPU 索引。
这授权一次文档定义的有界探索，不是任意 GPU 训练。

> 请阅读 CLAUDE.md、AGENTS.md 和 RUN_EXPERIMENT.md，按
> docs/rule_exposure_protocol.md 完成这一轮规则级曝光探索。
> 使用公司提供的单张 A800 80GB、我填写的本地模型目录与已分配 GPU 索引。
> 先运行 CPU 测试和模拟检查，在独立环境确认依赖后执行完整 run --execute，
> 连续完成四组训练、基础模型加四适配器评估和统计报告，不必逐阶段再问我。
> 遵守配置中的预算，不改标签和指标，不自动追加训练或启动旧 15-run。
> 失败保留产物，检查并报告具体原因；只对已解决的环境失败显式重试。
> 不动共享环境或他人进程，不上传公司数据、模型、密钥、原始日志和生成。
> 完成后给我脱敏汇总、运行版本和本地报告位置。

本地 MODEL_DIR 与 GPU_INDEX 是唯一必须由服务器侧提供的信息；
没有默认模型下载，也不能默认 GPU 0 属于本项目。
环境、依赖、完整运行和恢复命令都集中在 [RUN_EXPERIMENT.md](../RUN_EXPERIMENT.md)。

## GitHub 无法访问时

在允许联网且已有本公开仓库的电脑上：

~~~bash
git status --short
git rev-parse main
git bundle create skill-annealing-pilot.bundle main
git bundle verify skill-annealing-pilot.bundle
~~~

仅打包此分支，不用 --all。Git bundle 不含未提交文件或被忽略的数据/权重。
使用公司批准的 scp/SFTP 等方式传输；不要在仓库记录地址、账号或认证信息。

~~~bash
git clone --branch main skill-annealing-pilot.bundle skill-annealing-pilot
cd skill-annealing-pilot
git rev-parse HEAD
~~~

核对两端 SHA。离线 clone 的 origin 指向 bundle，后续更新要传新 bundle，
或在网络允许后明确改为正常 GitHub origin；不能假定 git pull 自动访问 GitHub。
依赖另用公司镜像或批准的离线 wheel。

## 本包与结果边界

已包含数据生成、配对训练、tokenizer 预检、推理、统计、失败恢复及文档。
数据是本地生成的模拟结构化事实；不是公司真实订单或正式论文验证集。
CPU 模拟的满分来自规则引擎，不能说成训练模型满分。A800 未实际试跑。
论文后续问题见 remaining_experiments.md；旧研究结论不变。
