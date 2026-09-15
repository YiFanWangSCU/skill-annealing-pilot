# 服务器环境说明

**完整实验直接从 [RUN_EXPERIMENT.md](RUN_EXPERIMENT.md) 开始。**
本文件只解释环境检查，不能把两步检查当完整实验或模型结果。

## 环境

在公司允许的独立环境中使用 Python 3.12。先安装与驱动匹配的 PyTorch 和
torchvision，再安装 configs/server_requirements.txt，执行 pip check。
默认 pytest 只用 CPU、合成输入和模拟子进程，不访问真实 GPU/数据或网络。
依赖安装可能联网；公司受限时用批准的镜像或离线 wheel。

模型必须是本地完整 Qwen3.5-4B dense 目录，包含配置、tokenizer/processor 与
safetensors 分片。代码检查类型并记录哈希，不能仅凭目录名认证型号。
不存在自动下载、自动选卡、自动替代模型。不要覆盖已有 GPU 调度分配。

## 可选的旧两步环境检查

只有需要单独排查训练环境时才使用 scripts.server_smoke。
它独立于新实验，不是新实验必须先跑的第五组训练。
用 --help 查看参数；无 --execute 时不训练，加 --execute 仅运行两个 updates。
检查只证明两步训练和适配器保存，不包含模型效果验证。
新实验的 run --execute 已有自己的 tokenizer 预检、训练记录及完整推理阶段。

## 当前范围

公司目标是单卡 A800、80 GB；真实运行尚待服务器验证。
公开仓库默认 main 已包含完整执行包；原私有仓库保持不变。
无需 GitHub 登录的 HTTPS clone 与离线 Git bundle 操作见
[公司交接](docs/company_handoff.md)。权重、公司数据、密钥和原始日志不放 GitHub。
历史研究证据库不在本分支；不自动启动旧 ABCD 或 MBPP 方案。
